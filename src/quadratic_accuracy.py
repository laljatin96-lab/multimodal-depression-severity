"""Nested regularized quadratic classification with the unchanged 49.28% language control."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import decide, score, summarize

OUTPUT = Path('results/quadratic_accuracy')
CONFIG = Path('configs/quadratic_accuracy.json')
PROTOCOL = 'training-only-regularized-quadratic-v1'
LANGUAGE_ROOT = Path('results/age_language_accuracy')


class QuadraticClassifier:
    """Shared dense covariance plus strongly pooled class-specific diagonal variance."""
    def __init__(self, candidate, settings):
        self.candidate, self.settings = dict(candidate), dict(settings)

    def fit(self, x, y):
        y = np.asarray(y)
        if y.shape != (len(x['semantic_av']),) or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('Five classes and real fitting participants required')
        if not np.isfinite(x['semantic_av']).all() or not np.isfinite(x['language']).all():
            raise ValueError('Invalid fitting features')
        y = y.astype(int); self.real_count = len(y); self.classes_ = np.arange(5)
        self.class_counts = np.bincount(y, minlength=5)
        self.base_scaler = StandardScaler().fit(x['semantic_av'])
        self.base_pca = PCA(n_components=self.candidate['components'], svd_solver='full').fit(self.base_scaler.transform(x['semantic_av']))
        self.language_scaler = StandardScaler().fit(x['language'])
        self.output_scaler = StandardScaler().fit(self.unscaled(x))
        z = self.transform(x)
        self.means_ = np.stack([z[y == c].mean(0) for c in self.classes_])
        residuals = z-self.means_[y]
        self.pooled_ = residuals.T @ residuals / len(y)
        diagonal = np.diag(self.pooled_)
        rho = self.candidate['pooled_diagonal_fraction']
        shared = (1-rho)*self.pooled_ + rho*np.diag(diagonal)
        alpha = self.candidate['class_variance_fraction']
        pseudo = self.settings['variance_prior_participants']
        covariances = []
        for c, n in enumerate(self.class_counts):
            # Only class-specific variances are free; cross-covariances are shared.
            variance = (np.square(residuals[y == c]).sum(0)+pseudo*diagonal)/(n+pseudo)
            covariance = (1-alpha)*shared+alpha*np.diag(variance)
            covariance += self.settings['variance_floor']*np.eye(z.shape[1])
            covariances.append(covariance)
        self.covariances_ = np.stack(covariances)
        self.cholesky_ = np.linalg.cholesky(self.covariances_)
        self.logdet_ = 2*np.log(np.diagonal(self.cholesky_, axis1=1, axis2=2)).sum(1)
        prior = self.class_counts.astype(float)**self.settings['prior_power']
        self.logprior_ = np.log(prior/prior.sum())
        return self

    def unscaled(self, x):
        return np.column_stack([self.base_pca.transform(self.base_scaler.transform(x['semantic_av'])),
                                self.language_scaler.transform(x['language'])])

    def transform(self, x):
        return self.output_scaler.transform(self.unscaled(x))

    def decision_function(self, x):
        from scipy.linalg import solve_triangular
        z = self.transform(x)
        distances = np.stack([np.square(solve_triangular(chol, (z-mean).T, lower=True)).sum(0)
                              for chol, mean in zip(self.cholesky_, self.means_)], axis=1)
        return self.logprior_[None, :]-.5*(self.logdet_[None, :]+distances)

    def predict_proba(self, x):
        from scipy.special import softmax
        p = softmax(self.decision_function(x), axis=1); decide(p, None)
        return p


def candidates(settings):
    return [{'family': 'language_control'}]+[{'family': 'regularized_quadratic', 'components': n,
        'pooled_diagonal_fraction': rho, 'class_variance_fraction': alpha}
        for n in settings['components'] for rho in settings['pooled_diagonal_fractions']
        for alpha in settings['class_variance_fractions']]


def options(inner, old, y, settings):
    control = old['language_control']; i = control['selected']['without_age']['selection']['index']
    predictions = [decide(np.asarray(control['inner_probabilities'])[i-1], None)]+[decide(p, None) for p in inner]
    return [{'candidate': c, **score(y, p)} for c, p in zip(candidates(settings), predictions)]


def choose(rows, policy):
    if policy == 'language_control':
        return 0
    if policy not in {'accuracy', 'quadratic_only'}:
        raise ValueError('Unknown policy')
    allowed = range(0 if policy == 'accuracy' else 1, len(rows))
    return max(allowed, key=lambda i: (rows[i]['correct'], rows[i]['macro_f1'], -i))


def predict_state(state, x):
    if state['candidate']['family'] in {'language_control', 'block_logistic'}:
        return state['estimator'].predict_proba({'base': x['semantic_av'], 'language': x['language']})
    return state['estimator'].predict_proba(x)


def subset(x, indices):
    return {k: np.asarray(v)[indices] for k, v in x.items()}


def job(x, frame, fold, old, settings, out):
    started = time.monotonic(); y = frame.severity_class.to_numpy(int); search = candidates(settings)
    inner = np.full((len(search)-1, len(y), 5), np.nan)
    for part in fold['inner']:
        for i, c in enumerate(search[1:]):
            model = QuadraticClassifier(c, settings).fit(subset(x, part['fit']), y[part['fit']])
            inner[i, part['heldout']] = model.predict_proba(subset(x, part['heldout']))
    inner = inner[:, fold['fit']]; rows = options(inner, old, y[fold['fit']], settings)
    out.mkdir(parents=True, exist_ok=True); selected, fitted = {}, {}
    for name in settings['policies']:
        i = choose(rows, name); c = search[i]
        if i not in fitted:
            if i:
                model = QuadraticClassifier(c, settings).fit(subset(x, fold['fit']), y[fold['fit']])
                state = {'estimator': model, 'candidate': c, 'feature': 'semantic_av'}
            else:
                source = LANGUAGE_ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'without_age.pkl'
                original = pickle.loads(source.read_bytes())
                state = {'estimator': original['estimator'], 'candidate': c,
                         'source_candidate': original['candidate']}
            fitted[i] = {**state, 'protocol': PROTOCOL, 'fit': fold['fit'], 'heldout': fold['heldout']}
        state = fitted[i]; heldout = {k: v[fold['heldout']] for k, v in x.items()}
        p = predict_state(state, heldout)
        if not i:
            np.testing.assert_allclose(p, old['language_control']['selected']['without_age']['probabilities'], rtol=0, atol=1e-14)
        path = out/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
        np.testing.assert_allclose(p, predict_state(pickle.loads(path.read_bytes()), heldout), rtol=0, atol=1e-14)
        training = {k: v[fold['fit']] for k, v in x.items()}
        selected[name] = {'selection': {'index': i}, 'candidate': c, 'threshold': None,
            'probabilities': p.tolist(), 'predictions': p.argmax(1).tolist(), 'checkpoint_sha256': sha256(path),
            'real_training_accuracy': float((predict_state(state, training).argmax(1) == y[fold['fit']]).mean())}
    return {**{k: fold[k] for k in ['seed', 'fold', 'fit', 'heldout']}, 'inner_probabilities': inner.tolist(),
        'options': rows, 'selected': selected, 'inner_receipts': fold['inner'],
        'fits': (len(search)-1)*len(fold['inner'])+sum(i > 0 for i in fitted), 'seconds': time.monotonic()-started}


def validate(record, fold, old, y, settings, directory):
    if (any(record[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout'])
            or record['inner_receipts'] != fold['inner'] or set(record['selected']) != set(settings['policies'])):
        raise ValueError('Fold or training ownership differs')
    if np.asarray(record['inner_probabilities']).shape != (len(candidates(settings))-1, len(fold['fit']), 5):
        raise ValueError('Incomplete inner coverage')
    rows = options(record['inner_probabilities'], old, y[fold['fit']], settings)
    if rows != record['options']:
        raise ValueError('Inner options changed')
    for name, result in record['selected'].items():
        i = choose(rows, name)
        if result['selection'] != {'index': i} or result['candidate'] != rows[i]['candidate']:
            raise ValueError('Selection changed')
        if np.asarray(result['probabilities']).shape != (len(fold['heldout']), 5) or decide(result['probabilities'], None).tolist() != result['predictions']:
            raise ValueError('Invalid probabilities')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')


def run(inputs, settings, output=OUTPUT, resume=False, max_new_jobs=None):
    x, frame, old_settings, old, provenance = inputs; out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing study')
    folds = nested_folds(frame, old_settings); y = frame.severity_class.to_numpy(int)
    if len(folds) != settings['expected_jobs'] or len(old) != len(folds) or frame.participant_id.duplicated().any():
        raise ValueError('Population or folds changed')
    for fold, baseline in zip(folds, old):
        if any(fold[k] != baseline[k] for k in ['seed', 'fold', 'fit', 'heldout']) or set(fold['fit']) & set(fold['heldout']):
            raise ValueError('Outer participant ownership differs')
        for part in fold['inner']:
            if set(part['fit']) & set(part['heldout']) or set(part['fit']+part['heldout']) != set(fold['fit']):
                raise ValueError('Inner participant ownership differs')
    maximum = sum((len(candidates(settings))-1)*len(f['inner'])+len(settings['policies']) for f in folds)
    if maximum > settings['max_nested_pipeline_fits']:
        raise ValueError('Declared fit budget exceeded')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
        'provenance': provenance, 'source_sha256': sha256(__file__), 'input_hashes': {k: digest(v.tolist()) for k,v in x.items()},
        'old_records_hash': digest(old), 'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'support_hashes': {str(Path('src')/name): sha256(Path('src')/name) for name in
            ['severe_case_diagnosis.py', 'severe_detection_study.py', 'raw_frame_uncertainty.py', 'batch_inference.py', 'quadratic_inference.py', 'age_language_accuracy.py']}}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        path = out/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Frozen inputs or code changed')
        elif resume:
            raise ValueError('Missing frozen protocol')
        else:
            atomic_json(path, envelope)
        records = []; new = fits = 0
        for fold, baseline in zip(folds, old):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; path = directory/'result.json'
            if path.exists():
                record = load_record(path, ch)
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                record = job(x, frame, fold, baseline, settings, directory)
                save_record(path, record, ch); new += 1; fits += record['fits']
            validate(record, fold, baseline, y, settings, directory); records.append(record)
            print(f"Regularized quadratic: {len(records)}/{len(folds)} jobs; {record['seconds']:.1f}s/job", flush=True)
        complete = len(records) == len(folds)
        # A completed resume validates caches and returns without rewriting evidence.
        if complete and not new and (out/'completion.json').exists():
            result = json.loads((out/'completion.json').read_text())
            return {**result, 'new_jobs': 0, 'new_fits': 0}
        targets = summarize(frame, records, old, settings, out) if complete else None
        result = {'complete': complete, 'context_hash': ch, 'completed_jobs': len(records),
            'new_jobs': new, 'new_fits': fits, 'total_fits': sum(r['fits'] for r in records),
            'targets': targets, 'old_models_refitted': 0, 'synthetic_rows': 0, 'validation_test_scored': False}
        atomic_json(out/'completion.json', result)
    return result


def training_choices(frame, records, old, settings):
    y = frame.severity_class.to_numpy(int); owners = np.zeros(len(y), int)
    if not records or len(records) != len(old):
        raise ValueError('Incomplete final selection')
    for record, baseline in zip(records, old):
        if any(record[k] != baseline[k] for k in ['seed', 'fold', 'fit', 'heldout']):
            raise ValueError('Final selection ownership differs')
        owners[record['fit']] += 1
    if not (owners == owners[0]).all() or not owners[0]:
        raise ValueError('Unequal final selection contribution')
    labels = np.concatenate([y[r['fit']] for r in records])
    baseline = np.concatenate([decide(np.asarray(r['language_control']['inner_probabilities'])[r['language_control']['selected']['without_age']['selection']['index']-1], None) for r in old])
    inner = np.concatenate([np.asarray(r['inner_probabilities']) for r in records], axis=1)
    rows = [{'candidate': {'family': 'language_control'}, **score(labels, baseline)}]
    rows += [{'candidate': c, **score(labels, decide(p, None))} for c, p in zip(candidates(settings)[1:], inner)]
    return {'rows': rows, 'choices': {n: choose(rows, n) for n in settings['policies']},
            'inner_predictions_per_participant': int(owners[0]), 'outer_results_used': False}


def freeze(inputs, settings, out=OUTPUT):
    from .best_accuracy_inference import load_bundle
    x, frame, old_settings, old, _ = inputs; out = Path(out)
    completion = json.loads((out/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Study incomplete')
    records = [load_record(out/'jobs'/f"seed{f['seed']}_fold{f['fold']}"/'result.json', completion['context_hash'])
               for f in nested_folds(frame, old_settings)]
    final = out/'final'
    if final.exists():
        raise FileExistsError('Preserving frozen final model')
    selection = training_choices(frame, records, old, settings)
    from .age_language_finalize import frozen_plan
    previous = frozen_plan()[0]['models']['without_age']
    if sha256(previous['path']) != previous['sha256']:
        raise ValueError('Baseline checkpoint changed')
    original = pickle.loads(Path(previous['path']).read_bytes())
    if original['fit_participants'] != frame.participant_id.tolist():
        raise ValueError('Baseline fitting participants changed')
    contract, _ = load_bundle(); final.mkdir(); models, fitted = {}, {}
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        atomic_json(final/'training_selection.json', selection)
        for name, i in selection['choices'].items():
            if not i:
                models[name] = {**previous, 'existing_language_control': True}
                continue
            if i not in fitted:
                c = candidates(settings)[i]
                estimator = QuadraticClassifier(c, settings).fit(x, frame.severity_class.to_numpy(int))
                fitted[i] = {'protocol': PROTOCOL, 'estimator': estimator, 'candidate': c, 'feature': 'semantic_av',
                             'fit_participants': frame.participant_id.tolist()}
            path = final/f'{name}.pkl'; path.write_bytes(pickle.dumps(fitted[i]))
            np.testing.assert_allclose(predict_state(fitted[i], x), predict_state(pickle.loads(path.read_bytes()), x), rtol=0, atol=1e-14)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': fitted[i]['candidate'], 'existing_language_control': False}
        bindings = {str(p): sha256(p) for p in [Path(__file__), CONFIG, out/'protocol.json', final/'training_selection.json',
            Path('src/quadratic_inference.py'), Path('src/age_language_accuracy.py')]}
        plan = {'protocol': PROTOCOL, 'context_hash': completion['context_hash'], 'models': models,
            'bindings': bindings, 'fit_participants': frame.participant_id.tolist(), 'feature_contract': contract,
            'feature_contract_hash': digest(contract), 'inference_feature_keys': ['semantic_av', 'language'], 'new_full_training_fits': len(fitted),
            'validation_test_scored': False, 'selection_uses_outer_results': False, 'independent_confirmation': False}
        atomic_json(final/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return plan


def load_inputs():
    from . import age_language_accuracy as language
    from .age_language_finalize import frozen_plan
    x, frame, old_settings, original, provenance = load_baseline()
    done = json.loads((LANGUAGE_ROOT/'completion.json').read_text())
    env = json.loads((LANGUAGE_ROOT/'protocol.json').read_text())
    if not done['complete'] or digest(env['context']) != env['context_hash'] or done['context_hash'] != env['context_hash']:
        raise ValueError('Incomplete or changed 49.28% control')
    frozen_plan()  # Read-only verification; never run the old study again.
    settings = json.loads(Path('configs/age_language_accuracy.json').read_text()); old = []
    bindings = dict(provenance['bindings'])
    for fold, baseline in zip(nested_folds(frame, old_settings), original):
        directory = LANGUAGE_ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
        r = load_record(directory/'result.json', done['context_hash'])
        language.validate(r, fold, baseline, frame.severity_class.to_numpy(int), settings, directory)
        old.append({**baseline, 'language_control': r})
        for path in [directory/'result.json', directory/'without_age.pkl']:
            bindings[str(path)] = sha256(path)
    docs = json.loads(Path('results/transcript_corpus/participants.json').read_text())
    text = {r['participant_id']: r for r in docs}
    if any(text[p]['split'] != 'train' for p in frame.participant_id):
        raise ValueError('Language participant split differs')
    x['language'] = np.stack([language.language_features(text[p]['text']) for p in frame.participant_id])
    x = {k: x[k] for k in ['semantic_av', 'language']}
    bindings['results/transcript_corpus/participants.json'] = sha256('results/transcript_corpus/participants.json')
    return x, frame, old_settings, old, {**provenance, 'bindings': bindings, 'language_control_context': done['context_hash']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'freeze']); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int); args = parser.parse_args()
    settings = json.loads(CONFIG.read_text()); inputs = load_inputs()
    result = (run(inputs, settings, resume=args.resume, max_new_jobs=args.max_new_jobs)
              if args.action == 'run' else freeze(inputs, settings))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    from .quadratic_accuracy import main as module_main
    module_main()
