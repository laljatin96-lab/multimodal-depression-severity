"""Nested eGeMAPS feature extension with the preserved 49.28% language procedure as control."""
import argparse
import json
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import decide, score, summarize

OUTPUT = Path('results/prosody_accuracy')
CONFIG = Path('configs/prosody_accuracy.json')
PROTOCOL = 'training-only-egemaps-language-extension-v1'
LANGUAGE_ROOT = Path('results/age_language_accuracy')


class ProsodyClassifier:
    def __init__(self, candidate, settings):
        self.candidate, self.settings = dict(candidate), dict(settings)

    def fit(self, x, y):
        y = np.asarray(y)
        if y.shape != (len(x['semantic_av']),) or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('Five classes and real fitting participants required')
        if not np.isfinite(x['semantic_av']).all() or not np.isfinite(x['language']).all() or np.isinf(x['prosody']).any():
            raise ValueError('Invalid fitting features')
        y = y.astype(int); self.real_count = len(y); self.classes_ = np.arange(5)
        self.class_counts = np.bincount(y, minlength=5)
        self.base_scaler = StandardScaler().fit(x['semantic_av'])
        self.base_pca = PCA(n_components=self.settings['base_components'], svd_solver='full').fit(self.base_scaler.transform(x['semantic_av']))
        self.language_scaler = StandardScaler().fit(x['language'])
        self.prosody_imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x['prosody'])
        p = self.prosody_imputer.transform(x['prosody'])
        self.prosody_scaler = StandardScaler().fit(p)
        self.prosody_pca = PCA(n_components=self.candidate['components'], svd_solver='full').fit(self.prosody_scaler.transform(p))
        weights = (len(y)/(5*self.class_counts[y]))**self.settings['weight_power']
        weights *= len(y)/weights.sum()
        self.model = LogisticRegression(C=self.candidate['C'], solver='lbfgs', max_iter=3000, random_state=self.settings['fit_seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.model.fit(self.transform(x), y, sample_weight=weights)
        return self

    def transform(self, x):
        parts = [self.base_pca.transform(self.base_scaler.transform(x['semantic_av']))]
        if self.candidate['language']:
            parts.append(self.settings['language_scale']*self.language_scaler.transform(x['language']))
        p = self.prosody_pca.transform(self.prosody_scaler.transform(self.prosody_imputer.transform(x['prosody'])))
        parts.append(self.candidate['block_scale']*p)
        return np.column_stack(parts)

    def predict_proba(self, x):
        p = self.model.predict_proba(self.transform(x)); decide(p, None)
        return p


def candidates(settings):
    return [{'family': 'language_control'}]+[{'family': 'prosody_logistic', 'language': language,
        'components': n, 'C': c, 'block_scale': scale}
        for language in settings['include_language'] for n in settings['components']
        for c in settings['C'] for scale in settings['block_scales']]


def options(inner, old, y, settings):
    control = old['language_control']; i = control['selected']['without_age']['selection']['index']
    predictions = [decide(np.asarray(control['inner_probabilities'])[i-1], None)]+[decide(p, None) for p in inner]
    return [{'candidate': c, **score(y, p)} for c, p in zip(candidates(settings), predictions)]


def choose(rows, policy):
    if policy == 'language_control':
        return 0
    if policy not in {'accuracy', 'prosody_only'}:
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
            model = ProsodyClassifier(c, settings).fit(subset(x, part['fit']), y[part['fit']])
            inner[i, part['heldout']] = model.predict_proba(subset(x, part['heldout']))
    inner = inner[:, fold['fit']]; rows = options(inner, old, y[fold['fit']], settings)
    out.mkdir(parents=True, exist_ok=True); selected, fitted = {}, {}
    for name in settings['policies']:
        i = choose(rows, name); c = search[i]
        if i not in fitted:
            if i:
                model = ProsodyClassifier(c, settings).fit(subset(x, fold['fit']), y[fold['fit']])
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
            ['severe_case_diagnosis.py', 'severe_detection_study.py', 'raw_frame_uncertainty.py', 'batch_inference.py', 'prosody_features.py', 'age_language_accuracy.py']}}
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
            print(f"Prosody extension: {len(records)}/{len(folds)} jobs; {record['seconds']:.1f}s/job", flush=True)
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
                estimator = ProsodyClassifier(c, settings).fit(x, frame.severity_class.to_numpy(int))
                fitted[i] = {'protocol': PROTOCOL, 'estimator': estimator, 'candidate': c, 'feature': 'semantic_av',
                             'fit_participants': frame.participant_id.tolist()}
            path = final/f'{name}.pkl'; path.write_bytes(pickle.dumps(fitted[i]))
            np.testing.assert_allclose(predict_state(fitted[i], x), predict_state(pickle.loads(path.read_bytes()), x), rtol=0, atol=1e-14)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': fitted[i]['candidate'], 'existing_language_control': False}
        from .prosody_features import NAMES
        bindings = {str(p): sha256(p) for p in [Path(__file__), CONFIG, out/'protocol.json', final/'training_selection.json',
            Path('src/prosody_features.py'), Path('src/age_language_accuracy.py')]}
        plan = {'protocol': PROTOCOL, 'context_hash': completion['context_hash'], 'models': models,
            'bindings': bindings, 'fit_participants': frame.participant_id.tolist(), 'feature_contract': contract,
            'feature_contract_hash': digest(contract), 'prosody_feature_names': NAMES, 'new_full_training_fits': len(fitted),
            'validation_test_scored': False, 'selection_uses_outer_results': False, 'independent_confirmation': False}
        atomic_json(final/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return plan


def load_inputs():
    from . import age_language_accuracy as language
    from .age_language_finalize import frozen_plan
    from .prosody_features import load_verified
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
    rows, feature_bindings = load_verified(); by_id = {r['participant_id']: r for r in rows}
    if set(by_id) != set(frame.participant_id) or len(rows) != len(frame):
        raise ValueError('Acoustic participant coverage differs')
    x['prosody'] = np.array([by_id[p]['features'] for p in frame.participant_id], dtype=float)
    bindings.update(feature_bindings)
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
    from .prosody_accuracy import main as module_main
    module_main()
