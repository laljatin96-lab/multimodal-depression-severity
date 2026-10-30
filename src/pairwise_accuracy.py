"""Nested pairwise logistic classification with the unchanged 49.28% control."""
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

OUTPUT = Path('results/pairwise_accuracy')
CONFIG = Path('configs/pairwise_accuracy.json')
PROTOCOL = 'training-only-regularized-pairwise-v1'
LANGUAGE_ROOT = Path('results/age_language_accuracy')


PAIRS = [(i,j) for i in range(5) for j in range(i+1,5)]


def combine_logits(logits):
    """Least-squares projection of ten pair log odds onto five centered scores.

    Each input column is log(P(j)/P(i)). The complete-graph Laplacian on
    centered scores is 5I, so the normal equations have this exact solution.
    This combines model scores; it does not calibrate them using held-out data.
    """
    from scipy.special import softmax
    logits = np.asarray(logits, float)
    if logits.ndim != 2 or logits.shape[1] != 10 or not np.isfinite(logits).all():
        raise ValueError('Ten finite pair log odds required')
    scores = np.zeros((len(logits),5))
    for column,(i,j) in enumerate(PAIRS):
        scores[:,i] -= logits[:,column]/5
        scores[:,j] += logits[:,column]/5
    return softmax(scores,axis=1)


class PairwiseClassifier:
    """Shared fold-local projection, ten small strongly penalized binary heads."""
    def __init__(self, candidate, settings):
        self.candidate, self.settings = dict(candidate), dict(settings)

    def fit(self,x,y):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        y=np.asarray(y); base,language=x['semantic_av'],x['language']
        if y.shape!=(len(base),) or not np.array_equal(y,y.astype(int)) or set(y)!=set(range(5)):
            raise ValueError('Five real fitting classes required')
        if not np.isfinite(base).all() or not np.isfinite(language).all():
            raise ValueError('Nonfinite fitting inputs')
        y=y.astype(int);self.real_count=len(y);self.class_counts=np.bincount(y,minlength=5)
        self.base_scaler=StandardScaler().fit(base)
        self.base_pca=PCA(n_components=self.settings['components'],svd_solver='full').fit(self.base_scaler.transform(base))
        self.language_scaler=StandardScaler().fit(language)
        z=self.transform(x);self.heads=[];self.pair_positions=[];self.pair_weights=[]
        for i,j in PAIRS:
            positions=np.flatnonzero((y==i)|(y==j));target=(y[positions]==j).astype(int)
            weights=1/np.sqrt(self.class_counts[y[positions]].astype(float));weights*=len(positions)/weights.sum()
            head=LogisticRegression(C=self.candidate['C'],solver='lbfgs',max_iter=3000,random_state=self.settings['fit_seed'])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(z[positions],target,sample_weight=weights)
            self.heads.append(head);self.pair_positions.append(positions);self.pair_weights.append(weights)
        self.classes_=np.arange(5)
        return self

    def transform(self,x):
        return np.column_stack([self.base_pca.transform(self.base_scaler.transform(x['semantic_av'])),
            .5*self.language_scaler.transform(x['language'])])

    def pair_logits(self,x):
        z=self.transform(x)
        return np.column_stack([h.decision_function(z) for h in self.heads])

    def predict_proba(self,x):
        return combine_logits(self.pair_logits(x))

    def diagnostics(self,x,y):
        from sklearn.metrics import log_loss
        logits=self.pair_logits(x);p=combine_logits(logits);z=self.transform(x)
        details=[]
        for (i,j),h,positions in zip(PAIRS,self.heads,self.pair_positions):
            target=(np.asarray(y)[positions]==j).astype(int)
            details.append({'classes':[i,j],'fitting_rows':len(positions),'positions':positions.tolist(),
                'binary_class_counts':np.bincount(target,minlength=2).tolist(),'iterations':int(h.n_iter_.max()),
                'fitting_accuracy':float((h.predict(z[positions])==target).mean()),
                'coefficient_l2_norm':float(np.linalg.norm(h.coef_))})
        return {'real_fitting_participants':self.real_count,'class_counts':self.class_counts.tolist(),
            'C':self.candidate['C'],'pca_dimensions':int(self.base_pca.n_components_),
            'head_dimensions':int(self.heads[0].n_features_in_),
            'parameter_count':sum(int(h.coef_.size+h.intercept_.size) for h in self.heads),
            'iterations':[int(h.n_iter_.max()) for h in self.heads], 'maximum_iterations':3000,
            'converged':all((h.n_iter_<3000).all() for h in self.heads),
            'unweighted_fitting_log_loss':float(log_loss(y,p,labels=range(5))),
            'fitting_accuracy':float((p.argmax(1)==y).mean()),'heads':details,
            'checkpoint_rule':'Converged binary logistic heads; configuration selected by inner predictions only'}


def candidates(settings):
    return [{'family': 'language_control'}]+[{'family': 'pairwise_logistic', 'C': c} for c in settings['C']]


def options(inner, old, y, settings):
    control = old['language_control']; i = control['selected']['without_age']['selection']['index']
    predictions = [decide(np.asarray(control['inner_probabilities'])[i-1], None)]+[decide(p, None) for p in inner]
    return [{'candidate': c, **score(y, p)} for c, p in zip(candidates(settings), predictions)]


def choose(rows, policy):
    if policy == 'language_control':
        return 0
    if policy not in {'accuracy', 'pairwise_only'}:
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
    inner = np.full((len(search)-1, len(y), 5), np.nan); diagnostics = []
    for part in fold['inner']:
        for i, c in enumerate(search[1:]):
            model = PairwiseClassifier(c, settings).fit(subset(x, part['fit']), y[part['fit']])
            inner[i, part['heldout']] = model.predict_proba(subset(x, part['heldout']))
            diagnostics.append({'candidate_index': i+1, 'fit': part['fit'], 'heldout': part['heldout'],
                'inner_accuracy': float((inner[i, part['heldout']].argmax(1) == y[part['heldout']]).mean()),
                **model.diagnostics(subset(x, part['fit']), y[part['fit']])})
    inner = inner[:, fold['fit']]; rows = options(inner, old, y[fold['fit']], settings)
    out.mkdir(parents=True, exist_ok=True); selected, fitted = {}, {}
    for name in settings['policies']:
        i = choose(rows, name); c = search[i]
        if i not in fitted:
            if i:
                model = PairwiseClassifier(c, settings).fit(subset(x, fold['fit']), y[fold['fit']])
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
        'options': rows, 'selected': selected, 'inner_receipts': fold['inner'], 'fitting_diagnostics': diagnostics,
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
    if 10*maximum > settings['max_binary_head_fits']:
        raise ValueError('Declared binary-head fit budget exceeded')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
        'provenance': provenance, 'source_sha256': sha256(__file__), 'input_hashes': {k: digest(v.tolist()) for k,v in x.items()},
        'old_records_hash': digest(old), 'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'support_hashes': {str(Path('src')/name): sha256(Path('src')/name) for name in
            ['severe_case_diagnosis.py', 'severe_detection_study.py', 'raw_frame_uncertainty.py', 'batch_inference.py', 'pairwise_inference.py', 'age_language_accuracy.py', 'checked_inference.py']}}
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
            print(f"Regularized pairwise: {len(records)}/{len(folds)} jobs; {record['seconds']:.1f}s/job", flush=True)
        complete = len(records) == len(folds)
        # A completed resume validates caches and returns without rewriting evidence.
        if complete and not new and (out/'completion.json').exists():
            result = json.loads((out/'completion.json').read_text())
            return {**result, 'new_jobs': 0, 'new_fits': 0}
        targets = summarize(frame, records, old, settings, out) if complete else None
        result = {'complete': complete, 'context_hash': ch, 'completed_jobs': len(records),
            'new_jobs': new, 'new_fits': fits, 'total_fits': sum(r['fits'] for r in records),
            'new_binary_head_fits': sum(r['fits'] for r in records)*10, 'targets': targets, 'old_models_refitted': 0, 'synthetic_rows': 0, 'validation_test_scored': False}
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
                estimator = PairwiseClassifier(c, settings).fit(x, frame.severity_class.to_numpy(int))
                fitted[i] = {'protocol': PROTOCOL, 'estimator': estimator, 'candidate': c, 'feature': 'semantic_av',
                             'fit_participants': frame.participant_id.tolist()}
            path = final/f'{name}.pkl'; path.write_bytes(pickle.dumps(fitted[i]))
            np.testing.assert_allclose(predict_state(fitted[i], x), predict_state(pickle.loads(path.read_bytes()), x), rtol=0, atol=1e-14)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': fitted[i]['candidate'], 'existing_language_control': False}
        bindings = {str(p): sha256(p) for p in [Path(__file__), CONFIG, out/'protocol.json', final/'training_selection.json',
            Path('src/pairwise_inference.py'), Path('src/age_language_accuracy.py'), Path('src/checked_inference.py')]}
        plan = {'protocol': PROTOCOL, 'context_hash': completion['context_hash'], 'models': models,
            'bindings': bindings, 'fit_participants': frame.participant_id.tolist(), 'feature_contract': contract,
            'feature_contract_hash': digest(contract), 'inference_feature_keys': ['semantic_av', 'language'], 'new_full_training_fits': len(fitted),
            'validation_test_scored': False, 'selection_uses_outer_results': False, 'independent_confirmation': False}
        atomic_json(final/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return plan


def load_inputs():
    # Reuse the original-feature loader; never refit or read filtered-text features.
    from .quadratic_accuracy import load_inputs as original_inputs
    x,frame,old_settings,old,provenance=original_inputs()
    bindings=dict(provenance['bindings']);bindings['src/quadratic_accuracy.py']=sha256('src/quadratic_accuracy.py')
    return x,frame,old_settings,old,{**provenance,'bindings':bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'freeze']); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int); args = parser.parse_args()
    settings = json.loads(CONFIG.read_text()); inputs = load_inputs()
    result = (run(inputs, settings, resume=args.resume, max_new_jobs=args.max_new_jobs)
              if args.action == 'run' else freeze(inputs, settings))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    from .pairwise_accuracy import main as module_main
    module_main()
