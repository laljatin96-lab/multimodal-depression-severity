"""Training-only supervised PLS and ridge continuation of the verified accuracy baseline."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.cross_decomposition import PLSRegression
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import decide, score, summarize

PROTOCOL = 'nested-supervised-pls-ridge-accuracy-v1'


class SupervisedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, family='pls', components=4, C=.1, alpha=100., weight_power=0.,
                 seed=42, max_iter=3000, pls_max_iter=2000):
        self.family, self.components, self.C, self.alpha = family, components, C, alpha
        self.weight_power, self.seed, self.max_iter = weight_power, seed, max_iter
        self.pls_max_iter = pls_max_iter

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('Finite real fitting rows and all five integer classes required')
        if self.family not in {'pls', 'ridge'} or self.weight_power not in {0., .5}:
            raise ValueError('Invalid supervised classifier family/weights')
        y = y.astype(int)
        self.classes_ = np.arange(5); self.n_features_in_ = x.shape[1]
        self.real_count_ = len(y); self.class_counts_ = np.bincount(y, minlength=5)
        self.scaler_ = StandardScaler().fit(x)
        z = self.scaler_.transform(x)
        self.pls_ = self.latent_scaler_ = None
        if self.family == 'pls':
            if not 1 <= self.components <= min(len(y)-1, x.shape[1]):
                raise ValueError('PLS component count exceeds fitting support')
            self.pls_ = PLSRegression(n_components=self.components, scale=False,
                                      max_iter=self.pls_max_iter, tol=1e-7)
            self.pls_.fit(z, np.eye(5)[y])
            self.latent_scaler_ = StandardScaler().fit(self.pls_.transform(z))
            z = self.latent_scaler_.transform(self.pls_.transform(z))
            self.model_ = LogisticRegression(C=self.C, solver='lbfgs', max_iter=self.max_iter, random_state=self.seed)
        else:
            self.model_ = RidgeClassifier(alpha=self.alpha, solver='cholesky')
        weights = (len(y)/(5*self.class_counts_[y]))**self.weight_power
        weights *= len(y)/weights.sum(); self.sample_weights_ = weights
        self.model_.fit(z, y, sample_weight=weights)
        return self

    def transform(self, x):
        z = self.scaler_.transform(x)
        if self.pls_ is not None:
            z = self.latent_scaler_.transform(self.pls_.transform(z))
        return z

    def predict_proba(self, x):
        z = self.transform(x)
        # Softmax preserves RidgeClassifier's decision-score argmax; it is not calibration.
        return softmax(self.model_.decision_function(z), axis=1) if self.family == 'ridge' else self.model_.predict_proba(z)

    def predict(self, x):
        return self.predict_proba(x).argmax(1)


def candidates(settings):
    return [{'family': 'baseline'}] + [
        {'family': 'pls', 'feature': f, 'components': k, 'C': c, 'weight_power': w}
        for f in settings['feature_inputs'] for k in settings['components']
        for c in settings['C_candidates'] for w in settings['weight_powers']] + [
        {'family': 'ridge', 'feature': f, 'alpha': a, 'weight_power': w}
        for f in settings['feature_inputs'] for a in settings['ridge_alpha'] for w in settings['weight_powers']]


def estimator(candidate, settings):
    return SupervisedClassifier(**{k:v for k,v in candidate.items() if k != 'feature'},
        seed=settings['fit_seed'], max_iter=settings['max_iter'], pls_max_iter=settings['pls_max_iter'])


def base_fits(candidate, settings):
    return 1


def baseline_inner(old):
    return old['candidates'][old['policies']['accuracy']['selection']['index']]['predictions']


def options(inner, old, y, settings):
    search = candidates(settings)
    rows = [{'candidate': search[0], **score(y, np.asarray(baseline_inner(old)))}]
    for candidate, p in zip(search[1:], inner):
        rows.append({'candidate': candidate, **score(y, decide(p, None))})
    return rows


def choose(rows):
    return max(range(len(rows)), key=lambda i: (rows[i]['correct'], rows[i]['macro_f1'], -i))


def job(x, frame, fold, old, settings, out):
    start = time.monotonic()
    y = frame.severity_class.to_numpy(int)
    search = candidates(settings)
    inner = []
    for candidate in search[1:]:
        p = np.full((len(y), 5), np.nan)
        for part in fold['inner']:
            model = estimator(candidate, settings)
            fit(model, x[candidate['feature']][part['fit']], y[part['fit']])
            p[part['heldout']] = five_probabilities(model, x[candidate['feature']][part['heldout']])
        inner.append(p[fold['fit']].tolist())
    rows = options(inner, old, y[fold['fit']], settings)
    i = choose(rows)
    candidate = search[i]
    if i:
        feature = candidate['feature']
        model = estimator(candidate, settings)
        fit(model, x[feature][fold['fit']], y[fold['fit']])
    else:
        feature = old['policies']['accuracy']['candidate']['feature']
        path = ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
        with path.open('rb') as f:
            model = pickle.load(f)['estimator']
    p = five_probabilities(model, x[feature][fold['heldout']])
    real_training_accuracy = float((model.predict(x[feature][fold['fit']]) == y[fold['fit']]).mean())
    if not i:
        np.testing.assert_allclose(p, old['policies']['accuracy']['probabilities'], rtol=0, atol=1e-14)
    state = {'protocol': PROTOCOL, 'estimator': model, 'feature': feature, 'candidate': candidate,
             'fit': fold['fit'], 'heldout': fold['heldout']}
    out.mkdir(parents=True, exist_ok=True)
    path = out/'accuracy.pkl'
    with path.open('wb') as f:
        pickle.dump(state, f)
    with path.open('rb') as f:
        restored = pickle.load(f)
    np.testing.assert_allclose(p, five_probabilities(restored['estimator'], x[feature][fold['heldout']]), rtol=0, atol=1e-14)
    selected = {'selection': {'index': i}, 'candidate': candidate, 'threshold': None,
                'probabilities': p.tolist(), 'predictions': decide(p, None).tolist(), 'checkpoint_sha256': sha256(path)}
    return {'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
            'inner_probabilities': inner, 'options': rows, 'selected': {'accuracy': selected},
            'real_training_accuracy': real_training_accuracy,
            'fits': (len(search)-1)*len(fold['inner']) + bool(i),
            'base_fits': sum(base_fits(c, settings) for c in search[1:])*len(fold['inner']) + (base_fits(candidate, settings) if i else 0),
            'seconds': time.monotonic()-start}


def validate(record, fold, old, y, settings, directory):
    if any(record[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']):
        raise ValueError('Fold changed')
    search = candidates(settings)
    if np.asarray(record['inner_probabilities']).shape != (len(search)-1, len(fold['fit']), 5):
        raise ValueError('Incomplete inner coverage')
    rows = options(record['inner_probabilities'], old, y[fold['fit']], settings)
    if record['options'] != rows or set(record['selected']) != {'accuracy'}:
        raise ValueError('Search changed')
    i = choose(rows)
    result = record['selected']['accuracy']
    if result['selection'] != {'index': i} or result['candidate'] != search[i] or result['threshold'] is not None:
        raise ValueError('Selection changed')
    p = np.asarray(result['probabilities'])
    if p.shape != (len(fold['heldout']), 5) or decide(p, None).tolist() != result['predictions']:
        raise ValueError('Predictions changed')
    if not i and result['probabilities'] != old['policies']['accuracy']['probabilities']:
        raise ValueError('Baseline changed')
    if sha256(directory/'accuracy.pkl') != result['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if record['fits'] != (len(search)-1)*len(fold['inner'])+bool(i) or record['base_fits'] != sum(base_fits(c, settings) for c in search[1:])*len(fold['inner'])+(base_fits(search[i], settings) if i else 0):
        raise ValueError('Fit budget changed')


def run(x, frame, old_settings, old_records, provenance, settings, output, resume=False, max_new_jobs=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing run')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires protocol')
    if max_new_jobs is not None and max_new_jobs < 1:
        raise ValueError('Positive pilot budget required')
    folds = nested_folds(frame, old_settings)
    search = candidates(settings)
    if len(folds) != settings['expected_jobs'] or len(old_records) != len(folds) or (len(search)-1)*old_settings['inner_folds'] != settings['inner_pipeline_fits_per_job'] or sum(base_fits(c, settings) for c in search[1:])*old_settings['inner_folds'] != settings['inner_base_fits_per_job'] or set(settings['policies']) != {'accuracy'}:
        raise ValueError('Budget or policy mismatch')
    y = frame.severity_class.to_numpy(int)
    if frame.participant_id.duplicated().any() or not np.array_equal(y, frame.severity_class) or set(y) != set(range(5)):
        raise ValueError('Invalid training participants')
    for a in x.values():
        if a.ndim != 2 or len(a) != len(y) or not np.isfinite(a).all():
            raise ValueError('Invalid features')
    for fold, old in zip(folds, old_records):
        if any(old[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']):
            raise ValueError('Baseline fold mismatch')
        if set(fold['fit']) & set(fold['heldout']):
            raise ValueError('Outer participant leakage')
        for part in fold['inner']:
            if set(part['fit']) & set(part['heldout']) or not set(part['fit']+part['heldout']) <= set(fold['fit']):
                raise ValueError('Inner participant leakage')
    context = {'protocol': PROTOCOL, 'settings': settings, 'old_settings': old_settings,
               'training': frame.to_dict('records'), 'folds': folds, 'baseline_records': old_records,
               'provenance': provenance, 'code_sha256': sha256(__file__),
               'feature_hashes': {n: digest(a.tolist()) for n, a in x.items()},
               'support_hashes': {n: sha256(Path(__file__).with_name(n)) for n in
                   ['severe_case_diagnosis.py', 'severe_detection_study.py', 'mpnet_features.py',
                    'crossvalidated_baselines.py', 'modality_diagnostics.py', 'evaluate.py',
                    'raw_frame_uncertainty.py', 'encoder_kernel_accuracy.py', 'batch_inference.py']},
               'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    ch = digest(context)
    envelope = {'context': context, 'context_hash': ch}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != envelope:
                raise ValueError('Inputs/code changed')
        else:
            atomic_json(out/'protocol.json', envelope)
        records = []
        new = new_fits = 0
        for fold, old in zip(folds, old_records):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            path = directory/'result.json'
            if path.exists():
                r = load_record(path, ch)
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                r = job(x, frame, fold, old, settings, directory)
                save_record(path, r, ch)
                new += 1
                new_fits += r['fits']
            validate(r, fold, old, y, settings, directory)
            records.append(r)
            print(f'Supervised accuracy: {len(records)}/{len(folds)} jobs; {new} new', flush=True)
        for path, h in provenance.get('bindings', {}).items():
            if sha256(path) != h:
                raise ValueError('Bound inputs changed')
        complete = len(records) == len(folds)
        targets = summarize(frame, records, old_records, settings, out) if complete else None
        receipt = {'complete': complete, 'context_hash': ch, 'completed_jobs': len(records),
                   'new_jobs': new, 'new_fits': new_fits, 'pipeline_fits': sum(r['fits'] for r in records),
                   'base_estimator_fits': sum(r['base_fits'] for r in records), 'targets': targets,
                   'validation_test_scored': False, 'baseline_replaced': False, 'synthetic_records': 0}
        atomic_json(out/'completion.json', receipt)
    return receipt



def load_inputs():
    from .mpnet_features import load_verified
    x, frame, old_settings, records, provenance = load_baseline()
    docs, _ = load_verified('results/final_evaluation_plan.json', 'results/transcript_corpus',
                            'checkpoints/pretrained_mpnet', 'results/mpnet_features')
    by_id = {r['participant_id']: r for r in docs}
    if len(by_id) != len(docs) or any(by_id[pid]['split'] != 'train' for pid in frame.participant_id):
        raise ValueError('MPNet training identity mismatch')
    larger = np.asarray([by_id[pid]['embedding'] for pid in frame.participant_id], dtype=float)
    if x['semantic'].shape != (len(frame), 384) or x['semantic_av'].shape != (len(frame), 512) or larger.shape != (len(frame), 768):
        raise ValueError('Feature schema mismatch')
    np.testing.assert_array_equal(x['semantic'], x['semantic_av'][:, :384])
    x.update(minilm=x['semantic'], minilm_av=x['semantic_av'], mpnet=larger,
             mpnet_av=np.column_stack([larger, x['semantic_av'][:, 384:]]))
    bindings = dict(provenance['bindings'])
    for p in [Path('results/mpnet_features')/name for name in ['protocol.json','completion.json','embeddings.json']]:
        bindings[str(p)] = sha256(p)
    return x, frame, old_settings, records, {**provenance, 'bindings': bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    settings = json.loads(Path('configs/supervised_accuracy.json').read_text())
    print(json.dumps(run(*load_inputs(), settings, 'results/supervised_accuracy',
                         resume=args.resume, max_new_jobs=args.max_new_jobs), indent=2))


if __name__ == '__main__':
    from .supervised_accuracy import main as module_main
    module_main()
