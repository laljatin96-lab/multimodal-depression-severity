"""Bounded training-only feature selection and balanced resampling of saved features."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .fold_feature_selection import FoldAnovaSelector
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import decide, score, summarize

PROTOCOL = 'baseline-anova-balanced-bootstrap-v1'


class SelectedLogistic(ClassifierMixin, BaseEstimator):
    def __init__(self, k=16, C=.01, intervention='none', members=3, seed=42, max_iter=2000):
        self.k, self.C, self.intervention = k, C, intervention
        self.members, self.seed, self.max_iter = members, seed, max_iter

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('Five real fitting classes and finite vectors required')
        if self.intervention not in {'none', 'sqrt_weight', 'balanced_bootstrap'} or self.members < 1:
            raise ValueError('Invalid balancing intervention')
        y = y.astype(int)
        self.classes_ = np.arange(5)
        self.real_count_ = len(y)
        self.class_counts_ = np.bincount(y, minlength=5)
        self.scaler_ = StandardScaler().fit(x)
        scaled = self.scaler_.transform(x)
        self.selector_ = FoldAnovaSelector(self.k).fit(scaled, y)
        z = self.selector_.transform(scaled)
        self.models_, self.draw_indices_, self.sample_weights_ = [], [], []
        count = self.members if self.intervention == 'balanced_bootstrap' else 1
        for member in range(count):
            indices = np.arange(len(y))
            weights = np.ones(len(y))
            if self.intervention == 'balanced_bootstrap':
                rng = np.random.default_rng(self.seed + member)
                extras = [rng.choice(np.flatnonzero(y == k), self.class_counts_.max() - n, replace=True)
                          for k, n in enumerate(self.class_counts_)]
                indices = np.concatenate([indices, *extras])
                weights = np.full(len(indices), len(y) / len(indices))
            elif self.intervention == 'sqrt_weight':
                weights = np.sqrt(len(y) / (5 * self.class_counts_[y]))
                weights *= len(y) / weights.sum()
            model = LogisticRegression(C=self.C, solver='lbfgs', max_iter=self.max_iter, random_state=self.seed)
            model.fit(z[indices], y[indices], sample_weight=weights)
            self.models_.append(model)
            self.draw_indices_.append(indices)
            self.sample_weights_.append(weights)
        return self

    def predict_proba(self, x):
        z = self.selector_.transform(self.scaler_.transform(x))
        return np.mean([five_probabilities(m, z) for m in self.models_], axis=0)

    def predict(self, x):
        return self.predict_proba(x).argmax(1)


def candidates(settings):
    return [{'family': 'baseline'}] + [
        {'family': 'selected_logistic', 'feature': f, 'k': k, 'C': c, 'intervention': v}
        for f in settings['feature_inputs'] for k in settings['selected_features']
        for c in settings['C_candidates'] for v in settings['interventions']]


def estimator(candidate, settings):
    return SelectedLogistic(candidate['k'], candidate['C'], candidate['intervention'],
                            settings['bootstrap_members'], settings['fit_seed'], settings['max_iter'])


def base_fits(candidate, settings):
    return settings['bootstrap_members'] if candidate.get('intervention') == 'balanced_bootstrap' else 1


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
    context = {'protocol': PROTOCOL, 'settings': settings, 'old_settings': old_settings,
               'training': frame.to_dict('records'), 'folds': folds, 'baseline_records': old_records,
               'provenance': provenance, 'code_sha256': sha256(__file__),
               'feature_hashes': {n: digest(a.tolist()) for n, a in x.items()},
               'support_hashes': {n: sha256(Path(__file__).with_name(n)) for n in
                   ['severe_case_diagnosis.py', 'severe_detection_study.py', 'fold_feature_selection.py',
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
            print(f'Baseline refinement: {len(records)}/{len(folds)} jobs; {new} new', flush=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    settings = json.loads(Path('configs/baseline_refinement.json').read_text())
    print(json.dumps(run(*load_baseline(), settings, 'results/baseline_refinement',
                         resume=args.resume, max_new_jobs=args.max_new_jobs), indent=2))


if __name__ == '__main__':
    from .baseline_refinement import main as module_main
    module_main()
