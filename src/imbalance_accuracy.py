"""Fold-local SMOTE/ROS with training-only selection and frozen validation inference."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .evaluate import write_report
from .final_evaluation import sha256
from .modality_diagnostics import modality_columns, nested_folds
from .severe_case_diagnosis import ROOT, load_baseline
from .severe_detection_study import score, summarize

PROTOCOL = 'fold-local-smote-ros-modality-accuracy-v1'
OUTPUT = Path('results/imbalance_accuracy')


def blocks(feature):
    names = {'text': ['text'], 'text_audio': ['text', 'audio'],
             'text_visual': ['text', 'visual'], 'text_audio_visual': ['text', 'audio', 'visual']}
    indices = {'text': np.arange(384), 'audio': 384+modality_columns('A'),
               'visual': 384+modality_columns('V')}
    return [(name, indices[name]) for name in names[feature]]


class FeatureMap:
    def __init__(self, feature='text_audio_visual', reduction='block32'):
        self.feature, self.reduction = feature, reduction

    def fit(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 512 or not np.isfinite(x).all() or len(x) < 2:
            raise ValueError('Expected finite real 512-column fitting features')
        pieces = blocks(self.feature)
        if self.reduction == 'global16':
            pieces = [('global', np.concatenate([v for _, v in pieces]))]
        elif self.reduction != 'block32':
            raise ValueError('Unknown feature reduction')
        self.transforms_ = []
        self.real_count_ = len(x)
        for name, indices in pieces:
            scaler = StandardScaler().fit(x[:, indices])
            k = 16 if name in {'global', 'text'} else 8
            pca = PCA(n_components=min(k, len(x)-1, len(indices)), svd_solver='full')
            pca.fit(scaler.transform(x[:, indices]))
            self.transforms_.append((indices, scaler, pca))
        return self

    def transform(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 512 or not np.isfinite(x).all():
            raise ValueError('Feature schema mismatch')
        return np.column_stack([pca.transform(scaler.transform(x[:, idx]))
                                for idx, scaler, pca in self.transforms_])


def resample(z, y, sampler, seed=42, neighbors=2):
    """Standard SMOTE interpolation or ROS; lineage refers only to real local rows."""
    z, y = np.asarray(z, float), np.asarray(y)
    if (z.ndim != 2 or len(z) != len(y) or not np.isfinite(z).all()
            or not np.array_equal(y, y.astype(int)) or not set(y) <= set(range(5)) or neighbors < 1):
        raise ValueError('Invalid training augmentation inputs')
    y = y.astype(int)
    targets = {}
    if sampler in {'ros20', 'ros30', 'smote20', 'smote30'}:
        targets = {4: int(sampler[-2:])}
    elif sampler == 'joint_smote30':
        targets = {1: 30, 2: 30, 3: 30, 4: 25}
    elif sampler not in {'none', 'quarter_weight', 'balanced_weight'}:
        raise ValueError('Unknown sampler')
    rng = np.random.default_rng(seed)
    generated, labels, lineage = [], [], []
    for label, target in targets.items():
        parents = np.flatnonzero(y == label)
        if not len(parents):
            raise ValueError('Cannot oversample absent class')
        use_smote = 'smote' in sampler and len(parents) >= 2
        if use_smote:
            d = ((z[parents, None, :]-z[None, parents, :])**2).sum(2)
            np.fill_diagonal(d, np.inf)
            nearest = np.argsort(d, axis=1, kind='stable')[:, :min(neighbors, len(parents)-1)]
        for _ in range(max(0, target-len(parents))):
            a = int(rng.integers(len(parents))); i = int(parents[a])
            j = int(parents[int(rng.choice(nearest[a]))]) if use_smote else i
            fraction = float(rng.random()) if use_smote else 0.0
            generated.append(z[i]+fraction*(z[j]-z[i])); labels.append(label)
            lineage.append({'class': label, 'parent_a': i, 'parent_b': j, 'fraction': fraction,
                            'method': 'smote' if use_smote else 'ros',
                            'singleton_fallback': 'smote' in sampler and not use_smote})
    if not generated:
        return z.copy(), y.copy(), lineage
    return np.vstack([z, generated]), np.concatenate([y, labels]), lineage


class ResampledLogistic(ClassifierMixin, BaseEstimator):
    def __init__(self, feature='text_audio_visual', reduction='block32', C=.01,
                 sampler='none', seed=42, neighbors=2, max_iter=3000):
        self.feature, self.reduction, self.C, self.sampler = feature, reduction, C, sampler
        self.seed, self.neighbors, self.max_iter = seed, neighbors, max_iter

    def fit(self, x, y):
        mapper = FeatureMap(self.feature, self.reduction).fit(x)
        return self.fit_prepared(mapper.transform(x), y, mapper)

    def fit_prepared(self, z, y, mapper):
        y = np.asarray(y)
        if not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('All five real fitting classes required')
        y = y.astype(int)
        if mapper.real_count_ != len(y) or len(z) != len(y):
            raise ValueError('Preprocessing must use exactly the real fitting rows')
        self.mapper_ = mapper
        self.real_count_ = len(y)
        self.real_counts_ = np.bincount(y, minlength=5)
        augmented, labels, self.lineage_ = resample(z, y, self.sampler, self.seed, self.neighbors)
        exponent = {'quarter_weight': .25, 'balanced_weight': 1.}.get(self.sampler, 0.)
        w = (len(y)/(5*self.real_counts_[labels]))**exponent
        w *= len(y)/w.sum()
        self.sample_weights_ = w
        self.fit_counts_ = np.bincount(labels, minlength=5)
        self.total_weight_ = float(w.sum())
        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver='lbfgs', random_state=self.seed)
        self.model_.fit(augmented, labels, sample_weight=w)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = 512
        return self

    def predict_proba(self, x):
        return self.model_.predict_proba(self.mapper_.transform(x))

    def predict(self, x):
        return self.predict_proba(x).argmax(1)


def candidates(settings):
    return [{'family': 'baseline'}] + [
        {'family': 'resampled_logistic', 'feature': f, 'reduction': r, 'C': c, 'sampler': s}
        for f in settings['features'] for r in settings['reductions']
        for c in settings['C'] for s in settings['samplers']]


def estimator(candidate, settings):
    return ResampledLogistic(**{k: candidate[k] for k in ['feature', 'reduction', 'C', 'sampler']},
                             seed=settings['fit_seed'], neighbors=settings['neighbors'], max_iter=settings['max_iter'])


def receipt(model, fit_indices):
    return {'fit_indices': list(fit_indices), 'real_count': model.real_count_,
            'real_counts': model.real_counts_.tolist(), 'fit_counts': model.fit_counts_.tolist(),
            'total_weight': model.total_weight_, 'lineage': model.lineage_}


def validate_receipt(r, fit_indices, y, candidate, settings):
    indices = np.asarray(fit_indices)
    local_y = y[indices]
    if (r['fit_indices'] != list(fit_indices) or r['real_count'] != len(indices)
            or r['real_counts'] != np.bincount(local_y, minlength=5).tolist()
            or not np.isclose(r['total_weight'], len(indices))):
        raise ValueError('Augmentation fitting ownership changed')
    # Nearest-parent geometry is tested against the fitting vectors before accepting this implementation.
    counts = np.bincount(local_y, minlength=5)
    for item in r['lineage']:
        a, b, label = item['parent_a'], item['parent_b'], item['class']
        if not (0 <= a < len(indices) and 0 <= b < len(indices) and local_y[a] == local_y[b] == label
                and 0 <= item['fraction'] < 1 and item['method'] in {'ros', 'smote'}):
            raise ValueError('Invalid fitting-only synthetic parents')
        if (item['method'] == 'smote' and a == b) or (item['method'] == 'ros' and (a != b or item['fraction'] != 0)):
            raise ValueError('Invalid interpolation/duplication lineage')
        counts[label] += 1
    expected = np.bincount(local_y, minlength=5)
    s = candidate['sampler']
    if s in {'ros20', 'ros30', 'smote20', 'smote30'}:
        expected[4] = max(expected[4], int(s[-2:]))
    elif s == 'joint_smote30':
        expected[1:4] = np.maximum(expected[1:4], 30); expected[4] = max(expected[4], 25)
    if counts.tolist() != r['fit_counts'] or not np.array_equal(counts, expected):
        raise ValueError('Training-only augmentation target mismatch')


def option_scores(inner, old, labels, settings):
    base = old['candidates'][old['policies']['accuracy']['selection']['index']]['predictions']
    return [{'candidate': candidates(settings)[0], **score(labels, np.asarray(base))}] + [
        {'candidate': c, **score(labels, np.asarray(p).argmax(1))}
        for c, p in zip(candidates(settings)[1:], inner)]


def choose(rows, policy, settings):
    pool = list(range(len(rows)))
    if policy in {'smote', 'ros'}:
        pool = [i for i, r in enumerate(rows) if policy in r['candidate'].get('sampler', '')]
    if policy == 'severe_guarded':
        best = max(r['accuracy'] for r in rows)
        pool = [i for i in pool if rows[i]['accuracy'] >= best-settings['severe_accuracy_tolerance']-1e-12]
        return max(pool, key=lambda i: (rows[i]['severe_true_positive'], rows[i]['correct'], rows[i]['macro_f1'], -i))
    if policy not in {'accuracy', 'ros', 'smote'} or not pool:
        raise ValueError('Invalid selection policy/pool')
    return max(pool, key=lambda i: (rows[i]['correct'], rows[i]['macro_f1'], -i))


def save_model(path, state, x):
    p = five_probabilities(state['estimator'], x)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(state, f)
    tmp.replace(path)
    with path.open('rb') as f:
        restored = pickle.load(f)
    np.testing.assert_allclose(p, five_probabilities(restored['estimator'], x), rtol=0, atol=1e-14)
    return p


def job(x, frame, fold, old, settings, directory):
    start = time.monotonic(); y = frame.severity_class.to_numpy(int)
    search = candidates(settings); inner = []; receipts = []; training_scores = []
    prepared = {}
    for feature in settings['features']:
        for reduction in settings['reductions']:
            for j, part in enumerate(fold['inner']):
                mapper = FeatureMap(feature, reduction).fit(x['semantic_av'][part['fit']])
                prepared[feature, reduction, j] = (mapper, mapper.transform(x['semantic_av'][part['fit']]),
                                                  mapper.transform(x['semantic_av'][part['heldout']]))
    for candidate in search[1:]:
        p = np.full((len(y), 5), np.nan); fitting_receipts = []; train_correct = train_n = 0
        for j, part in enumerate(fold['inner']):
            mapper, z, heldout_z = prepared[candidate['feature'], candidate['reduction'], j]
            model = estimator(candidate, settings).fit_prepared(z, y[part['fit']], mapper)
            p[part['heldout']] = model.model_.predict_proba(heldout_z)
            train_correct += int((model.model_.predict(z) == y[part['fit']]).sum()); train_n += len(z)
            fitting_receipts.append(receipt(model, part['fit']))
        inner.append(p[fold['fit']].tolist()); receipts.append(fitting_receipts)
        training_scores.append(train_correct/train_n)
    rows = option_scores(inner, old, y[fold['fit']], settings)
    selected = {}; directory.mkdir(parents=True, exist_ok=True)
    new_refits = 0
    for policy in settings['policies']:
        i = choose(rows, policy, settings); candidate = search[i]
        if i:
            model = estimator(candidate, settings); fit(model, x['semantic_av'][fold['fit']], y[fold['fit']])
            feature = 'semantic_av'; new_refits += 1; fit_receipt = receipt(model, fold['fit'])
        else:
            feature = old['policies']['accuracy']['candidate']['feature']
            path = ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
            with path.open('rb') as f:
                model = pickle.load(f)['estimator']
            fit_receipt = None
        state = {'protocol': PROTOCOL, 'estimator': model, 'feature': feature, 'candidate': candidate,
                 'fit': fold['fit'], 'heldout': fold['heldout']}
        path = directory/f'{policy}.pkl'
        p = save_model(path, state, x[feature][fold['heldout']])
        if not i:
            np.testing.assert_allclose(p, old['policies']['accuracy']['probabilities'], rtol=0, atol=1e-14)
        selected[policy] = {'selection': {'index': i}, 'candidate': candidate, 'threshold': None,
                            'probabilities': p.tolist(), 'predictions': p.argmax(1).tolist(),
                            'fit_receipt': fit_receipt, 'checkpoint_sha256': sha256(path)}
    return {**{k: fold[k] for k in ['seed', 'fold', 'fit', 'heldout']}, 'inner_probabilities': inner,
            'inner_receipts': receipts, 'inner_real_train_accuracies': training_scores,
            'options': rows, 'selected': selected,
            'fits': (len(search)-1)*len(fold['inner'])+new_refits, 'seconds': time.monotonic()-start}


def validate(r, fold, old, y, settings, directory):
    if any(r[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']):
        raise ValueError('Fold changed')
    search = candidates(settings)
    p = np.asarray(r['inner_probabilities'])
    if p.shape != (len(search)-1, len(fold['fit']), 5) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(2), 1):
        raise ValueError('Invalid inner predictions')
    rows = option_scores(p, old, y[fold['fit']], settings)
    if rows != r['options'] or set(r['selected']) != set(settings['policies']) or len(r['inner_receipts']) != len(search)-1:
        raise ValueError('Candidate results changed')
    for c, receipts in zip(search[1:], r['inner_receipts']):
        if len(receipts) != len(fold['inner']):
            raise ValueError('Missing inner provenance')
        for part, rec in zip(fold['inner'], receipts):
            validate_receipt(rec, part['fit'], y, c, settings)
    refits = 0
    for name, result in r['selected'].items():
        i = choose(rows, name, settings)
        if result['selection'] != {'index': i} or result['candidate'] != search[i]:
            raise ValueError('Selection changed')
        probs = np.asarray(result['probabilities'])
        if (probs.shape != (len(fold['heldout']), 5) or not np.isfinite(probs).all()
                or (probs < 0).any() or not np.allclose(probs.sum(1), 1)
                or result['predictions'] != probs.argmax(1).tolist()):
            raise ValueError('Invalid original-participant predictions')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        if i:
            validate_receipt(result['fit_receipt'], fold['fit'], y, search[i], settings); refits += 1
        elif result['probabilities'] != old['policies']['accuracy']['probabilities']:
            raise ValueError('Baseline changed')
    if r['fits'] != (len(search)-1)*len(fold['inner'])+refits:
        raise ValueError('Fit budget changed')


def run(inputs, settings, output=OUTPUT, resume=False, max_new_jobs=None):
    x, frame, old_settings, old_records, provenance = inputs
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving previous experiment; use --resume')
    if max_new_jobs is not None and max_new_jobs < 1:
        raise ValueError('Positive job budget required')
    y = frame.severity_class.to_numpy(int); folds = nested_folds(frame, old_settings)
    if (len(folds) != settings['expected_jobs'] or len(candidates(settings)) != settings['expected_candidates']
            or frame.participant_id.duplicated().any() or len(old_records) != len(folds)):
        raise ValueError('Invalid participants or declared budget')
    for fold, old in zip(folds, old_records):
        if any(old[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']):
            raise ValueError('Baseline fold mismatch')
        if set(fold['fit']) & set(fold['heldout']):
            raise ValueError('Outer participant leakage')
        for part in fold['inner']:
            if set(part['fit']) & set(part['heldout']) or not set(part['fit']+part['heldout']) <= set(fold['fit']):
                raise ValueError('Inner participant leakage')
    support = ['severe_case_diagnosis.py', 'semantic_severe_repair.py', 'severe_detection_study.py',
               'modality_diagnostics.py', 'crossvalidated_baselines.py', 'evaluate.py', 'raw_frame_uncertainty.py',
               'encoder_kernel_accuracy.py', 'batch_inference.py']
    context = {'protocol': PROTOCOL, 'settings': settings, 'training': frame.to_dict('records'),
               'folds': folds, 'provenance': provenance, 'baseline_records_hash': digest(old_records),
               'code_sha256': sha256(__file__), 'support_hashes': {n: sha256(Path(__file__).with_name(n)) for n in support},
               'feature_hashes': {k: digest(v.tolist()) for k, v in x.items()},
               'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__, 'pandas': pd.__version__}}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        path = out/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Inputs/code changed')
        elif resume:
            raise ValueError('Resume needs existing protocol')
        else:
            atomic_json(path, envelope)
        records = []; new = fits = 0
        for fold, old in zip(folds, old_records):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            path = directory/'result.json'
            if path.exists():
                r = load_record(path, ch)
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                r = job(x, frame, fold, old, settings, directory)
                save_record(path, r, ch); new += 1; fits += r['fits']
            validate(r, fold, old, y, settings, directory); records.append(r)
            print(f'Imbalance accuracy: {len(records)}/{len(folds)} jobs; {new} new; last {r["seconds"]:.1f}s', flush=True)
        for path, expected in provenance.get('bindings', {}).items():
            if sha256(path) != expected:
                raise ValueError('Previous artifacts changed')
        complete = len(records) == len(folds)
        targets = summarize(frame, records, old_records, settings, out) if complete else None
        result = {'complete': complete, 'context_hash': ch, 'new_jobs': new, 'new_fits': fits,
                  'completed_jobs': len(records), 'pipeline_fits': sum(r['fits'] for r in records),
                  'targets': targets, 'evaluation_population': 'Original participants only; no synthetic evaluation rows',
                  'validation_test_scored': False, 'baseline_replaced': False}
        atomic_json(out/'completion.json', result)
    return result


def finalize(inputs, settings, output=OUTPUT):
    """Freeze training-selected refits before loading official validation labels."""
    x, frame, old_settings, old_records, _ = inputs; out = Path(output)
    completion = json.loads((out/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Complete training experiment first')
    if (out/'final').exists():
        raise FileExistsError('Final models already frozen; use evaluate')
    records = [load_record(out/'jobs'/f"seed{f['seed']}_fold{f['fold']}"/'result.json', completion['context_hash'])
               for f in nested_folds(frame, old_settings)]
    y = frame.severity_class.to_numpy(int)
    labels = np.concatenate([y[r['fit']] for r in records])
    pooled_inner = np.concatenate([np.asarray(r['inner_probabilities']) for r in records], axis=1)
    counts = np.zeros(len(y), int)
    for r in records:
        counts[r['fit']] += 1
    if not (counts == counts[0]).all():
        raise ValueError('Unequal tuning contributions')
    baseline_candidates = [c['candidate'] for c in old_records[0]['candidates']]
    baseline_options = []
    for j, c in enumerate(baseline_candidates):
        pred = np.concatenate([r['candidates'][j]['predictions'] for r in old_records])
        baseline_options.append({'candidate': c, **score(labels, pred)})
    base_index = max(range(len(baseline_options)), key=lambda i: (baseline_options[i]['correct'], baseline_options[i]['macro_f1'], -i))
    # Baseline here is a single configuration selected across all training-only inner predictions.
    # The nested outer comparator remains the original baseline search procedure.
    pooled_old = {'candidates': [{'predictions': np.concatenate([r['candidates'][base_index]['predictions'] for r in old_records]).tolist()}],
                  'policies': {'accuracy': {'selection': {'index': 0}}}}
    rows = option_scores(pooled_inner, pooled_old, labels, settings)
    directory = out/'final'; directory.mkdir()
    selections = {n: choose(rows, n, settings) for n in settings['policies']}
    atomic_json(directory/'training_selection.json', {'options': rows, 'selected': selections,
                'baseline_single_candidate': baseline_options[base_index],
                'predictions_per_training_participant': int(counts[0]),
                'note': 'Selection scores only; optimistic and not validation performance.'})
    models = {}
    with threadpool_limits(limits=settings['cpu_threads']), offline_network():
        for name, i in selections.items():
            c = candidates(settings)[i]
            if i:
                model = estimator(c, settings); feature = 'semantic_av'
            else:
                from .semantic_severe_repair import estimator_for
                c = baseline_candidates[base_index]; model = estimator_for(c, old_settings); feature = c['feature']
            fit(model, x[feature], y)
            state = {'protocol': PROTOCOL, 'estimator': model, 'feature': feature, 'candidate': c,
                     'fit_participants': frame.participant_id.tolist()}
            path = directory/f'{name}.pkl'; save_model(path, state, x[feature])
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': c,
                            'fit_receipt': receipt(model, list(range(len(y)))) if i else None}
    baseline_path = Path('results/catboost_holdout/baseline49_refit.pkl')
    baseline_plan = json.loads(Path('results/catboost_holdout/plan.json').read_text())['plan']
    if sha256(baseline_path) != baseline_plan['models']['baseline49_refit']['sha256'] or frame.participant_id.tolist() != baseline_plan['training_ids']:
        raise ValueError('Original frozen full-training baseline changed')
    models['baseline_refit'] = {'path': str(baseline_path), 'sha256': sha256(baseline_path),
                              'candidate': baseline_plan['models']['baseline49_refit']['candidate']}
    bound_paths = [out/'protocol.json', directory/'training_selection.json', Path(__file__),
                   Path('configs/imbalance_accuracy.json'), Path('data/metadata/participants.csv'),
                   Path('results/semantic_features/embeddings.json'),
                   Path('results/temporal_summary_comparison/participant_features.npz')]
    plan = {'protocol': PROTOCOL, 'context_hash': completion['context_hash'], 'models': models,
            'training_ids': frame.participant_id.tolist(), 'primary': 'accuracy',
            'bindings': {str(p): sha256(p) for p in bound_paths},
            'validation_previously_exposed': True, 'independent_confirmation': False,
            'validation_may_select_models': False, 'test_scoring': False}
    atomic_json(directory/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return {'frozen': True, 'new_refits': len(selections), 'models': models}


def evaluate(output=OUTPUT):
    from .catboost_late_fusion import load_all_inputs
    out = Path(output); envelope = json.loads((out/'final'/'plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Frozen plan changed')
    for path, expected in plan['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen input changed')
    for model in plan['models'].values():
        if sha256(model['path']) != model['sha256']:
            raise ValueError('Frozen model changed')
    all_x, manifest = load_all_inputs()
    use = manifest.split.eq('validation').to_numpy(); frame = manifest.loc[use].reset_index(drop=True)
    if set(frame.participant_id) & set(plan['training_ids']) or frame.participant_id.duplicated().any():
        raise ValueError('Validation participant leakage')
    av = np.column_stack([all_x['audio'][use], all_x['visual'][use]])
    x = {'semantic': all_x['text'][use], 'semantic_av': np.column_stack([all_x['text'][use], av])}
    rows = []
    with output_lock(out), threadpool_limits(limits=2), offline_network():
        for name, model in plan['models'].items():
            directory = out/'validation'/name; path = directory/'result.json'
            if path.exists():
                r = load_record(path, envelope['plan_hash'])
                if r['participant_ids'] != frame.participant_id.tolist() or r['model_sha256'] != model['sha256']:
                    raise ValueError('Validation ownership changed')
                p = np.asarray(r['probabilities'])
            else:
                with Path(model['path']).open('rb') as f:
                    state = pickle.load(f)
                feature = state.get('feature', state['candidate']['feature'])
                p = five_probabilities(state['estimator'], x[feature])
                directory.mkdir(parents=True)
                save_record(path, {'participant_ids': frame.participant_id.tolist(), 'model_sha256': model['sha256'],
                                  'probabilities': p.tolist()}, envelope['plan_hash'])
            if p.shape != (len(frame), 5) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
                raise ValueError('Invalid validation probabilities')
            write_report(frame, p, directory, name, 42, 'reused_official_validation')
            rows.append({'model': name, **score(frame.severity_class.to_numpy(int), p.argmax(1))})
        pd.DataFrame(rows).to_csv(out/'validation_scores.csv', index=False)
        atomic_json(out/'validation_receipt.json', {'plan_hash': envelope['plan_hash'], 'participants': len(frame),
                    'original_participants_only': True, 'synthetic_evaluation_rows': 0,
                    'independent_confirmation': False, 'prior_holdout_exposure': True,
                    'test_scored': False, 'selected_using_validation': False})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    settings = json.loads(Path('configs/imbalance_accuracy.json').read_text())
    if args.evaluate:
        result = evaluate()
    else:
        inputs = load_baseline()
        result = finalize(inputs, settings) if args.finalize else run(inputs, settings, resume=args.resume, max_new_jobs=args.max_new_jobs)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    from .imbalance_accuracy import main as module_main
    module_main()
