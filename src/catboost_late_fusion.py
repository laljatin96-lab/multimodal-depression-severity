"""Participant-wise CatBoost late fusion with matched recorded-gender ablation."""
import argparse
import json
from pathlib import Path
import pickle
import time

import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest, joined_manifest
from .crossvalidated_baselines import five_probabilities
from .encoder_kernel_accuracy import offline_network
from .evaluate import metrics, write_report
from .final_evaluation import sha256, verify_plan
from .modality_diagnostics import nested_folds, modality_columns
from .raw_frame_uncertainty import METRICS, participant_counts, weighted_metrics
from .severe_case_diagnosis import load_baseline
from .semantic_pipeline_audit import load_reference
from .severe_detection_study import decide, score

PROTOCOL = 'catboost-ordered-late-fusion-recorded-gender-v1'
MODALITIES = ['text', 'audio', 'visual']
VARIANTS = {'non_gender': False, 'gender_aware': True}
OUTPUT = Path('results/catboost_late_fusion')


class FeatureMap:
    def __init__(self, components):
        self.components = components

    def fit(self, x, indices):
        self.fit_indices = list(indices)
        self.scaler = StandardScaler().fit(x['text'][indices])
        z = self.scaler.transform(x['text'][indices])
        self.pca = PCA(n_components=min(self.components, len(indices)-1, z.shape[1]), svd_solver='full').fit(z)
        return self

    def transform(self, x, indices):
        return {'text': self.pca.transform(self.scaler.transform(x['text'][indices])),
                'audio': x['audio'][indices], 'visual': x['visual'][indices]}


def profiles(settings):
    return [{**p, 'auto_class_weights': w} for p in settings['capacity_profiles'] for w in settings['auto_class_weights']]


def model_input(values, gender, enabled):
    if not enabled:
        return np.asarray(values)
    gender = np.asarray(gender)
    if len(gender) != len(values) or not set(gender) <= {'female', 'male', 'unknown'}:
        raise ValueError('Explicit recorded gender or unknown required')
    table = pd.DataFrame(values, columns=[f'x{i}' for i in range(values.shape[1])])
    table['recorded_gender'] = gender
    return table


def fit_head(values, y, gender, enabled, profile, settings):
    if set(y) != set(range(5)) or not np.isfinite(values).all():
        raise ValueError('All real fitting classes and finite features required')
    model = CatBoostClassifier(**profile, loss_function='MultiClass', classes_count=5,
        boosting_type=settings['boosting_type'], random_seed=settings['fit_seed'],
        thread_count=settings['cpu_threads'], border_count=settings['border_count'],
        random_strength=1, bootstrap_type='Bayesian', bagging_temperature=1,
        one_hot_max_size=3, use_best_model=False, allow_writing_files=False, verbose=False)
    model.fit(model_input(values, gender, enabled), y,
              cat_features=['recorded_gender'] if enabled else None)
    return model


def predict_head(model, values, gender, enabled):
    p = five_probabilities(model, model_input(values, gender, enabled))
    decide(p, None)
    return p


def fuse(probabilities, weights):
    p, w = np.asarray(probabilities), np.asarray(weights)
    if p.ndim != 3 or p.shape[0] != 3 or w.shape != (3,) or (w <= 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('Three modalities and positive normalized fusion weights required')
    for a in p:
        decide(a, None)
    return np.einsum('m,mnc->nc', w, p)


def options(inner, y, settings):
    p = np.asarray(inner)
    if p.shape != (len(profiles(settings)), 3, len(y), 5):
        raise ValueError('Incomplete inner modality coverage')
    return [{'profile_index': i, 'weights': w, **score(y, fuse(a, w).argmax(1))}
            for i, a in enumerate(p) for w in settings['fusion_weights']]


def choose(rows):
    return max(range(len(rows)), key=lambda i: (rows[i]['correct'], rows[i]['macro_f1'], -i))


def predict_state(state, x, gender, indices):
    z = state['feature_map'].transform(x, indices)
    parts = [predict_head(state['heads'][m], z[m], gender[indices], state['gender']) for m in MODALITIES]
    return fuse(parts, state['weights'])


def save_state(path, state):
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as f:
        pickle.dump(state, f)
    temporary.replace(path)


def job(x, frame, fold, settings, out):
    start = time.monotonic()
    y = frame.severity_class.to_numpy(int)
    gender = frame.gender.to_numpy(str)
    search = profiles(settings)
    arrays = {name: np.full((len(search), 3, len(y), 5), np.nan) for name in VARIANTS}
    for part in fold['inner']:
        mapper = FeatureMap(settings['text_components']).fit(x, part['fit'])
        train, held = mapper.transform(x, part['fit']), mapper.transform(x, part['heldout'])
        for name, enabled in VARIANTS.items():
            for i, profile in enumerate(search):
                for j, modality in enumerate(MODALITIES):
                    model = fit_head(train[modality], y[part['fit']], gender[part['fit']], enabled, profile, settings)
                    arrays[name][i, j, part['heldout'], :] = predict_head(model, held[modality], gender[part['heldout']], enabled)
    inner = {n: a[:, :, fold['fit'], :].tolist() for n, a in arrays.items()}
    rows = {n: options(p, y[fold['fit']], settings) for n, p in inner.items()}
    mapper = FeatureMap(settings['text_components']).fit(x, fold['fit'])
    train = mapper.transform(x, fold['fit'])
    out.mkdir(parents=True, exist_ok=True)
    selected = {}
    for name, enabled in VARIANTS.items():
        i = choose(rows[name]); option = rows[name][i]; profile = search[option['profile_index']]
        heads = {m: fit_head(train[m], y[fold['fit']], gender[fold['fit']], enabled, profile, settings) for m in MODALITIES}
        state = {'protocol': PROTOCOL, 'feature_map': mapper, 'heads': heads, 'profile': profile,
                 'weights': option['weights'], 'gender': enabled, 'fit': fold['fit'], 'heldout': fold['heldout']}
        p = predict_state(state, x, gender, fold['heldout'])
        path = out/f'{name}.pkl'; save_state(path, state)
        with path.open('rb') as f:
            restored = pickle.load(f)
        np.testing.assert_allclose(p, predict_state(restored, x, gender, fold['heldout']), rtol=0, atol=1e-14)
        selected[name] = {'index': i, 'profile': profile, 'weights': option['weights'],
                          'probabilities': p.tolist(), 'checkpoint_sha256': sha256(path)}
    return {k: fold[k] for k in ['seed', 'fold', 'fit', 'heldout']} | {
        'inner_probabilities': inner, 'options': rows, 'selected': selected,
        'head_fits': len(search)*3*len(VARIANTS)*len(fold['inner'])+6,
        'feature_map_fits': len(fold['inner'])+1, 'seconds': time.monotonic()-start}


def validate(r, fold, y, settings, directory):
    if any(r[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']):
        raise ValueError('Fold changed')
    if set(r['selected']) != set(VARIANTS) or set(r['inner_probabilities']) != set(VARIANTS):
        raise ValueError('Variant coverage changed')
    for name in VARIANTS:
        rows = options(r['inner_probabilities'][name], y[fold['fit']], settings)
        i = choose(rows); selected = r['selected'][name]
        if r['options'][name] != rows or selected['index'] != i or selected['profile'] != profiles(settings)[rows[i]['profile_index']] or selected['weights'] != rows[i]['weights']:
            raise ValueError('Training-only selection changed')
        p = np.asarray(selected['probabilities']); decide(p, None)
        if p.shape != (len(fold['heldout']), 5) or sha256(directory/f'{name}.pkl') != selected['checkpoint_sha256']:
            raise ValueError('Checkpoint or output coverage changed')
    if r['head_fits'] != len(profiles(settings))*3*len(VARIANTS)*len(fold['inner'])+6 or r['feature_map_fits'] != len(fold['inner'])+1:
        raise ValueError('Fit budget changed')


def summarize(frame, records, old49, old48, settings, out):
    y = frame.severity_class.to_numpy(int); rows = []; predictions = {}; matrices = {}
    for name in ['baseline_49', 'baseline_48', *VARIANTS]:
        repeated, cms = [], []
        for seed in sorted({r['seed'] for r in records}):
            p = np.full((len(y), 5), np.nan); owners = np.zeros(len(y), int)
            source = old49 if name == 'baseline_49' else old48 if name == 'baseline_48' else records
            for r in source:
                if r['seed'] != seed:
                    continue
                value = r['policies']['accuracy']['probabilities'] if name == 'baseline_49' else r['probabilities'] if name == 'baseline_48' else r['selected'][name]['probabilities']
                p[r['heldout']] = value; owners[r['heldout']] += 1
            if not (owners == 1).all():
                raise ValueError('Incomplete participant ownership')
            decide(p, None)
            report = write_report(frame, p, out/'oof'/f'{name}_seed{seed}', name, seed, 'train_nested_oof')
            rows.append({'model': name, 'seed': seed, **score(y, p.argmax(1))})
            repeated.append(p.argmax(1)); cms.append(report['confusion_matrix'])
        predictions[name] = np.stack(repeated); matrices[name] = np.asarray(cms).sum(0).tolist()
    scores = pd.DataFrame(rows); scores.to_csv(out/'repeat_scores.csv', index=False)
    summary = scores.groupby('model')[METRICS+['severe_recall', 'severe_false_positives']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]; summary.to_csv(out/'summary.csv')
    counts = participant_counts(len(y), 5000, 20261026)
    boot = {n: weighted_metrics(y, p, counts) for n, p in predictions.items()}
    comparisons = []
    for a, b in [(n, base) for n in VARIANTS for base in ['baseline_49', 'baseline_48']] + [('gender_aware', 'non_gender')]:
        intervals = np.percentile(boot[a]-boot[b], [2.5, 97.5], axis=0)
        for i, metric in enumerate(METRICS):
            comparisons.append({'model': a, 'reference': b, 'metric': metric,
                                'difference': float(summary.loc[a, metric+'_mean']-summary.loc[b, metric+'_mean']),
                                'ci95_low': float(intervals[0, i]), 'ci95_high': float(intervals[1, i])})
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv', index=False)
    atomic_json(out/'summed_confusion_matrices.json', {'note': 'Same 163 people in three repeats, not 489 independent participants; actual rows and predicted columns 0-4.', 'matrices': matrices})
    pd.DataFrame([{'seed': r['seed'], 'fold': r['fold'], 'variant': n, 'weights': json.dumps(v['weights']), **v['profile']}
                  for r in records for n, v in r['selected'].items()]).to_csv(out/'choices.csv', index=False)
    return {n: float(summary.loc[n, 'accuracy_mean']) for n in summary.index}


def load_inputs():
    baseline_x, frame, old_settings, old49, provenance = load_baseline()
    x48, frame48, settings48, old48, provenance48 = load_reference()
    if frame.to_dict('records') != frame48.to_dict('records') or nested_folds(frame, old_settings) != nested_folds(frame48, settings48):
        raise ValueError('Historical baseline participants/folds differ')
    for n in baseline_x:
        np.testing.assert_array_equal(baseline_x[n], x48[n])
    av = baseline_x['semantic_av'][:, baseline_x['semantic'].shape[1]:]
    x = {'text': baseline_x['semantic'], 'audio': av[:, modality_columns('A')], 'visual': av[:, modality_columns('V')]}
    bindings = {**provenance['bindings'], **provenance48['source_bindings']}
    paths = ['results/temporal_summary_comparison/participant_features.npz', 'results/semantic_features/embeddings.json',
             'data/metadata/participants.csv', 'data/splits/participants.csv', 'results/final_evaluation_plan.json']
    bindings.update({p: sha256(p) for p in paths})
    return x, frame, old_settings, old49, old48, {'baseline49': provenance['baseline_context_hash'],
        'baseline48': provenance48['source_context_hash'], 'bindings': bindings}


def load_all_inputs():
    """All cached features for final frozen inference; this function performs no fitting."""
    plan = verify_plan('results/final_evaluation_plan.json')
    frame = joined_manifest(plan['feature_config']).reset_index(drop=True)
    if digest(frame.to_dict('records')) != plan['manifest_hash']:
        raise ValueError('Original manifest changed')
    with np.load('results/temporal_summary_comparison/participant_features.npz', allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['participant_ids'], frame.participant_id)
        av = cache['cached_av'].copy()
    semantic = json.loads(Path('results/semantic_features/embeddings.json').read_text())
    if [r['participant_id'] for r in semantic] != frame.participant_id.tolist():
        raise ValueError('Semantic participant order changed')
    text = np.asarray([r['embedding'] for r in semantic])
    return {'text': text, 'audio': av[:, modality_columns('A')], 'visual': av[:, modality_columns('V')]}, frame


def run(x, frame, old_settings, old49, old48, provenance, settings, output, resume=False, max_new_jobs=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing experiment')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires protocol')
    if max_new_jobs is not None and max_new_jobs < 1:
        raise ValueError('Positive pilot budget required')
    folds = nested_folds(frame, old_settings); y = frame.severity_class.to_numpy(int)
    if len(folds) != settings['expected_jobs'] or len(profiles(settings))*3*2*old_settings['inner_folds'] != settings['inner_head_fits_per_job']:
        raise ValueError('Declared budget mismatch')
    if frame.participant_id.duplicated().any() or set(x) != set(MODALITIES) or not set(frame.gender) <= {'female', 'male', 'unknown'}:
        raise ValueError('Invalid real participants/features/gender')
    for a in x.values():
        if a.ndim != 2 or len(a) != len(y) or not np.isfinite(a).all():
            raise ValueError('Invalid feature matrix')
    for records in [old49, old48]:
        if len(records) != len(folds) or any(any(r[k] != f[k] for k in ['seed', 'fold', 'fit', 'heldout']) for r, f in zip(records, folds)):
            raise ValueError('Historical folds changed')
    context = {'protocol': PROTOCOL, 'settings': settings, 'old_settings': old_settings, 'training': frame.to_dict('records'),
        'folds': folds, 'old49': old49, 'old48': old48, 'provenance': provenance, 'code_sha256': sha256(__file__),
        'feature_hashes': {n: digest(a.tolist()) for n, a in x.items()},
        'support_hashes': {n: sha256(Path(__file__).with_name(n)) for n in ['severe_case_diagnosis.py', 'semantic_pipeline_audit.py',
            'severe_detection_study.py', 'crossvalidated_baselines.py', 'modality_diagnostics.py', 'raw_frame_uncertainty.py',
            'evaluate.py', 'batch_inference.py', 'encoder_kernel_accuracy.py']},
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'catboost': catboost.__version__}}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != envelope:
                raise ValueError('Inputs/code changed')
        else:
            atomic_json(out/'protocol.json', envelope)
        records = []; new = 0
        for fold in folds:
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; path = directory/'result.json'
            if path.exists():
                r = load_record(path, ch)
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                r = job(x, frame, fold, settings, directory); save_record(path, r, ch); new += 1
            validate(r, fold, y, settings, directory); records.append(r)
            print(f'CatBoost late fusion: {len(records)}/{len(folds)} jobs; {new} new', flush=True)
        for p, h in provenance.get('bindings', {}).items():
            if sha256(p) != h:
                raise ValueError('Bound inputs changed')
        complete = len(records) == len(folds)
        result = summarize(frame, records, old49, old48, settings, out) if complete else None
        receipt = {'complete': complete, 'context_hash': ch, 'completed_jobs': len(records), 'new_jobs': new,
            'new_head_fits': new*(settings['inner_head_fits_per_job']+6), 'head_fits': sum(r['head_fits'] for r in records),
            'feature_map_fits': sum(r['feature_map_fits'] for r in records), 'accuracy': result,
            'validation_test_scored': False, 'reference_replaced': False, 'synthetic_records': 0}
        atomic_json(out/'completion.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    settings = json.loads(Path('configs/catboost_late_fusion.json').read_text())
    print(json.dumps(run(*load_inputs(), settings, OUTPUT, args.resume, args.max_new_jobs), indent=2))


if __name__ == '__main__':
    from .catboost_late_fusion import main as module_main
    module_main()
