"""Nested accuracy comparison of cached moments and richer temporal summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .accuracy_model_comparison import choose_candidate, label_metrics
from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest, joined_manifest, write_json
from .crossvalidated_baselines import fit, five_probabilities
from .evaluate import write_report
from .final_evaluation import sha256, verify_plan
from .fold_feature_selection import FoldAnovaSelector
from .modality_diagnostics import nested_folds
from .raw_frame_features import input_state
from .raw_frame_uncertainty import METRICS, participant_counts, weighted_metrics
from .temporal_summaries import DIMENSIONS, STATISTICS, feature_names, prepare_all

PROTOCOL = 'post-final-temporal-summary-anova20-logistic-v1'


def select_training(matrices, manifest):
    if manifest.participant_id.duplicated().any() or set(manifest.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected unique participants and original split coverage')
    use = manifest.split.eq('train').to_numpy()
    training = manifest.loc[use, ['participant_id', 'severity_class', 'gender']].reset_index(drop=True)
    y = training.severity_class.to_numpy()
    if not np.isfinite(y).all() or not np.equal(y, y.astype(int)).all() or not set(y) <= set(range(5)):
        raise ValueError('Training targets must be integer classes 0–4')
    return {key: np.asarray(x)[use] for key, x in matrices.items()}, training


def load_inputs(settings):
    source, coverage, sequences, _ = input_state(settings['frozen_plan'])
    manifest = joined_manifest(source['feature_config'])
    plan = verify_plan(settings['frozen_plan'])
    if digest(manifest.to_dict('records')) != plan['manifest_hash'] or manifest[['participant_id', 'split']].to_dict('records') != coverage.to_dict('records'):
        raise ValueError('Original participant targets/splits changed')
    matrices, quality = prepare_all(sequences, coverage.participant_id.tolist())
    _, training = select_training(matrices, manifest)
    previous = json.loads(Path(settings['previous_protocol']).read_text())
    if digest(previous['context']) != previous['context_hash'] or previous['context']['folds'] != nested_folds(training, settings):
        raise ValueError('Comparison must retain the previous nested participant folds')
    for name, expected in previous['context']['implementation_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError('Previous experiment implementation changed')
    provenance = {'source_context': source, 'previous_protocol_sha256': sha256(settings['previous_protocol']),
                  'all_participants': len(coverage), 'all_sources': len(source['sources']),
                  'archives': sum(s['kind'] == 'archive' for s in source['sources']),
                  'folders': sum(s['kind'] == 'directory' for s in source['sources'])}
    return matrices, manifest, quality, provenance


def checked_fit(x, y, c, settings):
    estimator = make_pipeline(StandardScaler(), FoldAnovaSelector(settings['selector_k']),
                              LogisticRegression(C=c, max_iter=settings['max_iter'], solver='lbfgs', random_state=settings['fit_seed']))
    fit(estimator, x, y)
    np.testing.assert_array_equal(estimator.classes_, np.arange(5))
    scaler = estimator.named_steps['standardscaler']
    np.testing.assert_allclose(scaler.mean_, x.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, x.var(0), rtol=1e-12, atol=1e-12)
    return estimator


def selection_check(estimator, rows, y):
    return {'fit_count': len(rows), 'class_counts': np.bincount(y[rows], minlength=5).tolist(),
            'selected_columns': estimator.named_steps['foldanovaselector'].selected_.tolist(), 'scaler_verified': True}


def run_job(x, y, fold, feature, settings, directory):
    started = time.monotonic()
    candidates, checks = [], []
    for c in sorted(settings['C_candidates']):
        predictions = np.full(len(y), -1, int)
        for i, inner in enumerate(fold['inner']):
            rows, heldout = np.asarray(inner['fit']), np.asarray(inner['heldout'])
            estimator = checked_fit(x[rows], y[rows], c, settings)
            predictions[heldout] = estimator.predict(x[heldout])
            checks.append({'C': c, 'inner_fold': i, **selection_check(estimator, rows, y)})
        predicted = predictions[fold['fit']]
        score = label_metrics(y[fold['fit']], predicted)
        candidates.append({'C': c, 'inner_predictions': predicted.tolist(), 'inner_correct': int((predicted == y[fold['fit']]).sum()),
                           'inner_accuracy': score['accuracy'], 'inner_macro_f1': score['macro_f1']})
    chosen = choose_candidate(candidates)['C']
    estimator = checked_fit(x[fold['fit']], y[fold['fit']], chosen, settings)
    probabilities = five_probabilities(estimator, x[fold['heldout']])
    selection = selection_check(estimator, np.asarray(fold['fit']), y)
    path = Path(directory)/'model.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'estimator': estimator, 'protocol': PROTOCOL, 'feature_input': feature, 'feature_names': feature_names()[feature],
             'fit': fold['fit'], 'heldout': fold['heldout'], 'selected_C': chosen,
             'selected_feature_names': [feature_names()[feature][i] for i in selection['selected_columns']]}
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(state, handle)
    temporary.replace(path)
    with path.open('rb') as handle:
        restored = pickle.load(handle)  # Only this locally created checkpoint.
    np.testing.assert_array_equal(probabilities, five_probabilities(restored['estimator'], x[fold['heldout']]))
    return {'feature': feature, 'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
            'selected_C': chosen, 'inner_candidates': candidates, 'inner_preprocessing': checks, 'outer_preprocessing': selection,
            'outer_probabilities': probabilities.tolist(), 'model_sha256': sha256(path), 'checkpoint_parity': True,
            'fits': len(candidates)*len(fold['inner'])+1, 'seconds': time.monotonic()-started}


def validate_record(record, fold, feature, settings, directory, y):
    expected = {'feature': feature, 'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
                'fits': len(settings['C_candidates'])*len(fold['inner'])+1, 'checkpoint_parity': True}
    if any(record.get(k) != v for k, v in expected.items()) or sha256(Path(directory)/'model.pkl') != record['model_sha256']:
        raise ValueError('Saved job/checkpoint identity changed')
    candidates = record['inner_candidates']
    if [c['C'] for c in candidates] != sorted(settings['C_candidates']):
        raise ValueError('Candidate grid changed')
    for c in candidates:
        prediction = np.asarray(c['inner_predictions'])
        score = label_metrics(y[fold['fit']], prediction)
        if (c['inner_correct'] != int((prediction == y[fold['fit']]).sum()) or
                any(not np.isfinite(c['inner_'+key]) or abs(c['inner_'+key]-score[key]) > 1e-12 for key in ['accuracy', 'macro_f1'])):
            raise ValueError('Saved inner metrics disagree with predictions')
    if record['selected_C'] != choose_candidate(candidates)['C']:
        raise ValueError('Invalid accuracy-first C selection')
    expected_checks = [(c, i, inner['fit']) for c in sorted(settings['C_candidates']) for i, inner in enumerate(fold['inner'])]
    if len(record['inner_preprocessing']) != len(expected_checks):
        raise ValueError('Incomplete inner preprocessing checks')
    checks = []
    for saved, (c, i, rows) in zip(record['inner_preprocessing'], expected_checks):
        if saved['C'] != c or saved['inner_fold'] != i:
            raise ValueError('Inner preprocessing order changed')
        checks.append((saved, rows))
    checks.append((record['outer_preprocessing'], fold['fit']))
    for check, rows in checks:
        selected = check['selected_columns']
        if (check['fit_count'] != len(rows) or check['class_counts'] != np.bincount(y[rows], minlength=5).tolist()
                or check['scaler_verified'] is not True or not 1 <= len(selected) <= settings['selector_k']
                or sorted(set(selected)) != selected or any(not isinstance(i, int) or not 0 <= i < DIMENSIONS[feature] for i in selected)):
            raise ValueError('Invalid saved fold-local feature selection')
    p = np.asarray(record['outer_probabilities'])
    if p.shape != (len(fold['heldout']), 5) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid saved class probabilities')


def summarize(training, records, folds, settings, output, all_participants, all_sources):
    y, rows, predictions = training.severity_class.to_numpy(), [], {}
    for feature in settings['feature_inputs']+['majority']:
        repeated = []
        for seed in settings['fold_seeds']:
            p, owners = np.full((len(y), 5), np.nan), np.zeros(len(y), int)
            if feature == 'majority':
                for fold in [f for f in folds if f['seed'] == seed]:
                    p[fold['heldout']] = np.eye(5)[np.bincount(y[fold['fit']], minlength=5).argmax()]
                    owners[fold['heldout']] += 1
            else:
                for record in records:
                    if record['feature'] == feature and record['seed'] == seed:
                        p[record['heldout']] = record['outer_probabilities']
                        owners[record['heldout']] += 1
            if not (owners == 1).all() or not np.isfinite(p).all():
                raise ValueError('OOF predictions require one observation per participant per partition')
            model = feature if feature == 'majority' else f"LR-FS{settings['selector_k']}-{feature}"
            report = write_report(training, p, output/'oof'/f'{model}_seed{seed}', model, seed, 'train_nested_oof')
            predicted = p.argmax(1)
            repeated.append(predicted)
            rows.append({'feature': feature, 'model': model, 'seed': seed, **{m: report[m] for m in METRICS},
                         'severe_correct': int(((predicted == 4) & (y == 4)).sum()),
                         'predicted_class_counts': json.dumps(np.bincount(predicted, minlength=5).tolist())})
        predictions[feature] = np.stack(repeated)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'repeat_scores.csv', index=False)
    summary = scores.groupby(['feature', 'model'], sort=False)[METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output/'summary.csv', index=False)
    counts = participant_counts(len(y), settings['bootstrap_repetitions'], settings['bootstrap_seed'])
    boot = {name: weighted_metrics(y, p, counts) for name, p in predictions.items()}
    points = {name: weighted_metrics(y, p, np.ones((1, len(y)), int))[0] for name, p in predictions.items()}
    pairs = [('majority', name) for name in settings['feature_inputs']]
    pairs += [(f'cached_{m}', f'temporal_{m}') for m in ['visual', 'av'] if {f'cached_{m}', f'temporal_{m}'} <= predictions.keys()]
    comparisons = []
    for baseline, candidate in pairs:
        ci = np.percentile(boot[candidate]-boot[baseline], [2.5, 97.5], axis=0)
        for i, metric in enumerate(METRICS):
            comparisons.append({'baseline': baseline, 'candidate': candidate, 'metric': metric,
                                'mean_difference': points[candidate][i]-points[baseline][i], 'ci95_low': ci[0, i], 'ci95_high': ci[1, i]})
    pd.DataFrame(comparisons).to_csv(output/'paired_comparisons.csv', index=False)
    selections = []
    for r in records:
        for column in r['outer_preprocessing']['selected_columns']:
            selections.append({'feature': r['feature'], 'seed': r['seed'], 'fold': r['fold'], 'C': r['selected_C'],
                               'column': column, 'feature_name': feature_names()[r['feature']][column]})
    pd.DataFrame(selections).to_csv(output/'outer_selected_features.csv', index=False)
    plot_summary(summary, output/'comparison.png')
    lines = ['# Temporal-summary accuracy experiment', '',
             'Official training participants only; same nested folds and accuracy-first selection as the previous comparison. '
             'Every model uses the same fold-local top-20 ANOVA selector. Original moments are the matched representation controls.', '',
             '| Features | Accuracy | Macro-F1 | Balanced accuracy |', '|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.feature} | {r.accuracy_mean:.4f} ± {r.accuracy_std:.4f} | {r.macro_f1_mean:.4f} | {r.balanced_accuracy_mean:.4f} |')
    lines += ['', f'All {all_participants} dataset participants and {all_sources} physical sources remain accounted for upstream. Only {len(training)} official training participants enter fitting/scoring. '
              'No new validation/test evaluation, full-training refit or reference replacement.', '',
              'This post-final exploratory protocol does not provide independent confirmation. Accuracy must be read with class recall and the majority baseline. '
              'Partition SD is not independent replication uncertainty. Paired participant intervals condition on saved fitted models, share draws across partitions and are unadjusted for multiple comparisons.', '',
              'Richer features summarize existing 0.5-second bins and cannot recover sub-bin information or establish participant-only speech. '
              'Lag statistics never bridge missing bins; unsupported/constant correlations are zero with support in feature_support.csv. '
              'Original crop and unresolved speaker/timing limitations remain. Logistic probabilities are uncalibrated research outputs.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric in zip(axes, ['accuracy', 'macro_f1']):
        ax.barh(summary.feature, summary[metric+'_mean'], xerr=summary[metric+'_std'].fillna(0), capsize=3,
                color=['#bd753c' if f.startswith('temporal') else '#7598ac' if f != 'majority' else '#999999' for f in summary.feature])
        ax.invert_yaxis()
        ax.set_xlim(0, max(.6, (summary[metric+'_mean']+summary[metric+'_std'].fillna(0)).max()+.03))
        ax.set_xlabel('Accuracy' if metric == 'accuracy' else 'Five-class macro-F1')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Training-only nested comparison: fixed fold-local ANOVA top-20 + logistic\nBars: mean over three partitions; error bars: partition SD')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiment(matrices, manifest, quality, provenance, settings, output, resume=False, max_new_jobs=None, verify_inputs=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing output; use --resume with unchanged inputs')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires a saved protocol')
    features, grid = settings['feature_inputs'], settings['C_candidates']
    if (not features or len(set(features)) != len(features) or not set(features) <= DIMENSIONS.keys()
            or not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all() or min(grid) <= 0
            or settings['statistics'] != STATISTICS or settings['selector_k'] < 1 or settings['max_iter'] < 1
            or not 1 <= settings['cpu_threads'] <= 2 or (max_new_jobs is not None and max_new_jobs < 1)):
        raise ValueError('Invalid declared experiment settings')
    for feature in features:
        if np.shape(matrices[feature]) != (len(manifest), DIMENSIONS[feature]) or not np.isfinite(matrices[feature]).all():
            raise ValueError('Invalid all-participant feature matrix')
    selected, training = select_training(matrices, manifest)
    y, folds = training.severity_class.to_numpy(int), nested_folds(training, settings)
    jobs = [(fold, feature) for fold in folds for feature in features]
    fits_per_job = len(grid)*settings['inner_folds']+1
    if len(jobs) != settings['expected_jobs'] or len(jobs)*fits_per_job != settings['expected_fits']:
        raise ValueError('Declared compute budget does not match experiment')
    implementation = ['fold_feature_selection.py', 'temporal_summary_comparison.py', 'temporal_summaries.py', 'accuracy_model_comparison.py',
                      'raw_frame_features.py', 'raw_frame_uncertainty.py', 'modality_diagnostics.py', 'crossvalidated_baselines.py',
                      'batch_inference.py', 'exploratory_baselines.py', 'evaluate.py', 'common.py']
    context = {'protocol': PROTOCOL, 'settings': settings, 'training_participants': training.to_dict('records'), 'folds': folds,
               'coverage': manifest[['participant_id', 'split']].to_dict('records'), 'provenance': provenance, 'feature_names': feature_names(),
               'matrix_sha256': {f: hashlib.sha256(np.asarray(matrices[f], dtype=np.float64).tobytes()).hexdigest() for f in features},
               'quality_sha256': digest(quality), 'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
               'implementation_sha256': {name: sha256(Path(__file__).with_name(name)) for name in implementation}}
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']):
        envelope = {'context': context, 'context_hash': context_hash}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != envelope:
                raise ValueError('Inputs, code or protocol changed')
            with np.load(out/'participant_features.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['participant_ids'], manifest.participant_id)
                for f in features:
                    np.testing.assert_array_equal(saved[f], matrices[f])
        else:
            atomic_json(out/'protocol.json', envelope)
            np.savez_compressed(out/'participant_features.npz', participant_ids=manifest.participant_id.to_numpy(dtype=str), **{f: matrices[f] for f in features})
            manifest[['participant_id', 'split']].to_csv(out/'participant_coverage.csv', index=False)
            pd.DataFrame(quality).to_csv(out/'feature_support.csv', index=False)
            write_json(out/'feature_schema.json', feature_names())
        records, new, reused = [], 0, 0
        for fold, feature in jobs:
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/feature
            path = directory/'result.json'
            if path.exists():
                record = load_record(path, context_hash)
                reused += 1
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                record = run_job(selected[feature], y, fold, feature, settings, directory)
                save_record(path, record, context_hash)
                new += 1
            validate_record(record, fold, feature, settings, directory, y)
            records.append(record)
            if len(records) % len(features) == 0:
                print(f'Temporal summaries: {len(records)}/{len(jobs)} jobs; {new} new, {reused} reused', flush=True)
        if verify_inputs:
            verify_inputs()
        complete = len(records) == len(jobs)
        if complete:
            summarize(training, records, folds, settings, out, len(manifest), provenance.get('all_sources', 'unspecified'))
        result = {'complete': complete, 'context_hash': context_hash, 'training_participants': len(training),
                  'all_participants': len(manifest), 'all_sources': provenance.get('all_sources'), 'completed_jobs': len(records),
                  'expected_jobs': len(jobs), 'completed_fits': len(records)*fits_per_job, 'new_jobs': new, 'reused_jobs': reused,
                  'new_fits': new*fits_per_job, 'fit_seconds': sum(r['seconds'] for r in records),
                  'validation_test_scoring': False, 'full_training_refit': False, 'reference_replaced': False}
        atomic_json(out/'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/temporal_summary_comparison.json')
    parser.add_argument('--output', default='results/temporal_summary_comparison')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    if Path(args.output).exists() and not args.resume:
        parser.error('Preserving existing output; use --resume with unchanged inputs')
    settings = json.loads(Path(args.config).read_text())
    matrices, manifest, quality, provenance = load_inputs(settings)
    protected = ['data', provenance['source_context']['feature_config']['dataset'], 'checkpoints', str(Path(settings['previous_protocol']).parent)]
    if any(Path(args.output).resolve().is_relative_to(Path(p).resolve()) for p in protected):
        parser.error('Output must be outside existing data/model/previous comparison directories')
    def verify_inputs():
        current, labels, support, source = load_inputs(settings)
        if source != provenance or not labels.equals(manifest) or support != quality or any(not np.array_equal(matrices[k], current[k]) for k in matrices):
            raise ValueError('Source inputs changed during fitting')
    print(json.dumps(run_experiment(matrices, manifest, quality, provenance, settings, args.output,
                                   args.resume, args.max_new_jobs, verify_inputs), indent=2))


if __name__ == '__main__':
    main()
