"""Nested training-only modality ablations; preserve the frozen final study."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest, joined_manifest, load_config
from .crossvalidated_baselines import fit, five_probabilities, make_estimator, training_folds
from .evaluate import metrics, write_report
from .exploratory_baselines import load_summary_matrix
from .final_evaluation import sha256, verify_plan
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS

PROTOCOL = 'post-final-training-nested-modality-diagnostic-v1'
VARIANTS = [(modality, balanced) for modality in ['A', 'V', 'AV'] for balanced in [False, True]]
LIMITATIONS = [
    'Designed after previous validation and final-test results were seen; internal diagnostic, not independent confirmation',
    'Nested folds separate regularization selection from outer scoring, but do not erase earlier dataset exposure',
    'Repeat SD describes partition sensitivity; folds/repeats share people and are not independent model replications',
    'Only seven severe-class training participants in the real dataset; no significance or clinical-validity claim',
    'Original common-time crop, 0.5-second averaging, unverified event synchrony and possible interviewer speech remain',
    'No validation/test predictions, final full-training refit or replacement of the frozen reference model',
]


def variant_name(modality, balanced):
    return 'Nested-LR-'+modality+('-balanced' if balanced else '')


def modality_columns(modality):
    audio_count, visual_count = 2*len(AUDIO_COLUMNS), 2*len(VISUAL_COLUMNS)
    return {'A': np.arange(audio_count), 'V': np.arange(audio_count, audio_count+visual_count),
            'AV': np.arange(audio_count+visual_count)}[modality]


def nested_folds(training, settings):
    result = []
    for seed, fold, outer_fit, heldout in training_folds(training, settings['outer_folds'], settings['fold_seeds']):
        inner_seed = int(seed + 1009*(fold+1))
        subset = training.iloc[outer_fit].reset_index(drop=True)
        inner = [{'fit': outer_fit[fit_indices].tolist(), 'heldout': outer_fit[validation].tolist()}
                 for _, _, fit_indices, validation in training_folds(subset, settings['inner_folds'], [inner_seed])]
        result.append({'seed': seed, 'fold': fold, 'fit': outer_fit.tolist(), 'heldout': heldout.tolist(),
                       'inner_seed': inner_seed, 'inner': inner})
    return result


def checked_fit(x, y, c, balanced, seed):
    estimator = make_estimator(c, balanced, seed)
    fit(estimator, x, y)
    scaler = estimator.named_steps['standardscaler']
    np.testing.assert_allclose(scaler.mean_, x.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, x.var(0), rtol=1e-12, atol=1e-12)
    return estimator


def run_job(matrix, y, fold, modality, balanced, settings, directory):
    """Only inner held-out labels choose C; outer held-out labels are not read."""
    start = time.monotonic()
    x = matrix[:, modality_columns(modality)]
    outer_fit, heldout = np.asarray(fold['fit']), np.asarray(fold['heldout'])
    candidates, preprocessing = [], []
    for c in sorted(settings['C_candidates']):
        inner_predictions = np.full((len(matrix), 5), np.nan)
        for inner_fold, partition in enumerate(fold['inner']):
            fit_indices, validation = np.asarray(partition['fit']), np.asarray(partition['heldout'])
            estimator = checked_fit(x[fit_indices], y[fit_indices], c, balanced, settings['fit_seed'])
            inner_predictions[validation] = five_probabilities(estimator, x[validation])
            preprocessing.append({'C': c, 'inner_fold': inner_fold, 'fit_count': len(fit_indices),
                                  'fit_class_counts': np.bincount(y[fit_indices], minlength=5).tolist(),
                                  'scaler_mean_and_variance_verified': True})
        probabilities = inner_predictions[outer_fit]
        if not np.isfinite(probabilities).all():
            raise ValueError('Incomplete inner out-of-fold predictions')
        score = metrics(y[outer_fit], probabilities)
        candidates.append({'C': c, 'inner_macro_f1': score['macro_f1'], 'inner_accuracy': score['accuracy'],
                           'inner_oof_probabilities': probabilities.tolist()})
    selected = max(candidates, key=lambda item: item['inner_macro_f1'])  # Sorted grid: smaller C wins exact ties.
    estimator = checked_fit(x[outer_fit], y[outer_fit], selected['C'], balanced, settings['fit_seed'])
    probabilities = five_probabilities(estimator, x[heldout])
    state = {'estimator': estimator, 'modality': modality, 'balanced': balanced, 'selected_C': selected['C'],
             'outer_fit_indices': outer_fit.tolist(), 'outer_heldout_indices': heldout.tolist(),
             'feature_columns': modality_columns(modality).tolist(), 'purpose': PROTOCOL}
    path = Path(directory)/'model.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(state, handle)
    temporary.replace(path)
    with path.open('rb') as handle:
        restored = pickle.load(handle)  # The artifact just produced locally, never an external checkpoint.
    np.testing.assert_array_equal(five_probabilities(restored['estimator'], x[heldout]), probabilities)
    return {'model': variant_name(modality, balanced), 'modality': modality, 'balanced': balanced,
            'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
            'selected_C': selected['C'], 'inner_candidates': candidates, 'preprocessing_checks': preprocessing,
            'outer_probabilities': probabilities.tolist(), 'model_sha256': sha256(path),
            'fits': len(candidates)*len(fold['inner'])+1, 'seconds': time.monotonic()-start,
            'outer_scaler_verified': True, 'checkpoint_prediction_parity': True}


def validate_job(record, fold, modality, balanced, settings, directory):
    expected = {'model': variant_name(modality, balanced), 'modality': modality, 'balanced': balanced,
                'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
                'fits': len(settings['C_candidates'])*len(fold['inner'])+1}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved job identity/folds differ')
    if sha256(Path(directory)/'model.pkl') != record['model_sha256']:
        raise ValueError('Saved diagnostic checkpoint changed')
    probs = np.asarray(record['outer_probabilities'])
    if probs.shape != (len(fold['heldout']), 5) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1):
        raise ValueError('Invalid saved outer predictions')


def summarize(training, records, folds, settings, out, audit_coverage=None):
    score_rows, selections = [], []
    for seed in settings['fold_seeds']:
        probabilities = {variant_name(m, b): np.full((len(training), 5), np.nan) for m, b in VARIANTS}
        probabilities['majority'] = np.full((len(training), 5), np.nan)
        ownership = {name: np.zeros(len(training), dtype=int) for name in probabilities}
        for fold in [item for item in folds if item['seed'] == seed]:
            label = int(np.bincount(training.iloc[fold['fit']].severity_class, minlength=5).argmax())
            probabilities['majority'][fold['heldout']] = np.eye(5)[label]
            ownership['majority'][fold['heldout']] += 1
        for record in [item for item in records if item['seed'] == seed]:
            name = record['model']
            probabilities[name][record['heldout']] = record['outer_probabilities']
            ownership[name][record['heldout']] += 1
            selections.append({k: record[k] for k in ['model', 'modality', 'balanced', 'seed', 'fold', 'selected_C', 'fits', 'seconds']})
        for name, probs in probabilities.items():
            if not np.isfinite(probs).all() or not (ownership[name] == 1).all():
                raise ValueError('Outer OOF coverage must be exactly once per participant, model and repeat')
            report = write_report(training, probs, out/'oof'/f'{name}_seed{seed}', name, seed, 'train_nested_oof')
            score_rows.append({'model': name, 'seed': seed,
                               **{key: report[key] for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']}})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out/'repeat_scores.csv', index=False)
    pd.DataFrame(selections).to_csv(out/'outer_selections.csv', index=False)
    summary = scores.groupby('model', sort=False)[['accuracy', 'macro_f1', 'balanced_accuracy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    pairs = [(variant_name(m, b), variant_name('AV', b)) for b in [False, True] for m in ['A', 'V']]
    pairs += [(variant_name(m, False), variant_name(m, True)) for m in ['A', 'V', 'AV']]
    deltas = []
    for first, second in pairs:
        a, b = [scores[scores.model == name].set_index('seed') for name in [first, second]]
        for seed in settings['fold_seeds']:
            deltas.append({'baseline': first, 'candidate': second, 'seed': seed,
                           **{key+'_difference': float(b.loc[seed, key]-a.loc[seed, key]) for key in ['macro_f1', 'accuracy', 'balanced_accuracy']}})
    pd.DataFrame(deltas).to_csv(out/'paired_partition_differences.csv', index=False)
    lines = ['# Training-only modality diagnostic', '',
             f'{len(training)} official training participants were scored once per outer partition for each model. '
             'Regularization was selected inside each outer fitting partition. No validation/test scoring or final full-data refit occurred.', '',
             'This diagnostic was designed after the original final test had been evaluated. It does not supply a new independent test result '
             'or replace the frozen reference model. All original dataset sources remain represented in participant_coverage.csv and protocol.json.', '',
             '| Features / weighting | Mean macro-F1 | Partition SD | Mean accuracy | Mean balanced accuracy |',
             '|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        sd = f'{row.macro_f1_std:.4f}' if pd.notna(row.macro_f1_std) else 'not estimable (one partition)'
        lines.append(f'| {row.model} | {row.macro_f1_mean:.4f} | {sd} | {row.accuracy_mean:.4f} | {row.balanced_accuracy_mean:.4f} |')
    lines += ['', 'A uses 78 audio summary values; V uses 50 visual values; AV uses all 128. '
              'All six logistic variants share outer/inner partitions and the same C grid. Scalers and class weights fit each local training partition only. '
              'The majority label is also learned separately within each outer fitting partition.', '',
              'Per-repeat predictions, class/group metrics and confusion matrices are under oof/. '
              'The paired table subtracts model scores within the same partition seed. No confidence intervals or significance claims are made from these overlapping folds.', '',
              *['- '+item+'.' for item in LIMITATIONS], '',
              'Artifacts include fold membership, all inner candidate OOF probabilities/scores, preprocessing checks, outer checkpoints, '
              'selected C values and checksummed job records. Interrupted unchanged runs resume using --resume. All artifacts stay local.', '']
    if audit_coverage:
        lines += [f"The original source audit covers {audit_coverage['archives']} archives and {audit_coverage['folders']} extracted folders. "
                  'Their current inventory/stamps and all primary participant caches were verified before and after fitting. '
                  'Duplicate copies retain their original audit records and count once per participant. No archives were decompressed again.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plot_summary(summary, scores, out/'comparison.png')
    return summary


def plot_summary(summary, scores, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    labels = [name.replace('Nested-LR-', '').replace('-balanced', ' + weights') for name in summary.model]
    for ax, metric, title in zip(axes, ['macro_f1', 'accuracy'], ['Five-class macro-F1', 'Accuracy']):
        x = np.arange(len(summary))
        ax.barh(x, summary[metric+'_mean'], xerr=summary[metric+'_std'].fillna(0),
                color=['#397b96' if name != 'majority' else '#9a9a9a' for name in summary.model], alpha=.8, capsize=3)
        for i, name in enumerate(summary.model):
            ax.scatter(scores.loc[scores.model == name, metric], [i]*sum(scores.model == name), s=18, color='#183946', zorder=3)
        ax.set_yticks(x, labels)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.set_xlim(left=0)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Nested cross-validation within the official training set\nBars: mean; error bars: partition SD; dots: each partition', fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_experiment(matrix, manifest, settings, output, provenance=None, frozen_plan_hash=None, resume=False, verify_inputs=None, audit_coverage=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing modality diagnostic; use --resume for unchanged inputs')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires an existing protocol')
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(manifest), 128) or not np.isfinite(matrix).all() or manifest.participant_id.duplicated().any():
        raise ValueError('Expected unique participants and finite 128-feature summaries')
    if set(manifest.split) != {'train', 'validation', 'test'}:
        raise ValueError('Full official split coverage required')
    grid = settings['C_candidates']
    if not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all() or min(grid) <= 0:
        raise ValueError('C grid must contain unique positive finite values')
    mask = manifest.split.eq('train').to_numpy()
    training = manifest.loc[mask, ['participant_id', 'severity_class', 'gender']].reset_index(drop=True)
    y = training.severity_class.to_numpy()
    if not np.isfinite(y).all() or not np.equal(y, y.astype(int)).all() or (y < 0).any() or (y > 4).any():
        raise ValueError('Training targets must be integer classes 0 through 4')
    training['severity_class'] = y.astype(int)
    x, y = matrix[mask], y.astype(int)
    folds = nested_folds(training, settings)
    context = {'protocol': PROTOCOL, 'settings': settings, 'variants': VARIANTS, 'folds': folds,
               'training_participants': training.to_dict('records'),
               'training_features_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
               'coverage': manifest[['participant_id', 'split']].to_dict('records'),
               'source_provenance': provenance, 'original_frozen_plan_sha256': frozen_plan_hash,
               'source_audit_coverage': audit_coverage,
               'audio_columns': AUDIO_COLUMNS, 'visual_columns': VISUAL_COLUMNS,
               'implementation_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                        ['modality_diagnostics.py', 'crossvalidated_baselines.py', 'evaluate.py', 'common.py']},
               'numpy_version': np.__version__, 'pandas_version': pd.__version__, 'sklearn_version': sklearn.__version__,
               'limitations': LIMITATIONS, 'validation_test_scoring': False, 'final_refit': False}
    # Canonical JSON normalizes tuple variants before comparison with resumed files.
    context = json.loads(json.dumps(context))
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        if (out/'protocol.json').exists():
            saved = json.loads((out/'protocol.json').read_text())
            if saved != {'context': context, 'context_hash': context_hash}:
                raise ValueError('Diagnostic inputs/code/protocol changed; preserve this run and choose a new output')
        else:
            atomic_json(out/'protocol.json', {'context': context, 'context_hash': context_hash})
            manifest[['participant_id', 'split']].to_csv(out/'participant_coverage.csv', index=False)
            atomic_json(out/'folds.json', folds)
        records, reused = [], 0
        for fold in folds:
            for modality, balanced in VARIANTS:
                directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/variant_name(modality, balanced)
                path = directory/'result.json'
                if path.exists():
                    record = load_record(path, context_hash)
                    reused += 1
                else:
                    record = run_job(x, y, fold, modality, balanced, settings, directory)
                    save_record(path, record, context_hash)
                validate_job(record, fold, modality, balanced, settings, directory)
                records.append(record)
            print(f"Completed outer partition {fold['seed']}/{fold['fold']+1}: {len(records)}/{len(folds)*len(VARIANTS)} jobs", flush=True)
        if verify_inputs is not None:
            verify_inputs()
        summarize(training, records, folds, settings, out, audit_coverage)
        completion = {'complete': True, 'all_dataset_participants': len(manifest), 'training_participants': len(training),
                      'validation_participants_not_scored': int(manifest.split.eq('validation').sum()),
                      'test_participants_not_scored': int(manifest.split.eq('test').sum()), 'outer_jobs': len(records),
                      'total_logistic_fits': sum(r['fits'] for r in records), 'reused_jobs_this_invocation': reused,
                      'new_logistic_fits_this_invocation': (len(records)-reused)*(len(grid)*settings['inner_folds']+1),
                      'model_fit_seconds_total': sum(r['seconds'] for r in records),
                      'validation_test_predictions_created': False, 'final_reference_replaced': False,
                      'source_audit_coverage': audit_coverage,
                      'context_hash': context_hash}
        atomic_json(out/'completion.json', completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/modality_diagnostics.json')
    parser.add_argument('--output', default='results/modality_diagnostics')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    config = load_config(settings['feature_config'])
    if Path(args.output).resolve().is_relative_to(Path(config['dataset']).resolve()):
        parser.error('Diagnostic output must be outside the dataset')
    if Path(args.output).exists() and not args.resume:
        parser.error('Preserving existing output; use --resume for unchanged inputs')
    plan = verify_plan(settings['frozen_plan'])
    manifest = joined_manifest(config)
    matrix, provenance = load_summary_matrix(config, manifest)
    if provenance != plan['source_provenance'] or digest(manifest.to_dict('records')) != plan['manifest_hash']:
        raise ValueError('Current data differ from the frozen study; use a separately documented protocol')
    source_table_path = Path(config.get('full_dataset_report', 'results/full_dataset'))/'sources.csv'
    source_table = pd.read_csv(source_table_path)
    audit_coverage = {'sources': len(source_table), 'archives': int(source_table.kind.eq('archive').sum()),
                      'folders': int(source_table.kind.eq('directory').sum()),
                      'source_table': str(source_table_path), 'source_table_sha256': sha256(source_table_path)}
    def verify_inputs():
        verify_plan(settings['frozen_plan'])
        current_manifest = joined_manifest(config)
        _, current_sources = load_summary_matrix(config, current_manifest)
        if (current_sources != provenance or digest(current_manifest.to_dict('records')) != plan['manifest_hash']
                or sha256(source_table_path) != audit_coverage['source_table_sha256']):
            raise ValueError('Dataset changed during the diagnostic; completion cannot be confirmed')
    with threadpool_limits(limits=settings['cpu_threads']):
        result = run_experiment(matrix, manifest, settings, args.output, provenance, digest(plan), args.resume, verify_inputs, audit_coverage)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
