"""Matched nested training-only comparison of raw-frame acoustic summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest, joined_manifest
from .evaluate import metrics, write_report
from .final_evaluation import sha256, verify_plan
from .modality_diagnostics import nested_folds, run_job, validate_job, variant_name
from .raw_frame_features import FORMAT, FEATURE_NAMES, input_state, validate_record
from .exploratory_baselines import pool_modalities

PROTOCOL = 'post-final-raw-frame-nested-comparison-v1'
VARIANTS = [('A', False), ('A', True), ('AV', False), ('AV', True)]
SETTING_KEYS = ['C_candidates', 'outer_folds', 'inner_folds', 'fold_seeds', 'fit_seed']


def training_inputs(matrix, manifest):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(manifest), 128) or not np.isfinite(matrix).all() or manifest.participant_id.duplicated().any():
        raise ValueError('Expected unique participants and finite 128-feature summaries')
    if set(manifest.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected full official split coverage')
    mask = manifest.split.eq('train').to_numpy()
    training = manifest.loc[mask, ['participant_id', 'severity_class', 'gender']].reset_index(drop=True)
    y = training.severity_class.to_numpy()
    if not np.isfinite(y).all() or not np.equal(y, y.astype(int)).all() or (y < 0).any() or (y > 4).any():
        raise ValueError('Training targets must be integer classes 0 through 4')
    training['severity_class'] = y.astype(int)
    return matrix[mask], training


def verify_selection(record, fold, y, settings):
    candidates = record['inner_candidates']
    if [c['C'] for c in candidates] != sorted(settings['C_candidates']):
        raise ValueError('Saved candidate grid changed')
    for candidate in candidates:
        probabilities = np.asarray(candidate['inner_oof_probabilities'])
        if (probabilities.shape != (len(fold['fit']), 5) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1)):
            raise ValueError('Invalid inner probabilities')
        score = metrics(y[fold['fit']], probabilities)
        for key in ['macro_f1', 'accuracy']:
            if abs(score[key]-candidate['inner_'+key]) > 1e-12:
                raise ValueError('Saved inner score differs from probabilities')
    if max(candidates, key=lambda c: c['inner_macro_f1'])['C'] != record['selected_C']:
        raise ValueError('Saved C violates inner selection rule')


def load_inputs(settings):
    root = Path(settings['raw_features'])
    context, coverage, sequences, selected = input_state(settings['frozen_plan'])
    protocol = json.loads((root/'protocol.json').read_text())
    completion = json.loads((root/'completion.json').read_text())
    if protocol != {'context': context, 'context_hash': digest(context)} or not completion['complete']:
        raise ValueError('Raw-frame preparation is incomplete or has changed inputs')
    if completion['context_hash'] != protocol['context_hash'] or set(completion['record_sha256']) != set(coverage.participant_id):
        raise ValueError('Raw-frame completion coverage/context mismatch')
    bindings = {str(root/'protocol.json'): sha256(root/'protocol.json'), str(root/'completion.json'): sha256(root/'completion.json')}
    for name, expected in completion['artifacts'].items():
        if sha256(root/name) != expected:
            raise ValueError('Prepared raw-frame artifact checksum changed')
        bindings[str(root/name)] = expected
    rows = []
    for pid in coverage.participant_id:
        path = root/'participants'/f'{pid}.json'
        if sha256(path) != completion['record_sha256'][pid]:
            raise ValueError('Prepared raw-frame participant record changed')
        record = load_record(path, protocol['context_hash'])
        validate_record(record, pid, selected[pid])
        rows.append(record['summary'])
        bindings[str(path)] = sha256(path)
    matrix = np.asarray(rows, dtype=np.float64)
    with np.load(root/'participant_summaries.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['participant_ids'], coverage.participant_id)
        np.testing.assert_array_equal(data['feature_names'], FEATURE_NAMES)
        np.testing.assert_array_equal(data['features'], matrix)
    cached = np.stack([pool_modalities(sequences[pid]) for pid in coverage.participant_id])
    np.testing.assert_array_equal(matrix[:, 78:], cached[:, 78:])
    manifest = joined_manifest(context['feature_config'])
    plan = verify_plan(settings['frozen_plan'])
    if digest(manifest.to_dict('records')) != plan['manifest_hash'] or manifest[['participant_id', 'split']].to_dict('records') != coverage.to_dict('records'):
        raise ValueError('Manifest differs from frozen participant coverage')
    x, training = training_inputs(matrix, manifest)
    old_x, _ = training_inputs(cached, manifest)
    folds = nested_folds(training, settings)
    previous = Path(settings['cached_diagnostic'])
    envelope = json.loads((previous/'protocol.json').read_text())
    old_context = envelope['context']
    if (digest(old_context) != envelope['context_hash'] or old_context['folds'] != folds
            or old_context['training_participants'] != training.to_dict('records')
            or old_context['training_features_sha256'] != hashlib.sha256(old_x.tobytes()).hexdigest()
            or any(old_context['settings'][k] != settings[k] for k in SETTING_KEYS)
            or old_context['source_provenance'] != context['source_provenance']
            or old_context['original_frozen_plan_sha256'] != context['frozen_plan_sha256']):
        raise ValueError('Cached comparison does not match training inputs/folds/tuning budget')
    for name, expected in old_context['implementation_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError('Cached diagnostic implementation changed')
    bindings[str(previous/'protocol.json')] = sha256(previous/'protocol.json')
    records = []
    for fold in folds:
        for modality, balanced in VARIANTS:
            directory = previous/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/variant_name(modality, balanced)
            record = load_record(directory/'result.json', envelope['context_hash'])
            validate_job(record, fold, modality, balanced, settings, directory)
            verify_selection(record, fold, training.severity_class.to_numpy(), settings)
            records.append(record)
            for name in ['result.json', 'model.pkl']:
                bindings[str(directory/name)] = sha256(directory/name)
    return x, training, folds, records, coverage, bindings, context


def run_raw_job(x, y, fold, modality, balanced, settings, directory):
    result = run_job(x, y, fold, modality, balanced, settings, directory)
    # Distinguish new checkpoints from the identically shaped frozen summaries.
    path = Path(directory)/'model.pkl'
    with path.open('rb') as handle:
        state = pickle.load(handle)  # Only the artifact just created by run_job.
    state.update({'purpose': PROTOCOL, 'feature_format': FORMAT, 'feature_names': FEATURE_NAMES})
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(state, handle)
    temporary.replace(path)
    result.update({'feature_format': FORMAT, 'model_sha256': sha256(path)})
    return result


def summarize(training, records, baseline, folds, settings, out):
    scores = []
    for representation, jobs in [('cached_bins', baseline), ('raw_frames', records)]:
        for modality, balanced in VARIANTS:
            name = variant_name(modality, balanced)
            for seed in settings['fold_seeds']:
                probabilities = np.full((len(training), 5), np.nan)
                ownership = np.zeros(len(training), dtype=int)
                for record in jobs:
                    if record['model'] == name and record['seed'] == seed:
                        probabilities[record['heldout']] = record['outer_probabilities']
                        ownership[record['heldout']] += 1
                if not (ownership == 1).all() or not np.isfinite(probabilities).all():
                    raise ValueError('Each training participant must have exactly one OOF prediction per repeat')
                if representation == 'raw_frames':
                    report = write_report(training, probabilities, out/'oof'/f'{name}_seed{seed}',
                                          name.replace('Nested-LR', 'RawFrame-LR'), seed, 'train_nested_oof')
                else:
                    report = metrics(training.severity_class.to_numpy(), probabilities)
                scores.append({'representation': representation, 'modality': modality, 'balanced': balanced, 'seed': seed,
                               **{k: report[k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy']},
                               'severe_correct': int(((probabilities.argmax(1) == 4) & training.severity_class.eq(4).to_numpy()).sum())})
    scores = pd.DataFrame(scores)
    scores.to_csv(out/'repeat_scores.csv', index=False)
    summary = scores.groupby(['representation', 'modality', 'balanced'])[['macro_f1', 'accuracy', 'balanced_accuracy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    pairs = scores[scores.representation.eq('raw_frames')].merge(scores[scores.representation.eq('cached_bins')],
                                                               on=['modality', 'balanced', 'seed'], suffixes=('_raw', '_cached'), validate='one_to_one')
    deltas = pairs[['modality', 'balanced', 'seed']].copy()
    for metric in ['macro_f1', 'accuracy', 'balanced_accuracy']:
        deltas[metric+'_raw_minus_cached'] = pairs[metric+'_raw']-pairs[metric+'_cached']
    deltas.to_csv(out/'paired_partition_differences.csv', index=False)
    pd.DataFrame([{k: r[k] for k in ['model', 'seed', 'fold', 'selected_C', 'fits', 'seconds']} for r in records]).to_csv(out/'outer_selections.csv', index=False)
    lines = ['# Raw-frame versus cached acoustic summaries', '',
             'Post-final exploratory nested cross-validation: official training participants only. '
             'No validation/test predictions, final refit or frozen-reference replacement.', '',
             '| Input | Weights | Cached macro-F1 | Raw-frame macro-F1 | Mean difference |',
             '|---|---|---:|---:|---:|']
    for modality, balanced in VARIANTS:
        subset = summary[summary.modality.eq(modality) & summary.balanced.eq(balanced)].set_index('representation')
        old, new = subset.loc['cached_bins'], subset.loc['raw_frames']
        lines.append(f"| {modality} | {'balanced' if balanced else 'ordinary'} | {old.macro_f1_mean:.4f} ± {old.macro_f1_std:.4f} | {new.macro_f1_mean:.4f} ± {new.macro_f1_std:.4f} | {new.macro_f1_mean-old.macro_f1_mean:+.4f} |")
    lines += ['', f'{len(training)} training participants; {len(folds)} outer partitions; {len(records)} new outer jobs. '
              'Identical splits, class targets, four C candidates and fold-local scaling/weights. '
              'Saved cached-summary checkpoints/records were verified and reused. C selection was recomputed from every saved inner prediction vector.', '',
              '± values describe SD across three overlapping partitions, not independent replication uncertainty. '
              'This protocol was designed after the original test was seen. Seven severe training participants provide limited support. '
              'Raw-frame means/stds preserve variance but not temporal ordering; unverified speaker isolation and common crop remain. '
              'Raw frames and original bins use different sample weighting when counts vary. '
              'Improved internal scores would require independent confirmation.', '',
              'All 275 participants / 278 archives / three folders are represented in the separate feature preparation. '
              'Only training rows enter this experiment. Original studies, the portable model and the 40-row experiment log are preserved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plot_summary(summary, deltas, out/'comparison.png')


def plot_summary(summary, deltas, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    labels = [m+(' + weights' if b else '') for m, b in VARIANTS]
    for i, (modality, balanced) in enumerate(VARIANTS):
        rows = summary[summary.modality.eq(modality) & summary.balanced.eq(balanced)].set_index('representation')
        for offset, representation, color in [(-.18, 'cached_bins', '#7798aa'), (.18, 'raw_frames', '#c47543')]:
            row = rows.loc[representation]
            axes[0].barh(i+offset, row.macro_f1_mean, xerr=row.macro_f1_std, height=.32, color=color,
                         capsize=3, label=representation.replace('_', ' ') if i == 0 else None)
        differences = deltas[deltas.modality.eq(modality) & deltas.balanced.eq(balanced)].macro_f1_raw_minus_cached
        axes[1].scatter(differences, np.full(len(differences), i), color='#9b4c25', s=40)
    for ax in axes:
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    axes[0].set_xlabel('Mean five-class macro-F1 ± partition SD')
    axes[0].set_xlim(left=0)
    axes[0].legend(frameon=False)
    axes[1].axvline(0, color='gray', linewidth=1)
    axes[1].set_xlabel('Raw − cached macro-F1; one dot per partition')
    fig.suptitle('Raw-frame acoustic summaries: nested training-only comparison\nExploratory analysis after the original final test', fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_comparison(x, training, folds, baseline, settings, output, bindings=None, coverage=None, resume=False, verify_inputs=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving comparison; use --resume with unchanged inputs')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires an existing protocol')
    if np.asarray(x).shape != (len(training), 128) or not np.isfinite(x).all():
        raise ValueError('Expected finite training summaries')
    grid = settings['C_candidates']
    if not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all() or min(grid) <= 0:
        raise ValueError('C grid must contain unique positive finite values')
    if folds != nested_folds(training, settings):
        raise ValueError('Partitions differ from the predeclared folds')
    context = {'protocol': PROTOCOL, 'feature_format': FORMAT, 'feature_names': FEATURE_NAMES,
               'settings': settings, 'folds': folds, 'training_participants': training.to_dict('records'),
               'training_features_sha256': hashlib.sha256(np.asarray(x, dtype=np.float64).tobytes()).hexdigest(),
               'baseline_records_sha256': digest(baseline), 'input_bindings': bindings, 'coverage': coverage,
               'implementation_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                        ['raw_frame_comparison.py', 'raw_frame_features.py', 'modality_diagnostics.py',
                                         'crossvalidated_baselines.py', 'evaluate.py', 'common.py', 'batch_inference.py']},
               'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
               'validation_test_scoring': False, 'final_refit': False}
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        protocol = {'context': context, 'context_hash': context_hash}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != protocol:
                raise ValueError('Comparison inputs/code/protocol changed; choose a new output')
        else:
            atomic_json(out/'protocol.json', protocol)
            atomic_json(out/'folds.json', folds)
        records, reused = [], 0
        y = training.severity_class.to_numpy()
        for fold in folds:
            for modality, balanced in VARIANTS:
                directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/variant_name(modality, balanced)
                path = directory/'result.json'
                if path.exists():
                    record = load_record(path, context_hash)
                    reused += 1
                else:
                    record = run_raw_job(x, y, fold, modality, balanced, settings, directory)
                    save_record(path, record, context_hash)
                validate_job(record, fold, modality, balanced, settings, directory)
                verify_selection(record, fold, y, settings)
                if record['feature_format'] != FORMAT:
                    raise ValueError('Checkpoint uses a different representation')
                records.append(record)
            print(f"Completed {len(records)}/{len(folds)*len(VARIANTS)} raw-frame outer jobs", flush=True)
        if verify_inputs:
            verify_inputs()
        summarize(training, records, baseline, folds, settings, out)
        result = {'complete': True, 'context_hash': context_hash, 'training_participants': len(training),
                  'new_outer_jobs': len(records), 'total_logistic_fits': sum(r['fits'] for r in records),
                  'reused_jobs_this_invocation': reused,
                  'new_fits_this_invocation': (len(records)-reused)*(len(settings['C_candidates'])*settings['inner_folds']+1),
                  'model_fit_seconds_total': sum(r['seconds'] for r in records),
                  'validation_test_predictions_created': False, 'final_reference_replaced': False}
        atomic_json(out/'completion.json', result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/raw_frame_comparison.json')
    parser.add_argument('--output', default='results/raw_frame_comparison')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    if not 1 <= settings['cpu_threads'] <= 2:
        parser.error('Use one or two CPU threads for this bounded comparison')
    if Path(args.output).exists() and not args.resume:
        parser.error('Preserving existing comparison; use --resume for unchanged inputs')
    x, training, folds, baseline, coverage, bindings, context = load_inputs(settings)
    if Path(args.output).resolve().is_relative_to(Path(context['feature_config']['dataset']).resolve()):
        parser.error('Output must be outside the dataset')
    def verify_inputs():
        _, _, _, _, _, current_bindings, current_context = load_inputs(settings)
        if current_bindings != bindings or current_context != context:
            raise ValueError('Comparison inputs changed during execution')
    with threadpool_limits(limits=settings['cpu_threads']):
        result = run_comparison(x, training, folds, baseline, settings, args.output, bindings,
                                coverage.to_dict('records'), args.resume, verify_inputs)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
