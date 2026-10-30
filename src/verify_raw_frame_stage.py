"""Read-only verification of raw-frame preparation and saved nested results.

No raw interview archive decompression, model deserialization, fitting or prediction.
Existing training metrics are recomputed from saved probabilities for parity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .batch_inference import load_record
from .common import VERSION, digest
from .evaluate import metrics
from .final_evaluation import sha256
from .modality_diagnostics import validate_job, variant_name
from .raw_frame_comparison import PROTOCOL, VARIANTS, load_inputs, verify_selection
from .raw_frame_features import AUDIO_COLUMNS, FAMILIES, FEATURE_NAMES, FORMAT
from .verify_project import read_json, require

PROBABILITIES = [f'p{i}' for i in range(5)]


def table_matches(path, expected):
    expected = expected.copy()
    expected.columns = expected.columns.astype(str)
    actual = pd.read_csv(path, dtype={'participant_id': str, 'used_summary_participant_id': str})
    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True), check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-12)


def verify_features(root, context):
    root = Path(root)
    envelope = read_json(root/'protocol.json')
    require(envelope == {'context': context, 'context_hash': digest(context)}, 'Feature context changed')
    completion = read_json(root/'completion.json')
    coverage = context['coverage']
    ids = [r['participant_id'] for r in coverage]
    require({p.stem for p in (root/'participants').glob('*.json')} == set(ids), 'Missing or extra feature participant records')
    records = [load_record(root/'participants'/f'{pid}.json', envelope['context_hash']) for pid in ids]
    sources = context['sources']
    counts = {'participants': len(ids), 'prepared_participants': len(records), 'physical_sources': len(sources),
              'archives': sum(s['kind'] == 'archive' for s in sources), 'folders': sum(s['kind'] == 'directory' for s in sources),
              'raw_frames_in_cached_interval': sum(r['raw_frames_in_cached_interval'] for r in records)}
    require(all(completion.get(k) == v for k, v in counts.items()), 'Feature completion counts differ from records')
    require(completion['complete'] and not completion['failures'] and completion['model_fits'] == 0
            and completion['predictions_created'] is False, 'Feature completion status/claims differ')
    require(completion['new_raw_reads_this_invocation'] >= 0 and completion['reused_participants_this_invocation'] >= 0
            and completion['new_raw_reads_this_invocation']+completion['reused_participants_this_invocation'] == len(ids),
            'Feature resume counters disagree')
    for record in records:
        values = np.asarray(record['summary'], dtype=float)
        variance = record['variance']
        require(record['cache_reproduced_exactly'] and record['variance_decomposition_verified'], 'Missing recorded raw/cache checks')
        require(2 <= record['raw_frames_in_cached_interval'] <= record['raw_frames_total']
                and record['interval_end_exclusive'] > record['interval_start'] >= 0
                and record['observed_audio_bins'] > 0, 'Invalid recorded frame/crop counts')
        require(len(variance) == 39 and [v['feature'] for v in variance] == AUDIO_COLUMNS, 'Variance feature names/order differ')
        for i, v in enumerate(variance):
            require(v['family'] == FAMILIES[i//13], 'Variance family assignment differs')
            numbers = np.asarray([v['raw_variance'], v['within_bin_variance'], v['between_bin_variance']])
            require(np.isfinite(numbers).all() and (numbers >= 0).all() and values[39+i] >= 0, 'Invalid saved variance/std')
            np.testing.assert_allclose(numbers[0], numbers[1]+numbers[2], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(numbers[0], values[39+i]**2, rtol=1e-12, atol=1e-12)
            if numbers[0] == 0:
                require(v['retained_variance_fraction'] is None, 'Constant variance ratio must be undefined')
            else:
                np.testing.assert_allclose(v['retained_variance_fraction'], numbers[2]/numbers[0], rtol=1e-10, atol=1e-10)
    table_matches(root/'participant_coverage.csv', pd.DataFrame(coverage))
    table_matches(root/'source_coverage.csv', pd.DataFrame([{**s, 'used_summary_participant_id': s['participant_id'],
        'raw_mfcc_read_here': s['source_key'] == context['selected_sources'][s['participant_id']],
        'primary_features_match_selected': True} for s in sources]))
    table_matches(root/'quality.csv', pd.DataFrame([{k: v for k, v in r.items() if k not in ['summary', 'variance']} for r in records]))
    table_matches(root/'variance_retention.csv', pd.DataFrame([{'participant_id': r['participant_id'], **v} for r in records for v in r['variance']]))
    return counts


def verify_protocol(root, settings, inputs):
    x, training, folds, baseline, coverage, bindings, _ = inputs
    root = Path(root)
    envelope = read_json(root/'protocol.json')
    context = envelope['context']
    require(envelope['context_hash'] == digest(context), 'Comparison context checksum changed')
    expected = {'protocol': PROTOCOL, 'feature_format': FORMAT, 'feature_names': FEATURE_NAMES,
                'settings': settings, 'folds': folds, 'training_participants': training.to_dict('records'),
                'training_features_sha256': hashlib.sha256(np.asarray(x, dtype=np.float64).tobytes()).hexdigest(),
                'baseline_records_sha256': digest(baseline), 'input_bindings': bindings,
                'coverage': coverage.to_dict('records'), 'validation_test_scoring': False, 'final_refit': False,
                'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    require(all(context.get(k) == v for k, v in expected.items()), 'Comparison protocol differs from verified inputs/settings')
    require(set(context['implementation_sha256']) == {'raw_frame_comparison.py', 'raw_frame_features.py', 'modality_diagnostics.py',
            'crossvalidated_baselines.py', 'evaluate.py', 'common.py', 'batch_inference.py'}, 'Comparison implementation coverage differs')
    for name, expected_hash in context['implementation_sha256'].items():
        require(sha256(Path(__file__).with_name(name)) == expected_hash, f'Comparison implementation changed: {name}')
    require(read_json(root/'folds.json') == folds, 'Saved fold table changed')
    return envelope['context_hash']


def verify_jobs(root, context_hash, training, folds, settings):
    root = Path(root)
    records, directories = [], []
    y = training.severity_class.to_numpy()
    for fold in folds:
        for modality, balanced in VARIANTS:
            directory = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/variant_name(modality, balanced)
            directories.append(directory)
            record = load_record(directory/'result.json', context_hash)
            validate_job(record, fold, modality, balanced, settings, directory)
            verify_selection(record, fold, y, settings)
            require(record['feature_format'] == FORMAT and record['outer_scaler_verified']
                    and record['checkpoint_prediction_parity'], 'Job representation/verification claims differ')
            require(np.isfinite(record['seconds']) and record['seconds'] >= 0, 'Invalid fit duration')
            expected = [{'C': c, 'inner_fold': i, 'fit_count': len(partition['fit']),
                         'fit_class_counts': np.bincount(y[partition['fit']], minlength=5).tolist(),
                         'scaler_mean_and_variance_verified': True}
                        for c in sorted(settings['C_candidates']) for i, partition in enumerate(fold['inner'])]
            require(record['preprocessing_checks'] == expected, 'Saved fold preprocessing ownership/counts differ')
            records.append(record)
    require(set((root/'jobs').rglob('result.json')) == {d/'result.json' for d in directories}
            and set((root/'jobs').rglob('model.pkl')) == {d/'model.pkl' for d in directories}, 'Missing or extra comparison jobs/checkpoints')
    completion = read_json(root/'completion.json')
    expected = {'complete': True, 'context_hash': context_hash, 'training_participants': len(training),
                'new_outer_jobs': len(records), 'total_logistic_fits': sum(r['fits'] for r in records),
                'validation_test_predictions_created': False, 'final_reference_replaced': False}
    require(all(completion.get(k) == v for k, v in expected.items()), 'Comparison completion counts/claims differ')
    reused = completion['reused_jobs_this_invocation']
    require(0 <= reused <= len(records) and completion['new_fits_this_invocation'] ==
            (len(records)-reused)*(len(settings['C_candidates'])*settings['inner_folds']+1), 'Comparison resume counters disagree')
    np.testing.assert_allclose(completion['model_fit_seconds_total'], sum(r['seconds'] for r in records), rtol=1e-12)
    return records


def verify_saved_outputs(root, training, records, baseline, settings):
    root = Path(root)
    scores, prediction_paths = [], []
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
                require((ownership == 1).all() and np.isfinite(probabilities).all(), 'Incomplete or duplicated training OOF ownership')
                report = metrics(training.severity_class, probabilities)
                if representation == 'raw_frames':
                    directory = root/'oof'/f'{name}_seed{seed}'
                    prediction_paths.append(directory/'predictions.csv')
                    model = name.replace('Nested-LR', 'RawFrame-LR')
                    frame = training.copy()
                    frame['predicted_label'] = probabilities.argmax(1)
                    for i in range(5):
                        frame[f'p{i}'] = probabilities[:, i]
                    frame['split'], frame['seed'], frame['model_version'] = 'train_nested_oof', seed, VERSION+':'+model
                    frame = frame[['participant_id', 'gender', 'severity_class', 'predicted_label', *PROBABILITIES, 'split', 'seed', 'model_version']]
                    table_matches(directory/'predictions.csv', frame)
                    groups = {str(g or 'UNKNOWN'): metrics(rows.severity_class, probabilities[training.index.get_indexer(rows.index)])
                              for g, rows in training.groupby('gender', dropna=False)}
                    full_report = {**report, 'group_metrics': groups, 'model': model, 'seed': seed,
                                   'split': 'train_nested_oof', 'version': VERSION}
                    require(read_json(directory/'metrics.json') == full_report, 'Saved per-class/group metrics differ from OOF probabilities')
                    table_matches(directory/'per_class.csv', pd.DataFrame(report['per_class']))
                    table_matches(directory/'confusion_matrix.csv', pd.DataFrame(report['confusion_matrix']))
                    for i, group_report in enumerate(groups.values()):
                        table_matches(directory/f'group_{i}_confusion_matrix.csv', pd.DataFrame(group_report['confusion_matrix']))
                scores.append({'representation': representation, 'modality': modality, 'balanced': balanced, 'seed': seed,
                               **{k: report[k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy']},
                               'severe_correct': int(((probabilities.argmax(1) == 4) & training.severity_class.eq(4).to_numpy()).sum())})
    require(set(root.rglob('predictions.csv')) == set(prediction_paths), 'Missing or extra prediction tables in comparison')
    scores = pd.DataFrame(scores)
    table_matches(root/'repeat_scores.csv', scores)
    summary = scores.groupby(['representation', 'modality', 'balanced'])[['macro_f1', 'accuracy', 'balanced_accuracy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    table_matches(root/'summary.csv', summary.reset_index())
    pairs = scores[scores.representation.eq('raw_frames')].merge(scores[scores.representation.eq('cached_bins')],
        on=['modality', 'balanced', 'seed'], suffixes=('_raw', '_cached'), validate='one_to_one')
    deltas = pairs[['modality', 'balanced', 'seed']].copy()
    for metric in ['macro_f1', 'accuracy', 'balanced_accuracy']:
        deltas[metric+'_raw_minus_cached'] = pairs[metric+'_raw']-pairs[metric+'_cached']
    table_matches(root/'paired_partition_differences.csv', deltas)
    table_matches(root/'outer_selections.csv', pd.DataFrame([{k: r[k] for k in ['model', 'seed', 'fold', 'selected_C', 'fits', 'seconds']} for r in records]))
    return {'prediction_tables': len(prediction_paths), 'training_participants_per_table': len(training),
            'aggregate_rows': len(scores), 'validation_test_prediction_tables': 0}


def verify_stage(settings, comparison_root):
    checks, values = [], {}
    def check(name, dependencies, operation, details=lambda value: {}):
        missing = [d for d in dependencies if d not in values]
        if missing:
            checks.append({'check': name, 'status': 'blocked', 'reason': 'Prerequisite failed: '+', '.join(missing)})
            return
        try:
            value = operation()
            values[name] = value
            checks.append({'check': name, 'status': 'passed', **details(value)})
        except Exception as exc:
            checks.append({'check': name, 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
    check('source_and_input_bindings', [], lambda: load_inputs(settings),
          lambda x: {'participants': len(x[4]), 'sources': len(x[6]['sources'])})
    check('feature_records_and_tables', ['source_and_input_bindings'],
          lambda: verify_features(settings['raw_features'], values['source_and_input_bindings'][6]), lambda x: x)
    check('comparison_protocol', ['source_and_input_bindings'],
          lambda: verify_protocol(comparison_root, settings, values['source_and_input_bindings']))
    check('saved_nested_jobs', ['comparison_protocol'],
          lambda: verify_jobs(comparison_root, values['comparison_protocol'], values['source_and_input_bindings'][1],
                              values['source_and_input_bindings'][2], settings),
          lambda x: {'outer_jobs': len(x), 'recorded_logistic_fits': sum(r['fits'] for r in x)})
    check('saved_predictions_and_metrics', ['saved_nested_jobs'],
          lambda: verify_saved_outputs(comparison_root, values['source_and_input_bindings'][1], values['saved_nested_jobs'],
                                       values['source_and_input_bindings'][3], settings), lambda x: x)
    return {'format': 'raw-frame-stage-verification-v1', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'passed': all(c['status'] == 'passed' for c in checks), 'checks': checks,
            'models_fitted': 0, 'new_model_predictions': 0, 'models_deserialized': 0, 'raw_archives_decompressed': 0,
            'saved_training_metrics_recomputed': 'saved_predictions_and_metrics' in values,
            'limitations': ['Source freshness relies on the prior full audit and current source stamps; no fresh raw read',
                            'Saved variance/cache-parity receipts are checked for consistency, not reproduced from raw media here',
                            'Model checks use hashes and fitting records; no checkpoint deserialization or inference',
                            'Checksums are local consistency evidence, not independently signed evidence',
                            'This check does not establish model accuracy or erase earlier final-test exposure']}


def write_report(report, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Raw-frame stage verification', '', f"All checks passed: **{report['passed']}**.", '',
             '| Check | Status | Details |', '|---|---|---|']
    for item in report['checks']:
        detail = item.get('reason') or json.dumps({k: v for k, v in item.items() if k not in ['check', 'status']})
        lines.append(f"| {item['check']} | {item['status']} | {detail.replace('|', '/').replace(chr(10), ' ')} |")
    lines += ['', 'No model fitting, inference, deserialization or raw interview archive decompression. Saved training metrics were checked against saved probabilities. Input artifacts were preserved.', '',
              *['- '+limitation+'.' for limitation in report['limitations']], '']
    (root/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/raw_frame_comparison.json')
    parser.add_argument('--comparison', default='results/raw_frame_comparison')
    parser.add_argument('--output', help='Optional fresh directory; omit for a read-only check')
    args = parser.parse_args()
    settings = read_json(args.config)
    if args.output:
        path = Path(args.output).resolve()
        config = read_json(settings['frozen_plan'])['plan']['feature_config']
        protected = [Path(p).resolve() for p in [settings['raw_features'], settings['cached_diagnostic'], args.comparison,
                     config['dataset'], config.get('full_dataset_report', 'results/full_dataset'), config['cache']]]
        if path.exists() or any(path.is_relative_to(p) for p in protected):
            parser.error('Preserving inputs: use a fresh report directory outside data/experiment directories')
    report = verify_stage(settings, args.comparison)
    if args.output:
        write_report(report, args.output)
    print(json.dumps({'passed': report['passed'], 'checks': {c['check']: c['status'] for c in report['checks']},
                      'failures': [c for c in report['checks'] if c['status'] == 'failed'], 'report_directory': args.output,
                      'models_fitted': 0, 'new_model_predictions': 0, 'raw_archives_decompressed': 0}, indent=2))
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
