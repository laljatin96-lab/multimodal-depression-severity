"""Read-only verification of the completed local research and inference artifacts.

No estimator is loaded, fitted or run. Raw archives are checked by their audited
source stamps; their bytes are not read again. Optional reports are written only
to a new output directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import batch_inference as batch
from .common import digest
from .final_evaluation import verify_plan
from .reference_inference import DEFAULT_BUNDLE, load_bundle, sha256

PROBABILITY_COLUMNS = [f'p{i}' for i in range(5)]
# Existing neural/diagnostic outputs were computed and serialized as float32.
# This checks their probability simplex; reference-output parity remains 1e-12.
PROBABILITY_SUM_ATOL = 1e-6


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction_table(path, label_column='predicted_label'):
    frame = pd.read_csv(path, usecols=['participant_id', label_column, *PROBABILITY_COLUMNS],
                        dtype={'participant_id': str}, keep_default_na=False)
    require(len(frame) > 0 and frame.participant_id.ne('').all() and not frame.participant_id.duplicated().any(),
            f'Empty or duplicate prediction IDs: {path}')
    values = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
            and np.allclose(values.sum(axis=1), 1, rtol=0, atol=PROBABILITY_SUM_ATOL), f'Invalid probabilities: {path}')
    require(np.array_equal(frame[label_column].to_numpy(), values.argmax(axis=1)),
            f'Predicted classes disagree with probabilities: {path}')
    return frame.rename(columns={label_column: 'predicted_class'}).set_index('participant_id').sort_index()


def verify_splits(plan):
    frame = pd.read_csv(plan['feature_config']['splits'], usecols=['participant_id', 'split'],
                        dtype=str, keep_default_na=False)
    require(not frame.participant_id.duplicated().any(), 'Duplicate split IDs')
    require(set(frame.participant_id) == set(plan['source_provenance']), 'Split coverage differs from frozen participants')
    require(set(frame.split) == {'train', 'validation', 'test'}, 'Expected complete train/validation/test assignments')
    return dict(zip(frame.participant_id, frame.split))


def verify_bundle(bundle, plan):
    contract, _ = load_bundle(bundle)
    require(contract['frozen_plan_sha256'] == digest(plan), 'Portable bundle refers to a different frozen plan')
    reference = [item for item in plan['models'] if item['model'] == plan['reference_model_selected_before_test']]
    require(len(reference) == 1, 'Expected one preselected reference model')
    require(contract['model'] == reference[0]['model'] and
            contract['source_model_sha256'] == sha256(reference[0]['artifact']), 'Portable model source differs from frozen reference')
    return contract


def verify_dataset(dataset, audit, plan_path, contract):
    sources, rows, unknown = batch.inventory(dataset)
    require(not unknown, 'Dataset contains unrecognized items; inspect the source audit')
    cache_index, bindings = batch.audited_cache_index(dataset, sources, audit, plan_path, contract)
    require(all(row['participant_id'] is not None for row in rows), 'Source lacks a participant ID')
    return {'rows': rows, 'cache_index': cache_index, 'bindings': bindings}


def verify_csv_rows(path, expected, key, columns):
    actual = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(not actual[key].duplicated().any(), f'Duplicate records: {path}')
    expected_frame = pd.DataFrame(expected).reindex(columns=columns).astype(object).fillna('').astype(str)
    require(set(actual[key]) == set(expected_frame[key]), f'Record coverage differs: {path}')
    actual = actual.set_index(key).sort_index()
    expected_frame = expected_frame.set_index(key).sort_index()
    require(actual[expected_frame.columns].equals(expected_frame), f'Saved table differs from checked records: {path}')


def verify_batch(root, dataset, contract, sources, splits):
    root = Path(root)
    envelope = read_json(root/'run.json')
    context, context_hash = envelope['context'], envelope['context_hash']
    require(digest(context) == context_hash and context['format'] == batch.FORMAT, 'Batch context checksum/format differs')
    implementation = {name: sha256(Path(batch.__file__).with_name(name)) for name in
                      ['batch_inference.py', 'reference_inference.py', 'prepare_features.py',
                       'exploratory_baselines.py', 'full_dataset.py', 'common.py']}
    require(context['implementation_sha256'] == implementation, 'Batch implementation files changed')
    require(context['numpy_version'] == np.__version__ and context['pandas_version'] == pd.__version__,
            'Batch runtime versions changed; original resume context no longer matches')
    require(context['dataset'] == str(Path(dataset).resolve()) and context['sources'] == sources['rows']
            and context['unrecognized_items'] == [], 'Batch inventory differs from the current dataset')
    require(context['model_contract'] == contract and context['cache_index'] == sources['cache_index'],
            'Batch model or frozen cache bindings differ')
    require(context['declared_splits'] == splits, 'Batch split annotations differ from the frozen splits')
    for name, checksum in sources['bindings'].items():
        require(context['input_bindings'].get(name) == checksum, 'Batch audit bindings differ')
    for name, checksum in context['input_bindings'].items():
        require(sha256(name) == checksum, f'Batch bound input changed: {name}')
    require({p.stem for p in (root/'source_records').glob('*.json')} ==
            {row['source_key'] for row in sources['rows']}, 'Missing or extra batch source records')
    records, groups = [], {}
    for row in sources['rows']:
        record = batch.load_record(root/'source_records'/f"{row['source_key']}.json", context_hash)
        require(all(record.get(key) == value for key, value in row.items()), 'Batch source record identity differs')
        require(record['status'] == 'ready' and record['read_mode'] == 'verified_frozen_cache', 'Batch source is not ready from its frozen cache')
        bound = sources['cache_index'][row['source_key']]
        values = np.asarray(record['summary'], dtype=np.float64)
        require(values.shape == (128,) and np.isfinite(values).all(), 'Invalid saved source summary')
        require(hashlib.sha256(values.tobytes()).hexdigest() == record['summary_sha256'] == bound['summary_sha256'],
                'Source summary differs from its frozen features')
        require(record['primary_feature_sha256'] == bound['primary_feature_sha256']
                and record['source_sha256'] == bound['source_sha256'], 'Source feature hashes differ')
        records.append(record)
        groups.setdefault(row['participant_id'], []).append(record)
    require({p.stem for p in (root/'participant_records').glob('*.json')} == set(splits),
            'Missing or extra batch participant records')
    predictions = prediction_table(root/'predictions.csv', 'predicted_class')
    require(set(predictions.index) == set(splits), 'Batch prediction coverage differs')
    participants = []
    for pid, copies in sorted(groups.items()):
        record = batch.load_record(root/'participant_records'/f'{pid}.json', context_hash)
        require(record['participant_id'] == pid and record['split'] == splits[pid]
                and record['status'] == 'predicted', 'Batch participant identity/status differs')
        require(record['source_count'] == len(copies) and record['all_source_keys'] == [r['source_key'] for r in copies],
                'Batch participant source coverage differs')
        require(len({(digest(r['primary_feature_sha256']), r['sequence_sha256']) for r in copies}) == 1,
                'Conflicting duplicate copies were accepted')
        require(record['selected_source'] in {r['source'] for r in copies}
                and record['sequence_sha256'] == copies[0]['sequence_sha256']
                and record['summary_sha256'] == copies[0]['summary_sha256'], 'Participant feature provenance differs')
        require(record['model'] == contract['model'] and record['model_weights_sha256'] == contract['weights_sha256']
                and record['class_order'] == list(range(5)), 'Participant model/class order differs')
        saved = predictions.loc[pid]
        require(record['predicted_class'] == saved.predicted_class and
                np.allclose(record['probabilities'], saved[PROBABILITY_COLUMNS].to_numpy(float), rtol=0, atol=1e-12),
                'Participant record and predictions CSV disagree')
        require(record['predicted_questionnaire_category'] == contract['class_labels'][int(saved.predicted_class)],
                'Participant category differs from the model contract')
        participants.append(record)
    verify_csv_rows(root/'source_inventory.csv', sources['rows'], 'source_key',
                    ['source_key', 'source', 'participant_id', 'kind'])
    verify_csv_rows(root/'sources.csv', records, 'source_key',
                    ['source_key', 'source', 'participant_id', 'kind', 'status', 'read_mode', 'error', 'source_sha256'])
    verify_csv_rows(root/'participants.csv', participants, 'participant_id',
                    ['participant_id', 'status', 'split', 'source_count', 'reason', 'selected_source'])
    verify_csv_rows(root/'predictions.csv', participants, 'participant_id',
                    ['participant_id', 'split', 'source_count', 'predicted_class', 'predicted_questionnaire_category'])
    require(pd.read_csv(root/'source_errors.csv').empty, 'Completed batch still reports source errors')
    completion = read_json(root/'completion.json')
    cross_id = {}
    for record in participants:
        cross_id.setdefault(record['sequence_sha256'], []).append(record['participant_id'])
    expected = {'complete': True, 'sources_accounted_for': len(records),
                'archives_accounted_for': sum(r['kind'] == 'archive' for r in records),
                'folders_accounted_for': sum(r['kind'] == 'directory' for r in records),
                'source_failures': 0, 'unrecognized_items': [], 'unique_participants': len(splits),
                'predicted_participants': len(splits), 'blocked_participants': 0,
                'same_id_duplicate_groups': sum(len(copies) > 1 for copies in groups.values()),
                'cross_id_identical_feature_groups': [ids for ids in cross_id.values() if len(ids) > 1],
                'cache_reused_sources': len(records), 'raw_read_sources': 0,
                'prediction_counts_by_declared_split': pd.Series(splits).value_counts().to_dict(),
                'models_fitted': 0, 'questionnaire_labels_used': False, 'gender_used': False,
                'performance_metrics_computed': False}
    require(all(completion.get(key) == value for key, value in expected.items()), 'Batch completion counts/claims differ from verified records')
    return predictions


def verify_final(plan, splits):
    root = Path(plan['output'])
    require(read_json(root/'frozen_protocol.json') == {'plan': plan, 'plan_sha256': digest(plan)},
            'Saved final protocol differs from frozen plan')
    receipt = read_json(root/'test_access.json')
    require(receipt['test_set_exposed'] and receipt['plan_sha256'] == digest(plan), 'Final test access receipt differs')
    models = {(item['model'], item['seed']): item for item in plan['models']}
    require(len(models) == len(plan['models']), 'Duplicate frozen model/seed entries')
    saved = read_json(root/'saved_results_verification.json')
    checks = saved['checks']
    require(len(checks) == len(models) and {(c['model'], c['seed']) for c in checks} == set(models),
            'Saved final prediction checks do not cover the frozen models')
    test_ids = {pid for pid, split in splits.items() if split == 'test'}
    for check in checks:
        path = root/f"{check['model']}_seed{check['seed']}"/'test/predictions.csv'
        require(sha256(path) == check['prediction_sha256'], f'Final prediction checksum differs: {path}')
        require(set(prediction_table(path).index) == test_ids and check['participants'] == len(test_ids),
                'Final model prediction coverage differs')
    completion = read_json(root/'completion.json')
    expected = {'complete': True, 'models': len(models), 'test_evaluated': True,
                'all_dataset_participants': len(splits),
                **{f'{split}_participants': sum(value == split for value in splits.values()) for split in ['train', 'validation', 'test']}}
    require(all(completion.get(key) == value for key, value in expected.items()), 'Final completion counts differ')
    log = pd.read_csv(plan['feature_config'].get('experiment_log', 'EXPERIMENT_LOG.csv'), usecols=['model', 'seed', 'split'])
    require(len(log) == saved['experiment_log_entries'], 'Experiment log row count differs from final receipt')
    test_log = log[log.split == 'test']
    expected_log = {('FINAL-'+model, seed) for model, seed in models}
    require(len(test_log) == len(models) and set(zip(test_log.model, test_log.seed)) == expected_log,
            'Final experiment log model/seed entries differ')
    return {'models': len(models), 'experiment_log_entries': len(log)}


def verify_parity(plan, splits, predictions):
    reference = next(item for item in plan['models'] if item['model'] == plan['reference_model_selected_before_test'])
    checks = []
    for split in ['train', 'validation', 'test']:
        path = (Path(plan['output'])/f"{reference['model']}_seed{reference['seed']}"/'test/predictions.csv'
                if split == 'test' else Path(reference['artifact']).parent/split/'predictions.csv')
        saved = prediction_table(path)
        ids = sorted(pid for pid, value in splits.items() if value == split)
        require(saved.index.tolist() == ids, f'Reference {split} prediction coverage differs')
        current = predictions.loc[ids]
        require(np.array_equal(saved.predicted_class, current.predicted_class), f'Batch {split} predicted classes differ')
        difference = float(np.abs(saved[PROBABILITY_COLUMNS].to_numpy()-current[PROBABILITY_COLUMNS].to_numpy()).max())
        require(difference <= 1e-12, f'Batch {split} probabilities differ')
        checks.append({'split': split, 'participants': len(ids), 'max_absolute_probability_difference': difference})
    return checks


def verify_project(dataset='dataset', audit='results/full_dataset', plan_path='results/final_evaluation_plan.json',
                   bundle=DEFAULT_BUNDLE, batch_root='results/batch_inference_all'):
    checks, values = [], {}

    def check(name, dependencies, function, summarize=lambda value: {}):
        missing = [key for key in dependencies if key not in values]
        if missing:
            checks.append({'check': name, 'status': 'blocked', 'reason': 'Prerequisite failed: '+', '.join(missing)})
            return
        try:
            value = function()
            details = summarize(value)
        except Exception as exc:
            checks.append({'check': name, 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
        else:
            values[name] = value
            checks.append({'check': name, 'status': 'passed', **details})

    check('frozen_plan', [], lambda: verify_plan(plan_path), lambda p: {'bound_files': len(p['bound_files']), 'models': len(p['models'])})
    check('portable_bundle', ['frozen_plan'], lambda: verify_bundle(bundle, values['frozen_plan']))
    check('dataset_and_caches', ['portable_bundle'], lambda: verify_dataset(dataset, audit, plan_path, values['portable_bundle']),
          lambda s: {'sources': len(s['rows']), 'archives': sum(r['kind'] == 'archive' for r in s['rows']),
                     'folders': sum(r['kind'] == 'directory' for r in s['rows']),
                     'participants': len({r['participant_id'] for r in s['rows']})})
    check('split_coverage', ['frozen_plan'], lambda: verify_splits(values['frozen_plan']),
          lambda s: {'participants_by_split': pd.Series(s).value_counts().to_dict()})
    check('batch_outputs', ['dataset_and_caches', 'split_coverage'],
          lambda: verify_batch(batch_root, dataset, values['portable_bundle'], values['dataset_and_caches'], values['split_coverage']),
          lambda p: {'participant_predictions': len(p)})
    check('saved_final_results', ['frozen_plan', 'split_coverage'],
          lambda: verify_final(values['frozen_plan'], values['split_coverage']), lambda value: value)
    check('saved_prediction_parity', ['frozen_plan', 'split_coverage', 'batch_outputs', 'saved_final_results'],
          lambda: verify_parity(values['frozen_plan'], values['split_coverage'], values['batch_outputs']), lambda value: {'splits': value})
    return {'format': 'local-project-verification-v1', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'passed': all(c['status'] == 'passed' for c in checks), 'checks': checks,
            'models_fitted': 0, 'new_model_predictions': 0, 'new_performance_metrics': 0,
            'raw_archives_rehashed': False,
            'limitations': ['Source freshness uses the original audit and current filesystem stamps, not another full archive read',
                            'Local checksums detect drift relative to saved records; they are not independently signed evidence',
                            'Verification does not establish model accuracy or clinical validity; the final test has already been used']}


def write_report(report, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Local project verification', '', f"All checks passed: **{report['passed']}**.", '',
             '| Check | Status | Details |', '|---|---|---|']
    for check in report['checks']:
        details = check.get('reason') or json.dumps({k: v for k, v in check.items() if k not in {'check', 'status'}})
        lines.append(f"| {check['check']} | {check['status']} | {details.replace('|', '/').replace(chr(10), ' ')} |")
    lines += ['', 'This command performed no fitting, model inference or performance evaluation. It changed no input artifacts.', '',
              *[f'- {item}.' for item in report['limitations']], '', 'See verification.json for structured results.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--audit', default='results/full_dataset')
    parser.add_argument('--plan', default='results/final_evaluation_plan.json')
    parser.add_argument('--bundle', default=DEFAULT_BUNDLE)
    parser.add_argument('--batch', default='results/batch_inference_all')
    parser.add_argument('--output', help='Optional new directory for local JSON/Markdown reports; omit for a read-only check')
    args = parser.parse_args()
    if args.output:
        path = Path(args.output).resolve()
        protected = [Path(p).resolve() for p in [args.dataset, args.audit, args.bundle, args.batch]]
        if path.exists() or any(path.is_relative_to(p) for p in protected):
            parser.error('Preserving inputs/existing reports: choose a new output directory outside dataset/audit/bundle/batch')
    report = verify_project(args.dataset, args.audit, args.plan, args.bundle, args.batch)
    if args.output:
        write_report(report, args.output)
    print(json.dumps({'passed': report['passed'], 'checks': {c['check']: c['status'] for c in report['checks']},
                      'failures': [{'check': c['check'], 'reason': c['reason']} for c in report['checks'] if c['status'] == 'failed'],
                      'report_directory': args.output, 'new_model_predictions': 0, 'models_fitted': 0}, indent=2))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
