import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src import raw_frame_comparison as comparison
from src import raw_frame_features as raw
from src import verify_raw_frame_stage as verify
from src.batch_inference import save_record
from src.common import digest, write_json
from test_raw_frame_comparison import synthetic


@pytest.fixture
def stage(synthetic, tmp_path, monkeypatch):
    matrix, manifest, _, training, folds, baseline, original_settings = synthetic
    matrix[:, 39:78] = np.abs(matrix[:, 39:78])
    x, training = comparison.training_inputs(matrix, manifest)
    root = tmp_path/'raw'
    (root/'participants').mkdir(parents=True)
    coverage = manifest[['participant_id', 'split']]
    sources = [{'participant_id': pid, 'source_key': 'source'+pid, 'kind': 'archive', 'source': f'synthetic/{pid}.zip'}
               for pid in manifest.participant_id]
    context = {'coverage': coverage.to_dict('records'), 'sources': sources,
               'selected_sources': {s['participant_id']: s['source_key'] for s in sources}}
    context_hash = digest(context)
    write_json(root/'protocol.json', {'context': context, 'context_hash': context_hash})
    records = []
    for index, source in enumerate(sources):
        row = matrix[index]
        variance = [{'feature': name, 'family': raw.FAMILIES[i//13], 'raw_variance': row[i+39]**2,
                     'within_bin_variance': .8*row[i+39]**2, 'between_bin_variance': .2*row[i+39]**2,
                     'retained_variance_fraction': .2} for i, name in enumerate(raw.AUDIO_COLUMNS)]
        record = {**source, 'summary': row.tolist(), 'variance': variance,
                  'raw_frames_total': 100, 'raw_frames_in_cached_interval': 90,
                  'interval_start': 0., 'interval_end_exclusive': 1., 'observed_audio_bins': 2,
                  'cache_reproduced_exactly': True, 'variance_decomposition_verified': True}
        save_record(root/'participants'/f"{source['participant_id']}.json", record, context_hash)
        records.append(record)
    coverage.to_csv(root/'participant_coverage.csv', index=False)
    pd.DataFrame([{**s, 'used_summary_participant_id': s['participant_id'], 'raw_mfcc_read_here': True,
                   'primary_features_match_selected': True} for s in sources]).to_csv(root/'source_coverage.csv', index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ['summary', 'variance']} for r in records]).to_csv(root/'quality.csv', index=False)
    pd.DataFrame([{'participant_id': r['participant_id'], **v} for r in records for v in r['variance']]).to_csv(root/'variance_retention.csv', index=False)
    write_json(root/'completion.json', {'participants': 30, 'prepared_participants': 30, 'physical_sources': 30,
               'archives': 30, 'folders': 0, 'raw_frames_in_cached_interval': 2700, 'complete': True, 'failures': [],
               'model_fits': 0, 'predictions_created': False, 'new_raw_reads_this_invocation': 30, 'reused_participants_this_invocation': 0})
    settings = {**original_settings, 'raw_features': str(root)}
    out = tmp_path/'comparison'
    inputs = (x, training, folds, baseline, coverage, {}, context)
    comparison.run_comparison(x, training, folds, baseline, settings, out, {}, coverage.to_dict('records'))
    monkeypatch.setattr(verify, 'load_inputs', lambda _: inputs)
    return settings, out, inputs


def test_complete_verification_is_read_only_without_model_or_archive_execution(stage, tmp_path, monkeypatch):
    settings, out, _ = stage
    def forbidden(*args, **kwargs):
        raise AssertionError('Verification must not execute this operation')
    monkeypatch.setattr(pickle, 'load', forbidden)
    monkeypatch.setattr(raw, 'read_mfcc', forbidden)
    monkeypatch.setattr(comparison, 'run_raw_job', forbidden)
    before = {str(p): (verify.sha256(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    result = verify.verify_stage(settings, out)
    assert result['passed'] and len(result['checks']) == 5
    assert result['saved_training_metrics_recomputed'] and result['models_fitted'] == result['new_model_predictions'] == 0
    assert before == {str(p): (verify.sha256(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage,expected', [
    ('feature_count', 'feature_records_and_tables'), ('extra_feature_record', 'feature_records_and_tables'),
    ('variance', 'feature_records_and_tables'), ('feature_table', 'feature_records_and_tables'),
    ('protocol', 'comparison_protocol'), ('checkpoint', 'saved_nested_jobs'),
    ('predictions', 'saved_predictions_and_metrics'), ('metrics', 'saved_predictions_and_metrics'),
    ('summary', 'saved_predictions_and_metrics'), ('job_count', 'saved_nested_jobs')])
def test_corruption_identifies_stage_and_blocks_dependents(stage, damage, expected):
    settings, out, _ = stage
    root = raw.Path(settings['raw_features'])
    if damage == 'feature_count':
        path = root/'completion.json'
        value = verify.read_json(path)
        value['physical_sources'] += 1
        write_json(path, value)
    elif damage == 'extra_feature_record':
        (root/'participants/extra.json').write_text('{}')
    elif damage == 'variance':
        path = root/'participants/0.json'
        envelope = verify.read_json(path)
        envelope['record']['variance'][0]['within_bin_variance'] += 1
        save_record(path, envelope['record'], envelope['context_hash'])
    elif damage == 'feature_table':
        path = root/'source_coverage.csv'
        table = pd.read_csv(path)
        table.loc[0, 'used_summary_participant_id'] = 9999
        table.to_csv(path, index=False)
    elif damage == 'protocol':
        path = out/'protocol.json'
        value = verify.read_json(path)
        value['context']['validation_test_scoring'] = True
        write_json(path, value)
    elif damage == 'checkpoint':
        path = next((out/'jobs').rglob('model.pkl'))
        path.write_bytes(path.read_bytes()+b'changed')
    elif damage == 'predictions':
        path = next((out/'oof').glob('*/predictions.csv'))
        table = pd.read_csv(path)
        table.loc[0, 'participant_id'] = 9999
        table.to_csv(path, index=False)
    elif damage == 'metrics':
        path = next((out/'oof').glob('*/metrics.json'))
        value = verify.read_json(path)
        value['macro_f1'] += .1
        write_json(path, value)
    elif damage == 'summary':
        path = out/'summary.csv'
        table = pd.read_csv(path)
        table.loc[0, 'macro_f1_mean'] += .1
        table.to_csv(path, index=False)
    elif damage == 'job_count':
        path = out/'completion.json'
        value = verify.read_json(path)
        value['total_logistic_fits'] += 1
        write_json(path, value)
    result = verify.verify_stage(settings, out)
    assert not result['passed']
    statuses = {c['check']: c['status'] for c in result['checks']}
    assert statuses[expected] == 'failed'
    if expected in ['comparison_protocol', 'saved_nested_jobs']:
        assert statuses['saved_predictions_and_metrics'] == 'blocked'


def test_input_failure_blocks_checks_and_report_preserves_existing_output(tmp_path, monkeypatch):
    def failed(_):
        raise ValueError('synthetic changed dataset binding')
    monkeypatch.setattr(verify, 'load_inputs', failed)
    report = verify.verify_stage({}, tmp_path/'unused')
    assert not report['passed'] and report['checks'][0]['status'] == 'failed'
    assert all(c['status'] == 'blocked' for c in report['checks'][1:])
    out = tmp_path/'report'
    verify.write_report(report, out)
    before = (out/'verification.json').read_bytes()
    with pytest.raises(FileExistsError):
        verify.write_report(report, out)
    assert (out/'verification.json').read_bytes() == before
    assert verify.read_json(out/'verification.json')['passed'] is False
