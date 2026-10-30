import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import batch_inference as batch
from src import verify_project as verify
from src.common import digest, load_config, write_json
from src.exploratory_baselines import pool_modalities
from src.final_evaluation import PROTOCOL
from src.full_dataset import run_full_dataset
from src.reference_inference import load_bundle, probabilities, read_cache, sha256
from test_batch_inference import write_source
from test_reference_inference import bundle_fixture


@pytest.fixture
def completed_project(tmp_path):
    """Small synthetic completed workflow; no real participant data."""
    bundle, contract, _, _ = bundle_fixture(tmp_path)
    dataset, audit, output, final = [tmp_path/name for name in ['dataset', 'audit', 'batch', 'final']]
    dataset.mkdir()
    final.mkdir()
    for name, pid in [('123_P.zip', '123'), ('123_P 2', '123'), ('456_P.tar.gz', '456'), ('789_P', '789')]:
        write_source(dataset/name, pid, changed=pid != '123')
    splits = tmp_path/'splits.csv'
    splits.write_text('participant_id,split\n123,train\n456,validation\n789,test\n')
    log = tmp_path/'experiments.csv'
    log.write_text('model,seed,split\nFINAL-synthetic-reference,42,test\n')
    config = {**load_config('configs/default.json'), 'dataset': str(dataset), 'cache': str(tmp_path/'cache'),
              'splits': str(splits), 'experiment_log': str(log)}
    run_full_dataset(config, audit)
    participants = pd.read_csv(audit/'participants.csv', dtype={'participant_id': str})
    model_dir = tmp_path/'model'
    model_dir.mkdir()
    artifact = model_dir/'model.pkl'
    artifact.write_text('Verification must never deserialize this synthetic marker')
    features, frames = {}, {}
    _, parameters = load_bundle(bundle)
    for row, split in zip(participants.itertuples(), ['train', 'validation', 'test']):
        path = Path(row.audio_feature_path)
        sequence, _ = read_cache(path, contract)
        summary = pool_modalities(sequence)
        features[row.participant_id] = {'cache': str(path), 'npz_sha256': sha256(path),
                                       'source_fingerprint': verify.read_json(path.with_suffix('.json'))['fingerprint'],
                                       'summary_sha256': hashlib.sha256(summary.tobytes()).hexdigest()}
        probs = probabilities(parameters, summary[None, :])[0]
        frame = pd.DataFrame([{'participant_id': row.participant_id, 'predicted_label': int(probs.argmax()),
                               **{f'p{i}': p for i, p in enumerate(probs)}}])
        directory = final/'synthetic-reference_seed42'/'test' if split == 'test' else model_dir/split
        directory.mkdir(parents=True)
        frame.to_csv(directory/'predictions.csv', index=False)
        frames[split] = directory/'predictions.csv'
    plan = {'protocol': PROTOCOL, 'feature_config': config, 'output': str(final), 'source_provenance': features,
            'reference_model_selected_before_test': 'synthetic-reference',
            'models': [{'model': 'synthetic-reference', 'seed': 42, 'artifact': str(artifact),
                        'validation_predictions': str(frames['validation'])}],
            'bound_files': {str(p): sha256(p) for p in [artifact, splits, frames['validation']]}}
    plan_path = tmp_path/'plan.json'
    envelope = {'plan': plan, 'plan_sha256': digest(plan)}
    write_json(plan_path, envelope)
    write_json(final/'frozen_protocol.json', envelope)
    write_json(final/'test_access.json', {'test_set_exposed': True, 'plan_sha256': digest(plan)})
    write_json(final/'completion.json', {'complete': True, 'models': 1, 'test_evaluated': True,
                                       'all_dataset_participants': 3, 'train_participants': 1,
                                       'validation_participants': 1, 'test_participants': 1})
    write_json(final/'saved_results_verification.json', {'experiment_log_entries': 1,
               'checks': [{'model': 'synthetic-reference', 'seed': 42, 'participants': 1,
                           'prediction_sha256': sha256(frames['test'])}]})
    contract.update(frozen_plan_sha256=digest(plan), source_model_sha256=sha256(artifact))
    write_json(bundle/'model.json', {'contract': contract, 'contract_sha256': digest(contract)})
    batch.run_batch(dataset, bundle, output, audit, plan_path, splits)
    return dict(dataset=dataset, audit=audit, plan_path=plan_path, bundle=bundle, batch_root=output)


def test_completed_verification_never_predicts_fits_decompresses_or_writes(completed_project, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Verification must not predict, fit, deserialize or decompress')
    monkeypatch.setattr(batch, 'probabilities', forbidden)
    monkeypatch.setattr('src.reference_inference.probabilities', forbidden)
    monkeypatch.setattr('sklearn.linear_model.LogisticRegression.fit', forbidden)
    monkeypatch.setattr('pickle.load', forbidden)
    monkeypatch.setattr('torch.load', forbidden)
    monkeypatch.setattr(batch, 'read_features', forbidden)
    monkeypatch.setattr('src.full_dataset.read_source', forbidden)
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    result = verify.verify_project(**completed_project)
    assert result['passed'], result
    assert len(result['checks']) == 7
    assert result['new_model_predictions'] == result['models_fitted'] == result['new_performance_metrics'] == 0
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage,failed_check', [
    ('missing_archive', 'dataset_and_caches'),
    ('cache', 'dataset_and_caches'),
    ('source_swap', 'batch_outputs'),
    ('missing_participant', 'batch_outputs'),
    ('prediction', 'batch_outputs'),
    ('counts', 'batch_outputs'),
    ('final_prediction', 'saved_final_results'),
    ('experiment_log', 'saved_final_results'),
    ('model', 'frozen_plan'),
])
def test_damage_is_reported_without_repair_or_false_success(completed_project, damage, failed_check):
    p = completed_project
    if damage == 'missing_archive':
        (p['dataset']/'123_P.zip').unlink()
    elif damage == 'cache':
        plan = verify.read_json(p['plan_path'])['plan']
        Path(plan['source_provenance']['123']['cache']).write_bytes(b'corrupt')
    elif damage == 'source_swap':
        first, second = list((p['batch_root']/'source_records').glob('*.json'))[:2]
        a, b = first.read_bytes(), second.read_bytes()
        first.write_bytes(b)
        second.write_bytes(a)
    elif damage == 'missing_participant':
        (p['batch_root']/'participant_records/123.json').unlink()
    elif damage == 'prediction':
        path = p['batch_root']/'predictions.csv'
        table = pd.read_csv(path)
        table.loc[0, 'p0'] = 2
        table.to_csv(path, index=False)
    elif damage == 'counts':
        path = p['batch_root']/'completion.json'
        saved = verify.read_json(path)
        saved['sources_accounted_for'] = 999
        write_json(path, saved)
    elif damage == 'final_prediction':
        plan = verify.read_json(p['plan_path'])['plan']
        (Path(plan['output'])/'synthetic-reference_seed42/test/predictions.csv').write_text('damaged')
    elif damage == 'model':
        plan = verify.read_json(p['plan_path'])['plan']
        Path(plan['models'][0]['artifact']).write_bytes(b'changed')
    elif damage == 'experiment_log':
        plan = verify.read_json(p['plan_path'])['plan']
        Path(plan['feature_config']['experiment_log']).write_text('model,seed,split\nFINAL-other,42,test\n')
    result = verify.verify_project(**p)
    assert not result['passed']
    assert next(c for c in result['checks'] if c['check'] == failed_check)['status'] == 'failed'
    assert next(c for c in result['checks'] if c['check'] == 'saved_prediction_parity')['status'] == 'blocked'


def test_cli_failure_report_and_output_preservation(tmp_path, monkeypatch):
    out = tmp_path/'report'
    monkeypatch.setattr('sys.argv', ['verify_project', '--plan', str(tmp_path/'missing.json'), '--output', str(out)])
    with pytest.raises(SystemExit) as exc:
        verify.main()
    assert exc.value.code == 2
    result = verify.read_json(out/'verification.json')
    assert not result['passed'] and result['checks'][0]['status'] == 'failed'
    before = sha256(out/'verification.json')
    with pytest.raises(SystemExit) as exc:
        verify.main()
    assert exc.value.code == 2 and sha256(out/'verification.json') == before
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    monkeypatch.setattr('sys.argv', ['verify_project', '--dataset', str(dataset), '--output', str(dataset/'report')])
    with pytest.raises(SystemExit):
        verify.main()
    assert not (dataset/'report').exists()


def test_float32_serialization_tolerance_does_not_accept_invalid_probabilities(tmp_path):
    path = tmp_path/'predictions.csv'
    values = np.array([.4, .3, .15, .1, .05], dtype=np.float32)
    values[0] += np.float32(1e-7)
    row = {'participant_id': 'synthetic', 'predicted_label': 0,
           **{f'p{i}': float(p) for i, p in enumerate(values)}}
    pd.DataFrame([row]).to_csv(path, index=False)
    assert len(verify.prediction_table(path)) == 1
    pd.DataFrame([{**row, 'p0': .41}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Invalid probabilities'):
        verify.prediction_table(path)
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='duplicate'):
        verify.prediction_table(path)
