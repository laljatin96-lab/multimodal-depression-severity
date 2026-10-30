import io
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import batch_inference as batch
from src.common import digest, load_config, write_json
from src.exploratory_baselines import pool_modalities
from src.final_evaluation import PROTOCOL as FINAL_PROTOCOL
from src.full_dataset import run_full_dataset
from src.reference_inference import read_cache, sha256
from test_reference_inference import bundle_fixture, feature_pair


def write_source(path, pid, changed=False):
    audio, visual = feature_pair()
    if changed:
        frame = pd.read_csv(io.BytesIO(audio), sep=';')
        frame.iloc[:, 0] += .25
        audio = frame.to_csv(sep=';', index=False).encode()
    files = {f'{pid}_P/features/{pid}_OpenSMILE2.3.0_mfcc.csv': audio,
             f'{pid}_P/features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv': visual}
    if path.suffix == '.zip':
        with zipfile.ZipFile(path, 'w') as archive:
            for name, value in files.items():
                archive.writestr(name, value)
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path, 'w:gz') as archive:
            for name, value in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    else:
        for name, value in files.items():
            target = path/Path(name).relative_to(f'{pid}_P')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)


def setup_sources(tmp_path):
    bundle, contract, _, _ = bundle_fixture(tmp_path)
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    write_source(dataset/'123_P.zip', '123')
    write_source(dataset/'123_P 2', '123')
    write_source(dataset/'456_P.tar.gz', '456', changed=True)
    return dataset, bundle, contract


def test_every_source_one_prediction_per_person_and_unchanged_resume(tmp_path, monkeypatch):
    dataset, bundle, _ = setup_sources(tmp_path)
    output = tmp_path/'output'
    def forbidden(*args, **kwargs):
        raise AssertionError('No labels, fitting, or repeated processing expected')
    monkeypatch.setattr('src.common.read_metadata', forbidden)
    result = batch.run_batch(dataset, bundle, output)
    assert result['complete'] and result['sources_accounted_for'] == 3
    assert result['archives_accounted_for'] == 2 and result['folders_accounted_for'] == 1
    assert result['same_id_duplicate_groups'] == 1 and result['predicted_participants'] == 2
    assert result['raw_read_sources'] == 3 and not result['performance_metrics_computed']
    predictions = pd.read_csv(output/'predictions.csv', dtype={'participant_id': str})
    assert predictions.participant_id.tolist() == ['123', '456']
    times = {p: p.stat().st_mtime_ns for p in output.glob('*_records/*.json')}
    monkeypatch.setattr(batch, 'process_source', forbidden)
    monkeypatch.setattr(batch, 'probabilities', forbidden)
    resumed = batch.run_batch(dataset, bundle, output, resume=True)
    assert resumed['source_records_reused_this_invocation'] == 3
    assert resumed['participant_records_reused_this_invocation'] == 2
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in times.items())
    with pytest.raises(FileExistsError, match='Preserving'):
        batch.run_batch(dataset, bundle, output)


def test_interruption_resumes_only_unfinished_sources(tmp_path, monkeypatch):
    dataset, bundle, _ = setup_sources(tmp_path)
    output = tmp_path/'output'
    original = batch.process_source
    calls = []
    def interrupted(source, *args):
        calls.append(source['source_key'])
        if len(calls) == 2:
            raise KeyboardInterrupt('simulated interruption')
        return original(source, *args)
    monkeypatch.setattr(batch, 'process_source', interrupted)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(dataset, bundle, output)
    assert len(list((output/'source_records').glob('*.json'))) == 1
    resumed_calls = []
    def remaining(source, *args):
        resumed_calls.append(source['source_key'])
        return original(source, *args)
    monkeypatch.setattr(batch, 'process_source', remaining)
    result = batch.run_batch(dataset, bundle, output, resume=True)
    assert result['complete'] and result['source_records_reused_this_invocation'] == 1
    assert len(resumed_calls) == 2 and calls[0] not in resumed_calls


def test_conflicting_copies_block_prediction_and_partial_cli_has_nonzero_exit(tmp_path, monkeypatch):
    bundle, _, _, _ = bundle_fixture(tmp_path)
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    write_source(dataset/'123_P.zip', '123')
    write_source(dataset/'123_P (1).zip', '123', changed=True)
    output = tmp_path/'output'
    monkeypatch.setattr('sys.argv', ['batch', '--dataset', str(dataset), '--bundle', str(bundle), '--output', str(output)])
    with pytest.raises(SystemExit) as exc:
        batch.main()
    assert exc.value.code == 2
    report = json.loads((output/'completion.json').read_text())
    assert not report['complete'] and report['predicted_participants'] == 0 and report['blocked_participants'] == 1
    assert len(pd.read_csv(output/'predictions.csv')) == 0
    assert pd.read_csv(output/'participants.csv').iloc[0].status == 'conflicting_copies'


def test_bad_archive_and_unknown_files_are_reported_without_dropping_good_people(tmp_path):
    bundle, _, _, _ = bundle_fixture(tmp_path)
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    write_source(dataset/'123_P.zip', '123')
    (dataset/'456_P.zip').write_bytes(b'invalid zip')
    (dataset/'notes.txt').write_text('unrecognized item')
    output = tmp_path/'output'
    result = batch.run_batch(dataset, bundle, output)
    assert not result['complete'] and result['source_failures'] == 1
    assert result['unique_participants'] == 2 and result['predicted_participants'] == 1
    assert len(result['unrecognized_items']) == 1
    assert len(pd.read_csv(output/'source_errors.csv')) == 1
    assert len(pd.read_csv(output/'sources.csv')) == 2


def test_resume_rejects_changed_sources_and_corrupt_records(tmp_path):
    dataset, bundle, _ = setup_sources(tmp_path)
    output = tmp_path/'output'
    batch.run_batch(dataset, bundle, output)
    path = next((output/'source_records').glob('*.json'))
    envelope = json.loads(path.read_text())
    envelope['record']['summary'][0] += 1
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='integrity/context'):
        batch.run_batch(dataset, bundle, output, resume=True)
    write_source(dataset/'123_P.zip', '123', changed=True)
    with pytest.raises(ValueError, match='Resume inputs'):
        batch.run_batch(dataset, bundle, output, resume=True)


def test_cache_reuse_binds_every_copy_and_rejects_cache_changes(tmp_path, monkeypatch):
    dataset, bundle, contract = setup_sources(tmp_path)
    config = {**load_config('configs/default.json'), 'dataset': str(dataset), 'cache': str(tmp_path/'cache')}
    audit = tmp_path/'audit'
    run_full_dataset(config, audit)
    participants = pd.read_csv(audit/'participants.csv', dtype={'participant_id': str})
    features = {}
    for row in participants.itertuples():
        path = Path(row.audio_feature_path)
        sequence, _ = read_cache(path, contract)
        metadata = json.loads(path.with_suffix('.json').read_text())
        features[row.participant_id] = {'cache': str(path), 'npz_sha256': sha256(path),
                                        'source_fingerprint': metadata['fingerprint'],
                                        'summary_sha256': hashlib.sha256(pool_modalities(sequence).tobytes()).hexdigest()}
    plan = {'protocol': FINAL_PROTOCOL, 'bound_files': {}, 'source_provenance': features}
    plan_path = tmp_path/'plan.json'
    write_json(plan_path, {'plan': plan, 'plan_sha256': digest(plan)})
    contract['frozen_plan_sha256'] = digest(plan)
    write_json(bundle/'model.json', {'contract': contract, 'contract_sha256': digest(contract)})
    def forbidden(*args, **kwargs):
        raise AssertionError('Verified cache reuse must not decompress sources again')
    monkeypatch.setattr(batch, 'read_features', forbidden)
    output = tmp_path/'output'
    result = batch.run_batch(dataset, bundle, output, audit=audit, frozen_plan=plan_path)
    assert result['complete'] and result['cache_reused_sources'] == 3 and result['raw_read_sources'] == 0
    path = Path(next(iter(features.values()))['cache'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='cache checksum changed'):
        batch.run_batch(dataset, bundle, tmp_path/'changed', audit=audit, frozen_plan=plan_path)


def test_split_annotations_are_optional_exact_and_not_training_inputs(tmp_path):
    dataset, bundle, _ = setup_sources(tmp_path)
    splits = tmp_path/'splits.csv'
    splits.write_text('participant_id,split\n123,validation\n456,test\n')
    result = batch.run_batch(dataset, bundle, tmp_path/'output', splits=splits)
    assert result['prediction_counts_by_declared_split'] == {'validation': 1, 'test': 1}
    assert not result['questionnaire_labels_used'] and result['models_fitted'] == 0
    splits.write_text('participant_id,split\n123,validation\n')
    with pytest.raises(ValueError, match='cover each'):
        batch.run_batch(dataset, bundle, tmp_path/'invalid', splits=splits)
    with pytest.raises(ValueError, match='outside'):
        batch.run_batch(dataset, bundle, dataset/'outputs')


@pytest.mark.parametrize('directory', ['source_records', 'participant_records'])
def test_resume_rejects_swapped_valid_records(tmp_path, directory):
    dataset, bundle, _ = setup_sources(tmp_path)
    output = tmp_path/'output'
    batch.run_batch(dataset, bundle, output)
    first, second = list((output/directory).glob('*.json'))[:2]
    a, b = first.read_bytes(), second.read_bytes()
    first.write_bytes(b)
    second.write_bytes(a)
    with pytest.raises(ValueError, match='Resume record identity'):
        batch.run_batch(dataset, bundle, output, resume=True)
