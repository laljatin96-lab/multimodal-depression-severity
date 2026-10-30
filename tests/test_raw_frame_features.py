import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import raw_frame_features as raw
from src.full_dataset import source_stamp
from src.prepare_features import AUDIO_COLUMNS, bin_frames


def sample():
    times = np.array([0, .1, .6, 1., 1.1, 1.6, 2.])
    values = np.tile(np.array([0., 2., 4., 3., 5., 6., 999.])[:, None], (1, 39))
    values[:, 0] = 7
    frame = pd.DataFrame(values, columns=AUDIO_COLUMNS)
    frame['frameTime'] = times
    audio, mask = bin_frames(times, values, np.ones(len(times), bool), np.arange(5)*.5)
    seq = {'audio': audio, 'audio_mask': mask, 'timestamps': np.arange(4)*.5,
           'visual': np.arange(100, dtype=float).reshape(4, 25), 'visual_mask': np.array([True, False, True, True])}
    return frame.to_csv(sep=';', index=False).encode(), seq, values[:-1]


def report_for(path, name, blob, pid='123'):
    return {'source': str(path), 'source_key': path.name, 'participant_id': pid, 'kind': 'directory' if path.is_dir() else 'archive',
            'integrity_checked': True, 'source_stamp': source_stamp(path),
            'members': [{'member': name, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}],
            'feature_sha256': {Path(name).name: hashlib.sha256(blob).hexdigest()}}


def test_raw_summary_preserves_variance_exact_crop_and_visuals():
    blob, seq, x = sample()
    result = raw.summarize_raw(blob, seq, {'bin_seconds': .5, 'audio_time_offset': 0})
    np.testing.assert_allclose(result['summary'][:39], x.mean(0))
    np.testing.assert_allclose(result['summary'][39:78], x.std(0))
    np.testing.assert_array_equal(result['summary'][78:], raw.pool_modalities(seq)[78:])
    assert result['raw_frames_in_cached_interval'] == 6 and result['raw_frames_total'] == 7
    assert result['variance'][0]['retained_variance_fraction'] is None
    for v in result['variance']:
        assert v['within_bin_variance']+v['between_bin_variance'] == pytest.approx(v['raw_variance'])
    assert result['variance'][1]['within_bin_variance'] > 0
    seq['audio'][0, 0] += 1
    with pytest.raises(ValueError, match='exactly reproduce'):
        raw.summarize_raw(blob, seq, {'bin_seconds': .5, 'audio_time_offset': 0})


@pytest.mark.parametrize('kind', ['folder', 'zip', 'tar.gz'])
def test_targeted_reader_checks_raw_hash_and_stamps(tmp_path, kind):
    blob, _, _ = sample()
    name = 'features/123_OpenSMILE2.3.0_mfcc.csv'
    path = tmp_path / ('123_P' if kind == 'folder' else '123_P.'+kind)
    if kind == 'folder':
        (path/'features').mkdir(parents=True)
        (path/name).write_bytes(blob)
    elif kind == 'zip':
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(name, blob)
    else:
        with tarfile.open(path, 'w:gz') as archive:
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))
    report = report_for(path, name, blob)
    assert raw.read_mfcc(report) == blob
    report['members'][0]['sha256'] = 'bad'
    report['feature_sha256'][Path(name).name] = 'bad'
    with pytest.raises(ValueError, match='checksum'):
        raw.read_mfcc(report)
    report['source_stamp'] = 'changed'
    with pytest.raises(ValueError, match='Source changed'):
        raw.read_mfcc(report)


def test_unsafe_or_duplicate_audited_members_rejected(tmp_path):
    path = tmp_path/'123_P.tar'
    path.write_bytes(b'')
    report = report_for(path, '../123_OpenSMILE2.3.0_mfcc.csv', b'')
    with pytest.raises(ValueError, match='Unsafe'):
        raw.read_mfcc(report)
    report['members'][0]['member'] = '123_OpenSMILE2.3.0_mfcc.csv'
    report['members'] *= 2
    with pytest.raises(ValueError, match='one bounded'):
        raw.read_mfcc(report)


def test_preparation_resume_preserves_records_and_rejects_changed_context(tmp_path, monkeypatch):
    blob, seq, _ = sample()
    source = tmp_path/'dataset'/'123_P'
    source.mkdir(parents=True)
    name = '123_OpenSMILE2.3.0_mfcc.csv'
    (source/name).write_bytes(blob)
    report = report_for(source, name, blob)
    coverage = pd.DataFrame({'participant_id': ['123'], 'split': ['train']})
    context = {'feature_config': {'dataset': str(source.parent), 'bin_seconds': .5, 'audio_time_offset': 0},
               'sources': [{k: report[k] for k in ['participant_id', 'source_key', 'kind']}], 'version': 1}
    monkeypatch.setattr(raw, 'input_state', lambda _: (context.copy(), coverage, {'123': seq}, {'123': report}))
    out = tmp_path/'outputs'
    first = raw.prepare('synthetic', out)
    assert first['complete'] and first['new_raw_reads_this_invocation'] == 1
    record = out/'participants/123.json'
    before = record.read_bytes(), record.stat().st_mtime_ns
    def forbidden(*args):
        raise AssertionError('Completed sources must not be read again')
    monkeypatch.setattr(raw, 'process_participant', forbidden)
    second = raw.prepare('synthetic', out, resume=True)
    assert second['new_raw_reads_this_invocation'] == 0 and second['reused_participants_this_invocation'] == 1
    assert before == (record.read_bytes(), record.stat().st_mtime_ns)
    with pytest.raises(FileExistsError):
        raw.prepare('synthetic', out)
    context['version'] = 2
    with pytest.raises(ValueError, match='changed'):
        raw.prepare('synthetic', out, resume=True)
    context['version'] = 1
    envelope = json.loads(record.read_text())
    envelope['record']['summary'][0] += 1
    record.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='integrity'):
        raw.prepare('synthetic', out, resume=True)


def test_failed_participant_is_visible_and_retried(tmp_path, monkeypatch):
    blob, seq, _ = sample()
    source = tmp_path/'dataset'/'123_P'
    source.mkdir(parents=True)
    name = '123_OpenSMILE2.3.0_mfcc.csv'
    (source/name).write_bytes(blob)
    report = report_for(source, name, blob)
    coverage = pd.DataFrame({'participant_id': ['123'], 'split': ['train']})
    context = {'feature_config': {'dataset': str(source.parent), 'bin_seconds': .5, 'audio_time_offset': 0},
               'sources': [{k: report[k] for k in ['participant_id', 'source_key', 'kind']}]}
    monkeypatch.setattr(raw, 'input_state', lambda _: (context, coverage, {'123': seq}, {'123': report}))
    original = raw.process_participant
    def failed(*args):
        raise ValueError('synthetic read failure')
    monkeypatch.setattr(raw, 'process_participant', failed)
    out = tmp_path/'outputs'
    first = raw.prepare('synthetic', out)
    assert not first['complete'] and len(first['failures']) == 1
    assert not (out/'participant_summaries.npz').exists()
    monkeypatch.setattr(raw, 'process_participant', original)
    assert raw.prepare('synthetic', out, resume=True)['complete']
