import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import VERSION, digest, load_config, write_json
from src.prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS, align
from src.reference_inference import (
    FORMAT, LABELS, SETTINGS, load_bundle, parameters_from_estimator, predict,
    probabilities, read_cache, sha256, validate_sequence,
)


def bundle_fixture(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(25, 128))
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000)).fit(x, np.arange(25) % 5)
    arrays = parameters_from_estimator(estimator)
    root = tmp_path/'bundle'
    root.mkdir()
    np.savez_compressed(root/'weights.npz', **arrays)
    config = load_config('configs/default.json')
    contract = {'format': FORMAT, 'model': 'synthetic-reference', 'feature_version': VERSION,
                'audio_columns': AUDIO_COLUMNS, 'visual_columns': VISUAL_COLUMNS,
                'preprocessing': {key: config[key] for key in SETTINGS},
                'weights_sha256': sha256(root/'weights.npz'), 'class_labels': LABELS,
                'limitations': ['Synthetic correctness fixture only']}
    write_json(root/'model.json', {'contract': contract, 'contract_sha256': digest(contract)})
    return root, contract, estimator, x


def feature_pair():
    times = np.arange(0, 4, .1)
    audio = pd.DataFrame(np.sin(times[:, None]+np.arange(39)[None, :]), columns=AUDIO_COLUMNS)
    audio['frameTime'] = times
    visual = pd.DataFrame(np.cos(times[:, None]+np.arange(25)[None, :]), columns=VISUAL_COLUMNS)
    visual['timestamp'], visual['success'], visual['confidence'] = times, 1, .99
    visual.loc[10:14, 'confidence'] = .1
    return audio.to_csv(sep=';', index=False).encode(), visual.to_csv(index=False).encode()


def test_numeric_model_matches_original_probabilities_and_detects_modified_bundle(tmp_path):
    root, _, estimator, x = bundle_fixture(tmp_path)
    _, arrays = load_bundle(root)
    np.testing.assert_allclose(probabilities(arrays, x), estimator.predict_proba(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(probabilities(arrays, x).argmax(1), estimator.predict(x))
    (root/'weights.npz').write_bytes(b'modified')
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(root)


@pytest.mark.parametrize('source_kind', ['zip', 'tar.gz', 'directory'])
def test_unlabeled_archive_cache_and_feature_pair_match_without_fitting(tmp_path, monkeypatch, source_kind):
    root, contract, _, _ = bundle_fixture(tmp_path)
    audio, visual = feature_pair()
    sequence, info = align(audio, visual, contract['preprocessing'])
    cache = tmp_path/'777.npz'
    np.savez_compressed(cache, **sequence)
    write_json(cache.with_suffix('.json'), {**info, 'version': VERSION, 'participant_id': '777', 'settings': contract['preprocessing']})
    names = ['777_P/features/777_OpenSMILE2.3.0_mfcc.csv', '777_P/features/777_OpenFace2.1.0_Pose_gaze_AUs.csv']
    source = tmp_path/('777_P' if source_kind == 'directory' else f'777_P.{source_kind}')
    if source_kind == 'zip':
        with zipfile.ZipFile(source, 'w') as archive:
            for name, data in zip(names, [audio, visual]):
                archive.writestr(name, data)
    elif source_kind == 'tar.gz':
        with tarfile.open(source, 'w:gz') as archive:
            for name, data in zip(names, [audio, visual]):
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    else:
        for name, data in zip(names, [audio, visual]):
            destination = tmp_path/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    a, v = tmp_path/'audio.csv', tmp_path/'visual.csv'
    a.write_bytes(audio)
    v.write_bytes(visual)
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not fit or read questionnaire metadata')
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr('src.common.read_metadata', forbidden)
    first, second, third = [tmp_path/name for name in ['cache-result.json', 'source-result.json', 'pair-result.json']]
    predict(root, first, cache=cache)
    predict(root, second, source=source)
    predict(root, third, audio_csv=a, visual_csv=v)
    reports = [json.loads(path.read_text()) for path in [first, second, third]]
    for report in reports[1:]:
        np.testing.assert_array_equal(report['probabilities'], reports[0]['probabilities'])
        assert report['predicted_class'] == reports[0]['predicted_class']
        assert 'questionnaire_phq8_score' not in report and 'gender' not in report
    with pytest.raises(FileExistsError, match='Preserving'):
        predict(root, first, cache=cache)


def test_cache_rejects_column_reordering_and_preprocessing_changes(tmp_path):
    root, contract, _, _ = bundle_fixture(tmp_path)
    sequence, info = align(*feature_pair(), contract['preprocessing'])
    path = tmp_path/'777.npz'
    np.savez_compressed(path, **sequence)
    sidecar = {**info, 'version': VERSION, 'participant_id': '777', 'settings': contract['preprocessing']}
    sidecar['audio_columns'] = AUDIO_COLUMNS[::-1]
    write_json(path.with_suffix('.json'), sidecar)
    with pytest.raises(ValueError, match='schema'):
        read_cache(path, contract)
    sidecar['audio_columns'] = AUDIO_COLUMNS
    sidecar['settings'] = {**contract['preprocessing'], 'bin_seconds': .1}
    write_json(path.with_suffix('.json'), sidecar)
    with pytest.raises(ValueError, match='preprocessing'):
        read_cache(path, contract)


@pytest.mark.parametrize('problem', ['nonfinite', 'mask_type', 'empty_mask', 'wrong_width', 'timestamps'])
def test_malformed_arrays_are_rejected(tmp_path, problem):
    _, contract, _, _ = bundle_fixture(tmp_path)
    sequence, _ = align(*feature_pair(), contract['preprocessing'])
    if problem == 'nonfinite':
        sequence['audio'][0, 0] = np.nan
    elif problem == 'mask_type':
        sequence['audio_mask'] = sequence['audio_mask'].astype(int)
    elif problem == 'empty_mask':
        sequence['visual_mask'][:] = False
    elif problem == 'wrong_width':
        sequence['visual'] = sequence['visual'][:, :-1]
    else:
        sequence['timestamps'][1] += .2
    with pytest.raises(ValueError):
        validate_sequence(sequence, contract['preprocessing'])


def test_unsafe_archive_members_are_rejected_without_extraction(tmp_path):
    root, _, _, _ = bundle_fixture(tmp_path)
    source = tmp_path/'777_P.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('../escape.csv', b'a\n1\n')
    with pytest.raises(ValueError, match='Unsafe'):
        predict(root, tmp_path/'result.json', source=source)
    assert not (tmp_path/'result.json').exists()
    assert not (tmp_path/'escape.csv').exists()
