import json

import numpy as np
import pandas as pd
import pytest

from src import feature_diagnostics as feature
from src.prepare_features import AUDIO_COLUMNS, bin_frames


def sequence_from_raw(times, values, bins):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    audio, mask = bin_frames(times, values, np.ones(len(times), bool), np.arange(bins+1)*.5)
    sequence = {'audio': audio, 'audio_mask': mask, 'visual_mask': np.ones(bins, bool), 'timestamps': np.arange(bins)*.5}
    frame = pd.DataFrame(values, columns=AUDIO_COLUMNS)
    frame['frameTime'] = times
    return sequence, frame.to_csv(sep=';', index=False).encode()


def test_cache_profile_respects_masks_and_does_not_bridge_missing_bins():
    values = np.tile(np.array([0., 1., 999., 100., 102.])[:, None], (1, 39))
    sequence = {'audio': values, 'audio_mask': np.array([True, True, False, True, True]),
                'visual_mask': np.ones(5, bool)}
    quality, rows = feature.cache_statistics(sequence)
    assert quality['audio_observed_bins'] == 4 and quality['adjacent_observed_pairs'] == 2
    assert rows[0]['mean'] == (0+1+100+102)/4
    assert abs(rows[0]['adjacent_bin_correlation']-1) < 1e-12
    assert quality['identical_adjacent_audio_vector_fraction'] == 0


@pytest.mark.parametrize('invalid', ['mask', 'nonfinite'])
def test_invalid_cache_rejected(invalid):
    sequence = {'audio': np.ones((4, 39)), 'audio_mask': np.ones(4, bool), 'visual_mask': np.ones(4, bool)}
    if invalid == 'mask':
        sequence['audio_mask'] = np.ones(4, dtype=int)
    else:
        sequence['audio'][0, 0] = np.nan
    with pytest.raises(ValueError):
        feature.cache_statistics(sequence)


def test_training_geometry_ignores_reserved_rows_and_handles_constant_columns():
    rng = np.random.default_rng(12)
    base = rng.normal(size=(8, 2))
    matrix = np.tile(base, (1, 64))
    matrix[:6, 0] = 7
    coverage = pd.DataFrame({'participant_id': [str(i) for i in range(8)],
                             'split': ['train']*6+['validation', 'test']})
    before = feature.training_geometry(matrix, coverage)
    matrix[6:] *= 100000
    assert feature.training_geometry(matrix, coverage) == before
    audio = before[0][0]
    assert audio['constant_training_columns'] == 1 and audio['numerical_rank'] == 2
    assert audio['components_for_95_percent_variance'] <= 2
    all_constant, pairs = feature.training_geometry(np.ones((8, 128)), coverage)
    assert all(row['numerical_rank'] == 0 and row['components_for_95_percent_variance'] == 0 for row in all_constant)
    assert not pairs


def test_frame_weighted_variance_with_unequal_bin_counts_and_constant_feature():
    values = np.tile(np.array([0., 2., 4.])[:, None], (1, 39))
    values[:, 0] = 7
    sequence, blob = sequence_from_raw([0, .1, .6], values, 2)
    rows, check = feature.variance_retention(blob, sequence, .5)
    assert check['cached_audio_reproduced_exactly'] and check['raw_frames_in_cached_interval'] == 3
    assert rows[0]['raw_constant'] and rows[0]['retained_variance_fraction'] is None
    assert abs(rows[1]['retained_variance_fraction']-.75) < 1e-12


def test_high_frequency_variation_removed_and_between_bin_variation_retained():
    times = np.arange(40)*.05
    values = np.zeros((40, 39))
    values[:, :13] = (np.arange(40)//10)[:, None]
    values[:, 13:] = np.tile([-1., 1.], 20)[:, None]
    sequence, blob = sequence_from_raw(times, values, 4)
    rows, _ = feature.variance_retention(blob, sequence, .5)
    assert all(abs(row['retained_variance_fraction']-1) < 1e-12 for row in rows[:13])
    assert all(row['retained_variance_fraction'] == 0 for row in rows[13:])
    sequence['audio'][0, 0] += .1
    with pytest.raises(ValueError, match='exactly reproduce'):
        feature.variance_retention(blob, sequence, .5)


def test_raw_variance_uses_only_exact_cached_crop():
    times = np.arange(30)*.1
    values = np.tile(np.arange(30)[:, None], (1, 39)).astype(float)
    sequence, _ = sequence_from_raw(times, values, 4)
    values[times >= 2] = 10000
    frame = pd.DataFrame(values, columns=AUDIO_COLUMNS)
    frame['frameTime'] = times
    rows, check = feature.variance_retention(frame.to_csv(sep=';', index=False).encode(), sequence, .5)
    assert check['raw_frames_in_cached_interval'] == 20
    assert abs(rows[0]['raw_variance']-np.var(np.arange(20))) < 1e-12


def test_existing_outputs_preserved_and_modified_extracted_features_rejected(tmp_path):
    out = tmp_path/'existing'
    out.mkdir()
    (out/'note').write_text('preserve')
    with pytest.raises(FileExistsError):
        feature.run_diagnostic(tmp_path/'missing_plan', out)
    assert (out/'note').read_text() == 'preserve'
    source = tmp_path/'123_P'
    source.mkdir()
    name = '123_OpenSMILE2.3.0_mfcc.csv'
    (source/name).write_text('changed')
    audit = tmp_path/'audit'
    (audit/'sources').mkdir(parents=True)
    pd.DataFrame([{'source': str(source), 'kind': 'directory', 'participant_id': '123',
                   'source_key': 'synthetic'}]).to_csv(audit/'sources.csv', index=False)
    (audit/'sources/synthetic.json').write_text(json.dumps({'feature_sha256': {name: 'wrong'}}))
    with pytest.raises(ValueError, match='checksum'):
        feature.extracted_comparisons(audit, {}, {'bin_seconds': .5, 'audio_time_offset': 0})
