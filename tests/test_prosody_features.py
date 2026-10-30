import io

import numpy as np
import pandas as pd
import pytest

from src import prosody_features as s


def blob(values, times):
    f = pd.DataFrame(values, columns=s.COLUMNS)
    f.insert(0, 'frameTime', times); f.insert(0, 'name', 'unknown')
    return f.to_csv(index=False, sep=';').encode()


def test_order_invariant_summary_does_not_require_aligned_times():
    x = np.random.default_rng(2).normal(size=(20, 23)); x[:, 10] = np.arange(20)
    a = s.summarize(blob(x, np.arange(20)/100))
    # Same measured frames, arbitrary ordering and duplicate times: summary stays the same.
    b = s.summarize(blob(x[::-1], np.zeros(20)))
    np.testing.assert_allclose(a['features'], b['features'], rtol=0, atol=1e-12)
    assert b['nonincreasing_time_steps'] == 19 and not b['timestamps_used_as_features']
    assert a['features'][-1] == 19/20


def test_unvoiced_sentinels_are_excluded_only_from_voiced_descriptors():
    x = np.ones((6, 23)); x[:3, 10] = 0; x[:3, 11:] = -201
    a = s.summarize(blob(x, np.arange(6)))
    assert a['voiced_frames'] == 3 and a['features'][-1] == .5
    assert a['features'][11*5] == 1  # jitter mean comes from voiced frames only.
    assert a['features'][0] == 1  # loudness uses all rows.


def test_no_voiced_frames_are_explicit_missing_not_dropped_people():
    x = np.ones((4, 23)); x[:, 10] = 0
    a = s.summarize(blob(x, np.arange(4)))
    assert a['features'][50:-1] == [None]*65 and a['frames'] == 4
    assert np.isfinite(a['features'][:50]).all() and a['features'][-1] == 0
    x[0, 1] = np.inf
    with pytest.raises(ValueError, match='Nonfinite'):
        s.summarize(blob(x, np.arange(4)))


def test_duplicate_sources_are_order_independent_and_conflicts_rejected():
    a = {'participant_id': 'x', 'features': [1], 'source_key': 'a', 'member_sha256': 'same',
         'frames': 3, 'voiced_frames': 2, 'nonincreasing_time_steps': 1}
    b = dict(a, source_key='b')
    assert s.combine([a, b], ['x']) == s.combine([b, a], ['x'])
    with pytest.raises(ValueError, match='conflicting'):
        s.combine([a, dict(b, features=[2])], ['x'])
