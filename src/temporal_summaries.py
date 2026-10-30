"""Label-free distribution and within-modality dynamics from audited cached bins."""
import numpy as np

from .exploratory_baselines import pool_modalities
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS

STATISTICS = ['mean', 'std', 'median', 'iqr', 'p90_p10', 'mean_abs_step', 'lag1_correlation']
DIMENSIONS = {'cached_visual': 50, 'temporal_visual': 175, 'cached_av': 128, 'temporal_av': 448}


def feature_names():
    names = {}
    for modality, columns in [('audio', AUDIO_COLUMNS), ('visual', VISUAL_COLUMNS)]:
        names[modality] = [f'{modality}:{stat}:{column}' for stat in STATISTICS for column in columns]
    cached = [f'{modality}:{stat}:{column}' for modality, columns in [('audio', AUDIO_COLUMNS), ('visual', VISUAL_COLUMNS)]
              for stat in ['mean', 'std'] for column in columns]
    return {'cached_visual': cached[78:], 'cached_av': cached,
            'temporal_visual': names['visual'], 'temporal_av': names['audio']+names['visual']}


def summarize_modality(values, mask):
    values, mask = np.asarray(values, dtype=np.float64), np.asarray(mask)
    if values.ndim != 2 or mask.dtype != bool or mask.shape != (len(values),) or mask.sum() < 2:
        raise ValueError('Expected a nonempty matrix and at least two observed bins')
    observed = values[mask]
    if not np.isfinite(observed).all():
        raise ValueError('Observed bins must be finite')
    adjacent = mask[:-1] & mask[1:]
    left, right = values[:-1][adjacent], values[1:][adjacent]
    step, correlation = np.zeros(values.shape[1]), np.zeros(values.shape[1])
    if len(left):
        step = np.abs(right-left).mean(0)
    defined = np.zeros(values.shape[1], bool)
    if len(left) >= 2:
        a, b = left-left.mean(0), right-right.mean(0)
        denominator = np.sqrt((a*a).sum(0)*(b*b).sum(0))
        defined = denominator > 0
        correlation = np.divide((a*b).sum(0), denominator, out=correlation, where=defined)
        correlation = np.clip(correlation, -1, 1)
    p10, p25, median, p75, p90 = np.quantile(observed, [.1, .25, .5, .75, .9], axis=0, method='linear')
    result = np.concatenate([observed.mean(0), observed.std(0), median, p75-p25, p90-p10, step, correlation])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite temporal summary')
    return result, {'observed_bins': int(mask.sum()), 'adjacent_pairs': int(adjacent.sum()),
                    'undefined_lag_columns': int((~defined).sum())}


def summarize_participant(sequence):
    times = np.asarray(sequence['timestamps'], dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or not np.allclose(np.diff(times), .5, rtol=0, atol=1e-9):
        raise ValueError('Expected the audited half-second timestamp grid')
    parts, quality = {}, {}
    for modality, columns in [('audio', AUDIO_COLUMNS), ('visual', VISUAL_COLUMNS)]:
        if np.asarray(sequence[modality]).shape != (len(times), len(columns)):
            raise ValueError('Cached feature schema/length mismatch')
        parts[modality], quality[modality] = summarize_modality(sequence[modality], sequence[modality+'_mask'])
    cached = pool_modalities(sequence)
    result = {'cached_visual': cached[78:], 'cached_av': cached,
              'temporal_visual': parts['visual'], 'temporal_av': np.r_[parts['audio'], parts['visual']]}
    np.testing.assert_array_equal(parts['audio'][:78], cached[:78])
    np.testing.assert_array_equal(parts['visual'][:50], cached[78:])
    return result, quality


def prepare_all(sequences, participant_ids):
    if len(set(participant_ids)) != len(participant_ids) or set(sequences) != set(participant_ids):
        raise ValueError('Every unique audited participant must be retained exactly once')
    rows, quality = {key: [] for key in DIMENSIONS}, []
    for pid in participant_ids:
        summaries, support = summarize_participant(sequences[pid])
        for key in rows:
            rows[key].append(summaries[key])
        quality.extend({'participant_id': pid, 'modality': m, **counts} for m, counts in support.items())
    return {key: np.stack(values) for key, values in rows.items()}, quality
