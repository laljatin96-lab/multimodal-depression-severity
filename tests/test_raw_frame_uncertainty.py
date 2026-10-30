import numpy as np
import pandas as pd
import pytest

from src import raw_frame_uncertainty as uncertainty
from src.common import VERSION
from src.evaluate import metrics


def test_weighted_bootstrap_metrics_equal_explicit_participant_resampling():
    y = np.array([0, 0, 1, 2, 3, 4])
    predictions = np.array([[0, 1, 1, 0, 3, 2], [0, 0, 2, 2, 2, 4]])
    counts = uncertainty.participant_counts(6, 100, 42)
    actual = uncertainty.weighted_metrics(y, predictions, counts)
    expected = []
    for row in counts:
        indices = np.repeat(np.arange(6), row)
        expected.append(np.mean([[metrics(y[indices], np.eye(5)[p[indices]])[m] for m in uncertainty.METRICS]
                                 for p in predictions], axis=0))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_missing_classes_keep_five_class_macro_f1_denominator():
    scores = uncertainty.weighted_metrics(np.zeros(4, int), np.zeros((3, 4), int), np.ones((2, 4), int))
    np.testing.assert_allclose(scores, [[1., .2, 1.], [1., .2, 1.]])


def test_resampling_matches_one_shared_participant_draw_and_is_reproducible():
    counts = uncertainty.participant_counts(7, 50, 42)
    np.testing.assert_array_equal(counts, uncertainty.participant_counts(7, 50, 42))
    explicit = np.random.default_rng(42).integers(0, 7, size=(50, 7))
    np.testing.assert_array_equal(counts, np.stack([np.bincount(row, minlength=7) for row in explicit]))
    assert (counts.sum(1) == 7).all() and (counts >= 0).all()


def test_duplicate_partitions_do_not_create_more_independent_samples():
    y = np.array([0, 1, 2, 3, 4, 0])
    training = pd.DataFrame({'participant_id': list('abcdef'), 'severity_class': y})
    settings = {'bootstrap_repetitions': 100, 'bootstrap_seed': 42}
    old, new = np.array([[0, 0, 2, 2, 4, 0]]), np.array([[0, 1, 2, 2, 3, 0]])
    arrays = {(r, m, b): p for r, p in [('cached', old), ('raw', new)] for m, b in uncertainty.VARIANTS}
    single = uncertainty.analyze(training, [42], arrays, settings)
    repeated = uncertainty.analyze(training, [42, 137, 2026], {k: np.repeat(v, 3, axis=0) for k, v in arrays.items()}, settings)
    np.testing.assert_allclose(single[-1], repeated[-1], atol=1e-15)
    np.testing.assert_allclose(single[0].raw_minus_cached, repeated[0].raw_minus_cached, atol=1e-15)
    identical = uncertainty.analyze(training, [42], {k: old for k in arrays}, settings)
    assert not identical[-1].any()
    assert (identical[0][['raw_minus_cached', 'ci95_low', 'ci95_high']] == 0).all().all()


def test_paired_correctness_transitions_and_class_supports():
    training = pd.DataFrame({'participant_id': list('abcd'), 'severity_class': [0, 1, 2, 4]})
    transitions, classes, people = uncertainty.error_changes(training, [42], np.array([[0, 0, 2, 0]]),
                                                            np.array([[0, 1, 0, 0]]), 'AV', False)
    assert all(transitions[0][k] == 1 for k in ['both_correct', 'raw_only_correct', 'cached_only_correct', 'both_wrong'])
    assert sum(r['participants'] for r in classes) == 4
    assert classes[3]['participants'] == 0 and classes[3]['raw_recall'] is None
    assert classes[4]['raw_correct'] == classes[4]['cached_correct'] == 0
    assert len(people) == 4 and people[1]['raw_correct_partitions'] == 1


@pytest.mark.parametrize('y,p,c', [([0, 5], [[0, 0]], [[1, 1]]),
                                  ([0, 1], [[0]], [[1, 1]]),
                                  ([0, 1], [[0, 1]], [[1, 0]]),
                                  ([0, 1], [[0, 1]], [[-.5, 2.5]])])
def test_invalid_classes_shapes_and_resampling_counts_are_rejected(y, p, c):
    with pytest.raises(ValueError):
        uncertainty.weighted_metrics(y, p, c)


def test_saved_predictions_must_match_training_only_ids_and_job_probabilities(tmp_path):
    training = pd.DataFrame({'participant_id': ['1', '2'], 'severity_class': [0, 1]})
    probs = np.eye(5)[[0, 1]]
    frame = training.copy()
    frame['predicted_label'] = [0, 1]
    for i in range(5):
        frame[f'p{i}'] = probs[:, i]
    frame['split'], frame['seed'], frame['model_version'] = 'train_nested_oof', 42, VERSION+':RawFrame-LR-AV'
    path = tmp_path/'predictions.csv'
    frame.to_csv(path, index=False)
    np.testing.assert_array_equal(uncertainty.checked_predictions(path, training, probs, 42, 'RawFrame-LR-AV'), [0, 1])
    frame.loc[0, 'split'] = 'test'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='training IDs'):
        uncertainty.checked_predictions(path, training, probs, 42, 'RawFrame-LR-AV')
    frame.loc[0, 'split'] = 'train_nested_oof'
    frame.loc[0, ['p0', 'p1']] = [.9, .1]
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        uncertainty.checked_predictions(path, training, probs, 42, 'RawFrame-LR-AV')


def test_existing_output_preserved_and_invalid_bootstrap_budget_rejected(tmp_path):
    out = tmp_path/'existing'
    out.mkdir()
    (out/'note').write_text('preserve')
    with pytest.raises(FileExistsError):
        uncertainty.run({}, out)
    assert (out/'note').read_text() == 'preserve'
    for n, repetitions in [(0, 10), (2, 0), (2, 50001)]:
        with pytest.raises(ValueError):
            uncertainty.participant_counts(n, repetitions, 42)
