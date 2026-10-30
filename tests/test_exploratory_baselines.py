import pickle

import numpy as np
import pandas as pd
import pytest

from src.exploratory_baselines import fit_candidates, pool_modalities


def test_summary_is_independent_of_cross_modal_order_and_masked_values():
    sequence = {'audio': np.array([[1., 2.], [3., 6.], [999., 999.]]), 'visual': np.array([[4.], [2.]]), 'audio_mask': np.array([True, True, False]), 'visual_mask': np.array([True, True])}
    expected = pool_modalities(sequence)
    sequence['visual'] = sequence['visual'][::-1]
    sequence['audio'][2] = -100000
    np.testing.assert_allclose(pool_modalities(sequence), expected)
    sequence['visual_mask'][:] = False
    with pytest.raises(ValueError, match='empty observation'):
        pool_modalities(sequence)


def test_real_label_workflow_fits_train_only_and_never_evaluates_test(tmp_path):
    rng = np.random.default_rng(1)
    features = rng.normal(size=(25, 4))
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(25)], 'split': ['train']*15+['validation']*5+['test']*5, 'gender': ['female', 'male', 'unknown', 'female', 'male']*5, 'severity_class': list(range(5))*5})
    config = {'seed': 42, 'gender_categories': ['female', 'male']}
    output = tmp_path / 'first'
    result = fit_candidates(features, frame, config, output, grid=(1.0,))
    with (output / 'LR-AV/C_1/model.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    np.testing.assert_allclose(saved['estimator'].named_steps['standardscaler'].mean_, features[:15].mean(0))
    assert saved['fit_participant_ids'] == frame.participant_id[:15].tolist()
    assert not list(output.rglob('test'))
    assert len(pd.read_csv(output / 'LR-AV/C_1/validation/predictions.csv')) == 5
    features[20:] += 10000
    frame.loc[frame.split == 'test', 'severity_class'] = 4
    other = fit_candidates(features, frame, config, tmp_path / 'second', grid=(1.0,))
    pd.testing.assert_frame_equal(result.drop(columns='run'), other.drop(columns='run'))
    with pytest.raises(FileExistsError, match='Preserving existing'):
        fit_candidates(features, frame, config, output)
