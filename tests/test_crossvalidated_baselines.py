import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src.crossvalidated_baselines import run_experiment, training_folds
from src.diagnose_models import sensitivity


def fixture():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(40, 4))
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(40)],
                          'split': ['train']*30+['validation']*5+['test']*5,
                          'severity_class': list(range(5))*8,
                          'gender': ['female', 'male', 'unknown', 'male', 'female']*8})
    settings = {'C_candidates': [.01, 1.], 'folds': 3, 'fold_seeds': [42, 137],
                'fit_seed': 42, 'bootstrap_repetitions': 20}
    return matrix, frame, settings


def test_reserved_data_cannot_change_selection_or_fitted_models(tmp_path):
    matrix, frame, settings = fixture()
    config = {'gender_categories': ['female', 'male']}
    first, second = tmp_path/'first', tmp_path/'second'
    run_experiment(matrix, frame, config, settings, first)
    # Both validation and test are outside C selection and preprocessing.
    matrix[30:] += 10000
    frame.loc[30:, 'severity_class'] = (frame.loc[30:, 'severity_class']+1) % 5
    frame.loc[30:, 'gender'] = 'unknown'
    run_experiment(matrix, frame, config, settings, second)
    assert json.loads((first/'frozen_selections.json').read_text()) == json.loads((second/'frozen_selections.json').read_text())
    for path in first.glob('*/model.pkl'):
        with path.open('rb') as handle:
            a = pickle.load(handle)
        with (second/path.relative_to(first)).open('rb') as handle:
            b = pickle.load(handle)
        assert a['fit_participant_ids'] == frame.participant_id[:30].tolist()
        for field in ['mean_', 'var_']:
            np.testing.assert_array_equal(getattr(a['estimator'][0], field), getattr(b['estimator'][0], field))
        np.testing.assert_array_equal(a['estimator'][-1].coef_, b['estimator'][-1].coef_)
        pd.testing.assert_frame_equal(pd.read_csv(path.parent/'cv/oof_predictions.csv'), pd.read_csv(second/path.parent.relative_to(first)/'cv/oof_predictions.csv'))
    assert not list(first.rglob('test'))
    assert not list(second.rglob('test'))
    with pytest.raises(FileExistsError, match='Preserving existing'):
        run_experiment(matrix, frame, config, settings, first)


def test_training_folds_cover_each_person_once_and_require_class_support():
    _, frame, settings = fixture()
    training = frame[frame.split.eq('train')].reset_index(drop=True)
    folds = training_folds(training, settings['folds'], settings['fold_seeds'])
    for seed in settings['fold_seeds']:
        seen = []
        for fold_seed, _, fit_indices, heldout in folds:
            if fold_seed != seed:
                continue
            assert not set(fit_indices) & set(heldout)
            assert set(fit_indices) | set(heldout) == set(range(30))
            assert set(training.iloc[fit_indices].severity_class) == set(range(5))
            seen.extend(heldout)
        assert sorted(seen) == list(range(30))
    with pytest.raises(ValueError, match='five training classes'):
        training_folds(training, 7, [42])
    with pytest.raises(ValueError, match='unique'):
        training_folds(training, 3, [42, 42])


def test_leave_one_out_sensitivity_retains_five_class_denominator():
    frame = pd.DataFrame({'participant_id': list('abcde'), 'severity_class': range(5)})
    result = sensitivity(frame, np.eye(5))
    np.testing.assert_array_equal(result.full_macro_f1, np.ones(5))
    np.testing.assert_allclose(result.macro_f1_without_participant, .8)
