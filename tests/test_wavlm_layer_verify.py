"""Verify the actual selected input block of the preserved language checkpoint."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from src.age_language_accuracy import BlockClassifier
from src.wavlm_layer_verify import inspect_model


@pytest.mark.parametrize('power', [0., .5])
def test_unused_historical_age_projection_is_not_required_for_age_free_replay(power):
    rng = np.random.default_rng(97); n = 40
    x = {'base': rng.normal(size=(n, 8)), 'language': rng.normal(size=(n, 16)),
         'age': rng.normal(size=(n, 3)), 'age_language': rng.normal(size=(n, 19))}
    candidate = {'family': 'block_logistic', 'feature_set': 'language', 'block_scale': .5,
                 'C': .001, 'weight_power': power}
    y = np.repeat(np.arange(5), [16, 8, 6, 6, 4])
    with threadpool_limits(limits=1):
        model = BlockClassifier(candidate, {'components': 4, 'fit_seed': 42}).fit(x, y)
    selected = {'semantic_av': x['base'], 'language': x['language']}
    inspect_model(model, selected, y)
    before = model.predict_proba({'base': x['base'], 'language': x['language']})
    np.testing.assert_array_equal(before, model.predict_proba({**x, 'age': x['age']+100}))
    with pytest.raises(AssertionError):
        inspect_model(model, {**selected, 'language': x['language']+1}, y)
