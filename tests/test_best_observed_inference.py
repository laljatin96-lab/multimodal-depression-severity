import shutil

import numpy as np
import pytest

from src import best_observed_inference as s


def test_bundle_identifies_unchanged_age_free_checkpoint():
    contract, parameters = s.load()
    assert contract['source_model']['sha256'] == '599f1ec55d82a4a61f199aace525a23dd7f13f0e6afa77625086a4c4bc20b660'
    assert contract['observed_nested_procedure_accuracy'] == 241/489
    assert contract['export_refits'] == 0 and not contract['age_required']
    assert not contract['independent_confirmation']
    s.validate(parameters)


def test_invalid_or_questionnaire_extended_schema_is_rejected():
    _, parameters = s.load(); rng = np.random.default_rng(42)
    semantic = rng.normal(size=(2,384)); semantic /= np.linalg.norm(semantic,axis=1)[:,None]
    av = rng.normal(size=(2,128)); language = rng.normal(size=(2,16))
    p = s.predict_features(parameters, semantic, av, language)
    assert p.shape == (2,5) and np.allclose(p.sum(1),1)
    with pytest.raises(ValueError, match='aligned'):
        s.predict_features(parameters, semantic, av, np.column_stack([language,[0,1]]))
    with pytest.raises(ValueError, match='normalized'):
        s.predict_features(parameters, semantic*2, av, language)
    broken = {k:v.copy() for k,v in parameters.items()}; broken['base_scale'][0] = 0
    with pytest.raises(ValueError, match='scaling'):
        s.predict_features(broken, semantic, av, language)


def test_tampered_numeric_bundle_is_rejected(tmp_path):
    dest = tmp_path/'bundle'; shutil.copytree(s.BUNDLE,dest)
    path = dest/'weights.npz'; path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='integrity'):
        s.load(dest)
