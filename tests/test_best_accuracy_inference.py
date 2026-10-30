import json
from pathlib import Path

import numpy as np
import pytest

from src import best_accuracy_inference as s


@pytest.fixture
def bundle(tmp_path):
    rng = np.random.default_rng(17)
    params = {'scaler_mean': rng.normal(size=512), 'scaler_scale': rng.uniform(.5, 2, 512),
              'pca_mean': rng.normal(size=512), 'pca_components': rng.normal(size=(32, 512))*.05,
              'coef': rng.normal(size=(5, 32))*.1, 'intercept': rng.normal(size=5)}
    root = tmp_path/'bundle'; root.mkdir()
    np.savez_compressed(root/'weights.npz', **params)
    contract = {'format': s.FORMAT, 'weights_sha256': s.sha256(root/'weights.npz'),
                'classes': [0, 1, 2, 3, 4], 'source_model': {'sha256': 'fixture'}}
    (root/'model.json').write_text(json.dumps({'contract': contract, 'contract_hash': s.digest(contract)}))
    semantic = rng.normal(size=(3, 384)); semantic /= np.linalg.norm(semantic, axis=1, keepdims=True)
    av = rng.normal(size=(3, 128))
    return root, contract, params, semantic, av


def test_numeric_prediction_preserves_manual_pipeline(bundle):
    _, _, p, semantic, av = bundle
    z = (np.column_stack([semantic, av])-p['scaler_mean'])/p['scaler_scale']
    expected = s.softmax(((z-p['pca_mean']) @ p['pca_components'].T) @ p['coef'].T+p['intercept'], axis=1)
    np.testing.assert_allclose(s.predict_features(p, semantic, av), expected, rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match='normalized'):
        s.predict_features(p, semantic*2, av)
    with pytest.raises(ValueError, match='aligned'):
        s.predict_features(p, semantic, av[:2])


def test_inference_requires_contract_and_preserves_outputs(bundle, tmp_path):
    root, contract, params, semantic, av = bundle
    source = tmp_path/'features.npz'; output = tmp_path/'prediction.json'
    np.savez(source, semantic=semantic, av=av, contract_hash=np.array(s.digest(contract)))
    result = s.predict(root, output, features=source)
    assert result['participants'] == 3
    recorded = json.loads(output.read_text())
    np.testing.assert_allclose(recorded['decision_scores'], s.predict_features(params, semantic, av), rtol=0, atol=1e-14)
    assert 'actual_class' not in recorded and 'questionnaire' not in recorded
    with pytest.raises(FileExistsError):
        s.predict(root, output, features=source)
    np.savez(source, semantic=semantic, av=av, contract_hash=np.array('wrong'))
    with pytest.raises(ValueError, match='contract'):
        s.predict(root, tmp_path/'bad.json', features=source)


def test_model_tampering_is_rejected(bundle):
    root, _, _, _, _ = bundle
    s.load_bundle(root)
    path = root/'weights.npz'; path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='integrity'):
        s.load_bundle(root)


def test_incomplete_or_conflicting_source_inputs_rejected(bundle, tmp_path):
    root = bundle[0]
    with pytest.raises(ValueError, match='together'):
        s.predict(root, tmp_path/'a.json', transcript='missing.csv')
    with pytest.raises(ValueError, match='Choose'):
        s.predict(root, tmp_path/'a.json', features='unused.npz', transcript='unused.csv')
