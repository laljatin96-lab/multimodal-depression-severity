import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import decision_offset_accuracy as s


def test_offsets_fix_known_bias_and_penalty_limits_change():
    settings = json.loads(Path('configs/decision_offset_accuracy.json').read_text())
    y = np.repeat(np.arange(5), 10)
    logits = np.eye(5)[y]*.5; logits[:, 0] += .6
    p = s.softmax(logits, axis=1)
    before = p.copy(); bias = s.fit_offsets(p, y, settings)
    assert bias[0] == 0 and np.any(bias[1:] > 0)
    assert (s.adjust(p, bias).argmax(1) == y).sum() > (p.argmax(1) == y).sum()
    np.testing.assert_array_equal(p, before)
    np.testing.assert_allclose(s.adjust(p, np.zeros(5)), p, rtol=0, atol=1e-14)
    assert not s.fit_offsets(p, y, dict(settings, penalty=1e9)).any()
    with pytest.raises(ValueError, match='offsets'):
        s.adjust(p, np.ones(5))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    y = np.repeat(np.arange(5), 10); x = {'semantic': np.arange(len(y))[:, None].astype(float)}
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(len(y))], 'severity_class': y, 'gender': 'unknown'})
    old_settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    settings = json.loads(Path('configs/decision_offset_accuracy.json').read_text()); settings['expected_jobs'] = 2
    folds = s.nested_folds(frame, old_settings); old = []; root = tmp_path/'baseline'; monkeypatch.setattr(s, 'ROOT', root)
    for fold in folds:
        model = DummyClassifier(strategy='prior').fit(x['semantic'][fold['fit']], y[fold['fit']])
        directory = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; directory.mkdir(parents=True)
        (directory/'accuracy.pkl').write_bytes(pickle.dumps({'estimator': model}))
        logits = np.eye(5)[y[fold['fit']]]*.5; logits[:, 0] += .6
        old.append({**fold, 'cached_inner_probabilities': s.softmax(logits, axis=1).tolist(),
                    'policies': {'accuracy': {'candidate': {'feature': 'semantic'},
                        'probabilities': model.predict_proba(x['semantic'][fold['heldout']]).tolist()}}})
    return (x, frame, old_settings, old, {'bindings': {}}), settings, folds


def test_outer_labels_probabilities_do_not_affect_offsets(experiment, tmp_path):
    inputs, settings, folds = experiment; x, frame, _, old, _ = inputs; fold = folds[0]
    first = s.job(frame, fold, old[0], settings, tmp_path/'a')
    frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+1)%5
    old[0]['policies']['accuracy']['probabilities'] = np.tile([.1, .6, .1, .1, .1], (len(fold['heldout']), 1)).tolist()
    second = s.job(frame, fold, old[0], settings, tmp_path/'b')
    assert first['raw_offsets'] == second['raw_offsets']
    for name in settings['policies']:
        assert first['selected'][name]['checkpoint_sha256'] == second['selected'][name]['checkpoint_sha256']


def test_final_offsets_ignore_all_outer_outputs(experiment):
    inputs, settings, _ = experiment; _, frame, _, old, _ = inputs
    before = s.final_offsets(frame, old, settings)
    for r in old:
        r['policies'] = {'invalid_outer_output': 999}
    after = s.final_offsets(frame, old, settings)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_resume_has_zero_training_and_preserves_models(experiment, tmp_path, monkeypatch):
    inputs, settings, folds = experiment; out = tmp_path/'study'
    assert s.run(inputs, settings, out)['new_offset_fits'] == 2
    before = {p: [p.read_bytes(), p.stat().st_mtime_ns] for p in (out/'jobs').glob('*/*')}
    def forbidden(*a, **kw):
        raise AssertionError('No fits on completed resume')
    monkeypatch.setattr(s, 'fit_offsets', forbidden); monkeypatch.setattr(s, 'job', forbidden)
    assert s.run(inputs, settings, out, resume=True)['new_offset_fits'] == 0
    assert all([p.read_bytes(), p.stat().st_mtime_ns] == v for p, v in before.items())
    path = next((out/'jobs').glob('*/*.pkl')); path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        s.run(inputs, settings, out, resume=True)
