import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import conditional_hierarchy_accuracy as s


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def toy():
    rng = np.random.default_rng(9); y = np.repeat(np.arange(5), [24, 12, 12, 12, 8])
    x = {'semantic_av': rng.normal(size=(len(y), 12))}
    settings = json.loads(s.CONFIG.read_text())
    settings.update(components=[4], C=[.001], expected_jobs=2, cpu_threads=1)
    return x, y, settings


@pytest.mark.parametrize('boundary', [0, 1])
def test_conditional_ownership_and_probability_composition(toy, boundary):
    x, y, settings = toy
    c = {'boundary': boundary, 'components': 4, 'C': .001, 'family': 'conditional_hierarchy'}
    model = s.ConditionalHierarchy(c, settings).fit(x['semantic_av'], y)
    np.testing.assert_allclose(model.gate.scaler.mean_, x['semantic_av'].mean(0))
    for g, head in model.heads.items():
        if head is not None:
            positions = np.flatnonzero((y > boundary) == g)
            np.testing.assert_array_equal(head.positions, positions)
            np.testing.assert_allclose(head.scaler.mean_, x['semantic_av'][positions].mean(0))
            np.testing.assert_array_equal(head.classes_, np.unique(y[positions]))
    before = pickle.dumps(model); p = model.predict_proba(x['semantic_av'])
    assert p.shape == (len(y), 5) and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-14)
    gate = model.gate.predict_proba(x['semantic_av'])
    np.testing.assert_allclose(p[:, :boundary+1].sum(1), gate[:, 0], atol=1e-14)
    assert pickle.dumps(model) == before
    np.testing.assert_allclose(pickle.loads(before).predict_proba(x['semantic_av']), p, atol=1e-14)


@pytest.fixture
def experiment(toy, tmp_path, monkeypatch):
    x, y, settings = toy
    old_settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(len(y))], 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old_settings); old = []; root = tmp_path/'baseline'
    monkeypatch.setattr(s, 'ROOT', root)
    for f in folds:
        model = DummyClassifier(strategy='prior').fit(x['semantic_av'][f['fit']], y[f['fit']])
        directory = root/'jobs'/f"seed{f['seed']}_fold{f['fold']}"; directory.mkdir(parents=True)
        (directory/'accuracy.pkl').write_bytes(pickle.dumps({'estimator': model}))
        old.append({**f, 'candidates': [{'predictions': model.predict(x['semantic_av'][f['fit']]).tolist()}],
            'policies': {'accuracy': {'selection': {'index': 0}, 'candidate': {'feature': 'semantic_av'},
                'probabilities': model.predict_proba(x['semantic_av'][f['heldout']]).tolist()}}})
    return (x, frame, old_settings, old, {'bindings': {}}), settings, folds


def test_heldout_targets_and_features_cannot_affect_fit(experiment, tmp_path):
    inputs, settings, folds = experiment; x, frame, _, old, _ = inputs; f = folds[0]
    a = s.job(x, frame, f, old[0], settings, tmp_path/'a')
    # Force new-only to avoid replaying a deliberately changed baseline's input.
    settings = dict(settings, policies={'hierarchy_only': {}})
    a = s.job(x, frame, f, old[0], settings, tmp_path/'new_a')
    frame.loc[f['heldout'], 'severity_class'] = (frame.loc[f['heldout'], 'severity_class']+1)%5
    x['semantic_av'][f['heldout']] += 1000
    b = s.job(x, frame, f, old[0], settings, tmp_path/'b')
    assert a['options'] == b['options'] and a['inner_probabilities'] == b['inner_probabilities']
    assert a['selected']['hierarchy_only']['checkpoint_sha256'] == b['selected']['hierarchy_only']['checkpoint_sha256']


def test_resume_no_refit_no_rewrite_and_tamper_rejection(experiment, tmp_path, monkeypatch):
    inputs, settings, folds = experiment; out = tmp_path/'study'
    assert not s.run(inputs, settings, out, max_new_jobs=1)['complete']
    assert s.run(inputs, settings, out, resume=True)['complete']
    before = {p: [p.read_bytes(), p.stat().st_mtime_ns] for p in out.rglob('*') if p.is_file() and p.name != '.batch.lock'}
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed resume must not refit or predict')
    monkeypatch.setattr(s, 'job', forbidden); monkeypatch.setattr(s, 'predict_state', forbidden)
    assert s.run(inputs, settings, out, resume=True)['new_fits'] == 0
    assert all([p.read_bytes(), p.stat().st_mtime_ns] == v for p, v in before.items())
    with pytest.raises(ValueError, match='Frozen inputs'):
        s.run(inputs, dict(settings, fit_seed=99), out, resume=True)
    path = next((out/'jobs').glob('*/*.pkl')); path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        s.run(inputs, settings, out, resume=True)


def test_final_choice_independent_of_outer_results(experiment, tmp_path):
    inputs, settings, folds = experiment; x, frame, _, old, _ = inputs
    records = [s.job(x, frame, f, r, settings, tmp_path/f"fold{f['fold']}") for f, r in zip(folds, old)]
    original = s.training_choices(frame, records, old, settings)
    for r in records:
        r['selected'] = {'corrupt_outer': True}; r['options'] = []
    for r in old:
        r['policies']['accuracy']['probabilities'] = []
    assert s.training_choices(frame, records, old, settings) == original


def test_bounded_grid():
    settings = json.loads(s.CONFIG.read_text())
    assert len(s.candidates(settings)) == 9
    assert settings['expected_jobs'] == 15 and settings['max_nested_pipeline_fits'] == 390
    assert settings['target_accuracy'] == .56
