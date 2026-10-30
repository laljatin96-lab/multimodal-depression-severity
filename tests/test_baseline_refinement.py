import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import baseline_refinement as s


def test_preprocessing_precedes_balancing_and_draws_keep_real_rows():
    rng = np.random.default_rng(4)
    y = np.repeat(np.arange(5), [30, 16, 12, 9, 5])
    x = rng.normal(size=(len(y), 15)); x[:, -1] = 1
    models = [s.SelectedLogistic(k=7, intervention=v).fit(x, y)
              for v in ['none', 'sqrt_weight', 'balanced_bootstrap']]
    for model in models:
        np.testing.assert_allclose(model.scaler_.mean_, x.mean(0))
        np.testing.assert_array_equal(model.selector_.selected_, models[0].selector_.selected_)
        assert 14 not in model.selector_.selected_
        for indices, weights in zip(model.draw_indices_, model.sample_weights_):
            np.testing.assert_array_equal(indices[:len(y)], np.arange(len(y)))
            assert indices.min() >= 0 and indices.max() < len(y)
            assert weights.sum() == pytest.approx(len(y))
            if model.intervention == 'balanced_bootstrap':
                np.testing.assert_array_equal(np.bincount(y[indices]), [30]*5)
        p = model.predict_proba(x[:8])
        np.testing.assert_allclose(p.sum(1), 1)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict_proba(x[:8]), p)
    other = s.SelectedLogistic(k=7, intervention='balanced_bootstrap').fit(x, y)
    for a, b in zip(other.draw_indices_, models[-1].draw_indices_):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError): s.SelectedLogistic().fit(x[y != 4], y[y != 4])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    old = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    settings = json.loads(Path('configs/baseline_refinement.json').read_text())
    settings.update(feature_inputs=['semantic'], selected_features=[4], C_candidates=[.1],
                    expected_jobs=2, inner_pipeline_fits_per_job=6, inner_base_fits_per_job=10, cpu_threads=1)
    y = np.repeat(np.arange(5), [40, 20, 16, 16, 8])
    rng = np.random.default_rng(3)
    x = {'semantic': rng.normal(size=(100, 10))}
    # Deliberately strong training-only fixture makes new candidates selectable.
    x['semantic'][:, 0] = y * 3 + rng.normal(scale=.1, size=100)
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(100)], 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old); records = []; root = tmp_path/'baseline'
    monkeypatch.setattr(s, 'ROOT', root)
    for fold in folds:
        model = DummyClassifier(strategy='most_frequent').fit(x['semantic'][fold['fit']], y[fold['fit']])
        path = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
        path.parent.mkdir(parents=True)
        with path.open('wb') as f: pickle.dump({'estimator': model}, f)
        records.append({**fold, 'candidates': [{'predictions': model.predict(x['semantic'][fold['fit']]).tolist()}],
                        'policies': {'accuracy': {'selection': {'index': 0}, 'candidate': {'feature': 'semantic'},
                                                 'probabilities': model.predict_proba(x['semantic'][fold['heldout']]).tolist()}}})
    return x, frame, old, settings, records, folds


def test_outer_features_and_labels_cannot_change_selection_or_fit(fixture, tmp_path):
    x, frame, old, settings, records, folds = fixture; fold = folds[0]
    a = s.job(x, frame, fold, records[0], settings, tmp_path/'first')
    assert a['selected']['accuracy']['selection']['index'] > 0
    x['semantic'][fold['heldout']] += 100
    frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+1)%5
    b = s.job(x, frame, fold, records[0], settings, tmp_path/'second')
    assert a['options'] == b['options'] and a['inner_probabilities'] == b['inner_probabilities']
    for key in ['selection', 'candidate', 'checkpoint_sha256']:
        assert a['selected']['accuracy'][key] == b['selected']['accuracy'][key]


def test_resume_no_fit_inference_or_pickle_and_integrity(fixture, tmp_path, monkeypatch):
    x, frame, old, settings, records, folds = fixture; out = tmp_path/'run'
    args = (x, frame, old, records, {}, settings, out)
    assert not s.run(*args, max_new_jobs=1)['complete']
    assert s.run(*args, resume=True)['complete']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a, **kw): raise AssertionError('No fitting, inference or deserialization')
    monkeypatch.setattr(s, 'job', forbidden)
    monkeypatch.setattr(s, 'five_probabilities', forbidden)
    monkeypatch.setattr(s.pickle, 'load', forbidden)
    assert s.run(*args, resume=True)['new_fits'] == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    for path in (out/'oof').glob('*/predictions.csv'):
        table = pd.read_csv(path); assert len(table) == 100 and table.participant_id.is_unique
        np.testing.assert_allclose(table[[f'p{k}' for k in range(5)]].sum(1), 1)
    changed = dict(settings, fit_seed=123)
    with pytest.raises(ValueError, match='Inputs/code changed'):
        s.run(x, frame, old, records, {}, changed, out, resume=True)
    path = next((out/'jobs').glob('*/*.pkl')); path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Checkpoint changed'): s.run(*args, resume=True)


def test_predeclared_budget():
    settings = json.loads(Path('configs/baseline_refinement.json').read_text())
    search = s.candidates(settings)
    assert len(search) == 25
    assert (len(search)-1)*3 == settings['inner_pipeline_fits_per_job'] == 72
    assert sum(s.base_fits(c, settings) for c in search[1:])*3 == settings['inner_base_fits_per_job'] == 120
