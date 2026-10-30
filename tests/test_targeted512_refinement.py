import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from src import targeted512_refinement as s


def test_weighting_and_reduction_use_only_real_fitting_rows():
    y = np.repeat(np.arange(5), [30, 16, 12, 9, 5]); rng = np.random.default_rng(9)
    x = rng.normal(size=(len(y), 512))
    for k in [128, 256, 512, 'pca32']:
        m = s.TargetedLogistic(k=k, intervention='quarter_weight').fit(x, y)
        np.testing.assert_allclose(m.scaler_.mean_, x.mean(0))
        expected = (len(y)/(5*np.bincount(y)[y]))**.25; expected *= len(y)/expected.sum()
        np.testing.assert_allclose(m.sample_weights_, expected)
        assert m.real_count_ == len(y) and m.sample_weights_.sum() == pytest.approx(len(y))
        assert m.selector_.transform(m.scaler_.transform(x)).shape[1] == (32 if k == 'pca32' else k)
        np.testing.assert_allclose(m.predict_proba(x[:2]).sum(axis=1), 1)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(m)).predict_proba(x[:2]), m.predict_proba(x[:2]))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    old = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    settings = json.loads(Path('configs/targeted512_refinement.json').read_text())
    settings.update(selected_features=[16], C_candidates=[.03], expected_jobs=2,
                    inner_pipeline_fits_per_job=4, inner_base_fits_per_job=4, cpu_threads=1)
    y = np.repeat(np.arange(5), [40, 20, 16, 16, 8]); rng = np.random.default_rng(3)
    x = {'semantic_av': rng.normal(size=(100, 40))}; x['semantic_av'][:, 0] = y*5
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(100)], 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old); records = []; root = tmp_path/'baseline'; monkeypatch.setattr(s, 'ROOT', root)
    for f in folds:
        model = DummyClassifier(strategy='most_frequent').fit(x['semantic_av'][f['fit']], y[f['fit']])
        path = root/'jobs'/f"seed{f['seed']}_fold{f['fold']}"/'accuracy.pkl'; path.parent.mkdir(parents=True)
        with path.open('wb') as h: pickle.dump({'estimator': model}, h)
        records.append({**f, 'candidates': [{'predictions': model.predict(x['semantic_av'][f['fit']]).tolist()}],
            'policies': {'accuracy': {'selection': {'index': 0}, 'candidate': {'feature': 'semantic_av'},
                'probabilities': model.predict_proba(x['semantic_av'][f['heldout']]).tolist()}}})
    return x, frame, old, settings, records, folds


def test_outer_data_cannot_change_fit_or_choice(fixture, tmp_path):
    x, frame, old, settings, records, folds = fixture; fold = folds[0]
    first = s.job(x, frame, fold, records[0], settings, tmp_path/'first')
    x['semantic_av'][fold['heldout']] += 100
    frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+1)%5
    second = s.job(x, frame, fold, records[0], settings, tmp_path/'second')
    assert first['options'] == second['options'] and first['inner_probabilities'] == second['inner_probabilities']
    assert first['selected']['accuracy']['checkpoint_sha256'] == second['selected']['accuracy']['checkpoint_sha256']


def test_zero_fit_resume_and_tamper_rejection(fixture, tmp_path, monkeypatch):
    x, frame, old, settings, records, folds = fixture; out = tmp_path/'run'; args = (x, frame, old, records, {}, settings, out)
    assert not s.run(*args, max_new_jobs=1)['complete']; assert s.run(*args, resume=True)['complete']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a, **kw): raise AssertionError('No fitting, inference or deserialization')
    monkeypatch.setattr(s, 'job', forbidden); monkeypatch.setattr(s, 'five_probabilities', forbidden); monkeypatch.setattr(s.pickle, 'load', forbidden)
    assert s.run(*args, resume=True)['new_fits'] == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(ValueError, match='Inputs/code changed'): s.run(x, frame, old, records, {}, dict(settings, fit_seed=3), out, resume=True)
    p = next((out/'jobs').glob('*/*.pkl')); p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Checkpoint changed'): s.run(*args, resume=True)


def test_bounded_untried_grid():
    settings = json.loads(Path('configs/targeted512_refinement.json').read_text())
    search = s.candidates(settings)
    assert len(search) == 17 and (len(search)-1)*3 == settings['inner_pipeline_fits_per_job'] == 48
    assert {r['feature'] for r in search[1:]} == {'semantic_av'}
    assert {r['C'] for r in search[1:]} == {.003, .03}
