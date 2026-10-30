import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import quadratic_accuracy as s


class LanguageDummy:
    def fit(self, x, y):
        self.model = DummyClassifier(strategy='prior').fit(x, y)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x['base'])


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def toy():
    rng = np.random.default_rng(4); y = np.repeat(np.arange(5), [24, 12, 12, 12, 8])
    x = {'semantic_av': rng.normal(size=(len(y), 20)), 'language': rng.normal(size=(len(y), 16))}
    settings = json.loads(s.CONFIG.read_text())
    settings.update(components=[4], pooled_diagonal_fractions=[.5], class_variance_fractions=[.25], expected_jobs=2, cpu_threads=1)
    return x, y, settings


def test_quadratic_density_matches_scipy_and_pickle(toy):
    from scipy.stats import multivariate_normal
    from scipy.special import softmax
    x, y, settings = toy
    m = s.QuadraticClassifier(s.candidates(settings)[1], settings).fit(x, y)
    np.testing.assert_allclose(m.base_scaler.mean_, x['semantic_av'].mean(0))
    np.testing.assert_allclose(m.language_scaler.mean_, x['language'].mean(0))
    assert m.real_count == m.base_pca.n_samples_ == len(y)
    z = m.transform(x)
    expected = np.stack([multivariate_normal.logpdf(z, mean=mean, cov=cov)+prior
                        for mean,cov,prior in zip(m.means_,m.covariances_,m.logprior_)],axis=1)
    before = pickle.dumps(m); p = m.predict_proba(x)
    np.testing.assert_allclose(p, softmax(expected,axis=1), rtol=0, atol=1e-12)
    assert pickle.dumps(m) == before
    np.testing.assert_allclose(pickle.loads(before).predict_proba(x), p, atol=1e-14)


def test_scarce_class_constant_features_and_ignored_nonfeatures(toy):
    x,y,settings=toy
    x['language'][:,0]=0
    keep=np.r_[np.flatnonzero(y!=4), np.flatnonzero(y==4)[:1]]
    x=s.subset(x,keep); y=y[keep]
    m=s.QuadraticClassifier(s.candidates(settings)[1],settings).fit(x,y)
    p=m.predict_proba(x)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    assert np.linalg.eigvalsh(m.covariances_).min()>0
    changed={**x,'severity_class':np.full(len(y),4),'PHQ8_Score':np.full(len(y),24),'participant_id':np.arange(len(y))}
    np.testing.assert_array_equal(m.predict_proba(changed),p)
    assert not np.allclose(m.covariances_[0],m.covariances_[1])


@pytest.fixture
def experiment(toy, tmp_path, monkeypatch):
    x, y, settings = toy
    old_settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    frame = pd.DataFrame({'participant_id': list(map(str, range(len(y)))), 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old_settings); old = []; root = tmp_path/'language'
    monkeypatch.setattr(s, 'LANGUAGE_ROOT', root)
    for f in folds:
        model = LanguageDummy().fit(x['semantic_av'][f['fit']], y[f['fit']])
        directory = root/'jobs'/f"seed{f['seed']}_fold{f['fold']}"; directory.mkdir(parents=True)
        (directory/'without_age.pkl').write_bytes(pickle.dumps({'estimator': model, 'candidate': {'feature_set': 'language'}}))
        held = model.predict_proba({'base': x['semantic_av'][f['heldout']]})
        inner = np.full((len(y), 5), np.nan)
        for part in f['inner']:
            m = LanguageDummy().fit(x['semantic_av'][part['fit']], y[part['fit']])
            inner[part['heldout']] = m.predict_proba({'base': x['semantic_av'][part['heldout']]})
        control = {**f, 'inner_probabilities': [inner[f['fit']].tolist()],
            'selected': {'without_age': {'selection': {'index': 1}, 'probabilities': held.tolist()}}}
        old.append({**f, 'language_control': control,
            'policies': {'accuracy': {'probabilities': held.tolist()}},
            'candidates': [{'predictions': inner[f['fit']].argmax(1).tolist()}]})
    return (x, frame, old_settings, old, {'bindings': {}}), settings, folds


def test_heldout_features_and_targets_do_not_change_fit_or_selection(experiment, tmp_path):
    inputs, settings, folds = experiment; x, frame, _, old, _ = inputs; f = folds[0]
    a = s.job(x, frame, f, old[0], settings, tmp_path/'a')
    frame.loc[f['heldout'], 'severity_class'] = (frame.loc[f['heldout'], 'severity_class']+1)%5
    for k in x:
        x[k][f['heldout']] += 1000
    b = s.job(x, frame, f, old[0], settings, tmp_path/'b')
    assert a['inner_probabilities'] == b['inner_probabilities'] and a['options'] == b['options']
    for name in settings['policies']:
        assert a['selected'][name]['checkpoint_sha256'] == b['selected'][name]['checkpoint_sha256']


def test_resume_preserves_every_result_and_rejects_tampering(experiment, tmp_path, monkeypatch):
    inputs, settings, folds = experiment; out = tmp_path/'study'
    assert not s.run(inputs, settings, out, max_new_jobs=1)['complete']
    assert s.run(inputs, settings, out, resume=True)['complete']
    before = {p: [p.read_bytes(), p.stat().st_mtime_ns] for p in out.rglob('*') if p.is_file() and p.name != '.batch.lock'}
    def forbidden(*args, **kwargs):
        raise AssertionError('No retraining or inference on completed resume')
    monkeypatch.setattr(s, 'job', forbidden); monkeypatch.setattr(s, 'predict_state', forbidden)
    assert s.run(inputs, settings, out, resume=True)['new_fits'] == 0
    assert all([p.read_bytes(), p.stat().st_mtime_ns] == v for p, v in before.items())
    with pytest.raises(ValueError, match='Frozen inputs'):
        s.run(inputs, dict(settings, fit_seed=8), out, resume=True)
    path = next((out/'jobs').glob('*/*.pkl')); path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        s.run(inputs, settings, out, resume=True)


def test_final_choice_uses_inner_results_only(experiment, tmp_path):
    inputs, settings, folds = experiment; x, frame, _, old, _ = inputs
    records = [s.job(x, frame, f, o, settings, tmp_path/str(f['fold'])) for f, o in zip(folds, old)]
    expected = s.training_choices(frame, records, old, settings)
    for r in records:
        r['selected'] = {'corrupted_outer': True}; r['options'] = []
    for r in old:
        r['policies']['accuracy']['probabilities'] = []
        r['language_control']['selected']['without_age']['probabilities'] = []
    assert s.training_choices(frame, records, old, settings) == expected


def test_prespecified_grid_and_control():
    settings = json.loads(s.CONFIG.read_text()); candidates = s.candidates(settings)
    assert len(candidates) == 9 and candidates[0] == {'family': 'language_control'}
    assert s.choose([{'correct': 0}, {'correct': 20}], 'language_control') == 0
    assert settings['target_accuracy'] == .56
