import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import supervised_accuracy as s


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):
        yield


def test_pls_uses_only_training_labels_and_preserves_inference():
    rng = np.random.default_rng(10)
    y = np.repeat(np.arange(5), [20, 12, 10, 8, 6])
    x = rng.normal(size=(len(y), 40))
    model = s.SupervisedClassifier(components=4, weight_power=.5).fit(x, y)
    np.testing.assert_allclose(model.scaler_.mean_, x.mean(0))
    expected = (len(y)/(5*np.bincount(y)[y]))**.5
    expected *= len(y)/expected.sum()
    np.testing.assert_allclose(model.sample_weights_, expected)
    assert model.real_count_ == len(y) and model.pls_.x_rotations_.shape == (40, 4)
    changed = s.SupervisedClassifier(components=4).fit(x, rng.permutation(y))
    assert not np.allclose(changed.pls_.x_rotations_, model.pls_.x_rotations_)
    before = pickle.dumps(model)
    p = model.predict_proba(rng.normal(size=(3, 40)))
    assert p.shape == (3, 5) and np.allclose(p.sum(1), 1)
    assert pickle.dumps(model) == before
    np.testing.assert_allclose(pickle.loads(before).predict_proba(x), model.predict_proba(x), rtol=0, atol=1e-14)


def test_ridge_report_scores_preserve_decisions():
    rng = np.random.default_rng(21)
    y = np.repeat(np.arange(5), [20, 12, 10, 8, 6]); x = rng.normal(size=(len(y), 30))
    m = s.SupervisedClassifier(family='ridge', alpha=100, weight_power=.5).fit(x, y)
    test = rng.normal(size=(20, 30))
    np.testing.assert_array_equal(m.predict(test), m.model_.predict(m.transform(test)))
    assert m.pls_ is None and m.latent_scaler_ is None
    assert np.isfinite(m.predict_proba(test)).all()
    with pytest.raises(ValueError, match='five'):
        m.fit(x[y != 4], y[y != 4])
    with pytest.raises(ValueError, match='component'):
        s.SupervisedClassifier(components=500).fit(x, y)


@pytest.fixture
def study(tmp_path, monkeypatch):
    old = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    settings = json.loads(Path('configs/supervised_accuracy.json').read_text())
    settings.update(feature_inputs=['minilm'], components=[2], C_candidates=[.1], ridge_alpha=[10],
                    weight_powers=[0.], expected_jobs=2, inner_pipeline_fits_per_job=4,
                    inner_base_fits_per_job=4, cpu_threads=1)
    y = np.repeat(np.arange(5), [24, 12, 12, 12, 8])
    x = {'minilm': np.random.default_rng(6).normal(size=(len(y), 25))}; x['minilm'][:, 0] = y*4
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(len(y))], 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old); records = []; root = tmp_path/'baseline'; monkeypatch.setattr(s, 'ROOT', root)
    for fold in folds:
        model = DummyClassifier(strategy='most_frequent').fit(x['minilm'][fold['fit']], y[fold['fit']])
        directory = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; directory.mkdir(parents=True)
        with (directory/'accuracy.pkl').open('wb') as f:
            pickle.dump({'estimator': model}, f)
        records.append({**fold, 'candidates': [{'predictions': model.predict(x['minilm'][fold['fit']]).tolist()}],
            'policies': {'accuracy': {'selection': {'index': 0}, 'candidate': {'feature': 'minilm'},
                                     'probabilities': model.predict_proba(x['minilm'][fold['heldout']]).tolist()}}})
    return (x, frame, old, records, {}), settings, folds


def test_outer_labels_and_features_cannot_change_supervised_selection(study, tmp_path):
    inputs, settings, folds = study; x, frame, _, records, _ = inputs; fold = folds[0]
    first = s.job(x, frame, fold, records[0], settings, tmp_path/'first')
    x['minilm'][fold['heldout']] += 100
    frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+1)%5
    second = s.job(x, frame, fold, records[0], settings, tmp_path/'second')
    assert first['options'] == second['options'] and first['inner_probabilities'] == second['inner_probabilities']
    assert first['selected']['accuracy']['checkpoint_sha256'] == second['selected']['accuracy']['checkpoint_sha256']


def test_resume_integrity_without_fit_or_inference(study, tmp_path, monkeypatch):
    inputs, settings, _ = study; out = tmp_path/'study'
    assert not s.run(*inputs, settings, out, max_new_jobs=1)['complete']
    assert s.run(*inputs, settings, out, resume=True)['complete']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('Resume must not fit, infer or deserialize')
    monkeypatch.setattr(s, 'job', forbidden); monkeypatch.setattr(s, 'five_probabilities', forbidden); monkeypatch.setattr(s.pickle, 'load', forbidden)
    assert s.run(*inputs, settings, out, resume=True)['new_fits'] == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in before.items())
    with pytest.raises(ValueError, match='Inputs/code changed'):
        s.run(*inputs, dict(settings, fit_seed=10), out, resume=True)
    checkpoint = next((out/'jobs').glob('*/*.pkl')); checkpoint.write_bytes(checkpoint.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        s.run(*inputs, settings, out, resume=True)


def test_prespecified_grid_and_baseline_tie_priority():
    settings = json.loads(Path('configs/supervised_accuracy.json').read_text())
    candidates = s.candidates(settings)
    assert len(candidates) == 97
    assert sum(c['family'] == 'pls' for c in candidates) == 72
    assert sum(c['family'] == 'ridge' for c in candidates) == 24
    assert (len(candidates)-1)*3 == settings['inner_pipeline_fits_per_job']
    assert s.choose([{'correct': 50, 'macro_f1': .3}, {'correct': 50, 'macro_f1': .3}]) == 0


def test_final_selection_ignores_outer_results(study, tmp_path):
    from src.supervised_accuracy_finalize import training_choices
    inputs, settings, folds = study; x, frame, _, old, _ = inputs
    for r in old:
        r['candidates'][0]['candidate'] = {'family': 'majority', 'feature': 'minilm'}
    records = [s.job(x, frame, f, r, settings, tmp_path/f'fold{f["fold"]}') for f, r in zip(folds, old)]
    first = training_choices(frame, records, old, settings)
    for r in records:
        r['selected'] = {'arbitrary_outer_result': 1000}
        r['options'] = []; r['real_training_accuracy'] = 0.
    for r in old:
        r['policies'] = {'arbitrary_outer_result': 1000}
    assert training_choices(frame, records, old, settings) == first
    assert first['inner_predictions_per_participant'] == 1
    records[0]['inner_probabilities'][0][0][0] = float('nan')
    with pytest.raises(ValueError, match='probabilities'):
        training_choices(frame, records, old, settings)
