import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('catboost')
from src import catboost_late_fusion as s
from src import catboost_holdout as h
from sklearn.dummy import DummyClassifier


@pytest.fixture
def fixture():
    old = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    settings = json.loads(Path('configs/catboost_late_fusion.json').read_text())
    settings.update(capacity_profiles=[{'depth': 2, 'iterations': 5, 'learning_rate': .1, 'l2_leaf_reg': 10}],
                    auto_class_weights=['None', 'Balanced'], text_components=3, expected_jobs=2,
                    inner_head_fits_per_job=24, cpu_threads=1)
    y = np.repeat(np.arange(5), [40, 20, 16, 16, 8]); rng = np.random.default_rng(3)
    x = {n: rng.normal(size=(100, 6)) for n in s.MODALITIES}
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(100)], 'severity_class': y,
                          'gender': np.resize(['female', 'male', 'unknown'], 100)})
    folds = s.nested_folds(frame, old)
    records49, records48 = [], []
    for fold in folds:
        p = np.tile([1., 0, 0, 0, 0], (len(fold['heldout']), 1)).tolist()
        records49.append({**fold, 'policies': {'accuracy': {'probabilities': p}}})
        records48.append({**fold, 'probabilities': p})
    return x, frame, old, settings, folds, records49, records48


def test_fitting_only_reduction_and_gender_invariance(fixture):
    x, frame, old, settings, folds, *_ = fixture; part = folds[0]
    mapper = s.FeatureMap(3).fit(x, part['fit'])
    np.testing.assert_allclose(mapper.scaler.mean_, x['text'][part['fit']].mean(0))
    z = mapper.transform(x, part['fit']); g = frame.gender.to_numpy(str)[part['fit']]
    y = frame.severity_class.to_numpy()[part['fit']]
    model = s.fit_head(z['text'], y, g, False, s.profiles(settings)[0], settings)
    np.testing.assert_array_equal(s.predict_head(model, z['text'], g, False), s.predict_head(model, z['text'], g[::-1], False))
    gender = s.model_input(z['text'], g, True)
    assert gender.recorded_gender.tolist() == g.tolist()
    assert list(gender.columns)[-1] == 'recorded_gender'
    with pytest.raises(ValueError): s.model_input(z['text'], ['inferred']*len(g), True)
    p = np.tile([.5, .2, .1, .1, .1], (10, 1))
    np.testing.assert_allclose(s.fuse([p, p, p], [1/3]*3), p)
    with pytest.raises(ValueError): s.fuse([p, p, p], [1, 0, 0])


def test_outer_labels_features_gender_cannot_change_fitted_state(fixture, tmp_path):
    x, frame, old, settings, folds, *_ = fixture; fold = folds[0]
    a = s.job(x, frame, fold, settings, tmp_path/'a')
    other_x = {n: a.copy() for n, a in x.items()}; other_frame = frame.copy()
    for values in other_x.values(): values[fold['heldout']] += 100
    other_frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+1)%5
    other_frame.loc[fold['heldout'], 'gender'] = 'unknown'
    b = s.job(other_x, other_frame, fold, settings, tmp_path/'b')
    assert a['options'] == b['options'] and a['inner_probabilities'] == b['inner_probabilities']
    for name in s.VARIANTS:
        assert a['selected'][name]['index'] == b['selected'][name]['index']
        with (tmp_path/'a'/f'{name}.pkl').open('rb') as f: first = pickle.load(f)
        with (tmp_path/'b'/f'{name}.pkl').open('rb') as f: second = pickle.load(f)
        for modality in s.MODALITIES:
            np.testing.assert_array_equal(first['heads'][modality].get_leaf_values(), second['heads'][modality].get_leaf_values())
        np.testing.assert_array_equal(first['feature_map'].pca.components_, second['feature_map'].pca.components_)


def test_interrupted_resume_and_tampering(fixture, tmp_path, monkeypatch):
    x, frame, old, settings, folds, old49, old48 = fixture; out = tmp_path/'run'
    args = (x, frame, old, old49, old48, {}, settings, out)
    assert not s.run(*args, max_new_jobs=1)['complete']
    assert s.run(*args, resume=True)['complete']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a, **kw): raise AssertionError('Completed resume cannot train, infer or deserialize')
    monkeypatch.setattr(s, 'job', forbidden); monkeypatch.setattr(s, 'predict_state', forbidden)
    monkeypatch.setattr(s.pickle, 'load', forbidden)
    assert s.run(*args, resume=True)['new_head_fits'] == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    changed = copy.deepcopy(settings); changed['fit_seed'] = 123
    with pytest.raises(ValueError, match='Inputs/code changed'): s.run(x, frame, old, old49, old48, {}, changed, out, resume=True)
    path = next((out/'jobs').glob('*/*.pkl')); path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Checkpoint'): s.run(*args, resume=True)


def test_predeclared_budget():
    settings = json.loads(Path('configs/catboost_late_fusion.json').read_text())
    assert len(s.profiles(settings)) == 6 and len(settings['fusion_weights']) == 7
    assert len(s.profiles(settings))*3*2*3 == settings['inner_head_fits_per_job'] == 108


def add_baseline_candidates(frame, records):
    y = frame.severity_class.to_numpy(int)
    for r in records:
        r['candidates'] = [{'candidate': {'family': 'majority', 'feature': 'semantic'},
                            'predictions': [0]*len(r['fit'])}]


def test_final_choice_uses_inner_predictions_only(fixture):
    x, frame, old, settings, folds, old49, old48 = fixture
    add_baseline_candidates(frame, old49); add_baseline_candidates(frame, old48)
    records = []
    for f in folds:
        inner = np.tile([1., 0, 0, 0, 0], (len(s.profiles(settings)), 3, len(f['fit']), 1)).tolist()
        records.append({**f, 'inner_probabilities': {n: inner for n in s.VARIANTS}})
    first = h.pooled_training_choices(frame, records, old49, old48, settings)
    for r in records: r['selected'] = {'made_up_outer_scores': 1.0}
    for r in old49: r['policies']['accuracy']['probabilities'] = []
    for r in old48: r['probabilities'] = []
    assert first == h.pooled_training_choices(frame, records, old49, old48, settings)
    assert first['primary_catboost'] == 'non_gender'


def test_freeze_before_holdouts_and_zero_inference_resume(fixture, tmp_path, monkeypatch):
    x, frame, old, settings, folds, old49, old48 = fixture
    add_baseline_candidates(frame, old49); add_baseline_candidates(frame, old48)
    inputs = (x, frame, old, old49, old48, {'bindings': {}})
    study = tmp_path/'study'; out = tmp_path/'frozen'
    s.run(*inputs, settings, study)
    monkeypatch.setattr(s, 'OUTPUT', study); monkeypatch.setattr(s, 'load_inputs', lambda: inputs)
    # Freeze reads its settings file; substitute only that read with the bounded fixture settings.
    read_text = Path.read_text
    def configured(path, *a, **kw):
        if str(path) == 'configs/catboost_late_fusion.json': return json.dumps(settings)
        return read_text(path, *a, **kw)
    monkeypatch.setattr(Path, 'read_text', configured)
    monkeypatch.setattr(h.old49_module, 'estimator_for', lambda c, config: DummyClassifier(strategy='most_frequent'))
    monkeypatch.setattr(h.old48_module, 'estimator_for', lambda c, config: DummyClassifier(strategy='most_frequent'))
    def forbidden(*a, **kw): raise AssertionError('No holdout access during freeze or fitting during evaluation')
    monkeypatch.setattr(s, 'load_all_inputs', forbidden)
    plan = h.freeze(out)
    assert plan['training_ids'] == frame.participant_id.tolist()
    all_frame = pd.concat([frame.assign(split='train'),
        pd.DataFrame({'participant_id': [f'v{i}' for i in range(10)], 'severity_class': np.arange(10)%5, 'gender': 'unknown', 'split': 'validation'}),
        pd.DataFrame({'participant_id': [f't{i}' for i in range(10)], 'severity_class': np.arange(10)%5, 'gender': 'female', 'split': 'test'})], ignore_index=True)
    all_x = {n: np.concatenate([a, np.ones((20, a.shape[1]))]) for n, a in x.items()}
    monkeypatch.setattr(s, 'load_all_inputs', lambda: (all_x, all_frame))
    monkeypatch.setattr(s, 'fit_head', forbidden); monkeypatch.setattr(h, 'fit', forbidden)
    h.evaluate(out)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.glob('*/*/result.json')}
    assert len(before) == 10
    monkeypatch.setattr(h, 'predict_model', forbidden); monkeypatch.setattr(h.pickle, 'load', forbidden)
    h.evaluate(out)
    assert json.loads((out/'completion.json').read_text())['new_prediction_batches'] == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    path = Path(next(iter(plan['models'].values()))['path']); path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError, match='Frozen model changed'): h.evaluate(out)
