import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import imbalance_accuracy as s


@pytest.fixture(autouse=True)
def limited_threads():
    with threadpool_limits(limits=1):
        yield


def test_smote_uses_only_real_same_class_neighbors():
    y = np.repeat(np.arange(5), [8, 6, 5, 4, 3])
    z = np.random.default_rng(3).normal(size=(len(y), 4))
    for sampler in ['smote20', 'smote30', 'joint_smote30']:
        augmented, labels, lineage = s.resample(z, y, sampler)
        np.testing.assert_array_equal(augmented[:len(y)], z)
        for i, row in enumerate(lineage, len(y)):
            a, b = row['parent_a'], row['parent_b']
            assert a != b and labels[i] == y[a] == y[b] == row['class']
            distance = ((z[y == y[a]]-z[a])**2).sum(1)
            nearest = np.sort(distance[distance > 0])[:2]
            assert ((z[b]-z[a])**2).sum() <= nearest.max()+1e-12
            np.testing.assert_allclose(augmented[i], z[a]+row['fraction']*(z[b]-z[a]))
        assert (labels == 4).sum() == (25 if sampler == 'joint_smote30' else int(sampler[-2:]))


def test_ros_singleton_and_absent_class():
    y = np.array([0, 1, 2, 3, 4]); z = np.arange(15.).reshape(5, 3)
    for sampler in ['ros20', 'smote30']:
        augmented, labels, lineage = s.resample(z, y, sampler)
        np.testing.assert_array_equal(augmented[5:], np.tile(z[4], (len(lineage), 1)))
        assert (labels == 4).sum() == int(sampler[-2:])
        assert all(v['singleton_fallback'] == sampler.startswith('smote') for v in lineage)
    with pytest.raises(ValueError, match='absent class'):
        s.resample(z[:4], y[:4], 'smote20')


def test_preprocessing_precedes_augmentation_and_weight_normalization():
    y = np.repeat(np.arange(5), [20, 12, 10, 8, 3]); x = np.random.default_rng(8).normal(size=(len(y), 512))
    for sampler in ['none', 'quarter_weight', 'balanced_weight', 'ros30', 'smote20']:
        m = s.ResampledLogistic(sampler=sampler).fit(x, y)
        for indices, scaler, pca in m.mapper_.transforms_:
            np.testing.assert_allclose(scaler.mean_, x[:, indices].mean(0))
            assert pca.n_samples_ == len(y)
        assert m.real_count_ == m.mapper_.real_count_ == len(y)
        assert m.total_weight_ == pytest.approx(len(y))
        s.validate_receipt(s.receipt(m, list(range(len(y)))), list(range(len(y))), y,
                           {'sampler': sampler}, {})
        if sampler == 'balanced_weight':
            expected = len(y)/(5*np.bincount(y)[y])
            np.testing.assert_allclose(m.sample_weights_, expected)
        np.testing.assert_allclose(m.predict_proba(x).sum(1), 1)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(m)).predict_proba(x), m.predict_proba(x))


def test_block_feature_membership():
    y = np.repeat(np.arange(5), 6); x = np.random.default_rng(42).normal(size=(len(y), 512))
    for feature in ['text', 'text_audio', 'text_visual', 'text_audio_visual']:
        for reduction in ['global16', 'block32']:
            mapper = s.FeatureMap(feature, reduction).fit(x)
            expected = np.concatenate([v for _, v in s.blocks(feature)])
            actual = np.concatenate([idx for idx, _, _ in mapper.transforms_])
            np.testing.assert_array_equal(actual, expected)
            changed = x.copy(); excluded = np.setdiff1d(np.arange(512), expected); changed[:, excluded] += 10000
            np.testing.assert_allclose(mapper.transform(changed), mapper.transform(x))


@pytest.fixture
def study(tmp_path, monkeypatch):
    old = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    settings = json.loads(Path('configs/imbalance_accuracy.json').read_text())
    settings.update(features=['text'], reductions=['global16'], C=[.01],
                    samplers=['smote20', 'ros20'], expected_jobs=2, expected_candidates=3, cpu_threads=1)
    y = np.repeat(np.arange(5), [24, 12, 12, 12, 8]); x = {'semantic_av': np.random.default_rng(1).normal(size=(len(y), 512))}
    x['semantic_av'][:, 0] = y*5
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(len(y))], 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old); records = []; root = tmp_path/'baseline'; monkeypatch.setattr(s, 'ROOT', root)
    for fold in folds:
        m = DummyClassifier(strategy='most_frequent').fit(x['semantic_av'][fold['fit']], y[fold['fit']])
        directory = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; directory.mkdir(parents=True)
        with (directory/'accuracy.pkl').open('wb') as f:
            pickle.dump({'estimator': m}, f)
        records.append({**fold, 'candidates': [{'predictions': m.predict(x['semantic_av'][fold['fit']]).tolist()}],
                        'policies': {'accuracy': {'selection': {'index': 0}, 'candidate': {'feature': 'semantic_av'},
                                                 'probabilities': m.predict_proba(x['semantic_av'][fold['heldout']]).tolist()}}})
    return (x, frame, old, records, {}), settings, folds


def test_heldout_features_and_labels_cannot_change_selection_or_model(study, tmp_path):
    inputs, settings, folds = study; x, frame, old, records, _ = inputs; fold = folds[0]
    first = s.job(x, frame, fold, records[0], settings, tmp_path/'first')
    x['semantic_av'][fold['heldout']] += 100
    frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+2)%5
    second = s.job(x, frame, fold, records[0], settings, tmp_path/'second')
    assert first['options'] == second['options'] and first['inner_receipts'] == second['inner_receipts']
    assert first['inner_probabilities'] == second['inner_probabilities']
    for name in settings['policies']:
        assert first['selected'][name]['checkpoint_sha256'] == second['selected'][name]['checkpoint_sha256']


def test_resume_no_fitting_and_corruption_rejected(study, tmp_path, monkeypatch):
    inputs, settings, _ = study; out = tmp_path/'study'
    assert not s.run(inputs, settings, out, max_new_jobs=1)['complete']
    assert s.run(inputs, settings, out, resume=True)['complete']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('Resume must not fit, predict or deserialize')
    monkeypatch.setattr(s, 'job', forbidden); monkeypatch.setattr(s, 'five_probabilities', forbidden); monkeypatch.setattr(s.pickle, 'load', forbidden)
    assert s.run(inputs, settings, out, resume=True)['new_fits'] == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in before.items())
    with pytest.raises(ValueError, match='Inputs/code changed'):
        s.run(inputs, dict(settings, fit_seed=44), out, resume=True)
    checkpoint = next((out/'jobs').glob('*/*.pkl')); checkpoint.write_bytes(checkpoint.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        s.run(inputs, settings, out, resume=True)


def test_severe_policy_obeys_accuracy_budget():
    settings = {'severe_accuracy_tolerance': .02}
    rows = [{'candidate': {}, 'accuracy': .5, 'correct': 50, 'macro_f1': .2, 'severe_true_positive': 0},
            {'candidate': {}, 'accuracy': .48, 'correct': 48, 'macro_f1': .3, 'severe_true_positive': 1},
            {'candidate': {}, 'accuracy': .4, 'correct': 40, 'macro_f1': .4, 'severe_true_positive': 5}]
    assert s.choose(rows, 'accuracy', settings) == 0
    assert s.choose(rows, 'severe_guarded', settings) == 1


def test_declared_grid_is_bounded():
    settings = json.loads(Path('configs/imbalance_accuracy.json').read_text())
    assert len(s.candidates(settings)) == settings['expected_candidates'] == 129
