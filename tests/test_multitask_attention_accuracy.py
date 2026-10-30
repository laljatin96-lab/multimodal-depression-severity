import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import multitask_attention_accuracy as s


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def toy():
    rng = np.random.default_rng(13)
    y = np.repeat(np.arange(5), [24, 12, 12, 12, 8])
    clinical = np.ones((len(y), 25), int)
    for i, target in enumerate(y*5):
        clinical[i, :8] = [min(3, max(0, target-3*j)) for j in range(8)]
    data = {'vectors': [rng.normal(size=(1+i%4, 12)) for i in range(len(y))],
            'lengths': [[254]*(1+i%4) for i in range(len(y))], 'av': rng.normal(size=(len(y), 9))}
    settings = json.loads(Path('configs/multitask_attention_accuracy.json').read_text())
    settings.update(epochs=[2], use_av=[True], weight_powers=[0.], text_components=4, av_components=3,
                    hidden=4, expected_jobs=2, cpu_threads=1)
    return data, y, clinical, settings


def test_projection_weights_participants_equally(toy):
    data, y, clinical, settings = toy
    projection = s.Projection().fit(data, settings)
    np.testing.assert_allclose(projection.text[0], np.mean([v.mean(0) for v in data['vectors']], axis=0))
    np.testing.assert_allclose(projection.av[0], data['av'].mean(0))
    altered = copy.deepcopy(data)
    altered['vectors'][0] = np.repeat(altered['vectors'][0], 5, axis=0)
    altered['lengths'][0] = [254]*5
    other = s.Projection().fit(altered, settings)
    np.testing.assert_allclose(projection.text[0], other.text[0])


def test_auxiliary_targets_not_inference_inputs_and_class_only_ignores_them(toy):
    data, y, clinical, settings = toy
    candidate = s.candidates(settings)[1]
    a = s.AttentionClassifier(candidate, settings).fit(data, y, clinical)
    b = s.AttentionClassifier(candidate, settings).fit(data, y, np.full_like(clinical, -900))
    assert pickle.dumps(a) == pickle.dumps(b)
    encoded = pickle.dumps(a); p = a.predict_proba(data)
    assert p.shape == (len(y), 5) and np.allclose(p.sum(1), 1)
    assert encoded == pickle.dumps(a)
    np.testing.assert_allclose(pickle.loads(encoded).predict_proba(data), p, rtol=0, atol=0)
    c = s.AttentionClassifier(dict(candidate, mode='phq'), settings).fit(data, y, clinical)
    assert not np.allclose(c.predict_proba(data), p)
    bad = clinical.copy(); bad[0, 0] = 3; bad[0, 1] = 3
    with pytest.raises(ValueError, match='disagree'):
        s.AttentionClassifier(dict(candidate, mode='phq'), settings).fit(data, y, bad)


def test_padding_and_batch_composition_do_not_change_prediction(toy):
    data, y, clinical, settings = toy
    model = s.AttentionClassifier(s.candidates(settings)[-1], settings).fit(data, y, clinical)
    single = model.predict_proba(s.subset(data, [0]))
    np.testing.assert_allclose(single[0], model.predict_proba(data)[0], rtol=0, atol=1e-7)


@pytest.fixture
def experiment(toy, tmp_path, monkeypatch):
    data, y, clinical, settings = toy
    old_settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(len(y))], 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old_settings); x = {'semantic': data['av']}; old = []
    root = tmp_path/'baseline'; monkeypatch.setattr(s, 'ROOT', root)
    for fold in folds:
        model = DummyClassifier(strategy='uniform').fit(x['semantic'][fold['fit']], y[fold['fit']])
        directory = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; directory.mkdir(parents=True)
        (directory/'accuracy.pkl').write_bytes(pickle.dumps({'estimator': model}))
        old.append({**fold, 'candidates': [{'predictions': model.predict_proba(x['semantic'][fold['fit']]).argmax(1).tolist()}],
            'policies': {'accuracy': {'selection': {'index': 0}, 'candidate': {'feature': 'semantic'},
                                     'probabilities': model.predict_proba(x['semantic'][fold['heldout']]).tolist()}}})
    return (data, clinical, x, frame, old_settings, old, {'bindings': {}}), settings, folds


def test_outer_answers_labels_features_cannot_change_training_or_selection(experiment, tmp_path):
    inputs, settings, folds = experiment
    data, clinical, x, frame, _, old, _ = inputs; fold = folds[0]
    first = s.job(data, clinical, x, frame, fold, old[0], settings, tmp_path/'first')
    clinical[fold['heldout']] = -999
    frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+1)%5
    data['av'][fold['heldout']] += 999
    for i in fold['heldout']:
        data['vectors'][i] += 50
    second = s.job(data, clinical, x, frame, fold, old[0], settings, tmp_path/'second')
    assert first['inner_probabilities'] == second['inner_probabilities']
    assert first['options'] == second['options']
    for name in settings['policies']:
        assert first['selected'][name]['checkpoint_sha256'] == second['selected'][name]['checkpoint_sha256']


def test_resume_is_zero_fit_and_detects_tampering(experiment, tmp_path, monkeypatch):
    inputs, settings, folds = experiment; out = tmp_path/'run'
    assert not s.run(inputs, settings, out, max_new_jobs=1)['complete']
    assert s.run(inputs, settings, out, resume=True)['complete']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('No refit/replay on resume')
    monkeypatch.setattr(s, 'job', forbidden); monkeypatch.setattr(s, 'predict_state', forbidden)
    assert s.run(inputs, settings, out, resume=True)['new_fits'] == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in before.items())
    with pytest.raises(ValueError, match='protocol changed'):
        s.run(inputs, dict(settings, fit_seed=123), out, resume=True)
    path = next((out/'jobs').glob('*/*.pkl')); path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        s.run(inputs, settings, out, resume=True)


def test_fixed_search():
    settings = json.loads(Path('configs/multitask_attention_accuracy.json').read_text())
    assert len(s.candidates(settings)) == 25
    assert set(c['mode'] for c in s.candidates(settings)[1:]) == {'class_only', 'phq', 'phq_ptsd'}


def test_final_selection_ignores_outer_predictions(experiment, tmp_path):
    from src.multitask_attention_finalize import training_choices
    inputs, settings, folds = experiment
    data, clinical, x, frame, _, old, _ = inputs
    records = [s.job(data, clinical, x, frame, fold, baseline, settings, tmp_path/f'fold{fold["fold"]}')
               for fold, baseline in zip(folds, old)]
    first = training_choices(frame, records, old, settings)
    for r in records:
        r['selected'] = {'arbitrary': 'corrupted outer predictions'}
        r['options'] = []
    for r in old:
        r['policies']['accuracy']['probabilities'] = []
    assert training_choices(frame, records, old, settings) == first
    assert first['inner_predictions_per_participant'] == 1
