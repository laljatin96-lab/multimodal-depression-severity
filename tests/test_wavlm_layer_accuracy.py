import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import wavlm_layer_accuracy as s


class LanguageDummy:
    def fit(self, x, y):
        self.model = DummyClassifier(strategy='prior').fit(x, y)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x['base'])


class SpeechDummy(LanguageDummy):
    def predict_proba(self, x):
        return self.model.predict_proba(x['semantic_av'])


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def toy():
    rng = np.random.default_rng(14); y = np.repeat(np.arange(5), [24, 12, 12, 12, 8])
    x = {'semantic_av': rng.normal(size=(len(y), 20)), 'language': rng.normal(size=(len(y), 16))}
    for k in ['speech', 'final_nonzero', 'layer6_nonzero']:
        x[k] = rng.normal(size=(len(y), 30))
    settings = json.loads(s.CONFIG.read_text())
    settings.update(components=4, speech_components=3, expected_jobs=2, cpu_threads=1)
    return x, y, settings


@pytest.fixture
def experiment(toy, tmp_path, monkeypatch):
    x, y, settings = toy
    old_settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    frame = pd.DataFrame({'participant_id': list(map(str, range(len(y)))), 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old_settings); old = []; root = tmp_path/'previous'
    monkeypatch.setattr(s, 'PREVIOUS_ROOT', root)
    for f in folds:
        cls = LanguageDummy if f['fold'] == 0 else SpeechDummy
        model = cls().fit(x['semantic_av'][f['fit']], y[f['fit']])
        key = 'base' if cls is LanguageDummy else 'semantic_av'
        directory = root/'jobs'/f"seed{f['seed']}_fold{f['fold']}"; directory.mkdir(parents=True)
        state = {'estimator': model, 'candidate': {'family': 'language_control' if cls is LanguageDummy else 'wavlm_augmented'}}
        path = directory/'accuracy.pkl'; path.write_bytes(pickle.dumps(state))
        held = model.predict_proba({key: x['semantic_av'][f['heldout']]})
        inner = np.full((len(y), 5), np.nan)
        for part in f['inner']:
            m = cls().fit(x['semantic_av'][part['fit']], y[part['fit']])
            inner[part['heldout']] = m.predict_proba({key: x['semantic_av'][part['heldout']]})
        control = {**f, 'inner_probabilities': [inner[f['fit']].tolist()],
            'selected': {'without_age': {'selection': {'index': 1}, 'probabilities': held.tolist()}}}
        reference = {**f, 'inner_probabilities': [inner[f['fit']].tolist()],
            'selected': {'accuracy': {'selection': {'index': 0 if cls is LanguageDummy else 1},
                                     'probabilities': held.tolist(), 'checkpoint_sha256': s.sha256(path)}}}
        old.append({**f, 'language_control': control, 'previous_procedure': reference,
            'policies': {'accuracy': {'probabilities': held.tolist()}},
            'candidates': [{'predictions': inner[f['fit']].argmax(1).tolist()}]})
    return (x, frame, old_settings, old, {'bindings': {}}), settings, folds


def test_fitting_statistics_and_serialization(toy):
    x, y, settings = toy
    m = s.WavLMLayerClassifier(s.candidates(settings)[1], settings).fit(x, y)
    np.testing.assert_allclose(m.base_scaler.mean_, x['semantic_av'].mean(0))
    np.testing.assert_allclose(m.language_scaler.mean_, x['language'].mean(0))
    np.testing.assert_allclose(m.speech_scaler.mean_, x['final_nonzero'].mean(0))
    assert m.real_count == m.base_pca.n_samples_ == m.speech_pca.n_samples_ == len(y)
    before = pickle.dumps(m); p = m.predict_proba(x)
    assert pickle.dumps(m) == before
    np.testing.assert_allclose(pickle.loads(before).predict_proba(x), p, atol=1e-14)
    assert m.diagnostics(x, y)['converged']
    changed = {**x, 'severity_class': np.full(len(y), 4), 'PHQ8_Score': np.full(len(y), 24),
               'participant_id': np.arange(len(y)), 'speech': np.full_like(x['speech'], 1000),
               'layer6_nonzero': np.full_like(x['layer6_nonzero'], -1000)}
    np.testing.assert_array_equal(m.predict_proba(changed), p)


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


def test_resume_preserves_results_and_rejects_tampering(experiment, tmp_path, monkeypatch):
    inputs, settings, folds = experiment; out = tmp_path/'study'
    assert not s.run(inputs, settings, out, max_new_jobs=1)['complete']
    assert s.run(inputs, settings, out, resume=True)['complete']
    before = {p: [p.read_bytes(), p.stat().st_mtime_ns] for p in out.rglob('*') if p.is_file() and p.name != '.batch.lock'}
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed resume must not retrain or infer')
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
        r['previous_procedure']['selected']['accuracy']['probabilities'] = []
    assert s.training_choices(frame, records, old, settings) == expected


def test_declared_grid_budget_and_representation_policies():
    settings = json.loads(s.CONFIG.read_text()); grid = s.candidates(settings)
    assert len(grid) == 5 and grid[0] == {'family': 'previous_procedure'}
    assert settings['max_nested_pipeline_fits'] == 15*(3*4+2) == 210
    assert settings['max_full_training_fits'] == 2 and settings['target_accuracy'] == .56
    rows = [{'candidate': c, 'correct': i, 'macro_f1': 0} for i, c in enumerate(grid)]
    assert s.choose(rows, 'previous_procedure') == 0
    assert s.choose(rows, 'zero_only') == 2 and s.choose(rows, 'layer6_only') == 4
    with pytest.raises(ValueError, match='regularization budget'):
        s.candidates({**settings, 'C': [.01]})


def test_speech_whitening_and_representation_ownership(toy):
    x, y, settings = toy
    for i, key in [(1, 'final_nonzero'), (3, 'layer6_nonzero')]:
        m = s.WavLMLayerClassifier(s.candidates(settings)[i], settings).fit(x, y)
        np.testing.assert_allclose(m.speech_scaler.mean_, x[key].mean(0))
        projected = m.speech_pca.transform(m.speech_scaler.transform(x[key]))
        np.testing.assert_allclose(np.cov(projected, rowvar=False), np.eye(3), rtol=0, atol=1e-12)


def test_cached_reference_selects_language_or_previous_speech_inner(experiment):
    inputs, settings, folds = experiment; _, _, _, old, _ = inputs
    np.testing.assert_array_equal(s.previous_inner(old[0]), old[0]['language_control']['inner_probabilities'][0])
    changed = copy.deepcopy(old[1]); changed['language_control']['inner_probabilities'] = []
    np.testing.assert_array_equal(s.previous_inner(changed), changed['previous_procedure']['inner_probabilities'][0])


def test_raw_wav_identity_checked_before_model_or_input_io(tmp_path, monkeypatch):
    from src import wavlm_layer_inference as inference
    def forbidden(*args, **kwargs):
        raise AssertionError('Mismatched participant files must fail before model loading')
    monkeypatch.setattr(inference, 'frozen_model', forbidden)
    with pytest.raises(ValueError, match='Raw WAV participant ID differs'):
        inference.predict('layer6_only', tmp_path/'new.json', '487_Transcript.csv',
            '487_OpenSMILE2.3.0_mfcc.csv', '487_OpenFace2.1.0_Pose_gaze_AUs.csv', '306_AUDIO.wav')


def test_saved_state_routes_original_and_new_inference_features(toy):
    from src.wavlm_layer_inference import speech_requirement
    assert speech_requirement({'candidate': {'family': 'block_logistic'}}) is None
    assert speech_requirement({'candidate': {'family': 'previous_procedure'},
                               'source_candidate': {'family': 'wavlm_augmented'}}) == 'speech'
    assert speech_requirement({'candidate': {'family': 'wavlm_layer', 'representation': 'layer6_nonzero'}}) == 'layer6_nonzero'
