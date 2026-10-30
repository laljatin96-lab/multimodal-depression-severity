import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import pairwise_accuracy as s


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
    settings.update(components=4, C=[.001], expected_jobs=2, cpu_threads=1)
    return x, y, settings


def test_fitting_statistics_and_serialization(toy):
    x, y, settings = toy
    m = s.PairwiseClassifier(s.candidates(settings)[1], settings).fit(x, y)
    np.testing.assert_allclose(m.base_scaler.mean_, x['semantic_av'].mean(0))
    np.testing.assert_allclose(m.language_scaler.mean_, x['language'].mean(0))
    assert m.real_count == m.base_pca.n_samples_ == len(y)
    before = pickle.dumps(m); p = m.predict_proba(x)
    assert pickle.dumps(m) == before
    np.testing.assert_allclose(pickle.loads(before).predict_proba(x), p, atol=1e-14)
    assert m.diagnostics(x,y)['converged']
    changed={**x,'severity_class':np.full(len(y),4),'PHQ8_Score':np.full(len(y),24),'participant_id':np.arange(len(y))}
    np.testing.assert_array_equal(m.predict_proba(changed),p)


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
    assert len(candidates) == 3 and candidates[0] == {'family': 'language_control'}
    assert s.choose([{'correct': 0}, {'correct': 20}], 'language_control') == 0
    assert settings['target_accuracy'] == .60



def test_log_odds_projection_exact_for_coherent_probabilities():
    p=np.asarray([[.5,.2,.1,.15,.05],[.01,.02,.1,.27,.6]])
    logits=np.column_stack([np.log(p[:,j]/p[:,i]) for i,j in s.PAIRS])
    np.testing.assert_allclose(s.combine_logits(logits),p,rtol=0,atol=1e-14)
    np.testing.assert_allclose(s.combine_logits(np.zeros((1,10))),np.full((1,5),.2),atol=1e-14)
    with pytest.raises(ValueError):s.combine_logits(np.zeros((1,9)))


def test_pair_heads_use_only_their_classes(toy):
    x,y,settings=toy;m=s.PairwiseClassifier(s.candidates(settings)[1],settings).fit(x,y)
    for (i,j),head,positions,weights in zip(s.PAIRS,m.heads,m.pair_positions,m.pair_weights):
        np.testing.assert_array_equal(positions,np.flatnonzero((y==i)|(y==j)))
        assert set(y[positions])=={i,j} and np.isclose(weights.sum(),len(positions))
        assert np.array_equal(head.classes_,[0,1])

def test_checked_inference_rejects_cross_participant_files_before_io():
    from src.checked_inference import validate_identity, predict
    names = ['487_Transcript.csv','487_OpenSMILE2.3.0_mfcc.csv','487_OpenFace2.1.0_Pose_gaze_AUs.csv']
    assert validate_identity(*names) == '487'
    names[1] = names[1].replace('487','306')
    with pytest.raises(ValueError,match='participant IDs differ'):
        predict('/does/not/exist.json',*names)
    with pytest.raises(ValueError,match='participant-prefixed'):
        validate_identity('unknown.csv',*names[1:])
