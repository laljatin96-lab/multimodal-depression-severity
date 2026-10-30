import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import external_transfer_accuracy as s


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
def toy(monkeypatch):
    rng = np.random.default_rng(4); y = np.repeat(np.arange(5), [24, 12, 12, 12, 8])
    x = {'semantic_av': rng.normal(size=(len(y), 20)), 'language': rng.normal(size=(len(y), 16))}
    settings = json.loads(s.CONFIG.read_text())
    settings.update(external_masses=[.25,0.], adaptation_scales=[.5], C=[.001], semantic_dimensions=12, text_components=4, av_components=4, expected_jobs=2, cpu_threads=1)
    source_y=np.repeat(np.arange(5),[7,6,5,4,3])
    source={'semantic':rng.normal(size=(len(source_y),12)), 'language':rng.normal(size=(len(source_y),16))}
    monkeypatch.setattr(s,'external_inputs',lambda:(source,source_y,[f'emu:{i}' for i in range(len(source_y))],{}))
    return x, y, settings


def test_training_ownership_domain_weights_and_pickle(toy):
    from src.external_transfer_verify import inspect_model
    x,y,settings=toy
    m=s.ExternalTransferClassifier(s.candidates(settings)[1],settings).fit(x,y)
    inspect_model(m,x,y)
    assert m.external_count==25 and m.real_count==len(y)
    assert np.isclose(m.domain_weight_sums[1]/m.domain_weight_sums[0],.25)
    before=pickle.dumps(m);p=m.predict_proba(x)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    assert pickle.dumps(m)==before
    np.testing.assert_array_equal(pickle.loads(before).predict_proba(x),p)
    np.testing.assert_array_equal(m.predict_proba({**x,'PHQ8_Score':y*5,'severity_class':y}),p)


def test_no_external_ablation_cannot_see_source_values_or_targets(toy):
    x,y,settings=toy
    c=next(c for c in s.candidates(settings)[1:] if c['external_mass']==0)
    a=s.ExternalTransferClassifier(c,settings).fit(x,y)
    source,labels,_,_=s.external_inputs()
    for v in source.values():v[:]+=10000
    labels[:]=(labels+2)%5
    b=s.ExternalTransferClassifier(c,settings).fit(x,y)
    assert a.external_count==b.external_count==0
    assert pickle.dumps(a)==pickle.dumps(b)


def test_transfer_candidates_are_separate_from_ablation_policy():
    settings=json.loads(s.CONFIG.read_text())
    rows=[dict(candidate=c,correct=i,macro_f1=0) for i,c in enumerate(s.candidates(settings))]
    assert rows[s.choose(rows,'accuracy')]['candidate']['external_mass']>0
    assert rows[s.choose(rows,'transfer_only')]['candidate']['external_mass']>0
    assert rows[s.choose(rows,'no_external')]['candidate']['external_mass']==0


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
    assert len(candidates) == 13 and candidates[0] == {'family': 'language_control'}
    assert s.choose([{'correct': 0}, {'correct': 20}], 'language_control') == 0
    assert settings['target_accuracy'] == .54


def test_prediction_does_not_read_source_labels(toy,monkeypatch):
    x,y,settings=toy
    model=s.ExternalTransferClassifier(s.candidates(settings)[1],settings).fit(x,y)
    expected=model.predict_proba(x)
    def forbidden():
        raise AssertionError('External labels must never be read during inference')
    monkeypatch.setattr(s,'external_inputs',forbidden)
    np.testing.assert_array_equal(model.predict_proba(x),expected)
