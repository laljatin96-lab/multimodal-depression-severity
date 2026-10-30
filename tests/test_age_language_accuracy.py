import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import age_language_accuracy as s


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):
        yield


def test_fixed_language_features_preserve_negation_and_pronouns():
    values = dict(zip(s.LANGUAGE_NAMES, s.language_features("i don't feel happy i feel tired")))
    assert values['first_singular'] == 2/7
    assert values['negation'] == 1/7 and values['sleep_energy'] == 1/7
    assert len(values) == 16 and np.isfinite(list(values.values())).all()
    with pytest.raises(ValueError, match='Nonempty'):
        s.language_features('')


@pytest.fixture
def toy():
    rng = np.random.default_rng(9); y = np.repeat(np.arange(5), [24,12,12,12,8])
    n = len(y); base = rng.normal(size=(n, 12)); age = rng.normal(size=(n,3)); language = rng.normal(size=(n,16))
    x = {'base': base, 'semantic': base, 'semantic_av': base, 'age': age, 'language': language,
         'age_language': np.column_stack([age,language])}
    settings = json.loads(Path('configs/age_language_accuracy.json').read_text())
    settings.update(block_scales=[.5], C=[.001], weight_powers=[0.], components=4, expected_jobs=2, cpu_threads=1)
    return x, y, settings


def test_fold_local_preprocessing_and_missing_age(toy):
    x,y,settings = toy; x['age'][0,0] = np.nan; x['age_language'][0,0] = np.nan
    c = s.candidates(settings)[1]; m = s.BlockClassifier(c,settings).fit(x,y)
    assert m.real_count == len(y) and m.projection.real_count == len(y)
    np.testing.assert_allclose(m.projection.base_scaler.mean_,x['base'].mean(0))
    assert m.projection.imputers['age'].statistics_[0] == np.nanmedian(x['age'][:,0])
    before = pickle.dumps(m); p = m.predict_proba(x)
    assert p.shape == (len(y),5) and np.allclose(p.sum(1),1)
    assert pickle.dumps(m) == before
    np.testing.assert_allclose(pickle.loads(before).predict_proba(x), p, rtol=0, atol=1e-14)


def test_language_only_inference_needs_no_age(toy):
    x,y,settings = toy; c = next(c for c in s.candidates(settings)[1:] if c['feature_set']=='language')
    m = s.BlockClassifier(c,settings).fit(x,y)
    np.testing.assert_allclose(m.predict_proba({'base':x['base'],'language':x['language']}), m.predict_proba(x), rtol=0, atol=1e-14)


@pytest.fixture
def experiment(toy,tmp_path,monkeypatch):
    x,y,settings=toy
    old_settings=json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2,inner_folds=2,fold_seeds=[42])
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(len(y))], 'severity_class':y,'gender':'unknown','split':'train'})
    folds=s.nested_folds(frame,old_settings);old=[];root=tmp_path/'baseline';monkeypatch.setattr(s,'ROOT',root)
    for f in folds:
        model=DummyClassifier(strategy='prior').fit(x['base'][f['fit']],y[f['fit']])
        directory=root/'jobs'/f"seed{f['seed']}_fold{f['fold']}";directory.mkdir(parents=True)
        (directory/'accuracy.pkl').write_bytes(pickle.dumps({'estimator':model}))
        old.append({**f,'candidates':[{'predictions':model.predict(x['base'][f['fit']]).tolist()}],
          'policies':{'accuracy':{'selection':{'index':0},'candidate':{'feature':'semantic_av'},
            'probabilities':model.predict_proba(x['base'][f['heldout']]).tolist()}}})
    return (x,frame,old_settings,old,{'bindings':{}}),settings,folds


def test_heldout_age_language_targets_cannot_change_fitted_models(experiment,tmp_path):
    inputs,settings,folds=experiment;x,frame,_,old,_=inputs;f=folds[0]
    a=s.job(x,frame,f,old[0],settings,tmp_path/'a')
    frame.loc[f['heldout'],'severity_class']=(frame.loc[f['heldout'],'severity_class']+1)%5
    for name in ['age','language','age_language']:
        x[name][f['heldout']] += 1000
    b=s.job(x,frame,f,old[0],settings,tmp_path/'b')
    assert a['options']==b['options'] and a['inner_probabilities']==b['inner_probabilities']
    for name in settings['policies']:
        assert a['selected'][name]['checkpoint_sha256']==b['selected'][name]['checkpoint_sha256']


def test_completed_resume_zero_fit_and_tamper_detection(experiment,tmp_path,monkeypatch):
    inputs,settings,folds=experiment;out=tmp_path/'study'
    assert not s.run(inputs,settings,out,max_new_jobs=1)['complete']
    assert s.run(inputs,settings,out,resume=True)['complete']
    before={p:[p.read_bytes(),p.stat().st_mtime_ns] for p in (out/'jobs').glob('*/*')}
    def forbidden(*a,**kw):
        raise AssertionError('No real fits/inference during completed resume')
    monkeypatch.setattr(s,'job',forbidden);monkeypatch.setattr(s,'predict_state',forbidden)
    assert s.run(inputs,settings,out,resume=True)['new_fits']==0
    assert all([p.read_bytes(),p.stat().st_mtime_ns]==v for p,v in before.items())
    with pytest.raises(ValueError,match='Frozen inputs/code'):
        s.run(inputs,dict(settings,fit_seed=7),out,resume=True)
    path=next((out/'jobs').glob('*/*.pkl'));path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='Checkpoint changed'):
        s.run(inputs,settings,out,resume=True)


def test_prespecified_grid():
    settings=json.loads(Path('configs/age_language_accuracy.json').read_text())
    assert len(s.candidates(settings))==55
    assert settings['target_accuracy']>.51


def test_final_choice_ignores_outer_scores(experiment,tmp_path):
    from src.age_language_finalize import training_choices
    inputs,settings,folds=experiment;x,frame,_,old,_=inputs
    records=[s.job(x,frame,f,r,settings,tmp_path/f"fold{f['fold']}") for f,r in zip(folds,old)]
    original=training_choices(frame,records,old,settings)
    for r in records:
        r['selected']={'corrupted_outer_values':123};r['options']=[]
    for r in old:
        r['policies']['accuracy']['probabilities']=[]
    assert training_choices(frame,records,old,settings)==original
    assert original['inner_predictions_per_participant']==1
