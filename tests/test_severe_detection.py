import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import severe_detection_study as study
from src import semantic_severe_repair as baseline
from src.modality_diagnostics import nested_folds


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    old_settings=json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2,inner_folds=2,fold_seeds=[42],components=[None],C_candidates=[.1],interventions=['none'])
    settings=json.loads(Path('configs/severe_detection.json').read_text())
    settings.update(C=[.1],severe_weights=[2],ordinal_weight_powers=[.5],components=3,
                    expected_jobs=2,inner_pipeline_fits_per_job=10,cpu_threads=1,thresholds=[None,.1,.3,.5])
    y=np.repeat(np.arange(5),[40,20,16,16,8]);rng=np.random.default_rng(52)
    x={'semantic':rng.normal(size=(len(y),8)),'semantic_av':rng.normal(size=(len(y),12))}
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(len(y))],'severity_class':y,'gender':'unknown'})
    folds=nested_folds(frame,old_settings);root=tmp_path/'baseline';records=[]
    for fold in folds:
        records.append(baseline.run_job(x,y,fold,old_settings,root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"))
    monkeypatch.setattr(study,'ROOT',root)
    return x,frame,old_settings,settings,records,folds


@pytest.mark.parametrize('family,weight',[('weighted',4),('ordinal',0),('ordinal',.5)])
def test_ordinal_and_weighted_probabilities_preprocessing_and_pickle(family,weight):
    y=np.repeat(np.arange(5),10);x=np.random.default_rng(2).normal(size=(50,8))
    model=study.SeverityClassifier(family,.1,3,weight).fit(x,y)
    assert model.scaler_.n_samples_seen_==50 and model.pca_.n_samples_==50
    np.testing.assert_allclose(model.scaler_.mean_,x.mean(0))
    p=model.predict_proba(x)
    assert p.shape==(50,5) and (p>=0).all()
    np.testing.assert_allclose(p.sum(1),1)
    np.testing.assert_array_equal(p,pickle.loads(pickle.dumps(model)).predict_proba(x))
    assert len(model.models_)==(4 if family=='ordinal' else 1)


def test_thresholds_change_decisions_not_probabilities():
    p=np.array([[.45,.2,.1,.1,.15],[.1,.1,.1,.1,.6]])
    before=p.copy()
    assert study.decide(p,None).tolist()==[0,4]
    assert study.decide(p,.1).tolist()==[4,4]
    assert study.decide(p,.7).tolist()==[0,0]
    np.testing.assert_array_equal(p,before)
    with pytest.raises(ValueError):study.decide(p,-.1)


def test_selection_severe_priority_harm_bounds_and_fallback():
    y=np.array([0]*50+[1]*20+[2]*15+[3]*10+[4]*5)
    base=np.zeros(100,int);options=[]
    for true_positive,false_positive in [(0,0),(3,2),(5,10)]:
        pred=base.copy();pred[-true_positive:]=4 if true_positive else 0
        if true_positive==0:pred=base.copy()
        pred[:false_positive]=4
        options.append(study.score(y,pred))
    selected=study.select(options,options[0],{'max_accuracy_loss':.05,'max_severe_fpr':.05})
    assert selected['index']==1
    impossible=[{**options[0],'accuracy':0,'severe_fpr':1}]
    assert not study.select(impossible,options[0],{'max_accuracy_loss':0,'max_severe_fpr':0})['constraints_met']


def test_outer_heldout_cannot_influence_inner_selection(fixture,tmp_path):
    x,frame,old_settings,settings,records,folds=fixture;y=frame.severity_class.to_numpy(copy=True)
    fold=folds[0];old=copy.deepcopy(records[0])
    first=study.job(x,y,fold,old,settings,old_settings,tmp_path/'first')
    for a in x.values():a[fold['heldout']]+=100
    y[fold['heldout']]=(y[fold['heldout']]+1)%5
    with (study.ROOT/'jobs'/'seed42_fold0'/'accuracy.pkl').open('rb') as f:state=pickle.load(f)
    old['policies']['accuracy']['probabilities']=state['estimator'].predict_proba(x[state['candidate']['feature']][fold['heldout']]).tolist()
    second=study.job(x,y,fold,old,settings,old_settings,tmp_path/'second')
    assert first['inner_probabilities']==second['inner_probabilities'] and first['options']==second['options']
    for name in settings['policies']:
        for key in ['selection','candidate','threshold','checkpoint_sha256']:
            assert first['selected'][name][key]==second['selected'][name][key]


def test_resume_reporting_and_tamper_rejection(fixture,tmp_path,monkeypatch):
    x,frame,old_settings,settings,records,folds=fixture
    out=tmp_path/'study';args=(x,frame,old_settings,records,{},settings,out)
    first=study.run(*args,max_new_jobs=1);assert not first['complete']
    complete=study.run(*args,resume=True);assert complete['complete']
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    predictions={p:p.read_bytes() for p in (out/'oof').glob('*/predictions.csv')}
    def forbidden(*a,**kw):raise AssertionError('No fitting/prediction/deserialization')
    monkeypatch.setattr(study,'job',forbidden);monkeypatch.setattr(study,'five_probabilities',forbidden)
    monkeypatch.setattr(study.pickle,'load',forbidden)
    assert study.run(*args,resume=True)['new_fits']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    assert predictions=={p:p.read_bytes() for p in predictions}
    for p in predictions:
        table=pd.read_csv(p,dtype={'participant_id':str})
        assert table.participant_id.tolist()==frame.participant_id.tolist()
        assert table.split.eq('train_nested_oof').all()
        np.testing.assert_allclose(table[[f'p{k}' for k in range(5)]].sum(1),1)
    path=next((out/'jobs').glob('*/*.pkl'));path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Checkpoint changed'):study.run(*args,resume=True)


def test_budget():
    settings=json.loads(Path('configs/severe_detection.json').read_text())
    assert len(study.candidates(settings))==17
    assert len(study.candidates(settings))*3==settings['inner_pipeline_fits_per_job']==51
