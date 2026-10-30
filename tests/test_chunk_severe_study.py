import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import chunk_severe_study as study
from src import semantic_severe_repair as baseline


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    old_settings=json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2,inner_folds=2,fold_seeds=[42],components=[None],C_candidates=[.1],interventions=['none'])
    settings=json.loads(Path('configs/chunk_severe_detection.json').read_text())
    settings.update(C=[.1],components=[None,3],expected_jobs=2,inner_pipeline_fits_per_job=16,
                    cpu_threads=1,thresholds=[None,.1,.3,.5])
    y=np.repeat(np.arange(5),[40,20,16,16,8]);rng=np.random.default_rng(52)
    x={name:rng.normal(size=(len(y),dim)) for name,dim in [('semantic',8),('semantic_av',12),('chunk_av',14),('symptom_av',16)]}
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(len(y))],'severity_class':y,'gender':'unknown'})
    folds=baseline.nested_folds(frame,old_settings);root=tmp_path/'baseline';records=[]
    for fold in folds:
        old=baseline.run_job(x,y,fold,old_settings,root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}")
        candidate=old['policies']['accuracy']['candidate'];p=np.full((len(y),5),np.nan)
        for part in fold['inner']:
            model=baseline.estimator_for(candidate,old_settings)
            study.fit(model,x[candidate['feature']][part['fit']],y[part['fit']])
            p[part['heldout']]=study.five_probabilities(model,x[candidate['feature']][part['heldout']])
        old['cached_inner_probabilities']=p[fold['fit']].tolist();records.append(old)
    monkeypatch.setattr(study,'ROOT',root)
    return x,frame,old_settings,settings,records,folds


def test_cached_baseline_and_training_only_preprocessing(fixture,tmp_path,monkeypatch):
    x,frame,old_settings,settings,records,folds=fixture
    original=study.estimator;fitted=[]
    def checked(candidate,*args):
        assert candidate['family']!='baseline'
        model=original(candidate,*args);fitted.append(model);return model
    monkeypatch.setattr(study,'estimator',checked)
    fold=folds[0];result=study.job(x,frame.severity_class.to_numpy(),fold,records[0],settings,old_settings,tmp_path/'job')
    assert result['inner_probabilities'][0]==records[0]['cached_inner_probabilities']
    assert len(fitted)==result['fits']==16+sum(r['candidate']['family']!='baseline' for r in result['selected'].values())
    for i,candidate in enumerate(study.candidates(settings)[1:]):
        for j,part in enumerate(fold['inner']):
            model=fitted[i*2+j]
            assert model.real_count_==len(part['fit']) and model.lineage_==[]
            np.testing.assert_allclose(model.scaler_.mean_,x[candidate['feature']][part['fit']].mean(0))
            if model.pca_ is not None:assert model.pca_.n_samples_==len(part['fit'])


def test_heldout_changes_do_not_change_inner_choices(fixture,tmp_path):
    x,frame,old_settings,settings,records,folds=fixture;y=frame.severity_class.to_numpy(copy=True)
    fold=folds[0];old=copy.deepcopy(records[0])
    first=study.job(x,y,fold,old,settings,old_settings,tmp_path/'first')
    for a in x.values():a[fold['heldout']]+=100
    y[fold['heldout']]=(y[fold['heldout']]+1)%5
    with (study.ROOT/'jobs'/'seed42_fold0'/'accuracy.pkl').open('rb') as f:state=pickle.load(f)
    old['policies']['accuracy']['probabilities']=study.five_probabilities(state['estimator'],x[state['candidate']['feature']][fold['heldout']]).tolist()
    second=study.job(x,y,fold,old,settings,old_settings,tmp_path/'second')
    assert first['inner_probabilities']==second['inner_probabilities'] and first['options']==second['options']
    for policy in settings['policies']:
        for key in ['selection','candidate','threshold','checkpoint_sha256']:
            assert first['selected'][policy][key]==second['selected'][policy][key]


def test_partial_resume_reporting_and_tamper_rejection(fixture,tmp_path,monkeypatch):
    x,frame,old_settings,settings,records,folds=fixture
    out=tmp_path/'study';args=(x,frame,old_settings,records,{},settings,out)
    assert not study.run(*args,max_new_jobs=1)['complete']
    assert study.run(*args,resume=True)['complete']
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
        np.testing.assert_allclose(table[[f'p{k}' for k in range(5)]].sum(axis=1),1)
    saved=x['chunk_av'][0,0];x['chunk_av'][0,0]+=1
    with pytest.raises(ValueError,match='Inputs/code changed'):study.run(*args,resume=True)
    x['chunk_av'][0,0]=saved
    path=next((out/'jobs').glob('*/*.pkl'));path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Checkpoint changed'):study.run(*args,resume=True)


def test_declared_budget():
    settings=json.loads(Path('configs/chunk_severe_detection.json').read_text())
    assert len(study.candidates(settings))==25
    assert (len(study.candidates(settings))-1)*3==settings['inner_pipeline_fits_per_job']==72
    assert all(c.get('intervention')!='severe_interpolation' for c in study.candidates(settings))
