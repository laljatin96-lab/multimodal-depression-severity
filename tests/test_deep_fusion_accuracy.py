import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from src import deep_fusion_accuracy as s


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    old=json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old.update(outer_folds=2,inner_folds=2,fold_seeds=[42])
    settings=json.loads(Path('configs/deep_fusion_accuracy.json').read_text())
    settings.update(C=[.1],deep_components=3,text_av_components=3,expected_jobs=2,inner_head_fits_per_job=16,cpu_threads=1)
    y=np.repeat(np.arange(5),[40,20,16,16,8]);rng=np.random.default_rng(3)
    x={n:rng.normal(size=(100,10)) for n in ['densenet','vgg_audio','resnet_face','semantic','av']}
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(100)],'severity_class':y,'gender':'unknown'})
    folds=s.cached.baseline.nested_folds(frame,old);root=tmp_path/'baseline';records=[]
    monkeypatch.setattr(s,'ROOT',root)
    for fold in folds:
        model=DummyClassifier(strategy='most_frequent').fit(x['semantic'][fold['fit']],y[fold['fit']])
        path=root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl';path.parent.mkdir(parents=True)
        with path.open('wb') as f:pickle.dump({'estimator':model},f)
        records.append({**fold,'cached_inner_probabilities':model.predict_proba(x['semantic'][fold['fit']]).tolist(),
            'policies':{'accuracy':{'candidate':{'feature':'semantic'},'probabilities':model.predict_proba(x['semantic'][fold['heldout']]).tolist()}}})
    return x,frame,old,settings,records,folds


def test_train_only_projector_and_heldout_invariance(fixture,tmp_path):
    x,frame,old,settings,records,folds=fixture;fold=folds[0];y=frame.severity_class.to_numpy(copy=True)
    projector=s.Projector(settings).fit(x,fold['fit'])
    for name,p in projector.transforms.items():
        np.testing.assert_allclose(p.steps[0][1].mean_,x[name][fold['fit']].mean(0))
        assert p.steps[1][1].n_samples_==len(fold['fit'])
    first=s.job(x,y,fold,records[0],settings,tmp_path/'first')
    for a in x.values():a[fold['heldout']]+=100
    y[fold['heldout']]=(y[fold['heldout']]+1)%5
    second=s.job(x,y,fold,records[0],settings,tmp_path/'second')
    assert first['inner_probabilities']==second['inner_probabilities'] and first['options']==second['options']
    for policy in settings['policies']:
        for key in ['selection','candidate','checkpoint_sha256']:assert first['selected'][policy][key]==second['selected'][policy][key]


def test_resume_parity_metrics_and_corruption(fixture,tmp_path,monkeypatch):
    x,frame,old,settings,records,folds=fixture;out=tmp_path/'run';args=(x,frame,old,records,{},settings,out)
    assert not s.run(*args,max_new_jobs=1)['complete']
    assert s.run(*args,resume=True)['complete']
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a,**kw):raise AssertionError('No refit or inference')
    monkeypatch.setattr(s,'job',forbidden);monkeypatch.setattr(s,'predict_state',forbidden);monkeypatch.setattr(s.pickle,'load',forbidden)
    assert s.run(*args,resume=True)['new_head_fits']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    for p in (out/'oof').glob('*/predictions.csv'):
        t=pd.read_csv(p);assert len(t)==100 and t.participant_id.is_unique
        np.testing.assert_allclose(t[[f'p{k}' for k in range(5)]].sum(axis=1),1)
    p=next((out/'jobs').glob('*/*.pkl'));p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Checkpoint changed'):s.run(*args,resume=True)


def test_budget_and_fusion():
    settings=json.loads(Path('configs/deep_fusion_accuracy.json').read_text())
    assert len(s.candidates(settings))==33
    assert (len(s.candidates(settings))-1)*3==settings['inner_head_fits_per_job']==96
    p=np.array([[.6,.4,0,0,0],[.2,.8,0,0,0]])
    inner=[p.tolist()]*33;options=s.search_options(inner,np.array([0,1]),settings)
    assert len(options)==129 and s.choose(options,'accuracy_fusion')['index']==0
    assert s.choose(options,'single_model')['index']==0
