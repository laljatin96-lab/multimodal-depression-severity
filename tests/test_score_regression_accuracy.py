import itertools
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from src import score_regression_accuracy as s


def test_ordered_cut_solver_matches_exhaustive_search():
    rng=np.random.default_rng(22);values=rng.uniform(0,24,40);y=rng.integers(0,5,40)
    settings={'cut_radius':.5,'cut_step':.5,'minimum_cut_gap':1.}
    grid=[np.arange(c-.5,c+1,.5) for c in s.CUTS]
    choices=list(itertools.product(*grid))
    expected=max(choices,key=lambda c:(int((s.decode(values,c)==y).sum()),-sum(abs(np.asarray(c)-s.CUTS)),tuple(-v for v in c)))
    assert s.optimize_cuts(values,y,settings)==list(expected)
    assert s.decode(np.array([0,4,5,9,10,14,15,19,20,24]),s.CUTS).tolist()==[0,0,1,1,2,2,3,3,4,4]
    with pytest.raises(ValueError):s.decode(np.array([np.nan]),s.CUTS)


def test_kernels_are_valid_and_fitting_only_scaling():
    rng=np.random.default_rng(2);x={n:rng.normal(size=(12,5)) for n in ['semantic','av','densenet','vgg_audio','resnet_face']}
    fit=list(range(8));scaler=s.BlockScaling().fit(x,fit)
    for n,m in scaler.scalers.items():np.testing.assert_allclose(m.mean_,x[n][fit].mean(0))
    a=s.matrix(scaler.transform(x,fit),'deep_av_text')
    for kind in ['linear','rbf']:
        k=s.kernel(a,a,kind);np.testing.assert_allclose(k,k.T)
        assert np.linalg.eigvalsh(k).min()>-1e-10
    np.testing.assert_allclose(np.diag(s.kernel(a,a,'rbf')),1)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    old=json.loads(Path('configs/semantic_severe_repair.json').read_text());old.update(outer_folds=2,inner_folds=2,fold_seeds=[42])
    settings=json.loads(Path('configs/score_regression_accuracy.json').read_text())
    settings.update(feature_inputs=['semantic','deep_av_text'],alpha=[1.],expected_jobs=2,inner_regressor_fits_per_job=8,cpu_threads=1)
    y=np.repeat(np.arange(5),[40,20,16,16,8]);totals=y.astype(float)*5+2;rng=np.random.default_rng(3)
    x={n:rng.normal(size=(100,10)) for n in ['semantic','av','densenet','vgg_audio','resnet_face']}
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(100)],'severity_class':y,'gender':'unknown'})
    folds=s.prior.cached.baseline.nested_folds(frame,old);root=tmp_path/'baseline';records=[];monkeypatch.setattr(s.prior,'ROOT',root)
    for fold in folds:
        model=DummyClassifier(strategy='most_frequent').fit(x['semantic'][fold['fit']],y[fold['fit']])
        p=root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl';p.parent.mkdir(parents=True)
        with p.open('wb') as f:pickle.dump({'estimator':model},f)
        records.append({**fold,'cached_inner_probabilities':model.predict_proba(x['semantic'][fold['fit']]).tolist(),'policies':{'accuracy':{'candidate':{'feature':'semantic'},'probabilities':model.predict_proba(x['semantic'][fold['heldout']]).tolist()}}})
    return x,frame,totals,old,settings,records,folds


def test_no_heldout_score_class_or_feature_influence(fixture,tmp_path):
    x,frame,totals,old,settings,records,folds=fixture;fold=folds[0]
    first=s.job(x,frame,totals,fold,records[0],settings,tmp_path/'first')
    for a in x.values():a[fold['heldout']]+=100
    frame.loc[fold['heldout'],'severity_class']=(frame.loc[fold['heldout'],'severity_class']+1)%5
    totals[fold['heldout']]=24-totals[fold['heldout']]
    second=s.job(x,frame,totals,fold,records[0],settings,tmp_path/'second')
    assert first['inner_score_predictions']==second['inner_score_predictions'] and first['options']==second['options']
    for policy in settings['policies']:
        for key in ['selection_index','candidate','cuts','checkpoint_sha256']:assert first['selected'][policy][key]==second['selected'][policy][key]


def test_complete_resume_true_targets_and_tamper(fixture,tmp_path,monkeypatch):
    x,frame,totals,old,settings,records,folds=fixture;out=tmp_path/'run';args=(x,frame,totals,old,records,{},settings,out)
    assert not s.run(*args,max_new_jobs=1)['complete'];assert s.run(*args,resume=True)['complete']
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a,**kw):raise AssertionError('No fitting or inference')
    monkeypatch.setattr(s,'job',forbidden);monkeypatch.setattr(s,'predict_state',forbidden);monkeypatch.setattr(s.pickle,'load',forbidden)
    assert s.run(*args,resume=True)['new_regressor_fits']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    for p in (out/'oof').glob('*/predictions.csv'):
        t=pd.read_csv(p);assert t.participant_id.is_unique and len(t)==100
        assert 'predicted_phq8' in t and 'p0' not in t
    saved=totals[0];totals[0]=24
    with pytest.raises(ValueError,match='PHQ'):s.run(*args,resume=True)
    totals[0]=saved;p=next((out/'jobs').glob('*/*.pkl'));p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Checkpoint changed'):s.run(*args,resume=True)


def test_budget():
    settings=json.loads(Path('configs/score_regression_accuracy.json').read_text())
    assert len(s.candidates(settings))==25 and (len(s.candidates(settings))-1)*3==settings['inner_regressor_fits_per_job']==72
