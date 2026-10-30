import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import ensemble_accuracy_target as study
from src.batch_inference import load_record, save_record
from src.modality_diagnostics import nested_folds

@pytest.fixture
def fixture():
    settings=json.loads(Path('configs/ensemble_accuracy_target.json').read_text())
    settings.update(feature_inputs=['cached_visual'], outer_folds=2, inner_folds=2, fold_seeds=[42], expected_jobs=2, expected_fits=14, bootstrap_repetitions=20, cpu_threads=1)
    settings['extra_trees'].update(n_estimators=8, min_samples_leaf=[1], max_features=[1.0])
    settings['hist_gradient_boosting'].update(max_iter=5, min_samples_leaf=2, max_depth=[2], l2_regularization=[1.0])
    y=np.tile(np.arange(5),10)
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(len(y))], 'severity_class':y, 'gender':'unknown'})
    x=np.random.default_rng(9).normal(size=(len(y),8))
    x[:,0]=y*10
    return {'cached_visual':x},frame,settings

def test_candidate_space_and_accuracy_first_ties(fixture):
    _,_,settings=fixture
    assert len(study.candidates(settings))==3
    assert study.chosen_index([{'correct':2,'macro_f1':.9},{'correct':3,'macro_f1':.2},{'correct':3,'macro_f1':.2}])==1
    config=json.loads(Path('configs/ensemble_accuracy_target.json').read_text())
    assert len(study.candidates(config))==33
    assert 15*(33*3+1)==config['expected_fits']

def test_outer_data_cannot_change_model_or_inner_selection(fixture,tmp_path):
    x,frame,settings=fixture
    fold=nested_folds(frame,settings)[0]
    y=frame.severity_class.to_numpy(copy=True)
    first=study.run_job(x,y,fold,settings,tmp_path/'first')
    x['cached_visual'][fold['heldout']]+=999
    y[fold['heldout']]=(y[fold['heldout']]+1)%5
    second=study.run_job(x,y,fold,settings,tmp_path/'second')
    for key in ['candidates','selected_index','selected_candidate','model_sha256']:
        assert first[key]==second[key]

def test_pilot_resume_coverage_target_and_preservation(fixture,tmp_path,monkeypatch):
    x,frame,settings=fixture
    out=tmp_path/'out'
    args=(x,frame,{},settings,out)
    pilot=study.run_experiment(*args,max_new_jobs=1)
    assert not pilot['complete'] and pilot['completed_fits']==7
    assert not (out/'summary.csv').exists()
    complete=study.run_experiment(*args,resume=True)
    assert complete['complete'] and complete['completed_fits']==14 and complete['reused_jobs']==1
    assert complete['target']['internal_target_reached']==(complete['target']['observed_nested_accuracy']>=.55)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a,**kw): raise AssertionError('No refits/inference/deserialization on completed resume')
    monkeypatch.setattr(study,'run_job',forbidden)
    monkeypatch.setattr(study,'five_probabilities',forbidden)
    monkeypatch.setattr(study.pickle,'load',forbidden)
    result=study.run_experiment(*args,resume=True)
    assert result['new_fits']==0 and result['reused_jobs']==2
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    for path in (out/'oof').glob('*/predictions.csv'):
        saved=pd.read_csv(path,dtype={'participant_id':str})
        assert saved.participant_id.tolist()==frame.participant_id.tolist()
        assert saved.split.eq('train_nested_oof').all()
    with pytest.raises(FileExistsError): study.run_experiment(*args)
    x['cached_visual'][0,0]+=1
    with pytest.raises(ValueError,match='changed'): study.run_experiment(*args,resume=True)

@pytest.mark.parametrize('kind',['checkpoint','choice'])
def test_corruption_rejected(fixture,tmp_path,kind):
    x,frame,settings=fixture
    out=tmp_path/'out'
    args=(x,frame,{},settings,out)
    study.run_experiment(*args,max_new_jobs=1)
    path=next((out/'jobs').rglob('result.json'))
    if kind=='checkpoint': path.with_name('model.pkl').write_bytes(b'changed')
    else:
        context=json.loads((out/'protocol.json').read_text())['context_hash']
        record=load_record(path,context)
        record['selected_index']=99
        save_record(path,record,context)
    with pytest.raises(ValueError): study.run_experiment(*args,resume=True)

def test_input_change_prevents_completion(fixture,tmp_path):
    x,frame,settings=fixture
    def changed(): raise ValueError('inputs changed')
    with pytest.raises(ValueError,match='inputs changed'):
        study.run_experiment(x,frame,{},settings,tmp_path/'out',verify_inputs=changed)
    assert not (tmp_path/'out/completion.json').exists()
