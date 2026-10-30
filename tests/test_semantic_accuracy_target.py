import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import semantic_accuracy_target as study
from src.modality_diagnostics import nested_folds
from src.batch_inference import load_record, save_record


@pytest.fixture
def fixture():
    settings=json.loads(Path('configs/semantic_accuracy_target.json').read_text())
    settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42], components=[None,3], C_candidates=[.1],
                    expected_jobs=2, expected_fits=22, bootstrap_repetitions=20, cpu_threads=1)
    y=np.tile(np.arange(5),10)
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(len(y))],'severity_class':y,'gender':'unknown'})
    rng=np.random.default_rng(9)
    inputs={'semantic':rng.normal(size=(len(y),12)),'semantic_av':rng.normal(size=(len(y),18))}
    return inputs,frame,settings


def test_real_budget_and_fold_local_scaler_pca(fixture):
    x,frame,settings=fixture
    actual=json.loads(Path('configs/semantic_accuracy_target.json').read_text())
    assert len(study.candidates(actual))==17 and 15*(17*3+1)==actual['expected_fits']
    fold=nested_folds(frame,settings)[0]
    for components in [None,3]:
        candidate={'family':'logistic','feature':'semantic','C':.1,'components':components}
        estimator=study.estimator_for(candidate,settings)
        estimator.fit(x['semantic'][fold['fit']],frame.severity_class.iloc[fold['fit']])
        np.testing.assert_allclose(estimator.named_steps['standardscaler'].mean_,x['semantic'][fold['fit']].mean(axis=0))
        p=estimator.predict_proba(x['semantic'][fold['heldout']])
        assert np.isfinite(p).all() and p.shape==(25,5)
        if components is not None:assert estimator.named_steps['pca'].n_samples_==25


def test_outer_data_cannot_change_selected_pipeline(fixture,tmp_path):
    x,frame,settings=fixture
    fold=nested_folds(frame,settings)[0]
    y=frame.severity_class.to_numpy(copy=True)
    first=study.run_job(x,y,fold,settings,tmp_path/'first')
    for data in x.values():
        data[fold['heldout']]+=999
    y[fold['heldout']]=(y[fold['heldout']]+1)%5
    second=study.run_job(x,y,fold,settings,tmp_path/'second')
    for key in ['candidates','selected_index','selected_candidate','model_sha256']:
        assert first[key]==second[key]


def test_pilot_complete_resume_and_saved_coverage(fixture,tmp_path,monkeypatch):
    x,frame,settings=fixture
    out=tmp_path/'out';args=(x,frame,{},settings,out)
    result=study.run_experiment(*args,max_new_jobs=1)
    assert not result['complete'] and result['completed_fits']==11
    result=study.run_experiment(*args,resume=True)
    assert result['complete'] and result['completed_fits']==22 and result['reused_jobs']==1
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    preds={p:p.read_bytes() for p in (out/'oof').glob('*/predictions.csv')}
    def forbidden(*a,**kw): raise AssertionError('No refitting/inference/deserialization on resume')
    monkeypatch.setattr(study,'run_job',forbidden)
    monkeypatch.setattr(study,'five_probabilities',forbidden)
    monkeypatch.setattr(study.pickle,'load',forbidden)
    result=study.run_experiment(*args,resume=True)
    assert result['new_fits']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    assert preds=={p:p.read_bytes() for p in preds}
    for p in preds:
        table=pd.read_csv(p,dtype={'participant_id':str})
        assert table.participant_id.tolist()==frame.participant_id.tolist() and table.split.eq('train_nested_oof').all()
    with pytest.raises(FileExistsError): study.run_experiment(*args)
    x['semantic'][0,0]+=1
    with pytest.raises(ValueError,match='changed'): study.run_experiment(*args,resume=True)


def test_changed_checkpoint_and_input_callback_rejected(fixture,tmp_path):
    x,frame,settings=fixture
    out=tmp_path/'out';args=(x,frame,{},settings,out)
    study.run_experiment(*args,max_new_jobs=1)
    model=next((out/'jobs').rglob('model.pkl'));model.write_bytes(model.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checkpoint'): study.run_experiment(*args,resume=True)
    def changed(): raise ValueError('inputs changed')
    with pytest.raises(ValueError,match='inputs changed'):
        study.run_experiment(x,frame,{},settings,tmp_path/'other',verify_inputs=changed)
    assert not (tmp_path/'other/completion.json').exists()
