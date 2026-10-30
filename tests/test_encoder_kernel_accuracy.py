import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import importlib.util
from src import encoder_kernel_accuracy as study
from src.modality_diagnostics import nested_folds
from src.batch_inference import load_record, save_record

def assert_same_fitted_predictor(first_path, second_path, probe):
    # Torch pickle bytes contain storage identifiers; compare learned values and
    # behavior on identical inputs, not two independently serialized byte streams.
    with first_path.open('rb') as handle:
        first = pickle.load(handle)
    with second_path.open('rb') as handle:
        second = pickle.load(handle)
    assert first['candidate'] == second['candidate'] and first['fit'] == second['fit']
    name = first['candidate']['feature']
    with study.offline_network():
        np.testing.assert_allclose(first['estimator'].predict_proba(probe[name]),
                                   second['estimator'].predict_proba(probe[name]), rtol=0, atol=1e-14)
    if hasattr(first['estimator'], 'named_steps'):
        for name in ['standardscaler', 'pca']:
            if name in first['estimator'].named_steps:
                a, b = first['estimator'].named_steps[name], second['estimator'].named_steps[name]
                np.testing.assert_array_equal(a.mean_, b.mean_)
                if name == 'pca':
                    np.testing.assert_array_equal(a.components_, b.components_)


@pytest.fixture
def fixture():
    settings=json.loads(Path('configs/encoder_kernel_accuracy.json').read_text())
    settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42], components=[None,3], C_candidates=[.1],
                    expected_jobs=2, expected_fits=30, bootstrap_repetitions=20, cpu_threads=1,
                    svm_C_candidates=[1.], calibration_folds=2, feature_inputs=['semantic','semantic_av'])
    y=np.tile(np.arange(5),10)
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(len(y))],'severity_class':y,'gender':'unknown'})
    rng=np.random.default_rng(9)
    inputs={'semantic':rng.normal(size=(len(y),12)),'semantic_av':rng.normal(size=(len(y),18))}
    return inputs,frame,settings


def test_real_budget_and_fold_local_scaler_pca(fixture):
    x,frame,settings=fixture
    actual=json.loads(Path('configs/encoder_kernel_accuracy.json').read_text())
    assert len(study.candidates(actual))==45 and 15*(45*3+1)==actual['expected_fits']
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
    probe={k:v.copy() for k,v in x.items()}
    fold=nested_folds(frame,settings)[0]
    y=frame.severity_class.to_numpy(copy=True)
    first=study.run_job(x,y,fold,settings,tmp_path/'first')
    for data in x.values():
        data[fold['heldout']]+=999
    y[fold['heldout']]=(y[fold['heldout']]+1)%5
    second=study.run_job(x,y,fold,settings,tmp_path/'second')
    for key in ['candidates','selected_index','selected_candidate']:
        assert first[key]==second[key]
    assert_same_fitted_predictor(tmp_path/'first/model.pkl', tmp_path/'second/model.pkl', probe)


def test_pilot_complete_resume_and_saved_coverage(fixture,tmp_path,monkeypatch):
    x,frame,settings=fixture
    out=tmp_path/'out';args=(x,frame,{},settings,out)
    result=study.run_experiment(*args,max_new_jobs=1)
    assert not result['complete'] and result['completed_fits']==15
    result=study.run_experiment(*args,resume=True)
    assert result['complete'] and result['completed_fits']==30 and result['reused_jobs']==1
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


def test_all_new_families_offline_probability_and_checkpoint(fixture):
    x,frame,settings=fixture
    candidates=[{'family':'svm_rbf','feature':'semantic','C':1.}]
    fold=nested_folds(frame,settings)[0]
    for candidate in candidates:
        with study.offline_network():
            estimator=study.estimator_for(candidate,settings)
            estimator.fit(x['semantic'][fold['fit']],frame.severity_class.iloc[fold['fit']])
            p=estimator.predict_proba(x['semantic'][fold['heldout']])
            restored=pickle.loads(pickle.dumps(estimator))
            q=restored.predict_proba(x['semantic'][fold['heldout']])
        assert p.shape==(25,5) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        np.testing.assert_array_equal(p,q)
        fitted=estimator.calibrated_classifiers_[0].estimator
        assert not fitted.named_steps['svc']._effective_probability
        np.testing.assert_allclose(fitted.named_steps['standardscaler'].mean_,x['semantic'][fold['fit']].mean(axis=0))


def test_calibrated_svm_heldout_invariance(fixture,tmp_path,monkeypatch):
    x,frame,settings=fixture
    probe={k:v.copy() for k,v in x.items()}
    monkeypatch.setattr(study,'candidates',lambda s:[{'family':'svm_rbf','feature':'semantic','C':1.}])
    fold=nested_folds(frame,settings)[0]
    y=frame.severity_class.to_numpy(copy=True)
    with study.offline_network():
        first=study.run_job(x,y,fold,settings,tmp_path/'a')
        x['semantic'][fold['heldout']]+=999
        y[fold['heldout']]=(y[fold['heldout']]+1)%5
        second=study.run_job(x,y,fold,settings,tmp_path/'b')
    for key in ['candidates','selected_index','selected_candidate']:
        assert first[key]==second[key]
    assert_same_fitted_predictor(tmp_path/'a/model.pkl', tmp_path/'b/model.pkl', probe)


def test_network_guard_restores_socket():
    import socket
    before=socket.socket.connect
    with study.offline_network():
        with pytest.raises(RuntimeError,match='Network access forbidden'):
            socket.create_connection(('example.com',443))
    assert socket.socket.connect is before


def test_previous_reference_predictions_are_reported(fixture,tmp_path):
    x,frame,settings=fixture
    folds=nested_folds(frame,settings)
    y=frame.severity_class.to_numpy(int)
    records=[{'seed':f['seed'],'fold':f['fold'],'fit':f['fit'],'heldout':f['heldout'],
              'probabilities':np.eye(5)[y[f['heldout']]].tolist()} for f in folds]
    result=study.run_experiment(x,frame,{'reference_records':records},settings,tmp_path/'out')
    assert result['target']['previous_semantic_accuracy']==1.0
    comparisons=pd.read_csv(tmp_path/'out/paired_comparisons.csv')
    assert set(comparisons.reference)=={'majority','previous_semantic'} and len(comparisons)==6


def test_calibration_support_and_base_fit_counts(fixture,tmp_path):
    x,frame,settings=fixture
    candidate={'family':'svm_rbf','feature':'semantic','C':1.}
    assert study.base_fits(candidate,settings)==3
    assert study.base_fits({'family':'logistic'},settings)==1
    settings['calibration_folds']=10
    with pytest.raises(ValueError,match='support the calibration folds'):
        study.run_experiment(x,frame,{},settings,tmp_path/'bad')
