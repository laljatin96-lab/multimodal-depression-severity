import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import text_accuracy_target as study
from src.modality_diagnostics import nested_folds
from src.batch_inference import load_record, save_record


@pytest.fixture
def fixture():
    settings=json.loads(Path('configs/text_accuracy_target.json').read_text())
    settings.update(feature_inputs=['text_word','text_av','av'], outer_folds=2, inner_folds=2, fold_seeds=[42],
                    expected_jobs=2, expected_fits=18, bootstrap_repetitions=20, cpu_threads=1)
    settings['C_candidates']={f:[10.] for f in settings['feature_inputs']}
    settings['text'].update(min_df=1,max_df=1.0,word_max_features=100,char_max_features=100)
    y=np.tile(np.arange(5),10)
    frame=pd.DataFrame({'participant_id':[str(i) for i in range(len(y))],'severity_class':y,'gender':'unknown'})
    words=['walking sunshine','reading literature','cooking recipes','drawing pictures','singing music']
    text=pd.DataFrame({'text':[words[i]+' interview discussion' for i in y]})
    av=pd.DataFrame(np.random.default_rng(9).normal(size=(len(y),128)),columns=[f'av{i}' for i in range(128)])
    inputs={'text_word':text.copy(),'text_char':text.copy(),'text_av':pd.concat([text,av],axis=1),'av':av}
    return inputs,frame,settings


def test_real_budget_and_probability_models(fixture):
    x,frame,settings=fixture
    actual=json.loads(Path('configs/text_accuracy_target.json').read_text())
    assert len(study.candidates(actual))==13 and 15*(13*3+1)==actual['expected_fits']
    for feature in ['text_word','text_char','text_av','av']:
        estimator=study.estimator_for({'family':'logistic','feature':feature,'C':10.},settings)
        estimator.fit(x[feature],frame.severity_class)
        p=estimator.predict_proba(x[feature])
        assert p.shape==(50,5) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1),1)


def test_vocabulary_idf_and_fusion_scaler_use_only_fit_rows(fixture):
    x,frame,settings=fixture
    fold=nested_folds(frame,settings)[0]
    data=x['text_av'].copy()
    data.loc[fold['heldout'],'text']+=' heldoutsecretword'
    estimator=study.estimator_for({'family':'logistic','feature':'text_av','C':10.},settings)
    estimator.fit(data.iloc[fold['fit']],frame.severity_class.iloc[fold['fit']])
    transform=estimator.named_steps['columntransformer']
    vectorizer=transform.named_transformers_['text']
    assert 'heldoutsecretword' not in vectorizer.vocabulary_
    scaler=transform.named_transformers_['av'].named_steps['standardscaler']
    np.testing.assert_allclose(scaler.mean_,x['av'].iloc[fold['fit']].mean(axis=0))
    data.loc[fold['heldout'],'text']='totally different vocabulary'
    estimator2=study.estimator_for({'family':'logistic','feature':'text_av','C':10.},settings)
    estimator2.fit(data.iloc[fold['fit']],frame.severity_class.iloc[fold['fit']])
    other=estimator2.named_steps['columntransformer'].named_transformers_['text']
    assert other.vocabulary_==vectorizer.vocabulary_
    np.testing.assert_array_equal(other.idf_,vectorizer.idf_)


def test_outer_data_cannot_change_selected_pipeline(fixture,tmp_path):
    x,frame,settings=fixture
    fold=nested_folds(frame,settings)[0]
    y=frame.severity_class.to_numpy(copy=True)
    first=study.run_job(x,y,fold,settings,tmp_path/'first')
    for data in x.values():
        if 'text' in data:
            data.loc[fold['heldout'],'text']='unseen heldout text'
        for col in data.select_dtypes(include='number').columns:
            data.loc[fold['heldout'],col]+=999
    y[fold['heldout']]=(y[fold['heldout']]+1)%5
    second=study.run_job(x,y,fold,settings,tmp_path/'second')
    for key in ['candidates','selected_index','selected_candidate','model_sha256']:
        assert first[key]==second[key]


def test_pilot_complete_resume_and_saved_coverage(fixture,tmp_path,monkeypatch):
    x,frame,settings=fixture
    out=tmp_path/'out';args=(x,frame,{},settings,out)
    result=study.run_experiment(*args,max_new_jobs=1)
    assert not result['complete'] and result['completed_fits']==9
    result=study.run_experiment(*args,resume=True)
    assert result['complete'] and result['completed_fits']==18 and result['reused_jobs']==1
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
    x['text_word'].loc[0,'text']+=' changed'
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
