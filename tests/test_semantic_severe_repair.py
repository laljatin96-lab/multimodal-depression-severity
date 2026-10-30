import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import semantic_severe_repair as study
from src import semantic_accuracy_target as previous
from src.batch_inference import load_record, save_record
from src.modality_diagnostics import nested_folds


@pytest.fixture
def fixture():
    settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    settings.update(outer_folds=2,inner_folds=2,fold_seeds=[42],components=[None,3],C_candidates=[.1],
                    expected_jobs=2,expected_fits=72,bootstrap_repetitions=20,cpu_threads=1)
    y = np.repeat(np.arange(5),[30,20,16,12,8]); rng = np.random.default_rng(73)
    x = {'semantic':rng.normal(size=(len(y),8)), 'semantic_av':rng.normal(size=(len(y),12))}
    frame = pd.DataFrame({'participant_id':[str(i) for i in range(len(y))],'severity_class':y,'gender':'unknown'})
    return x,frame,settings


def test_declared_budget_and_matching_original_candidates():
    settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    candidates = study.candidates(settings)
    assert len(candidates) == 65 and 15*(65*3+2) == settings['expected_fits'] == 2955
    original = previous.candidates(json.loads(Path('configs/semantic_accuracy_target.json').read_text()))
    assert [{k:v for k,v in c.items() if k != 'intervention'} for c in candidates[:17]] == original


def test_interpolation_is_deterministic_bounded_and_only_real_severe_parents():
    y = np.repeat(np.arange(5),[10,8,6,9,3]); x = np.arange(len(y)*4,dtype=float).reshape(-1,4)
    a,b,lineage = study.interpolate_severe(x,y,42)
    c,d,again = study.interpolate_severe(x,y,42)
    np.testing.assert_array_equal(a,c); np.testing.assert_array_equal(b,d); assert lineage == again
    np.testing.assert_array_equal(a[:len(y)],x); np.testing.assert_array_equal(b[:len(y)],y)
    assert len(lineage) == 6 and (b[len(y):] == 4).all()
    for value,item in zip(a[len(y):],lineage):
        i,j,w = item['parent_a'],item['parent_b'],item['fraction']
        assert i != j and y[i] == y[j] == 4 and 0 <= w < 1
        np.testing.assert_allclose(value,x[i]+w*(x[j]-x[i]))
    with pytest.raises(ValueError,match='two real severe'):study.interpolate_severe(x[:34],y[:34],42)


@pytest.mark.parametrize('mode',['none','sqrt_weight','balanced_weight','severe_interpolation'])
def test_preprocessing_uses_real_fitting_data_and_effective_weight_is_fixed(fixture,mode):
    x,frame,settings = fixture; y = frame.severity_class.to_numpy()
    model = study.ImbalanceLogistic(C=.1,components=3,intervention=mode).fit(x['semantic'],y)
    np.testing.assert_allclose(model.scaler_.mean_,x['semantic'].mean(0))
    assert model.scaler_.n_samples_seen_ == len(y) and model.pca_.n_samples_ == len(y)
    assert model.total_weight_ == pytest.approx(len(y))
    if mode == 'balanced_weight':np.testing.assert_allclose(model.effective_class_weights_,len(y)/5)
    assert (len(model.lineage_) > 0) == (mode == 'severe_interpolation')
    receipt = study.fit_receipt(model,{'family':'logistic'})
    study.validate_receipt(receipt,{'family':'logistic','intervention':mode},y,settings)
    p = model.predict_proba(x['semantic']); assert p.shape == (len(y),5)
    np.testing.assert_allclose(p.sum(1),1); np.testing.assert_array_equal(model.predict(x['semantic']),p.argmax(1))


def test_unweighted_control_matches_original_pipeline(fixture):
    x,frame,settings = fixture; y = frame.severity_class.to_numpy()
    for components in [None,3]:
        candidate = {'family':'logistic','feature':'semantic','components':components,'C':.1,'intervention':'none'}
        new = study.estimator_for(candidate,settings).fit(x['semantic'],y)
        old = previous.estimator_for(candidate,settings).fit(x['semantic'],y)
        np.testing.assert_allclose(new.predict_proba(x['semantic']),old.predict_proba(x['semantic']),rtol=0,atol=1e-12)


def test_original_control_discrepancy_blocks_completion():
    candidate = {'family':'logistic','feature':'semantic','components':3,'C':.1}
    old = {'candidates':[{'candidate':candidate,'predictions':[0,1,4]}]}
    new = {'candidates':[{'candidate':{**candidate,'intervention':'none'},'predictions':[0,1,4]}]}
    study.validate_controls(new,old)
    new['candidates'][0]['predictions'][0] = 1
    with pytest.raises(ValueError,match='Matched control predictions'):study.validate_controls(new,old)


def test_severe_gate_requires_true_detection_and_explicit_fallback():
    settings = {'minimum_severe_recall':.2,'minimum_severe_precision':.2}
    scores = [dict(correct=90,macro_f1=.2,severe_recall=0,severe_precision=0),
              dict(correct=82,macro_f1=.3,severe_recall=.4,severe_precision=.25),
              dict(correct=89,macro_f1=.3,severe_recall=.4,severe_precision=.1)]
    assert study.choose(scores,'accuracy',settings)['index'] == 0
    assert study.choose(scores,'severe_aware',settings)['index'] == 1
    scores[1]['severe_recall'] = 0
    result = study.choose(scores,'severe_aware',settings)
    assert result['index'] == 0 and result['fallback_used'] and not result['gate_satisfied']


def test_outer_data_cannot_change_inner_selection_augmentation_or_fitted_state(fixture,tmp_path):
    x,frame,settings = fixture; y = frame.severity_class.to_numpy(copy=True)
    fold = nested_folds(frame,settings)[0]
    first = study.run_job(x,y,fold,settings,tmp_path/'first')
    probes = {k:v[:5].copy() for k,v in x.items()}
    for data in x.values():data[fold['heldout']] += 1000
    y[fold['heldout']] = (y[fold['heldout']]+1)%5
    second = study.run_job(x,y,fold,settings,tmp_path/'second')
    assert first['candidates'] == second['candidates']
    for name in settings['policies']:
        for key in ['candidate','selection','fit_receipt','model_sha256']:
            assert first['policies'][name][key] == second['policies'][name][key]
        with (tmp_path/'first'/f'{name}.pkl').open('rb') as f:a = pickle.load(f)
        with (tmp_path/'second'/f'{name}.pkl').open('rb') as f:b = pickle.load(f)
        probe = probes[a['candidate']['feature']]
        np.testing.assert_array_equal(a['estimator'].predict_proba(probe),b['estimator'].predict_proba(probe))


def test_pilot_resume_no_fit_and_real_only_evaluation(fixture,tmp_path,monkeypatch):
    x,frame,settings = fixture; out = tmp_path/'out'; args = (x,frame,{},settings,out)
    first = study.run_experiment(*args,max_new_jobs=1)
    assert not first['complete'] and first['new_fits'] == 36
    done = study.run_experiment(*args,resume=True)
    assert done['complete'] and done['completed_fits'] == 72 and done['reused_jobs'] == 1
    files = {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    predictions = {p:p.read_bytes() for p in (out/'oof').glob('*/predictions.csv')}
    def forbidden(*a,**kw):raise AssertionError('No repeat fitting/prediction/deserialization')
    monkeypatch.setattr(study,'run_job',forbidden); monkeypatch.setattr(study,'five_probabilities',forbidden)
    monkeypatch.setattr(study.pickle,'load',forbidden)
    resumed = study.run_experiment(*args,resume=True)
    assert resumed['new_fits'] == 0 and resumed['reused_jobs'] == 2
    assert files == {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
    assert predictions == {p:p.read_bytes() for p in predictions}
    for path in predictions:
        data = pd.read_csv(path,dtype={'participant_id':str})
        assert data.participant_id.tolist() == frame.participant_id.tolist()
        assert data.split.eq('train_nested_oof').all()
    with pytest.raises(FileExistsError):study.run_experiment(*args)
    x['semantic'][0,0] += 1
    with pytest.raises(ValueError,match='changed'):study.run_experiment(*args,resume=True)


def test_corrupted_synthetic_lineage_checkpoint_and_callback_rejected(fixture,tmp_path):
    x,frame,settings = fixture; out = tmp_path/'out'; args = (x,frame,{},settings,out)
    study.run_experiment(*args,max_new_jobs=1)
    context = json.loads((out/'protocol.json').read_text())
    fold = context['context']['folds'][0]; job = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
    original = load_record(job/'result.json',context['context_hash']); changed = copy.deepcopy(original)
    entry = next(c for c in changed['candidates'] if c['candidate'].get('intervention') == 'severe_interpolation')
    receipt = next(r for r in entry['inner_fit_receipts'] if r['lineage'])
    receipt['lineage'][0]['parent_a'] = len(frame)+1
    save_record(job/'result.json',changed,context['context_hash'])
    with pytest.raises(ValueError,match='parent'):study.run_experiment(*args,resume=True)
    save_record(job/'result.json',original,context['context_hash'])
    model = job/'accuracy.pkl'; model.write_bytes(model.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='checkpoint'):study.run_experiment(*args,resume=True)
    def changed_inputs():raise ValueError('inputs changed')
    with pytest.raises(ValueError,match='inputs changed'):
        study.run_experiment(x,frame,{},settings,tmp_path/'second',max_new_jobs=1,verify_inputs=changed_inputs)
    assert not (tmp_path/'second/completion.json').exists()
