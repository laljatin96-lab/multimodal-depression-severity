"""Nested complete-chunk and symptom-contrast features for severe detection."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from . import semantic_severe_repair as baseline
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .severe_case_diagnosis import load_baseline, ROOT

PROTOCOL = 'chunk-feature-severe-priority-v1'


from .severe_detection_study import decide, score, select, summarize
from . import severe_detection_study as previous
from .chunk_semantic_features import load_verified as chunk_features


def candidates(settings):
    return [{'family':'baseline'}]+[
        {'family':'logistic','feature':feature,'C':c,'components':components,'intervention':intervention}
        for feature in settings['feature_inputs'] for components in settings['components']
        for c in settings['C'] for intervention in settings['interventions']]


def estimator(candidate,old_candidate,settings,old_settings):
    if candidate['family']=='baseline':return baseline.estimator_for(old_candidate,old_settings)
    return baseline.estimator_for(candidate,{**old_settings,'max_iter':settings['max_iter'],'fit_seed':settings['fit_seed']})


def job(x,y,fold,old,settings,old_settings,out):
    start=time.monotonic();search=candidates(settings);inner=[];options=[];fits=base_fits=0
    old_candidate=old['policies']['accuracy']['candidate']
    for candidate in search:
        feature=old_candidate['feature'] if candidate['family']=='baseline' else candidate['feature']
        if candidate['family']=='baseline':
            p=np.asarray(old['cached_inner_probabilities'])
        else:
            p=np.full((len(y),5),np.nan)
            for part in fold['inner']:
                model=estimator(candidate,old_candidate,settings,old_settings)
                fit(model,x[feature][part['fit']],y[part['fit']]);fits+=1;base_fits+=1
                p[part['heldout']]=five_probabilities(model,x[feature][part['heldout']])
            p=p[fold['fit']]
        inner.append(p.tolist())
        for threshold in settings['thresholds']:
            options.append({'candidate':candidate,'threshold':threshold,**score(y[fold['fit']],decide(p,threshold))})
    expected=old['candidates'][old['policies']['accuracy']['selection']['index']]['predictions']
    np.testing.assert_array_equal(np.asarray(inner[0]).argmax(1),expected)
    selected={};out.mkdir(parents=True,exist_ok=True)
    for policy,limits in settings['policies'].items():
        choice=select(options,options[0],limits);option=options[choice['index']];candidate=option['candidate']
        feature=old_candidate['feature'] if candidate['family']=='baseline' else candidate['feature']
        if candidate['family']=='baseline':
            path=ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
            with path.open('rb') as f:model=pickle.load(f)['estimator']
            p=five_probabilities(model,x[feature][fold['heldout']])
            np.testing.assert_allclose(p,old['policies']['accuracy']['probabilities'],rtol=0,atol=1e-14)
        else:
            model=estimator(candidate,old_candidate,settings,old_settings)
            fit(model,x[feature][fold['fit']],y[fold['fit']]);fits+=1
            base_fits+=1
            p=five_probabilities(model,x[feature][fold['heldout']])
        pred=decide(p,option['threshold'])
        state={'estimator':model,'feature':feature,'threshold':option['threshold'],'fit':fold['fit'],
               'heldout':fold['heldout'],'candidate':candidate,'protocol':PROTOCOL}
        path=out/f'{policy}.pkl'
        with path.open('wb') as f:pickle.dump(state,f)
        with path.open('rb') as f:restored=pickle.load(f)
        np.testing.assert_allclose(p,five_probabilities(restored['estimator'],x[feature][fold['heldout']]),rtol=0,atol=1e-14)
        np.testing.assert_array_equal(pred,decide(p,restored['threshold']))
        selected[policy]={'selection':choice,'candidate':candidate,'threshold':option['threshold'],
                          'probabilities':p.tolist(),'predictions':pred.tolist(),'checkpoint_sha256':sha256(path)}
    return {'seed':fold['seed'],'fold':fold['fold'],'fit':fold['fit'],'heldout':fold['heldout'],'inner_probabilities':inner,
            'options':options,'selected':selected,'fits':fits,'base_fits':base_fits,'seconds':time.monotonic()-start}


def validate(record,fold,y,settings,directory):
    if any(record[k]!=fold[k] for k in ['seed','fold','fit','heldout']):raise ValueError('Fold changed')
    expected=[]
    if len(record['inner_probabilities'])!=len(candidates(settings)):raise ValueError('Missing inner probabilities')
    for candidate,p in zip(candidates(settings),record['inner_probabilities']):
        if np.asarray(p).shape!=(len(fold['fit']),5):raise ValueError('Inner coverage changed')
        for threshold in settings['thresholds']:
            expected.append({'candidate':candidate,'threshold':threshold,**score(y[fold['fit']],decide(p,threshold))})
    if record['options']!=expected:raise ValueError('Inner search metrics changed')
    fits=(len(candidates(settings))-1)*len(fold['inner'])
    base_fits=fits
    if set(record['selected'])!=set(settings['policies']):raise ValueError('Policies changed')
    for policy,result in record['selected'].items():
        choice=select(expected,expected[0],settings['policies'][policy]);option=expected[choice['index']]
        if result['selection']!=choice or result['candidate']!=option['candidate'] or result['threshold']!=option['threshold']:
            raise ValueError('Selection changed')
        p=np.asarray(result['probabilities'])
        if p.shape!=(len(fold['heldout']),5) or decide(p,result['threshold']).tolist()!=result['predictions']:
            raise ValueError('Decision output mismatch')
        if sha256(directory/f'{policy}.pkl')!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if result['candidate']['family']!='baseline':
            fits+=1;base_fits+=1
    if record['fits']!=fits or record['base_fits']!=base_fits:raise ValueError('Fit budget changed')


def run(x,frame,old_settings,old_records,provenance,settings,output,resume=False,max_new_jobs=None):
    out=Path(output)
    if max_new_jobs is not None and max_new_jobs<1:raise ValueError('Positive pilot count required')
    if out.exists() and not resume:raise FileExistsError('Preserving experiment; use --resume')
    if resume and not (out/'protocol.json').exists():raise ValueError('Resume needs protocol')
    folds=baseline.nested_folds(frame,old_settings);y=frame.severity_class.to_numpy(int)
    if len(old_records)!=len(folds):raise ValueError('Incomplete baseline coverage')
    if len(folds)!=settings['expected_jobs'] or (len(candidates(settings))-1)*old_settings['inner_folds']!=settings['inner_pipeline_fits_per_job']:
        raise ValueError('Declared budget mismatch')
    if frame.participant_id.duplicated().any() or not np.array_equal(y,frame.severity_class):raise ValueError('Invalid real participants')
    for value in x.values():
        if len(value)!=len(y) or not np.isfinite(value).all():raise ValueError('Invalid features')
    context={'protocol':PROTOCOL,'settings':settings,'old_settings':old_settings,'training':frame.to_dict('records'),
        'folds':folds,'baseline_records':old_records,'provenance':provenance,'code_sha256':sha256(__file__),
        'input_hashes':{k:digest(v.tolist()) for k,v in x.items()},
        'support_hashes':{n:sha256(Path(__file__).with_name(n)) for n in ['severe_case_diagnosis.py','semantic_severe_repair.py','severe_detection_study.py','chunk_semantic_features.py','evaluate.py','encoder_kernel_accuracy.py','raw_frame_uncertainty.py']},
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    ch=digest(context);envelope={'context':context,'context_hash':ch};out.mkdir(parents=True,exist_ok=True)
    with output_lock(out),threadpool_limits(limits=settings['cpu_threads']),offline_network():
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text())!=envelope:raise ValueError('Inputs/code changed')
        else:atomic_json(out/'protocol.json',envelope)
        records=[];new=0;new_fits=0
        for fold,old in zip(folds,old_records):
            directory=out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}";path=directory/'result.json'
            if path.exists():record=load_record(path,ch)
            elif max_new_jobs is not None and new>=max_new_jobs:continue
            else:
                record=job(x,y,fold,old,settings,old_settings,directory);save_record(path,record,ch);new+=1;new_fits+=record['fits']
            validate(record,fold,y,settings,directory)
            np.testing.assert_array_equal(record['inner_probabilities'][0],old['cached_inner_probabilities'])
            records.append(record)
            print(f'Severe detection: {len(records)}/{len(folds)} jobs; {new} new',flush=True)
        # Bound source hashes are checked again without fitting or writing them.
        for name,h in provenance.get('bindings',{}).items():
            if sha256(name)!=h:raise ValueError('Baseline changed during experiment')
        complete=len(records)==len(folds)
        targets=summarize(frame,records,old_records,settings,out) if complete else None
        receipt={'complete':complete,'context_hash':ch,'completed_jobs':len(records),'new_jobs':new,'new_fits':new_fits,
                 'pipeline_fits':sum(r['fits'] for r in records),'base_estimator_fits':sum(r['base_fits'] for r in records),
                 'targets':targets,'validation_test_scored':False,'baseline_replaced':False,'synthetic_evaluation_participants':0}
        atomic_json(out/'completion.json',receipt)
    return receipt


def load_inputs():
    x,frame,old_settings,old_records,provenance=load_baseline()
    root=Path('results/severe_detection')
    settings=json.loads(Path('configs/severe_detection.json').read_text())
    folds=baseline.nested_folds(frame,old_settings)
    expected={'protocol':previous.PROTOCOL,'settings':settings,'old_settings':old_settings,
        'training':frame.to_dict('records'),'folds':folds,'baseline_records':old_records,'provenance':provenance,
        'code_sha256':sha256(previous.__file__),'input_hashes':{k:digest(v.tolist()) for k,v in x.items()},
        'support_hashes':{n:sha256(Path(__file__).with_name(n)) for n in
            ['severe_case_diagnosis.py','semantic_severe_repair.py','evaluate.py','encoder_kernel_accuracy.py','raw_frame_uncertainty.py']},
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    envelope=json.loads((root/'protocol.json').read_text())
    if envelope!={'context':expected,'context_hash':digest(expected)} or not json.loads((root/'completion.json').read_text())['complete']:
        raise ValueError('Previous threshold study changed/incomplete')
    for fold,old in zip(folds,old_records):
        directory=root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
        record=load_record(directory/'result.json',envelope['context_hash'])
        previous.validate(record,fold,frame.severity_class.to_numpy(int),settings,directory)
        p=np.asarray(record['inner_probabilities'][0])
        np.testing.assert_array_equal(p.argmax(1),old['candidates'][old['policies']['accuracy']['selection']['index']]['predictions'])
        old['cached_inner_probabilities']=p.tolist()
    records,feature_bindings=chunk_features()
    by_id={r['participant_id']:r for r in records}
    dispersion=np.asarray([by_id[pid]['dispersion'] for pid in frame.participant_id])
    symptom=np.asarray([by_id[pid]['symptom'] for pid in frame.participant_id])
    np.testing.assert_allclose([by_id[pid]['mean'] for pid in frame.participant_id],x['semantic'],rtol=0,atol=2e-7)
    av=x['semantic_av'][:,384:]
    x['chunk_av']=np.column_stack([x['semantic'],dispersion,av])
    x['symptom_av']=np.column_stack([x['semantic'],symptom,av])
    bindings={**provenance['bindings'],**feature_bindings,**{str(p):sha256(p) for p in root.rglob('*') if p.is_file()}}
    return x,frame,old_settings,old_records,{**provenance,'bindings':bindings,'feature_coverage':len(records),
        'prior_threshold_context':envelope['context_hash'],'cached_inner_probabilities_reused':len(old_records)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/chunk_severe_detection')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--max-new-jobs',type=int)
    args=parser.parse_args()
    if any(Path(args.output).resolve().is_relative_to(Path(p).resolve()) for p in ['data','dataset','checkpoints',str(ROOT),'results/semantic_features','results/transcript_corpus','results/chunk_semantic_features','results/severe_detection']):
        parser.error('Output must be outside existing inputs')
    x,frame,old_settings,old_records,provenance=load_inputs()
    settings=json.loads(Path('configs/chunk_severe_detection.json').read_text())
    print(json.dumps(run(x,frame,old_settings,old_records,provenance,settings,args.output,args.resume,args.max_new_jobs),indent=2))

if __name__=='__main__':
    from .chunk_severe_study import main as module_main
    module_main()
