"""Nested ordinal/weighted severe detection and inner-only decision thresholds."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import semantic_severe_repair as baseline
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .evaluate import metrics
from .final_evaluation import sha256
from .raw_frame_uncertainty import participant_counts, weighted_metrics, METRICS
from .severe_case_diagnosis import load_baseline, ROOT

PROTOCOL = 'severe-priority-ordinal-threshold-v1'


class SeverityClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, family='ordinal', C=.01, components=32, weight=0., max_iter=2000):
        self.family,self.C,self.components,self.weight,self.max_iter=family,C,components,weight,max_iter

    def fit(self,x,y):
        x,y=np.asarray(x,float),np.asarray(y)
        if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.array_equal(y,y.astype(int)) or set(y)!=set(range(5)):
            raise ValueError('Five real fitting classes and finite vectors required')
        y=y.astype(int);self.classes_=np.arange(5);self.real_count_=len(y)
        self.scaler_=StandardScaler();z=self.scaler_.fit_transform(x)
        self.pca_=PCA(n_components=self.components,svd_solver='full')
        z=self.pca_.fit_transform(z);self.models_=[];self.class_counts_=np.bincount(y,minlength=5)
        targets=[y] if self.family=='weighted' else [(y>k).astype(int) for k in range(4)]
        if self.family not in ['weighted','ordinal']:raise ValueError('Unknown family')
        for target in targets:
            if self.family=='weighted':
                w=np.where(y==4,self.weight,1.)
            else:
                counts=np.bincount(target,minlength=2);w=(len(y)/(2*counts[target]))**self.weight
            w=w*len(y)/w.sum()
            model=LogisticRegression(C=self.C,max_iter=self.max_iter,solver='lbfgs')
            model.fit(z,target,sample_weight=w);self.models_.append(model)
        return self

    def predict_proba(self,x):
        z=self.pca_.transform(self.scaler_.transform(x))
        if self.family=='weighted':return self.models_[0].predict_proba(z)
        q=np.column_stack([m.predict_proba(z)[:,1] for m in self.models_])
        q=np.stack([isotonic_regression(row,increasing=False,y_min=0,y_max=1) for row in q])
        return np.column_stack([1-q[:,0],q[:,:-1]-q[:,1:],q[:,-1]])

    def predict(self,x):return self.predict_proba(x).argmax(1)


def candidates(settings):
    return [{'family':'baseline'}]+[
        {'family':family,'feature':feature,'C':c,'weight':weight}
        for family,key in [('weighted','severe_weights'),('ordinal','ordinal_weight_powers')]
        for feature in settings['feature_inputs'] for c in settings['C'] for weight in settings[key]]


def estimator(candidate,old_candidate,settings,old_settings):
    if candidate['family']=='baseline':return baseline.estimator_for(old_candidate,old_settings)
    return SeverityClassifier(candidate['family'],candidate['C'],settings['components'],candidate['weight'],settings['max_iter'])


def decide(p,threshold):
    p=np.asarray(p,float)
    if p.ndim!=2 or p.shape[1]!=5 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):
        raise ValueError('Invalid probabilities')
    if threshold is None:return p.argmax(1)
    if not 0<=threshold<=1:raise ValueError('Invalid severe threshold')
    return np.where(p[:,4]>=threshold,4,p[:,:4].argmax(1))


def score(y,pred):
    result=baseline.selection_metrics(y,pred)
    result['nonsevere_errors']=int(((y!=4)&(pred!=y)).sum())
    result['severe_false_positives']=int(((y!=4)&(pred==4)).sum())
    result['severe_fpr']=result['severe_false_positives']/int((y!=4).sum())
    return result


def select(options,baseline_score,limits):
    eligible=[i for i,r in enumerate(options) if r['accuracy']+1e-12>=baseline_score['accuracy']-limits['max_accuracy_loss']
              and r['severe_fpr']<=limits['max_severe_fpr']+1e-12]
    if not eligible:return {'index':0,'constraints_met':False}
    index=max(eligible,key=lambda i:(options[i]['severe_true_positive'],-options[i]['nonsevere_errors'],
                                     options[i]['correct'],options[i]['macro_f1'],-i))
    return {'index':index,'constraints_met':True}


def job(x,y,fold,old,settings,old_settings,out):
    start=time.monotonic();search=candidates(settings);inner=[];options=[];fits=base_fits=0
    old_candidate=old['policies']['accuracy']['candidate']
    for candidate in search:
        feature=old_candidate['feature'] if candidate['family']=='baseline' else candidate['feature']
        p=np.full((len(y),5),np.nan)
        for part in fold['inner']:
            model=estimator(candidate,old_candidate,settings,old_settings)
            fit(model,x[feature][part['fit']],y[part['fit']]);fits+=1
            base_fits+=4 if candidate['family']=='ordinal' else 1
            p[part['heldout']]=five_probabilities(model,x[feature][part['heldout']])
        p=p[fold['fit']];inner.append(p.tolist())
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
            base_fits+=4 if candidate['family']=='ordinal' else 1
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
    fits=len(candidates(settings))*len(fold['inner'])
    base_fits=sum(4 if c['family']=='ordinal' else 1 for c in candidates(settings))*len(fold['inner'])
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
            fits+=1;base_fits+=4 if result['candidate']['family']=='ordinal' else 1
    if record['fits']!=fits or record['base_fits']!=base_fits:raise ValueError('Fit budget changed')


def summarize(frame,records,old_records,settings,out):
    y=frame.severity_class.to_numpy(int);rows=[];predictions={};matrices={}
    for name in ['baseline']+list(settings['policies']):
        repeated=[];cms=[]
        for seed in sorted({r['seed'] for r in records}):
            p=np.full((len(y),5),np.nan);pred=np.full(len(y),-1);owners=np.zeros(len(y),int)
            for r in (old_records if name=='baseline' else records):
                if r['seed']!=seed:continue
                result=r['policies']['accuracy'] if name=='baseline' else r['selected'][name]
                p[r['heldout']]=result['probabilities']
                pred[r['heldout']]=np.asarray(result['probabilities']).argmax(1) if name=='baseline' else result['predictions']
                owners[r['heldout']]+=1
            if not (owners==1).all() or not np.isfinite(p).all():raise ValueError('Incomplete real-participant evaluation')
            # One-hot decisions are used only to calculate label metrics; original probabilities are saved separately.
            report=metrics(y,np.eye(5)[pred]);report['group_metrics']={g:metrics(part.severity_class,np.eye(5)[pred[frame.index.get_indexer(part.index)]]) for g,part in frame.groupby('gender')}
            directory=out/'oof'/f'{name}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            table=frame.copy();table['predicted_label']=pred;table['probability_argmax']=p.argmax(1)
            for k in range(5):table[f'p{k}']=p[:,k]
            table['split']='train_nested_oof';table.to_csv(directory/'predictions.csv',index=False)
            atomic_json(directory/'metrics.json',report);pd.DataFrame(report['confusion_matrix']).to_csv(directory/'confusion_matrix.csv',index=False)
            pd.DataFrame(report['per_class']).to_csv(directory/'per_class.csv',index=False)
            rows.append({'model':name,'seed':seed,**score(y,pred)});repeated.append(pred);cms.append(report['confusion_matrix'])
        predictions[name]=np.stack(repeated);matrices[name]=np.asarray(cms).sum(0).tolist()
    scores=pd.DataFrame(rows);scores.to_csv(out/'repeat_scores.csv',index=False)
    summary=scores.groupby('model')[METRICS+['severe_recall','severe_precision','severe_false_positives']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(out/'summary.csv')
    counts=participant_counts(len(y),5000,20261025)
    boot={n:weighted_metrics(y,p,counts) for n,p in predictions.items()};differences=[]
    for n in settings['policies']:
        interval=np.percentile(boot[n]-boot['baseline'],[2.5,97.5],axis=0)
        for i,k in enumerate(METRICS):
            differences.append({'model':n,'metric':k,'difference':float(summary.loc[n,k+'_mean']-summary.loc['baseline',k+'_mean']),
                                'ci95_low':float(interval[0,i]),'ci95_high':float(interval[1,i])})
    pd.DataFrame(differences).to_csv(out/'paired_comparisons.csv',index=False)
    atomic_json(out/'summed_confusion_matrices.json',{'note':'Sum of three repeats; 489 predictions on the same 163 people, including the same seven severe people three times. Rows actual, columns predicted, classes 0-4.','matrices':matrices})
    pd.DataFrame([{'seed':r['seed'],'fold':r['fold'],'policy':n,'threshold':s['threshold'],**s['candidate'],**s['selection']} for r in records for n,s in r['selected'].items()]).to_csv(out/'choices.csv',index=False)
    return {n:{'accuracy':float(summary.loc[n,'accuracy_mean']),'severe_recall':float(summary.loc[n,'severe_recall_mean']),
               'target_accuracy_reached':bool(summary.loc[n,'accuracy_mean']>=settings['target_accuracy']),
               'independent_confirmation':False} for n in settings['policies']}


def run(x,frame,old_settings,old_records,provenance,settings,output,resume=False,max_new_jobs=None):
    out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserving experiment; use --resume')
    if resume and not (out/'protocol.json').exists():raise ValueError('Resume needs protocol')
    folds=baseline.nested_folds(frame,old_settings);y=frame.severity_class.to_numpy(int)
    if len(folds)!=settings['expected_jobs'] or len(candidates(settings))*old_settings['inner_folds']!=settings['inner_pipeline_fits_per_job']:
        raise ValueError('Declared budget mismatch')
    if frame.participant_id.duplicated().any() or not np.array_equal(y,frame.severity_class):raise ValueError('Invalid real participants')
    for value in x.values():
        if len(value)!=len(y) or not np.isfinite(value).all():raise ValueError('Invalid features')
    context={'protocol':PROTOCOL,'settings':settings,'old_settings':old_settings,'training':frame.to_dict('records'),
        'folds':folds,'baseline_records':old_records,'provenance':provenance,'code_sha256':sha256(__file__),
        'input_hashes':{k:digest(v.tolist()) for k,v in x.items()},
        'support_hashes':{n:sha256(Path(__file__).with_name(n)) for n in ['severe_case_diagnosis.py','semantic_severe_repair.py','evaluate.py','encoder_kernel_accuracy.py','raw_frame_uncertainty.py']},
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
            validate(record,fold,y,settings,directory);records.append(record)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/severe_detection')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--max-new-jobs',type=int)
    args=parser.parse_args()
    if any(Path(args.output).resolve().is_relative_to(Path(p).resolve()) for p in ['data','dataset','checkpoints',str(ROOT),'results/semantic_features','results/transcript_corpus']):
        parser.error('Output must be outside existing inputs')
    x,frame,old_settings,old_records,provenance=load_baseline()
    settings=json.loads(Path('configs/severe_detection.json').read_text())
    print(json.dumps(run(x,frame,old_settings,old_records,provenance,settings,args.output,args.resume,args.max_new_jobs),indent=2))

if __name__=='__main__':
    from .severe_detection_study import main as module_main
    module_main()
