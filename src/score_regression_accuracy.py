"""Nested true-PHQ regression with fixed or inner-learned five-class decisions."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import deep_fusion_accuracy as prior
from .batch_inference import atomic_json,load_record,save_record,output_lock
from .common import digest
from .evaluate import metrics
from .final_evaluation import sha256
from .raw_frame_uncertainty import METRICS,participant_counts,weighted_metrics
from .severe_detection_study import score
from .encoder_kernel_accuracy import offline_network

PROTOCOL='true-phq-kernel-regression-five-class-v1'
FEATURES={'semantic':['semantic'],'semantic_av':['semantic','av'],
    'deep_av':['densenet','vgg_audio','resnet_face'],
    'deep_av_text':['densenet','vgg_audio','resnet_face','semantic','av']}
CUTS=np.array([4.5,9.5,14.5,19.5])


class BlockScaling:
    def fit(self,x,indices):
        self.indices=list(indices);self.scalers={n:StandardScaler().fit(x[n][indices]) for n in sorted({n for v in FEATURES.values() for n in v})}
        return self
    def transform(self,x,indices):return {n:s.transform(x[n][indices])/np.sqrt(x[n].shape[1]) for n,s in self.scalers.items()}


def matrix(z,feature):return np.column_stack([z[n] for n in FEATURES[feature]])/np.sqrt(len(FEATURES[feature]))


def kernel(a,b,kind):
    product=a@b.T
    if kind=='linear':return product
    if kind!='rbf':raise ValueError('Unknown kernel')
    distance=np.maximum(np.square(a).sum(1)[:,None]+np.square(b).sum(1)[None,:]-2*product,0)
    return np.exp(-.5*distance)


def candidates(settings):
    return [{'family':'baseline'}]+[{'family':'score_regression','feature':f,'kernel':k,'alpha':a}
        for f in settings['feature_inputs'] for k in settings['kernels'] for a in settings['alpha']]


def decode(values,cuts):
    a=np.asarray(values,float);c=np.asarray(cuts,float)
    if a.ndim!=1 or not np.isfinite(a).all() or c.shape!=(4,) or not np.isfinite(c).all() or not (np.diff(c)>0).all():raise ValueError('Invalid scores or ordered cuts')
    return np.searchsorted(c,a,side='right')


def optimize_cuts(values,y,settings):
    """Exact dynamic program: max inner correct, then min displacement from fixed bins."""
    values,y=np.asarray(values),np.asarray(y)
    decode(values,CUTS)
    width,step=settings['cut_radius'],settings['cut_step']
    grids=[np.arange(c-width,c+width+step/2,step) for c in CUTS]
    states=[]
    rank=lambda r:(r[0],-r[1],tuple(-v for v in r[2]))
    for c in grids[0]:states.append((int(((y==0)&(values<c)).sum()),abs(c-CUTS[0]),[float(c)]))
    for k in range(1,4):
        new=[]
        for c in grids[k]:
            choices=[(r[0]+int(((y==k)&(values>=r[2][-1])&(values<c)).sum()),r[1]+abs(c-CUTS[k]),r[2]+[float(c)]) for r in states if c-r[2][-1]>=settings['minimum_cut_gap']-1e-12]
            if choices:new.append(max(choices,key=rank))
        states=new
    final=[(r[0]+int(((y==4)&(values>=r[2][-1])).sum()),r[1],r[2]) for r in states]
    if not final:raise ValueError('No feasible ordered cuts')
    return max(final,key=rank)[2]


def options(inner,baseline_predictions,y,settings):
    search=candidates(settings);result={}
    for policy in settings['policies']:
        rows=[{'candidate_index':0,'cuts':None,**score(y,np.asarray(baseline_predictions))}]
        for i,values in enumerate(inner,1):
            cuts=CUTS.tolist() if policy=='fixed_score_bins' else optimize_cuts(values,y,settings)
            rows.append({'candidate_index':i,'cuts':cuts,**score(y,decode(values,cuts))})
        result[policy]=rows
    return result


def choose(rows):return max(range(len(rows)),key=lambda i:(rows[i]['correct'],rows[i]['macro_f1'],-i))


def predict_state(state,x,indices):
    if state['candidate']['family']=='baseline':
        p=prior.cached.five_probabilities(state['baseline'],x[state['baseline_feature']][indices])
        return p.argmax(1),None
    z=matrix(state['scaler'].transform(x,indices),state['candidate']['feature'])
    values=np.clip(state['regressor'].predict(kernel(z,state['fitting_vectors'],state['candidate']['kernel']))+state['target_mean'],0,24)
    return decode(values,state['cuts']),values


def job(x,frame,totals,fold,old,settings,out):
    start=time.monotonic();y=frame.severity_class.to_numpy(int);search=candidates(settings);inner=[]
    predictions=[np.full(len(y),np.nan) for _ in search[1:]]
    for part in fold['inner']:
        scaler=BlockScaling().fit(x,part['fit']);train=scaler.transform(x,part['fit']);held=scaler.transform(x,part['heldout'])
        matrices={f:(matrix(train,f),matrix(held,f)) for f in settings['feature_inputs']}
        kernels={(f,k):(kernel(a,a,k),kernel(b,a,k)) for f,(a,b) in matrices.items() for k in settings['kernels']}
        mean=float(totals[part['fit']].mean())
        for candidate,values in zip(search[1:],predictions):
            fitting,testing=kernels[(candidate['feature'],candidate['kernel'])]
            model=KernelRidge(alpha=candidate['alpha'],kernel='precomputed').fit(fitting,totals[part['fit']]-mean)
            values[part['heldout']]=np.clip(model.predict(testing)+mean,0,24)
    inner=[v[fold['fit']].tolist() for v in predictions]
    base=np.asarray(old['cached_inner_probabilities']).argmax(1)
    all_options=options(inner,base,y[fold['fit']],settings);choices={p:choose(rows) for p,rows in all_options.items()}
    selected_indices={i for i in choices.values() if i};scaler=BlockScaling().fit(x,fold['fit']) if selected_indices else None
    z=scaler.transform(x,fold['fit']) if scaler else None;fitted={};mean=float(totals[fold['fit']].mean())
    for i in selected_indices:
        candidate=search[i];a=matrix(z,candidate['feature'])
        model=KernelRidge(alpha=candidate['alpha'],kernel='precomputed').fit(kernel(a,a,candidate['kernel']),totals[fold['fit']]-mean)
        fitted[i]=(model,a)
    with (prior.ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl').open('rb') as f:baseline=pickle.load(f)['estimator']
    feature=old['policies']['accuracy']['candidate']['feature']
    np.testing.assert_allclose(prior.cached.five_probabilities(baseline,x[feature][fold['heldout']]),old['policies']['accuracy']['probabilities'],rtol=0,atol=1e-14)
    selected={};out.mkdir(parents=True,exist_ok=True)
    for policy,i in choices.items():
        candidate=search[i];cuts=all_options[policy][i]['cuts']
        state={'protocol':PROTOCOL,'candidate':candidate,'cuts':cuts,'baseline':baseline,'baseline_feature':feature,
            'scaler':scaler if i else None,'regressor':fitted[i][0] if i else None,'fitting_vectors':fitted[i][1] if i else None,
            'target_mean':mean,'fit':fold['fit'],'heldout':fold['heldout']}
        labels,values=predict_state(state,x,fold['heldout']);path=out/f'{policy}.pkl'
        with path.open('wb') as f:pickle.dump(state,f)
        with path.open('rb') as f:restored=pickle.load(f)
        other,other_values=predict_state(restored,x,fold['heldout']);np.testing.assert_array_equal(labels,other)
        if values is not None:np.testing.assert_allclose(values,other_values,rtol=0,atol=1e-12)
        selected[policy]={'candidate':candidate,'selection_index':i,'cuts':cuts,'predictions':labels.tolist(),
            'score_predictions':None if values is None else values.tolist(),'checkpoint_sha256':sha256(path)}
    return {'seed':fold['seed'],'fold':fold['fold'],'fit':fold['fit'],'heldout':fold['heldout'],'inner_score_predictions':inner,
        'options':all_options,'selected':selected,'regressor_fits':(len(search)-1)*len(fold['inner'])+len(selected_indices),
        'scaler_fits':len(fold['inner'])+bool(selected_indices),'seconds':time.monotonic()-start}


def validate(r,fold,old,y,settings,out):
    if any(r[k]!=fold[k] for k in ['seed','fold','fit','heldout']):raise ValueError('Fold changed')
    if len(r['inner_score_predictions'])!=len(candidates(settings))-1 or np.asarray(r['inner_score_predictions']).shape!=(len(candidates(settings))-1,len(fold['fit'])):raise ValueError('Incomplete score coverage')
    expected=options(r['inner_score_predictions'],np.asarray(old['cached_inner_probabilities']).argmax(1),y[fold['fit']],settings)
    if r['options']!=expected or set(r['selected'])!=set(settings['policies']):raise ValueError('Search changed')
    indices=set()
    for policy,result in r['selected'].items():
        i=choose(expected[policy]);candidate=candidates(settings)[i]
        if result['selection_index']!=i or result['candidate']!=candidate or result['cuts']!=expected[policy][i]['cuts']:raise ValueError('Selection changed')
        if len(result['predictions'])!=len(fold['heldout']) or not set(result['predictions'])<=set(range(5)):raise ValueError('Invalid predictions')
        if i:
            if decode(result['score_predictions'],result['cuts']).tolist()!=result['predictions']:raise ValueError('Score decision mismatch')
            indices.add(i)
        elif result['score_predictions'] is not None or result['predictions']!=np.asarray(old['policies']['accuracy']['probabilities']).argmax(1).tolist():raise ValueError('Baseline mismatch')
        if sha256(out/f'{policy}.pkl')!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if r['regressor_fits']!=(len(candidates(settings))-1)*len(fold['inner'])+len(indices) or r['scaler_fits']!=len(fold['inner'])+bool(indices):raise ValueError('Fit budget changed')


def summarize(frame,totals,records,old_records,settings,out):
    y=frame.severity_class.to_numpy(int);rows=[];repeated={};matrices={}
    for name in ['baseline']+list(settings['policies']):
        predictions=[];cms=[]
        for seed in sorted({r['seed'] for r in records}):
            pred=np.full(len(y),-1);values=np.full(len(y),np.nan);owners=np.zeros(len(y),int)
            for r in (old_records if name=='baseline' else records):
                if r['seed']!=seed:continue
                result=r['policies']['accuracy'] if name=='baseline' else r['selected'][name]
                pred[r['heldout']]=np.asarray(result['probabilities']).argmax(1) if name=='baseline' else result['predictions']
                if name!='baseline' and result['score_predictions'] is not None:values[r['heldout']]=result['score_predictions']
                owners[r['heldout']]+=1
            if not (owners==1).all() or (pred<0).any():raise ValueError('Incomplete evaluation')
            report=metrics(y,np.eye(5)[pred]);report['group_metrics']={g:metrics(part.severity_class,np.eye(5)[pred[frame.index.get_indexer(part.index)]]) for g,part in frame.groupby('gender')}
            directory=out/'oof'/f'{name}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            table=frame.copy();table['predicted_label']=pred;table['predicted_phq8']=values;table['true_phq8']=totals
            table['prediction_kind']=np.where(np.isnan(values),'baseline_classifier','score_regressor');table['split']='train_nested_oof';table.to_csv(directory/'predictions.csv',index=False)
            atomic_json(directory/'metrics.json',report);pd.DataFrame(report['confusion_matrix']).to_csv(directory/'confusion_matrix.csv',index=False);pd.DataFrame(report['per_class']).to_csv(directory/'per_class.csv',index=False)
            rows.append({'model':name,'seed':seed,**score(y,pred),'regression_predictions':int(np.isfinite(values).sum())});predictions.append(pred);cms.append(report['confusion_matrix'])
        repeated[name]=np.stack(predictions);matrices[name]=np.asarray(cms).sum(0).tolist()
    table=pd.DataFrame(rows);table.to_csv(out/'repeat_scores.csv',index=False)
    summary=table.groupby('model')[METRICS+['severe_recall','severe_precision','severe_false_positives']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(out/'summary.csv')
    counts=participant_counts(len(y),5000,20261026);boot={n:weighted_metrics(y,p,counts) for n,p in repeated.items()};diff=[]
    for n in settings['policies']:
        interval=np.percentile(boot[n]-boot['baseline'],[2.5,97.5],axis=0)
        for i,k in enumerate(METRICS):diff.append({'model':n,'metric':k,'difference':float(summary.loc[n,k+'_mean']-summary.loc['baseline',k+'_mean']),'ci95_low':float(interval[0,i]),'ci95_high':float(interval[1,i])})
    pd.DataFrame(diff).to_csv(out/'paired_comparisons.csv',index=False)
    atomic_json(out/'summed_confusion_matrices.json',{'note':'Same 163 participants repeated three times, including seven severe people; rows actual, columns predicted, classes 0-4.','matrices':matrices})
    pd.DataFrame([{'seed':r['seed'],'fold':r['fold'],'policy':n,'cuts':json.dumps(v['cuts']),**v['candidate']} for r in records for n,v in r['selected'].items()]).to_csv(out/'choices.csv',index=False)
    return {n:{'accuracy':float(summary.loc[n,'accuracy_mean']),'severe_recall':float(summary.loc[n,'severe_recall_mean']),'target_accuracy_reached':bool(summary.loc[n,'accuracy_mean']>=settings['target_accuracy']),'independent_confirmation':False} for n in settings['policies']}


def run(x,frame,totals,old_settings,old_records,provenance,settings,output,resume=False,max_new_jobs=None):
    out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserving existing run')
    if resume and not (out/'protocol.json').exists():raise ValueError('Resume requires protocol')
    if max_new_jobs is not None and max_new_jobs<1:raise ValueError('Positive pilot budget required')
    y=frame.severity_class.to_numpy(int);folds=prior.cached.baseline.nested_folds(frame,old_settings)
    if len(folds)!=settings['expected_jobs'] or len(old_records)!=len(folds) or (len(candidates(settings))-1)*old_settings['inner_folds']!=settings['inner_regressor_fits_per_job']:raise ValueError('Budget mismatch')
    if frame.participant_id.duplicated().any() or np.asarray(totals).shape!=(len(y),) or not np.isfinite(totals).all() or (totals<0).any() or (totals>24).any() or not np.array_equal(totals,totals.astype(int)) or not np.array_equal(np.minimum(totals//5,4),y):raise ValueError('Invalid genuine PHQ totals/class mapping')
    for a in x.values():
        if len(a)!=len(y) or not np.isfinite(a).all():raise ValueError('Invalid features')
    ctx={'protocol':PROTOCOL,'settings':settings,'old_settings':old_settings,'training':frame.to_dict('records'),'phq8_totals':totals.tolist(),
        'folds':folds,'baseline_records':old_records,'provenance':provenance,'code_sha256':sha256(__file__),'input_hashes':{n:digest(a.tolist()) for n,a in x.items()},
        'support_hashes':{n:sha256(Path(__file__).with_name(n)) for n in ['deep_fusion_accuracy.py','supplied_deep_features.py','severe_detection_study.py','evaluate.py','raw_frame_uncertainty.py']},
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    ch=digest(ctx);envelope={'context':ctx,'context_hash':ch};out.mkdir(parents=True,exist_ok=True)
    with output_lock(out),threadpool_limits(limits=settings['cpu_threads']),offline_network():
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text())!=envelope:raise ValueError('Inputs/code changed')
        else:atomic_json(out/'protocol.json',envelope)
        records=[];new=new_fits=0
        for fold,old in zip(folds,old_records):
            directory=out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}";path=directory/'result.json'
            if path.exists():r=load_record(path,ch)
            elif max_new_jobs is not None and new>=max_new_jobs:continue
            else:r=job(x,frame,totals,fold,old,settings,directory);save_record(path,r,ch);new+=1;new_fits+=r['regressor_fits']
            validate(r,fold,old,y,settings,directory);records.append(r);print(f'Score regression: {len(records)}/{len(folds)} jobs; {new} new',flush=True)
        for path,h in provenance.get('bindings',{}).items():
            if sha256(path)!=h:raise ValueError('Bound inputs changed')
        complete=len(records)==len(folds);targets=summarize(frame,totals,records,old_records,settings,out) if complete else None
        receipt={'complete':complete,'context_hash':ch,'completed_jobs':len(records),'new_jobs':new,'new_regressor_fits':new_fits,'regressor_fits':sum(r['regressor_fits'] for r in records),'scaler_fits':sum(r['scaler_fits'] for r in records),'targets':targets,'validation_test_scored':False,'baseline_replaced':False,'synthetic_records':0}
        atomic_json(out/'completion.json',receipt)
    return receipt


def load_inputs():
    x,frame,old_settings,old_records,provenance=prior.load_inputs()
    path='data/metadata/participants.csv';metadata=pd.read_csv(path,dtype={'participant_id':str}).set_index('participant_id')
    totals=metadata.loc[frame.participant_id,'phq8_score'].to_numpy(float)
    return x,frame,totals,old_settings,old_records,{**provenance,'bindings':{**provenance['bindings'],path:sha256(path)}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');p.add_argument('--max-new-jobs',type=int);a=p.parse_args()
    settings=json.loads(Path('configs/score_regression_accuracy.json').read_text());print(json.dumps(run(*load_inputs(),settings,'results/score_regression_accuracy',resume=a.resume,max_new_jobs=a.max_new_jobs),indent=2))


if __name__=='__main__':
    from .score_regression_accuracy import main as module_main
    module_main()
