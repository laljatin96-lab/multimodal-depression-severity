"""Accuracy-first nested selection of supplied deep features and baseline soft voting."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import chunk_severe_study as cached
from . import supplied_deep_features as deep
from .batch_inference import atomic_json,load_record,save_record,output_lock
from .common import digest
from .final_evaluation import sha256
from .severe_case_diagnosis import ROOT
from .severe_detection_study import score,summarize
from .encoder_kernel_accuracy import offline_network

PROTOCOL='supplied-deep-nested-accuracy-fusion-v1'
FEATURES={'deep_audio':['densenet','vgg_audio'],'deep_visual':['resnet_face'],
          'deep_av':['densenet','vgg_audio','resnet_face'],
          'deep_av_text':['densenet','vgg_audio','resnet_face','semantic','av']}


class Projector:
    def __init__(self,settings):self.settings=settings
    def fit(self,x,indices):
        self.transforms={};self.fit_indices=list(indices)
        for name in sorted({n for names in FEATURES.values() for n in names}):
            k=self.settings['text_av_components'] if name in ['semantic','av'] else self.settings['deep_components']
            model=make_pipeline(StandardScaler(),PCA(n_components=k,svd_solver='full'),StandardScaler())
            model.fit(x[name][indices]);self.transforms[name]=model
        return self
    def transform(self,x,indices):return {n:m.transform(x[n][indices]) for n,m in self.transforms.items()}


def matrix(z,feature):return np.column_stack([z[n] for n in FEATURES[feature]])


def candidates(settings):
    return [{'family':'baseline'}]+[{'family':'deep_logistic','feature':feature,'C':c,'weight':w}
        for feature in settings['feature_inputs'] for c in settings['C'] for w in settings['weight_powers']]


def fit_head(z,y,candidate,settings):
    model=LogisticRegression(C=candidate['C'],solver='lbfgs',max_iter=settings['max_iter'])
    counts=np.bincount(y,minlength=5)
    if (counts==0).any():raise ValueError('All fitting classes required')
    weights=(len(y)/(5*counts[y]))**candidate['weight'];weights*=len(y)/weights.sum()
    model.fit(matrix(z,candidate['feature']),y,sample_weight=weights if candidate['weight'] else None)
    return model


def search_options(inner,y,settings):
    options=[]
    for i,candidate in enumerate(candidates(settings)):
        weights=[1.] if i==0 else settings['baseline_weights']
        for weight in weights:
            p=weight*np.asarray(inner[0])+(1-weight)*np.asarray(inner[i])
            if p.shape!=(len(y),5) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid inner probabilities')
            options.append({'candidate_index':i,'baseline_weight':weight,**score(y,p.argmax(1))})
    return options


def choose(options,policy):
    if policy not in ['accuracy_fusion','single_model']:raise ValueError('Unknown policy')
    eligible=[i for i,r in enumerate(options) if policy=='accuracy_fusion' or r['baseline_weight'] in [0,1]]
    index=max(eligible,key=lambda i:(options[i]['correct'],options[i]['macro_f1'],-i))
    return {'index':index}


def predict_state(state,x,indices):
    w=state['baseline_weight'];p=np.zeros((len(indices),5))
    if w:
        p+=w*cached.five_probabilities(state['baseline'],x[state['baseline_feature']][indices])
    if w<1:
        z=state['projector'].transform(x,indices)
        p+=(1-w)*state['head'].predict_proba(matrix(z,state['candidate']['feature']))
    return p


def job(x,y,fold,old,settings,out):
    start=time.monotonic();search=candidates(settings);inner=[old['cached_inner_probabilities']]
    arrays=[np.full((len(y),5),np.nan) for _ in search[1:]]
    for part in fold['inner']:
        projector=Projector(settings).fit(x,part['fit'])
        train=projector.transform(x,part['fit']);held=projector.transform(x,part['heldout'])
        for candidate,p in zip(search[1:],arrays):
            model=fit_head(train,y[part['fit']],candidate,settings)
            p[part['heldout']]=model.predict_proba(matrix(held,candidate['feature']))
    inner.extend(p[fold['fit']].tolist() for p in arrays)
    options=search_options(inner,y[fold['fit']],settings)
    choices={name:choose(options,name) for name in settings['policies']}
    indices={options[c['index']]['candidate_index'] for c in choices.values()}-{0}
    projector=Projector(settings).fit(x,fold['fit']) if indices else None
    transformed=projector.transform(x,fold['fit']) if projector else None
    fitted={i:fit_head(transformed,y[fold['fit']],search[i],settings) for i in indices}
    baseline_feature=old['policies']['accuracy']['candidate']['feature']
    with (ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl').open('rb') as f:baseline=pickle.load(f)['estimator']
    np.testing.assert_allclose(cached.five_probabilities(baseline,x[baseline_feature][fold['heldout']]),old['policies']['accuracy']['probabilities'],rtol=0,atol=1e-14)
    out.mkdir(parents=True,exist_ok=True);selected={}
    for name,choice in choices.items():
        option=options[choice['index']];i=option['candidate_index'];candidate=search[i]
        state={'protocol':PROTOCOL,'candidate':candidate,'baseline_weight':option['baseline_weight'],
            'baseline':baseline,'baseline_feature':baseline_feature,'projector':projector if i else None,'head':fitted.get(i),
            'fit':fold['fit'],'heldout':fold['heldout']}
        p=predict_state(state,x,fold['heldout']);path=out/f'{name}.pkl'
        with path.open('wb') as f:pickle.dump(state,f)
        with path.open('rb') as f:restored=pickle.load(f)
        np.testing.assert_allclose(p,predict_state(restored,x,fold['heldout']),rtol=0,atol=1e-14)
        selected[name]={'selection':choice,'candidate':{**candidate,'baseline_weight':option['baseline_weight']},
            'threshold':None,'probabilities':p.tolist(),'predictions':p.argmax(1).tolist(),'checkpoint_sha256':sha256(path)}
    return {'seed':fold['seed'],'fold':fold['fold'],'fit':fold['fit'],'heldout':fold['heldout'],
        'inner_probabilities':inner,'options':options,'selected':selected,
        'head_fits':(len(search)-1)*len(fold['inner'])+len(indices),
        'projector_fits':len(fold['inner'])+bool(indices),'seconds':time.monotonic()-start}


def validate(r,fold,old,y,settings,out):
    if any(r[k]!=fold[k] for k in ['seed','fold','fit','heldout']):raise ValueError('Fold changed')
    if len(r['inner_probabilities'])!=len(candidates(settings)):raise ValueError('Incomplete inner search')
    np.testing.assert_array_equal(r['inner_probabilities'][0],old['cached_inner_probabilities'])
    expected=search_options(r['inner_probabilities'],y[fold['fit']],settings)
    if r['options']!=expected or set(r['selected'])!=set(settings['policies']):raise ValueError('Search changed')
    indices=set()
    for name,result in r['selected'].items():
        choice=choose(expected,name);option=expected[choice['index']];i=option['candidate_index']
        candidate={**candidates(settings)[i],'baseline_weight':option['baseline_weight']}
        if result['selection']!=choice or result['candidate']!=candidate:raise ValueError('Choice changed')
        p=np.asarray(result['probabilities'])
        if p.shape!=(len(fold['heldout']),5) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1) or p.argmax(1).tolist()!=result['predictions']:raise ValueError('Invalid outer predictions')
        if sha256(out/f'{name}.pkl')!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if i:indices.add(i)
    if r['head_fits']!=(len(candidates(settings))-1)*len(fold['inner'])+len(indices) or r['projector_fits']!=len(fold['inner'])+bool(indices):raise ValueError('Fit budget changed')


def run(x,frame,old_settings,old_records,provenance,settings,output,resume=False,max_new_jobs=None):
    out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserving existing run')
    if resume and not (out/'protocol.json').exists():raise ValueError('Resume requires protocol')
    if max_new_jobs is not None and max_new_jobs<1:raise ValueError('Positive pilot budget required')
    folds=cached.baseline.nested_folds(frame,old_settings);y=frame.severity_class.to_numpy(int)
    if len(folds)!=settings['expected_jobs'] or len(folds)!=len(old_records) or (len(candidates(settings))-1)*old_settings['inner_folds']!=settings['inner_head_fits_per_job']:raise ValueError('Budget/coverage mismatch')
    if frame.participant_id.duplicated().any() or set(y)!=set(range(5)):raise ValueError('Invalid real participants')
    for a in x.values():
        if len(a)!=len(y) or not np.isfinite(a).all():raise ValueError('Invalid features')
    ctx={'protocol':PROTOCOL,'settings':settings,'old_settings':old_settings,'training':frame.to_dict('records'),
        'folds':folds,'baseline_records':old_records,'provenance':provenance,'code_sha256':sha256(__file__),
        'input_hashes':{n:digest(a.tolist()) for n,a in x.items()},
        'support_hashes':{n:sha256(Path(__file__).with_name(n)) for n in ['chunk_severe_study.py','supplied_deep_features.py','severe_detection_study.py','evaluate.py','raw_frame_uncertainty.py']},
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
            else:
                r=job(x,y,fold,old,settings,directory);save_record(path,r,ch);new+=1;new_fits+=r['head_fits']
            validate(r,fold,old,y,settings,directory);records.append(r)
            print(f'Deep fusion: {len(records)}/{len(folds)} jobs; {new} new',flush=True)
        for path,h in provenance.get('bindings',{}).items():
            if sha256(path)!=h:raise ValueError('Bound input changed during fitting')
        complete=len(records)==len(folds)
        targets=summarize(frame,records,old_records,settings,out) if complete else None
        receipt={'complete':complete,'context_hash':ch,'completed_jobs':len(records),'new_jobs':new,'new_head_fits':new_fits,
            'head_fits':sum(r['head_fits'] for r in records),'projector_fits':sum(r['projector_fits'] for r in records),
            'targets':targets,'validation_test_scored':False,'baseline_replaced':False,'synthetic_records':0}
        atomic_json(out/'completion.json',receipt)
    return receipt


def load_inputs():
    x,frame,old_settings,old_records,provenance=cached.load_inputs()
    records,bindings=deep.load_verified();by_id={r['participant_id']:r for r in records}
    for name in deep.SPECS:x[name]=np.asarray([by_id[pid]['features'][name]['summary'] for pid in frame.participant_id])
    x['av']=x['semantic_av'][:,384:]
    return x,frame,old_settings,old_records,{**provenance,'bindings':{**provenance['bindings'],**bindings},'deep_feature_coverage':len(records)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');p.add_argument('--max-new-jobs',type=int)
    a=p.parse_args();inputs=load_inputs();settings=json.loads(Path('configs/deep_fusion_accuracy.json').read_text())
    print(json.dumps(run(*inputs,settings,'results/deep_fusion_accuracy',resume=a.resume,max_new_jobs=a.max_new_jobs),indent=2))


if __name__=='__main__':
    from .deep_fusion_accuracy import main as module_main
    module_main()
