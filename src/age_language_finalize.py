"""Training-only final selection, reused validation, and preservation audit."""
import argparse
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import age_language_accuracy as s
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .evaluate import write_report
from .final_evaluation import sha256

OUTPUT = s.OUTPUT


def training_choices(frame, records, old, settings):
    y = frame.severity_class.to_numpy(int); owners = np.zeros(len(y), int)
    if len(records) != len(old) or not records:
        raise ValueError('Incomplete training selection')
    for r, baseline in zip(records, old):
        if any(r[k] != baseline[k] for k in ['seed','fold','fit','heldout']):
            raise ValueError('Training selection ownership differs')
        owners[r['fit']] += 1
    if not (owners == owners[0]).all() or not owners[0]:
        raise ValueError('Unequal tuning contribution')
    labels = np.concatenate([y[r['fit']] for r in records])
    baseline_predictions = np.concatenate([r['candidates'][r['policies']['accuracy']['selection']['index']]['predictions'] for r in old])
    inner = np.concatenate([np.asarray(r['inner_probabilities']) for r in records], axis=1)
    if inner.shape != (len(s.candidates(settings))-1, len(labels), 5):
        raise ValueError('Incomplete inner predictions')
    rows = [{'candidate': {'family':'baseline'}, **s.score(labels, baseline_predictions)}]
    rows += [{'candidate': c, **s.score(labels, s.decide(p, None))} for c,p in zip(s.candidates(settings)[1:], inner)]
    return {'rows': rows, 'choices': {n:s.choose(rows,n) for n in settings['policies']},
            'inner_predictions_per_participant': int(owners[0]),
            'note': 'Optimistic tuning scores; only outer predictions estimate procedure performance.'}


def records_for(inputs):
    completion = json.loads((OUTPUT/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Nested study incomplete')
    _,frame,old_settings,_,_ = inputs
    records = [load_record(OUTPUT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'result.json', completion['context_hash'])
               for fold in s.nested_folds(frame,old_settings)]
    return completion, records


def freeze():
    settings = json.loads(Path('configs/age_language_accuracy.json').read_text())
    inputs = s.load_inputs(); x,frame,_,old,_ = inputs; completion,records = records_for(inputs)
    directory = OUTPUT/'final'
    if directory.exists():
        raise FileExistsError('Preserving frozen final models')
    if s.run(inputs, settings, resume=True)['new_fits']:
        raise ValueError('Unexpected incomplete study')
    selection = training_choices(frame,records,old,settings)
    previous = json.loads(Path('results/project_accuracy_audit_20261027/best_model.json').read_text())['best_model']
    if sha256(previous['path']) != previous['sha256']:
        raise ValueError('Existing best model changed')
    original = pickle.loads(Path(previous['path']).read_bytes())
    if original['fit_participants'] != frame.participant_id.tolist():
        raise ValueError('Existing full-training ownership differs')
    directory.mkdir(); models = {}; fitted = {}; y = frame.severity_class.to_numpy(int)
    with output_lock(OUTPUT), threadpool_limits(limits=settings['cpu_threads']), s.offline_network():
        atomic_json(directory/'training_selection.json', selection)
        projection = s.BlockProjection().fit(x,settings['components'])
        for name,i in selection['choices'].items():
            if not i:
                models[name] = {**previous,'existing_baseline':True}
                continue
            if i not in fitted:
                c = s.candidates(settings)[i]; model = s.BlockClassifier(c,settings).fit(x,y,projection)
                fitted[i] = {'protocol':s.PROTOCOL,'estimator':model,'candidate':c,'fit_participants':frame.participant_id.tolist()}
            state = fitted[i]; path = directory/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
            np.testing.assert_allclose(s.predict_state(pickle.loads(path.read_bytes()),x),s.predict_state(state,x),rtol=0,atol=1e-14)
            models[name] = {'path':str(path),'sha256':sha256(path),'candidate':state['candidate'],'existing_baseline':False,
                            'recorded_age_required':state['candidate']['feature_set']!='language'}
        models['baseline_refit'] = {**previous,'existing_baseline':True}
        paths = [OUTPUT/'protocol.json',directory/'training_selection.json',Path(__file__),Path(s.__file__),
                 Path('configs/age_language_accuracy.json'),s.METADATA,s.CORPUS,
                 Path('data/metadata/participants.csv'),Path('results/semantic_features/embeddings.json'),
                 Path('results/temporal_summary_comparison/participant_features.npz')]
        plan = {'protocol':s.PROTOCOL,'training_ids':frame.participant_id.tolist(),'models':models,
                'context_hash':completion['context_hash'],'bindings':{str(p):sha256(p) for p in paths},
                'new_full_training_fits':len(fitted),'primary':'accuracy','validation_previously_exposed':True,
                'validation_selected_model':False,'test_scoring':False,'independent_confirmation':False}
        atomic_json(directory/'plan.json',{'plan':plan,'plan_hash':digest(plan)})
    return {'new_full_training_fits':len(fitted),'models':models}


def frozen_plan():
    envelope = json.loads((OUTPUT/'final'/'plan.json').read_text());plan=envelope['plan']
    if digest(plan)!=envelope['plan_hash']:
        raise ValueError('Plan changed')
    for p,h in plan['bindings'].items():
        if sha256(p)!=h:
            raise ValueError(f'Bound input changed: {p}')
    for r in plan['models'].values():
        if sha256(r['path'])!=r['sha256']:
            raise ValueError('Final model changed')
    return plan,envelope['plan_hash']


def validation_inputs(plan):
    from .supervised_accuracy_finalize import validation_inputs as original_validation
    x,frame = original_validation(plan)
    x.update(base=x['semantic_av'],**s.extra_features(frame))
    return x,frame


def predict_record(record,x):
    state=pickle.loads(Path(record['path']).read_bytes())
    if record['existing_baseline']:
        return state['estimator'].predict_proba(x[state['candidate']['feature']])
    return s.predict_state(state,x)


def evaluate():
    plan,ph=frozen_plan();x,frame=validation_inputs(plan);rows=[]
    with output_lock(OUTPUT),threadpool_limits(limits=2),s.offline_network():
        for name,r in plan['models'].items():
            directory=OUTPUT/'validation'/name;path=directory/'result.json'
            if path.exists():
                saved=load_record(path,ph)
                if saved['participant_ids']!=frame.participant_id.tolist() or saved['model_sha256']!=r['sha256']:
                    raise ValueError('Validation ownership changed')
                p=np.asarray(saved['probabilities'])
            else:
                p=predict_record(r,x);s.decide(p,None)
                save_record(path,{'participant_ids':frame.participant_id.tolist(),'model_sha256':r['sha256'],'probabilities':p.tolist()},ph)
            write_report(frame,p,directory,name,42,'reused_official_validation')
            rows.append({'model':name,**s.score(frame.severity_class.to_numpy(int),p.argmax(1))})
        pd.DataFrame(rows).to_csv(OUTPUT/'validation_scores.csv',index=False)
        atomic_json(OUTPUT/'validation_receipt.json',{'plan_hash':ph,'participants':len(frame),'original_participants_only':True,
            'validation_used_for_selection':False,'independent_confirmation':False,'test_scored':False,'synthetic_rows':0})
    return rows


def verify():
    settings=json.loads(Path('configs/age_language_accuracy.json').read_text())
    inputs=s.load_inputs();x,frame,old_settings,old,_=inputs;completion,records=records_for(inputs)
    plan,ph=frozen_plan();vx,vf=validation_inputs(plan)
    paths=list((OUTPUT/'jobs').glob('*/*'))+list((OUTPUT/'final').glob('*'))+list((OUTPUT/'validation').glob('*/result.json'))
    before={str(p):[sha256(p),p.stat().st_mtime_ns] for p in paths if p.is_file()}
    def forbidden(*a,**kw):
        raise AssertionError('No fitting/inference during completed resume')
    with patch.object(s,'job',forbidden),patch.object(s,'predict_state',forbidden),patch.object(pickle,'loads',forbidden):
        result=s.run(inputs,settings,resume=True);evaluate()
    if result['new_fits'] or before!={p:[sha256(p),Path(p).stat().st_mtime_ns] for p in before}:
        raise ValueError('Completed resume mutated saved artifacts')
    if training_choices(frame,records,old,settings)!=json.loads((OUTPUT/'final'/'training_selection.json').read_text()):
        raise ValueError('Final selection differs')
    replayed=0;rows=[]
    with threadpool_limits(limits=2),s.offline_network():
        for fold,r in zip(s.nested_folds(frame,old_settings),records):
            for name,selected in r['selected'].items():
                state=pickle.loads((OUTPUT/'jobs'/f"seed{r['seed']}_fold{r['fold']}"/f'{name}.pkl').read_bytes())
                if state['fit']!=fold['fit'] or state['heldout']!=fold['heldout']:
                    raise ValueError('Model fit ownership changed')
                if state['candidate']['family']!='baseline':
                    model=state['estimator'];train=s.subset(x,fold['fit'])
                    if model.real_count!=len(fold['fit']):
                        raise ValueError('Real fit count changed')
                    np.testing.assert_allclose(model.projection.base_scaler.mean_,train['base'].mean(0),rtol=0,atol=1e-12)
                    for block in settings['feature_sets']:
                        np.testing.assert_allclose(model.projection.imputers[block].statistics_,np.nanmedian(train[block],axis=0),rtol=0,atol=1e-12)
                        z=model.projection.imputers[block].transform(train[block])
                        np.testing.assert_allclose(model.projection.scalers[block].mean_,z.mean(0),rtol=0,atol=1e-12)
                p=s.predict_state(state,s.subset(x,fold['heldout']))
                np.testing.assert_allclose(p,selected['probabilities'],rtol=0,atol=1e-14);replayed+=1
                fit_accuracy=float((s.predict_state(state,s.subset(x,fold['fit'])).argmax(1)==frame.severity_class.to_numpy(int)[fold['fit']]).mean())
                if fit_accuracy!=selected['real_training_accuracy']:
                    raise ValueError('Fitting accuracy differs')
                rows.append({'policy':name,'seed':r['seed'],'fold':r['fold'],'training_accuracy':fit_accuracy,
                    'inner_accuracy':r['options'][selected['selection']['index']]['accuracy'],
                    'outer_accuracy':s.score(frame.severity_class.to_numpy(int)[fold['heldout']],p.argmax(1))['accuracy']})
        for name,r in plan['models'].items():
            saved=load_record(OUTPUT/'validation'/name/'result.json',ph)
            np.testing.assert_allclose(predict_record(r,vx),saved['probabilities'],rtol=0,atol=1e-14);replayed+=1
    pd.DataFrame(rows).to_csv(OUTPUT/'overfitting_audit.csv',index=False)
    summary=pd.read_csv(OUTPUT/'summary.csv').set_index('model')
    previous=json.loads(Path('results/project_accuracy_audit_20261027/best_model.json').read_text())['best_model']
    improved=bool(summary.loc['accuracy','accuracy_mean']>summary.loc['baseline','accuracy_mean'])
    manifest={'retained':'age_language_exploratory' if improved else 'original_verified_49.08_baseline',
        'replacement':improved,'original_model_overwritten':False,'primary':'accuracy',
        'baseline_nested_accuracy':float(summary.loc['baseline','accuracy_mean']),
        'new_nested_accuracy':float(summary.loc['accuracy','accuracy_mean']),
        'best_model':plan['models']['accuracy'] if improved else previous,'research_models':plan['models'],
        'above_50_percent':bool(summary.loc['accuracy','accuracy_mean']>.5),
        'two_percentage_point_target_reached':bool(summary.loc['accuracy','accuracy_mean']>=settings['target_accuracy']),
        'selected_using_validation':False,'independent_confirmation':False}
    atomic_json(OUTPUT/'best_valid_model.json',manifest)
    before_all=json.loads(Path('results/age_language_preservation/before.json').read_text())
    changed=[p for p,v in before_all.items() if not Path(p).is_file() or Path(p).stat().st_mtime_ns!=v['mtime_ns'] or sha256(p)!=v['sha256']]
    if changed:
        raise ValueError(f'Pre-existing files changed: {changed}')
    result={'passed':True,'checkpoints_replayed':replayed,'zero_fit_resume':True,
        'previous_files_and_mtimes_preserved':len(before_all),'saved_artifacts_preserved_on_resume':len(before),
        'new_classifier_fits':completion['total_fits']+plan['new_full_training_fits'],'old_models_refitted':0,
        'validation_participants':len(vf),'test_scored':False}
    atomic_json(OUTPUT/'verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','evaluate','verify'])
    args=parser.parse_args();print(json.dumps(globals()[args.action](),indent=2))


if __name__=='__main__':
    main()
