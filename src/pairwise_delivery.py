"""Save final paired-model diagnostics, reproducible evidence and a human report."""
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from . import pairwise_accuracy as s
from .batch_inference import atomic_json, load_record
from .final_evaluation import sha256
from .pairwise_inference import frozen_model, predict_arrays


def deliver():
    out=s.OUTPUT
    if (out/'delivery_manifest.json').exists():
        raise FileExistsError('Preserving completed delivery')
    inputs=s.load_inputs();x,frame,old_settings,old,_=inputs;y=frame.severity_class.to_numpy(int)
    complete=json.loads((out/'completion.json').read_text());verified=json.loads((out/'verification.json').read_text())
    assert complete['complete'] and verified['passed']
    plan=json.loads((out/'final/plan.json').read_text())['plan']
    diagnostics=[];head_rows=[];outer_diagnostics=[]
    with threadpool_limits(limits=2):
        for f in s.nested_folds(frame,old_settings):
            directory=out/'jobs'/f"seed{f['seed']}_fold{f['fold']}"
            r=load_record(directory/'result.json',complete['context_hash'])
            for d in r['fitting_diagnostics']:
                assert d['converged'] and d['real_fitting_participants']==len(d['fit'])
                assert len(d['heads'])==10
                inner_y=y[d['fit']]
                for head in d['heads']:
                    i,j=head['classes'];expected=np.flatnonzero((inner_y==i)|(inner_y==j))
                    assert head['positions']==expected.tolist()
                    assert head['binary_class_counts']==np.bincount((inner_y[expected]==j).astype(int),minlength=2).tolist()
                    head_rows.append({'seed':f['seed'],'outer_fold':f['fold'],'candidate_index':d['candidate_index'],
                        'inner_fit_hash':sha256_of_indices(d['fit']),**head})
                diagnostics.append({'seed':f['seed'],'outer_fold':f['fold'],**d})
            for policy,selection in r['selected'].items():
                if selection['candidate']['family']=='pairwise_logistic':
                    state=pickle.loads((directory/f'{policy}.pkl').read_bytes())
                    outer_diagnostics.append({'seed':f['seed'],'outer_fold':f['fold'],'policy':policy,
                        **state['estimator'].diagnostics(s.subset(x,f['fit']),y[f['fit']])})
        record,state,_=frozen_model('pairwise_only')
        p=predict_arrays(record,state,x)
        index=frame.participant_id.tolist().index('487')
        raw=json.loads((out/'raw_training_example.json').read_text())
        np.testing.assert_allclose(raw['decision_scores'],p[index:index+1],rtol=0,atol=1e-12)
        final_diag=state['estimator'].diagnostics(x,y)
    np.savez_compressed(out/'final_training_predictions.npz',participant_id=frame.participant_id.to_numpy(str),decision_scores=p)
    atomic_json(out/'inner_fitting_diagnostics.json',diagnostics)
    atomic_json(out/'binary_head_diagnostics.json',head_rows)
    atomic_json(out/'outer_fitting_diagnostics.json',outer_diagnostics)
    atomic_json(out/'final_fitting_diagnostics.json',final_diag)
    compact=[{k:v for k,v in d.items() if k not in ['heads','fit','heldout','iterations','class_counts']} |
        {'iterations_max':max(d['iterations'])} for d in diagnostics]
    pd.DataFrame(compact).to_csv(out/'inner_fitting_diagnostics.csv',index=False)
    atomic_json(out/'raw_inference_verification.json',{'passed':True,'training_participant':'487',
        'maximum_probability_difference':float(np.max(np.abs(np.asarray(raw['decision_scores'])-p[index:index+1]))),
        'validation_test_scored':False,'new_fits':0,'candidate_sha256':record['sha256']})
    summary=pd.read_csv(out/'summary.csv').set_index('model')
    matrices=json.loads((out/'summed_confusion_matrices.json').read_text())['matrices']
    names=['None/minimal','Mild','Moderate','Moderately severe','Severe']
    class_rows=[]
    for policy in ['language_control','accuracy','pairwise_only']:
        tables=[pd.read_csv(out/'oof'/f'{policy}_seed{seed}/predictions.csv',dtype={'participant_id':str}) for seed in [42,137,2026]]
        for table in tables:
            assert table.participant_id.tolist()==frame.participant_id.tolist()
            np.testing.assert_array_equal(table.severity_class,y)
        labels=np.tile(y,3);pred=np.concatenate([t.predicted_label.to_numpy(int) for t in tables])
        precision,recall,f1,support=precision_recall_fscore_support(labels,pred,labels=range(5),zero_division=0)
        for c in range(5):
            class_rows.append({'model':policy,'class':c,'class_name':names[c],'unique_participants':int((y==c).sum()),
                'repeated_support':int(support[c]),'precision_pooled':float(precision[c]),'recall':float(recall[c]),'f1_pooled':float(f1[c])})
    pd.DataFrame(class_rows).to_csv(out/'per_class_precision_recall.csv',index=False)
    previous=Path('results/confidence_accuracy/experiment_ledger_extended.csv');ledger=pd.read_csv(previous)
    new=[{'study':'pairwise_accuracy','variant':json.dumps({'model':policy}),'nested_accuracy':summary.loc[policy,'accuracy_mean'],
        'macro_f1':summary.loc[policy,'macro_f1_mean'],'summary_path':str(out/'summary.csv'),'summary_sha256':sha256(out/'summary.csv'),
        'evidence_scope':'Original163 training participants, identical nested folds, exploratory long-history comparison; no validation/test.',
        'repeat_action':'Do not repeat two-config pairwise logistic unchanged. Retained49.28 model; no promotion.'} for policy in ['accuracy','pairwise_only']]
    pd.concat([ledger,pd.DataFrame(new)],ignore_index=True).to_csv(out/'experiment_ledger_extended.csv',index=False)
    old_files=json.loads(Path('results/pairwise_preservation/before.json').read_text())
    changed=[p for p,v in old_files.items() if not Path(p).is_file() or Path(p).stat().st_mtime_ns!=v['mtime_ns'] or sha256(p)!=v['sha256']]
    assert not changed,changed[:5]
    logdir=out/'logs';logdir.mkdir()
    for source in Path('/tmp').glob('project_v2_pairwise_*.log'):
        if 'delivery' not in source.name:
            shutil.copyfile(source,logdir/source.name)
    manifest={'complete':True,'target_accuracy':.60,'target_reached':False,
        'best_observed_accuracy':float(summary.loc['language_control','accuracy_mean']),
        'new_primary_accuracy':float(summary.loc['accuracy','accuracy_mean']),
        'new_pairwise_only_accuracy':float(summary.loc['pairwise_only','accuracy_mean']),
        'retained_checkpoint':plan['models']['language_control'],'retained_numeric_bundle':'checkpoints/best_observed_4928',
        'promotion':False,'independent_confirmation':False,'new_nested_pipeline_fits':complete['total_fits'],
        'new_nested_binary_head_fits':complete['new_binary_head_fits'],
        'new_full_training_pipeline_fits':plan['new_full_training_fits'],'new_full_training_binary_head_fits':10*plan['new_full_training_fits'],
        'focused_tests_passed':8,'old_models_refitted':0,'old_files_and_mtimes_preserved':len(old_files),
        'validation_test_scored':False,'external_data_used':False,'speaker_identity_guessed':False,
        'prior_pipeline_verification_reused':'results/confidence_accuracy/pipeline_verification.json',
        'next_step':'Verified participant-speech annotations and more compatible labeled people, with a fresh evaluation cohort; current data/model evidence does not establish60%.',
        'bindings':{str(path):sha256(path) for path in [s.CONFIG,Path(s.__file__),out/'verification.json',out/'raw_inference_verification.json',
            out/'language_paired_comparisons.csv',out/'experiment_ledger_extended.csv',Path(__file__)]}}
    atomic_json(out/'delivery_manifest.json',manifest)
    write_report(out,summary,matrices,pd.DataFrame(class_rows),pd.DataFrame(compact),manifest)
    print(json.dumps(manifest,indent=2))


def sha256_of_indices(indices):
    from .common import digest
    return digest(indices)


def write_report(out,summary,matrices,classes,diagnostics,manifest):
    report=Path('PAIRWISE_ACCURACY_RESULTS_20261028.md')
    if report.exists():
        raise FileExistsError('Preserving previous report')
    def link(path,label=None):
        return f'[{label or Path(path).name}]({Path(path).resolve()})'
    comparisons=pd.read_csv(out/'language_paired_comparisons.csv')
    changes=pd.read_csv(out/'language_error_changes.csv').set_index('model')
    fit=pd.read_csv(out/'overfitting_audit.csv').groupby('policy').training_accuracy.mean()
    text='''**Project_V2: pairwise logistic continuation — 28 October 2026**

The retained best observed repeated nested-CV accuracy remains **49.28%**. The new primary procedure scored **49.08%**, and pairwise-only scored **46.01%**. The requested **60% was not achieved**. Neither candidate replaces the preserved model. These figures describe repeated model-selection procedures on the original 163 training participants, not independent unseen-test accuracy or a guarantee for individual predictions.

**Implemented change and bounded budget**

The existing language control misclassifies many Mild/Moderate cases as None/minimal (151 majority predictions among their 186 repeated predictions). A pairwise logistic objective was tested to learn separate class contrasts without a hierarchical gate. The hypothesis is that these contrasts might better preserve minority separation; it is not a verified explanation for prior errors.

This specific implementation was absent from the ledger/source search. Prior RBF SVMs also used pairwise class decomposition, so pairwise classification in general is not claimed as a new idea for this project. Unlike those models or the failed conditional hierarchy, this implementation has ten linear logistic heads, shared fold-local preprocessing and deterministic least-squares combination of their pair log odds.

Original semantic/AV512 features and fixed age-free language16 are unchanged. Within every fitting fold, fit a scaler and shared PCA16 to the original base features; separately scale language features and multiply that block by 0.5. Each binary head fits only members of its two classes, with square-root frequency weights normalized over that pair. This gives 32 coordinates per head, 330 total coefficients/intercepts and strong L2 regularization. The shared projection avoids estimating separate PCAs from tiny rare-class pairs.

Exactly two configurations were frozen before training: C=0.001 and C=0.01. Maximum budget: 135 nested pipelines / 1,350 binary heads and two full-training pipelines. Actual: 105 nested pipelines / 1,050 binary heads and one full-training pipeline / ten heads. No old model was retrained, no external people or synthetic rows were added, and no follow-up parameter search was made after outer results.

Original five outer / three inner participant folds, seeds 42/137/2026. Learned preprocessing and class weights fit within the relevant fitting fold. Inner held-out correct count, then macro-F1 and declared order select the configuration; the primary procedure may choose the cached control. Full-training choice uses cached inner predictions only (12 held-out contributions per participant). No outer, validation or test labels select a checkpoint. No questionnaire scores, answers, IDs or label-derived features enter inference.

**Matched evaluation**

There are 489 repeated predictions on **163 independent people**, including the same seven Severe participants three times. Aggregate metrics below average the three complete repetitions.

| Procedure | Correct /489 | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Always predict majority | 231 | 47.24% | 0.1283 | 20.00% |
'''
    for name,label in [('baseline','Older preserved baseline'),('language_control','Retained language control'),('accuracy','Primary inner-selected procedure'),('pairwise_only','Pairwise-only procedure')]:
        r=summary.loc[name];text+=f'| {label} | {int(np.trace(matrices[name]))} | {100*r.accuracy_mean:.2f}% | {r.macro_f1_mean:.4f} | {100*r.balanced_accuracy_mean:.2f}% |\n'
    text+='\nThe new primary score happens to equal the older baseline score; the two procedures and their predictions are different. All new comparisons below use the **49.28%** language control.\n\n'
    for name,label in [('accuracy','Primary'),('pairwise_only','Pairwise-only')]:
        c=changes.loc[name];ci=comparisons[(comparisons.model==name)&(comparisons.metric=='accuracy')].iloc[0]
        text+=f'{label}: **{int(c.gained)} correct predictions gained, {int(c.lost)} lost**, net {int(c.gained-c.lost):+d}/489. {int(c.net_better_people)} participants improve and {int(c.net_worse_people)} worsen across repeats. Accuracy change {100*ci.difference:+.2f} percentage points, paired 95% interval **[{100*ci.ci95_low:+.2f}, {100*ci.ci95_high:+.2f}]**.\n\n'
    text+='''Uncertainty uses 5,000 participant-bootstrap resamples, seed 20261025, carrying all three predictions with each sampled participant. Intervals are conditional on the saved fitted models/folds; they do not include retraining uncertainty or correct for the long adaptive experiment history. There is no supported improvement. The pairwise-only result is worse in this conditional comparison. No newly untouched original test set is established after historical reuse; no validation or test scoring occurred here.

The primary result has a net two additional Mild correct predictions, offset by two fewer None/minimal and one fewer Moderately severe correct predictions. Moderate and Severe correct counts are unchanged. Primary majority predictions decline from 411 to 405; pairwise-only increases them to 445/489 and reduces minority recall. The hypothesis did not produce a useful overall improvement.

**Per-class performance**

P/R means pooled precision / recall in percent, with unique participant supports. Pooled precision is computed from summed confusion matrices, not averaged across folds. Recall equals the mean across repetitions because each repetition has identical class supports.

| Class | People | Retained P/R | Primary P/R | Pairwise-only P/R |
|---|---:|---:|---:|---:|
'''
    for c in range(5):
        rows=[classes[(classes.model==n)&(classes['class']==c)].iloc[0] for n in ['language_control','accuracy','pairwise_only']]
        text+=f'| {rows[0].class_name} | {int(rows[0].unique_participants)} | '+' | '.join(f'{100*r.precision_pooled:.2f} / {100*r.recall:.2f}' for r in rows)+' |\n'
    text+='\nSummed confusion matrices: rows actual, columns predicted, order None/minimal, Mild, Moderate, Moderately severe, Severe.\n'
    for name in ['language_control','accuracy','pairwise_only']:
        text+=f'\n{name}:\n\n```text\n'+'\n'.join(' '.join(f'{v:3d}' for v in row) for row in matrices[name])+'\n```\n'
    text+='\n**Fitting diagnostics and checkpoint verification**\n\n'
    for name,label in [('language_control','Retained control'),('accuracy','Primary'),('pairwise_only','Pairwise-only')]:
        text+=f'{label}: mean outer-fitting accuracy **{100*fit[name]:.2f}%**, held-out **{100*summary.loc[name,"accuracy_mean"]:.2f}%**.\n\n'
    for C,group in diagnostics.groupby('C'):
        text+=f'New inner fits with C={C:g}: mean training {100*group.fitting_accuracy.mean():.2f}%, inner held out {100*group.inner_accuracy.mean():.2f}%.\n\n'
    text+=f'''All new binary-head fits converged; convergence warnings are treated as errors. The saved inner runs have maximum per-pipeline iteration counts from {int(diagnostics.iterations_max.min())} to {int(diagnostics.iterations_max.max())}, below the 3,000 cap. These are converged penalized logistic fits, not neural epochs; validation-based early stopping is not used. Saved diagnostics include head-specific class ownership, sample counts, iterations, coefficient norms, training accuracy, overall fitting log loss and inner held-out performance. No high training score is substituted for held-out accuracy.

Eight focused tests pass, including held-out feature/target invariance, class-pair fitting ownership, exact reconstruction of coherent pair log odds, checkpoint serialization, inner-only final selection, corruption rejection and zero-fit resume. Verification replays 48 checkpoints and confirms fitting-only statistics and pair weights. Raw-file candidate inference for original training participant 487 matches the cached feature prediction within 1e−12; this is an inference check, not a new evaluation. {manifest['old_files_and_mtimes_preserved']:,} pre-existing source/config/test/result/checkpoint/report files retain their hashes and modification times.

The full-training primary choice is the unchanged control. A separate pairwise-only C=0.001 checkpoint, including its preprocessing and ten fitted heads, is archived for reproducibility and is not promoted. The current model's previous 49.28% versus 49.08% advantage remains a small, unsupported historical point gain; no independent confirmation is claimed.

**Remaining limitations and next step**

The prior pipeline audit and latest verification were reused rather than repeated. Participant/class mapping and the original preprocessing bindings remain valid. The source still has no verified speaker labels; no speaker identities were guessed. No StudentSADD location has become available, and the failed EMU transfer was not repeated. Current Severe support remains seven people, and the original Mild/Moderate features show weak held-out separation. A different model name or additional synthetic rows cannot establish a 60% result.

Retain the 49.28% model and stop this bounded search. The evidence-supported next step is verified participant-speech annotations and/or additional compatible, genuinely labeled participants, especially minority classes, with a fresh independent evaluation cohort reserved before development. This is a data and evaluation requirement, not a promise that more data will automatically yield 60%.

**Artifacts and inference**

'''
    for path,label in [(out/'delivery_manifest.json','Final decision and preservation manifest'),(out/'experiment_ledger_extended.csv','Updated experiment ledger (100 rows)'),
        (s.CONFIG,'Frozen two-configuration budget'),(out/'protocol.json','Frozen participant folds and input/code hashes'),
        (out/'final/plan.json','Final checkpoints and preprocessing contract'),(out/'oof','All held-out predictions'),
        (out/'language_paired_comparisons.csv','Participant-paired confidence intervals against49.28'),
        (out/'language_participant_changes.csv','Participant-level gains and losses'),(out/'per_class_precision_recall.csv','Every class precision and recall'),
        (out/'inner_fitting_diagnostics.json','Inner fitting diagnostics'),(out/'binary_head_diagnostics.json','Binary-head ownership and convergence'),
        (out/'verification.json','Checkpoint replay and preservation verification'),(out/'raw_inference_verification.json','Raw inference parity'),
        (Path('checkpoints/best_observed_4928'),'Retained numeric49.28 bundle'),(Path('results/age_language_accuracy/final/without_age.pkl'),'Unchanged retained checkpoint')]:
        text+='- '+link(path,label)+'\n'
    text+='''
Run the retained model from the project folder using the checked participant-ID entry point. Example using training participant 487 only to demonstrate inference; use a new output filename:

```bash
cd /Users/lalja/Documents/Project_V2
.venv/bin/python -m src.checked_inference \\
  --transcript /Users/lalja/Documents/Project_V2/dataset/487_P/487_Transcript.csv \\
  --audio-csv /Users/lalja/Documents/Project_V2/dataset/487_P/features/487_OpenSMILE2.3.0_mfcc.csv \\
  --visual-csv /Users/lalja/Documents/Project_V2/dataset/487_P/features/487_OpenFace2.1.0_Pose_gaze_AUs.csv \\
  --output /private/tmp/project_v2_retained_pairwise_round_prediction.json
```

For the archived research candidate, use the same arguments with `python -m src.pairwise_inference --policy pairwise_only`. No questionnaire inputs are required. `python -m src.pairwise_accuracy run --resume` verifies completed training records with zero new fits. Both inference entry points preserve existing prediction outputs.
'''
    report.write_text(text)
    atomic_json(out/'report_receipt.json',{'report':str(report.resolve()),'sha256':sha256(report),
        'delivery_manifest_sha256':sha256(out/'delivery_manifest.json')})


if __name__=='__main__':
    deliver()
