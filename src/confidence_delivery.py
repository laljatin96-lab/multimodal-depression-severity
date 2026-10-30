"""Finalize confidence experiment evidence without training or test evaluation."""
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from . import confidence_accuracy as s
from .batch_inference import atomic_json, load_record
from .common import digest, severity
from .final_evaluation import sha256
from .raw_frame_uncertainty import weighted_metrics


def finalize():
    out = s.OUTPUT
    if (out/'delivery_manifest.json').exists():
        raise FileExistsError('Preserving completed delivery')
    x, frame, old_settings, old, provenance = s.load_inputs()
    y = frame.severity_class.to_numpy(int)
    verification = json.loads((out/'verification.json').read_text())
    if not verification['passed']:
        raise ValueError('Verification incomplete')
    # Reuse full source audit; check live source-label bindings and current training mapping.
    official = Path('data/metadata/official_edaic')
    imported = json.loads((official/'import_report.json').read_text())
    for name, expected in imported['input_sha256'].items():
        if sha256(official/name) != expected:
            raise ValueError('Official label source changed since reconciliation')
    metadata = pd.read_csv('data/metadata/participants.csv', dtype={'participant_id':str})
    if metadata.participant_id.duplicated().any():
        raise ValueError('Duplicate participant in metadata')
    training = metadata.set_index('participant_id').loc[frame.participant_id]
    if not training.split.eq('train').all():
        raise ValueError('Unexpected original evaluation participants')
    np.testing.assert_array_equal(training.severity_class, y)
    np.testing.assert_array_equal(training.phq8_score.map(severity), y)
    detail = pd.read_csv(official/'detailed_lables.csv',dtype={'Participant':str}).set_index('Participant').loc[frame.participant_id]
    item_sum = detail.filter(regex='^PHQ8_[1-8]_').apply(pd.to_numeric).sum(axis=1).to_numpy()
    np.testing.assert_array_equal(item_sum, training.phq8_score)
    from .semantic_pipeline_audit import check_folds
    fold_checks = check_folds(s.nested_folds(frame, old_settings), y, old_settings['fold_seeds'])
    corpus = {r['participant_id']:r for r in json.loads(Path('results/transcript_corpus/participants.json').read_text())}
    assert all(corpus[p]['words'] > 0 and corpus[p]['split'] == 'train' for p in frame.participant_id)
    assert len({corpus[p]['text_sha256'] for p in frame.participant_id}) == len(y)
    assert all(np.isfinite(a).all() and len(np.unique(a, axis=0)) == len(y) for a in x.values())
    controls = 0; diagnostics = []
    context_hash = json.loads((out/'completion.json').read_text())['context_hash']
    with threadpool_limits(limits=2):
        for fold in s.nested_folds(frame,old_settings):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            r = load_record(directory/'result.json', context_hash)
            control = pickle.loads((directory/'language_control.pkl').read_bytes())['estimator']
            fit_x = s.subset(x, fold['fit']); fit_y = y[fold['fit']]
            proj = control.projection
            assert control.real_count == proj.real_count == proj.pca.n_samples_ == len(fit_y)
            np.testing.assert_array_equal(control.class_counts, np.bincount(fit_y,minlength=5))
            np.testing.assert_allclose(proj.base_scaler.mean_,fit_x['semantic_av'].mean(0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(proj.imputers['language'].statistics_,np.median(fit_x['language'],axis=0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(proj.scalers['language'].mean_,fit_x['language'].mean(0),rtol=0,atol=1e-12)
            controls += 1
            for d in r['fitting_diagnostics']:
                if not d['converged'] or d['real_fitting_participants'] != len(d['fit']) or set(d['fit']) & set(d['heldout']):
                    raise ValueError('Invalid fitting diagnostics')
                diagnostics.append({'seed':fold['seed'],'fold':fold['fold'],**d})
    atomic_json(out/'inner_fitting_diagnostics.json',diagnostics)
    compact = [{k:v for k,v in d.items() if k not in ['fit','heldout','class_counts','iterations']} |
               {'iterations_max':max(d['iterations']), 'class_counts':json.dumps(d['class_counts'])} for d in diagnostics]
    pd.DataFrame(compact).to_csv(out/'inner_fitting_diagnostics.csv',index=False)
    coverage = json.loads(Path('results/confidence_features/annotation_coverage.json').read_text())
    quality = {'passed':True,'new_full_audit':False,'metadata_participants':len(metadata),
        'original_split_counts':metadata.split.value_counts().to_dict(), 'training_participants':len(y),
        'training_supports':np.bincount(y,minlength=5).tolist(), 'training_item_sum_class_mapping_matches':True,
        'source_label_bindings_checked':len(imported['input_sha256']), 'fold_checks':fold_checks,
        'control_preprocessing_checkpoints_checked':controls, 'original_training_text_empty_or_duplicate':False,
        'feature_arrays_finite_and_distinct':True,'training_only_estimated_preprocessing':True,
        'source_questionnaire_columns_in_features':False,'synthetic_rows':0,'confidence_annotation_coverage':coverage,
        'confirmed_bug_fixed':'New checked_inference entry point rejects cross-participant modality filename IDs before feature extraction.',
        'fix_changes_training_evaluation':False,'old_entry_points_preserved':True,
        'input_identity_limit':'Matching filename IDs do not authenticate file contents. Existing training alignment is independently bound to audited source member hashes.',
        'hypotheses_not_verified_bugs':['Low recognition confidence causes misclassification','Whole-interview speaker mixture dilutes severity information'],
        'limits':['No speaker labels; no participant-only filtering claimed','No verified event-level AV alignment',
            'No independent clinical label authentication','Direct score expressions filtered; indirect questionnaire cues cannot be completely excluded',
            'Historical evaluation reuse prevents independent confirmation'],
        'source_receipts':{p:sha256(p) for p in ['PROJECT_ACCURACY_AUDIT_20261027.md',
            'results/age_language_recovery_20261027/recovery.json','results/timing_audit/timing_assessment.json',
            'results/external_transfer_accuracy/verification.json','data/metadata/official_edaic/import_report.json']}}
    atomic_json(out/'pipeline_verification.json',quality)
    # Verify numeric retained model and both new raw-file entry points, without scoring new cohorts.
    from .best_observed_inference import load, predict_features
    from .confidence_inference import frozen_model, predict_arrays
    contract, parameters = load()
    original_state = pickle.loads(Path(contract['source_model']['path']).read_bytes())
    with threadpool_limits(limits=2):
        numeric = predict_features(parameters,x['semantic_av'][:,:384],x['semantic_av'][:,384:],x['language'])
        cached = original_state['estimator'].predict_proba({'base':x['semantic_av'],'language':x['language']})
        np.testing.assert_allclose(numeric,cached,rtol=0,atol=1e-12)
        index = frame.participant_id.tolist().index('487')
        baseline_raw = json.loads((out/'checked_baseline_example.json').read_text())
        np.testing.assert_allclose(baseline_raw['decision_scores'],numeric[index:index+1],rtol=0,atol=1e-12)
        record,state,_ = frozen_model('confidence_only')
        expected = predict_arrays(record,state,s.subset(x,[index]))
        actual = json.loads((out/'candidate_raw_example.json').read_text())['decision_scores']
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        final_diag = state['estimator'].diagnostics(x,y)
    atomic_json(out/'final_fitting_diagnostics.json',final_diag)
    atomic_json(out/'raw_inference_verification.json',{'passed':True,'numeric_control_participants_replayed':len(y),
        'raw_training_participant':'487','baseline_max_difference':float(np.abs(np.asarray(baseline_raw['decision_scores'])-numeric[index:index+1]).max()),
        'candidate_max_difference':float(np.abs(np.asarray(actual)-expected).max()),
        'new_fits':0,'validation_test_scored':False})
    # Mean across repeats for aggregate metrics; pooled confusion for transparent per-class precision.
    names = ['baseline','language_control','accuracy','confidence_only']
    class_names = ['None/minimal','Mild','Moderate','Moderately severe','Severe']
    tables=[]; class_rows=[]; matrices={}; paired=[]
    for name in names+['majority']:
        pred = np.zeros((3,len(y)),int) if name=='majority' else np.stack([
            pd.read_csv(out/'oof'/f'{name}_seed{seed}/predictions.csv').predicted_label.to_numpy(int) for seed in [42,137,2026]])
        metric = weighted_metrics(y,pred,np.ones((1,len(y)),int))[0]
        cm = confusion_matrix(np.tile(y,3),pred.ravel(),labels=range(5));matrices[name]=cm.tolist()
        tables.append({'model':name,'correct_repeated_predictions':int(np.trace(cm)), 'predictions':int(cm.sum()),
            'independent_participants':len(y),'accuracy':metric[0],'macro_f1_mean_over_repeats':metric[1],
            'balanced_accuracy_mean_over_repeats':metric[2],'majority_predictions':int(cm[:,0].sum())})
        precision,recall,f1,support=precision_recall_fscore_support(np.tile(y,3),pred.ravel(),labels=range(5),zero_division=0)
        for c in range(5):
            class_rows.append({'model':name,'class':c,'class_name':class_names[c],'unique_participants':int((y==c).sum()),
                'repeated_support':int(support[c]),'precision_pooled':float(precision[c]),'recall':float(recall[c]),'f1_pooled':float(f1[c])})
        if name!='majority':
            for j,seed in enumerate([42,137,2026]):
                paired.extend({'model':name,'seed':seed,'participant_id':pid,'true_class':int(y[i]),'predicted_class':int(pred[j,i])}
                              for i,pid in enumerate(frame.participant_id))
    pd.DataFrame(tables).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(class_rows).to_csv(out/'per_class_precision_recall.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'reproducible_paired_predictions.csv',index=False)
    atomic_json(out/'confusion_matrices_with_majority.json',{'rows':'actual','columns':'predicted','class_names':class_names,
        'note':'489 repeated predictions from163 people; never489 independent participants','matrices':matrices})
    previous_ledger = Path('results/external_transfer_accuracy/experiment_ledger_extended.csv')
    ledger = pd.read_csv(previous_ledger)
    new = [{'study':'confidence_accuracy','variant':json.dumps({'model':r['model']}),'nested_accuracy':r['accuracy'],
        'macro_f1':r['macro_f1_mean_over_repeats'],'summary_path':str(out/'comparison.csv'),'summary_sha256':sha256(out/'comparison.csv'),
        'evidence_scope':'Same163 training people, original repeated nested folds; exploratory. No validation/test scoring.',
        'repeat_action':'Do not repeat confidence>=0.8 filtering unchanged; failed primary comparison; retain49.28.'}
        for r in tables if r['model'] in ['accuracy','confidence_only']]
    pd.concat([ledger,pd.DataFrame(new)],ignore_index=True).to_csv(out/'experiment_ledger_extended.csv',index=False)
    logdir=out/'logs'; logdir.mkdir()
    for source in Path('/tmp').glob('project_v2_*'):
        if source.is_file() and source.suffix=='.log' and ('confidence' in source.name or 'checked_inference' in source.name):
            shutil.copyfile(source,logdir/source.name)
    preserved=json.loads(Path('results/confidence_preservation/before.json').read_text())
    if any(not Path(p).is_file() or Path(p).stat().st_mtime_ns!=v['mtime_ns'] or sha256(p)!=v['sha256'] for p,v in preserved.items()):
        raise ValueError('An earlier artifact changed')
    manifest={'complete':True,'target_above_54_reached':False,'best_observed_accuracy':241/489,
        'retained_checkpoint':contract['source_model'],'numeric_bundle':'checkpoints/best_observed_4928',
        'recommended_inference':'python -m src.checked_inference --transcript PID_Transcript.csv --audio-csv PID_OpenSMILE2.3.0_mfcc.csv --visual-csv PID_OpenFace2.1.0_Pose_gaze_AUs.csv --output new_prediction.json',
        'new_candidate_promoted':False,'new_primary_accuracy':239/489,'new_feature_only_accuracy':235/489,
        'statistically_supported_improvement':False,'old_files_sha_and_mtime_preserved':len(preserved),
        'nested_pipeline_fits':105,'full_training_fits':1,'previous_experiments_refitted':0,'focused_tests_passed':7,
        'student_sadd_location':'Unavailable; user answered no after one location request. No import or access request sent.',
        'external_data_used_in_new_fits':False,'independent_confirmation':False,
        'evaluation_scope':'Original163 training participants, three matched nested-CV repetitions. No new validation/test scoring.',
        'next_step':'Verified participant speech annotations and additional genuine in-domain Mild/Moderate/Severe participants with a fresh independent evaluation cohort.',
        'bindings':{str(p):sha256(p) for p in [out/'pipeline_verification.json',out/'verification.json',out/'raw_inference_verification.json',
            out/'comparison.csv',out/'language_paired_comparisons.csv',out/'experiment_ledger_extended.csv',Path(__file__)]}}
    atomic_json(out/'delivery_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    finalize()
