"""Replay preserved evidence and record readiness for verified participant inputs.

This command cannot fit a model, extract encoder features, change a label or
search a configuration.  A blocked readiness receipt is not an accuracy claim.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record
from .common import digest, severity
from .final_evaluation import sha256
from .raw_frame_uncertainty import weighted_metrics


OUTPUT = Path('results/participant_input_readiness_20261029')
PRESERVATION = Path('results/participant_input_preservation_20261029/before.json')
PROTOCOL = 'preserved-model-replay-participant-input-readiness-v1'
CLASS_NAMES = ['None/minimal', 'Mild', 'Moderate', 'Moderately severe', 'Severe']
STUDIES = {
    'retained_original_procedure': (Path('results/age_language_accuracy'), 'without_age'),
    'best_observed_procedure': (Path('results/wavlm_accuracy'), 'accuracy'),
    'latest_unsuccessful_procedure': (Path('results/wavlm_layer_accuracy'), 'accuracy'),
}


def bound_file(path, bindings):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required final artifact is missing: {path}')
    bindings[str(path)] = sha256(path)
    return path


def read_json(path, bindings):
    return json.loads(bound_file(path, bindings).read_text())


def check_bindings(bindings):
    for path, expected in bindings.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f'Bound evidence changed: {path}')


def preserved_files():
    snapshot = json.loads(PRESERVATION.read_text())
    for path, expected in snapshot.items():
        if (not Path(path).is_file() or Path(path).stat().st_mtime_ns != expected['mtime_ns']
                or sha256(path) != expected['sha256']):
            raise ValueError(f'Previous project evidence changed: {path}')
    return len(snapshot)


def verify_fallback_cache(speaker_contract, x, frame, bindings):
    contract = read_json(speaker_contract, bindings)
    context = contract['context']
    if (digest(context) != contract['context_hash'] or context['targets_used']
            or context['validation_test_processed'] or context['training_ids'] != frame.participant_id.tolist()
            or sha256('src/participant_speech.py') != context['source_code_sha256']):
        raise ValueError('Participant input contract identity differs')
    dependencies = {**context['support_hashes'], **context['provenance']['bindings'], **context['annotation_hashes']}
    for path, expected in dependencies.items():
        if sha256(bound_file(path, bindings)) != expected:
            raise ValueError('Participant input dependency changed')
    root = Path(speaker_contract).parent
    completion = read_json(root/'completion.json', bindings)
    if (not completion['complete'] or completion['context_hash'] != contract['context_hash']
            or completion['dropped_participants'] != 0 or completion['new_encoder_runs'] != 0
            or completion['new_fits'] != 0):
        raise ValueError('Expected the zero-encoding, zero-fit complete fallback cache')
    for path, expected in completion['artifacts'].items():
        if sha256(bound_file(path, bindings)) != expected:
            raise ValueError('Participant fallback artifact changed')
    with np.load(root/'features.npz', allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['participant_id'], frame.participant_id.to_numpy(str))
        for key in ['semantic_av', 'language', 'speech']:
            np.testing.assert_array_equal(cache[key], x[key])
    rows = read_json(root/'participant_inputs.json', bindings)
    if [row['participant_id'] for row in rows] != frame.participant_id.tolist():
        raise ValueError('Fallback participant ownership differs')
    if any(row['decision']['text_verified'] or row['decision']['audio_verified'] or not row['document']['fallback'] for row in rows):
        raise ValueError('Unexpected verified speaker claim in a fallback cache')
    return {'passed': True, 'participants': len(rows), 'unchanged_feature_blocks': ['semantic_av512', 'language16', 'speech1536'],
        'all_fallback_features_bit_identical': True, 'new_fits': 0, 'new_encoder_extractions': 0}


def verify_population(frame, folds, metadata=None):
    """Validate the original participants and every inner/outer ownership receipt."""
    if (len(frame) != 163 or frame.participant_id.duplicated().any()
            or not frame.participant_id.map(lambda value: isinstance(value, str) and value.isdigit()).all()):
        raise ValueError('Expected the original163 unique training participants')
    if metadata is None:
        metadata=pd.read_csv('data/metadata/participants.csv',dtype={'participant_id':str})
    if metadata.participant_id.duplicated().any():
        raise ValueError('Duplicate participant metadata')
    training=metadata[metadata.split.eq('train')].set_index('participant_id')
    if set(training.index)!=set(frame.participant_id):
        raise ValueError('Official original training ownership changed')
    y = frame.severity_class.to_numpy(int)
    if set(y) != set(range(5)) or np.bincount(y, minlength=5).tolist() != [77,36,26,17,7]:
        raise ValueError('Original class mapping/supports changed')
    np.testing.assert_array_equal(training.loc[frame.participant_id,'severity_class'],y)
    np.testing.assert_array_equal([severity(v) for v in training.loc[frame.participant_id,'phq8_score']],y)
    if len(folds) != 15 or sorted({f['seed'] for f in folds}) != [42,137,2026]:
        raise ValueError('Expected the original15 outer folds and three seeds')
    all_indices = set(range(len(frame)))
    for seed in [42,137,2026]:
        seed_folds = [f for f in folds if f['seed'] == seed]
        if sorted(f['fold'] for f in seed_folds) != list(range(5)):
            raise ValueError('Incomplete outer-fold repetition')
        owners = np.zeros(len(frame), int)
        for fold in seed_folds:
            fit, held = fold['fit'], fold['heldout']
            if (len(set(fit)) != len(fit) or len(set(held)) != len(held)
                    or set(fit) & set(held) or set(fit+held) != all_indices):
                raise ValueError('Outer participant separation differs')
            owners[held] += 1
            inner_owners = np.zeros(len(frame), int)
            if len(fold['inner']) != 3:
                raise ValueError('Expected three inner folds')
            for part in fold['inner']:
                if (set(part['fit']) & set(part['heldout'])
                        or set(part['fit']+part['heldout']) != set(fit)
                        or len(set(part['fit'])) != len(part['fit'])
                        or len(set(part['heldout'])) != len(part['heldout'])):
                    raise ValueError('Inner participant separation differs')
                inner_owners[part['heldout']] += 1
            if not (inner_owners[fit] == 1).all() or inner_owners[held].any():
                raise ValueError('Inner participant coverage differs')
        if not (owners == 1).all():
            raise ValueError('Each person requires exactly one held-out prediction per repetition')
    return {'passed': True, 'unique_original_training_participants': len(frame),
        'class_supports': np.bincount(y, minlength=5).tolist(), 'outer_folds': 15,
        'inner_folds_per_outer': 3, 'seeds': [42,137,2026],
        'participant_ids_and_target_alignment_checked': True, 'all_recording_chunks_inherit_participant_ownership': True,
        'questionnaire_total_used_only_to_verify_training_target_mapping': True}


def verify_frozen_context(path, frame, folds, x, bindings):
    envelope = read_json(path, bindings); context = envelope['context']
    if digest(context) != envelope['context_hash'] or context['folds'] != folds:
        raise ValueError('Frozen context/folds changed')
    if context['training'] != frame.to_dict('records'):
        raise ValueError('Frozen training IDs or targets changed')
    source = context.get('source_sha256')
    if source:
        module = Path('src')/(path.parent.name+'.py')
        if sha256(bound_file(module, bindings)) != source:
            raise ValueError('Frozen study source changed')
    for source_path, expected in context.get('support_hashes', {}).items():
        if sha256(bound_file(source_path, bindings)) != expected:
            raise ValueError('Frozen support code changed')
    for key, values in x.items():
        if key in context.get('input_hashes', {}) and digest(np.asarray(values).tolist()) != context['input_hashes'][key]:
            raise ValueError(f'Frozen input array changed: {key}')
    completion = read_json(path.parent/'completion.json', bindings)
    if not completion['complete'] or completion['context_hash'] != envelope['context_hash']:
        raise ValueError('Frozen study is incomplete')
    return context


def cached_predictions(root, policy, frame, folds, bindings):
    y = frame.severity_class.to_numpy(int); repeated = []; all_rows = []
    protocol = read_json(root/'protocol.json', bindings)
    for seed in [42,137,2026]:
        path = bound_file(root/'oof'/f'{policy}_seed{seed}'/'predictions.csv', bindings)
        table = pd.read_csv(path, dtype={'participant_id': str})
        if table.participant_id.tolist() != frame.participant_id.tolist() or not table.split.eq('train_nested_oof').all():
            raise ValueError('Cached prediction ownership changed')
        np.testing.assert_array_equal(table.severity_class, y)
        scores = table[[f'p{i}' for i in range(5)]].to_numpy(float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('Invalid cached decision scores')
        np.testing.assert_allclose(scores.sum(1), 1, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(scores.argmax(1), table.predicted_label)
        expected = np.full_like(scores, np.nan); owners = np.zeros(len(frame), int)
        for fold in [f for f in folds if f['seed'] == seed]:
            path = bound_file(root/'jobs'/f"seed{seed}_fold{fold['fold']}"/'result.json', bindings)
            record = load_record(path, protocol['context_hash'])
            if any(record[key] != fold[key] for key in ['seed','fold','fit','heldout']):
                raise ValueError('Cached job participant ownership changed')
            expected[fold['heldout']] = record['selected'][policy]['probabilities']
            owners[fold['heldout']] += 1
        if not (owners == 1).all():
            raise ValueError('Cached predictions have incomplete outer ownership')
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-14)
        repeated.append(table.predicted_label.to_numpy(int))
        all_rows.extend({'participant_id': pid, 'severity_class': int(y[i]), 'seed': seed,
            'predicted_class': int(table.predicted_label.iloc[i]),
            **{f'p{c}':float(scores[i,c]) for c in range(5)}} for i,pid in enumerate(frame.participant_id))
    return np.stack(repeated), all_rows


def cached_metrics(frame, folds, bindings):
    y = frame.severity_class.to_numpy(int); labels = np.tile(y, 3)
    metrics, classes, matrices, prediction_rows = [], [], {}, []
    for name, (root, policy) in STUDIES.items():
        predictions, rows = cached_predictions(root, policy, frame, folds, bindings)
        point = weighted_metrics(y, predictions, np.ones((1,len(y)), int))[0]
        matrix = confusion_matrix(labels, predictions.reshape(-1), labels=range(5))
        saved = pd.read_csv(bound_file(root/'summary.csv', bindings)).set_index('model').loc[policy]
        np.testing.assert_allclose(point, saved[['accuracy_mean','macro_f1_mean','balanced_accuracy_mean']].to_numpy(float), rtol=0, atol=1e-12)
        precision, recall, f1, support = precision_recall_fscore_support(labels, predictions.reshape(-1), labels=range(5), zero_division=0)
        metrics.append({'procedure':name,'source':str(root),'source_policy':policy,'unique_participants':len(y),
            'repeated_predictions':len(labels),'correct_repeated_predictions':int((predictions==y).sum()),
            'accuracy':float(point[0]),'macro_f1':float(point[1]),'balanced_accuracy':float(point[2]),
            'majority_class_predictions':int((predictions==np.bincount(y).argmax()).sum()),
            'newly_measured_accuracy':False,'independent_confirmation':False})
        for c in range(5):
            classes.append({'procedure':name,'class':c,'class_name':CLASS_NAMES[c],
                'unique_participants':int((y==c).sum()),'repeated_support':int(support[c]),
                'precision_pooled':float(precision[c]),'recall':float(recall[c]),'f1_pooled':float(f1[c])})
        matrices[name] = matrix.tolist()
        prediction_rows.extend({'procedure':name,**row} for row in rows)
    return pd.DataFrame(metrics), pd.DataFrame(classes), matrices, pd.DataFrame(prediction_rows)


def replay_retained(x, frame, bindings):
    from .best_observed_inference import load, predict_features, BUNDLE
    contract, parameters = load(); record = contract['source_model']
    for path in [BUNDLE/'model.json', BUNDLE/'weights.npz', BUNDLE/'verification.json']:
        bound_file(path, bindings)
    model_path = bound_file(record['path'], bindings)
    if sha256(model_path) != record['sha256']:
        raise ValueError('Retained checkpoint changed')
    blob = model_path.read_bytes(); state = pickle.loads(blob)
    if (state['fit_participants'] != frame.participant_id.tolist()
            or state['candidate']['feature_set'] != 'language'
            or contract['questionnaire_required'] or contract['age_required']):
        raise ValueError('Retained inference schema or fitting ownership differs')
    model = state['estimator']; y = frame.severity_class.to_numpy(int)
    if model.real_count != len(y) or model.projection.pca.n_samples_ != len(y):
        raise ValueError('Retained preprocessing fitting ownership differs')
    np.testing.assert_array_equal(model.class_counts,np.bincount(y,minlength=5))
    np.testing.assert_allclose(model.projection.base_scaler.mean_,x['semantic_av'].mean(0),rtol=0,atol=1e-12)
    lang = model.projection.imputers['language'].transform(x['language'])
    np.testing.assert_allclose(model.projection.scalers['language'].mean_,lang.mean(0),rtol=0,atol=1e-12)
    with threadpool_limits(limits=2):
        actual = model.predict_proba({'base':x['semantic_av'],'language':x['language']})
        serialized = pickle.loads(pickle.dumps(state))['estimator'].predict_proba({'base':x['semantic_av'],'language':x['language']})
        numeric = predict_features(parameters,x['semantic_av'][:,:384],x['semantic_av'][:,384:],x['language'])
    np.testing.assert_allclose(actual,serialized,rtol=0,atol=1e-14)
    np.testing.assert_allclose(actual,numeric,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(actual.argmax(1),numeric.argmax(1))
    for root, key in [(Path('results/wavlm_accuracy'),'language_control_decision_scores'),
                      (Path('results/wavlm_layer_accuracy'),'previous_procedure_decision_scores')]:
        path = bound_file(root/'final_training_predictions.npz',bindings)
        with np.load(path,allow_pickle=False) as saved:
            if saved['participant_id'].tolist() != frame.participant_id.tolist():
                raise ValueError('Cached final fitting prediction ownership differs')
            np.testing.assert_allclose(actual,saved[key],rtol=0,atol=1e-14)
    receipt = {'passed':True,'original_training_participants':len(y),'new_fits':0,
        'source_checkpoint':record,'serialization_predictions_match':True,'numeric_bundle_predictions_match':True,
        'recorded_full_fitting_predictions_match':True,'maximum_numeric_probability_difference':float(np.abs(actual-numeric).max()),
        'fitting_accuracy':float((actual.argmax(1)==y).mean()),
        'fitting_accuracy_is_validation_accuracy':False,'iterations':model.model.n_iter_.tolist(),
        'maximum_iterations':int(model.model.max_iter),'converged':bool((model.model.n_iter_<model.model.max_iter).all()),
        'classifier_dimensions':int(model.model.n_features_in_),'parameter_count':int(model.model.coef_.size+model.model.intercept_.size),
        'questionnaire_inference_columns':[],'inference_feature_blocks':['semantic_av512','fixed_language16'],
        'speaker_annotations_verified':False}
    return receipt, actual


def fitting_evidence(bindings):
    outer, inner = [], []
    for name, (root, policy) in STUDIES.items():
        if name == 'retained_original_procedure':
            root = Path('results/wavlm_accuracy'); policy = 'language_control'
        table = pd.read_csv(bound_file(root/'overfitting_audit.csv',bindings))
        selected = table[table.policy.eq(policy)]
        if len(selected)!=15:
            raise ValueError('Incomplete saved fitting diagnostics')
        outer.append({'procedure':name,'mean_outer_fitting_accuracy':float(selected.training_accuracy.mean()),
            'mean_inner_selection_accuracy':float(selected.inner_accuracy.mean()),
            'mean_outer_fold_accuracy_unweighted':float(selected.outer_accuracy.mean()),
            'folds':len(selected),'means_are_not_additional_validation_participants':True})
    for root in [Path('results/wavlm_accuracy'),Path('results/wavlm_layer_accuracy')]:
        table = pd.read_csv(bound_file(root/'inner_fitting_diagnostics.csv',bindings))
        if not table.converged.all() or not (table.iterations_max<table.maximum_iterations).all():
            raise ValueError('Saved diagnostic records an unconverged fit')
        group_keys = [key for key in ['representation','C','speech_scale'] if key in table]
        for keys,group in table.groupby(group_keys):
            if not isinstance(keys,tuple):
                keys=(keys,)
            inner.append({'study':root.name,**dict(zip(group_keys,keys)), 'inner_fits':len(group),
                'mean_fitting_accuracy':float(group.fitting_accuracy.mean()),
                'mean_inner_heldout_accuracy':float(group.inner_accuracy.mean()),
                'mean_fitting_log_loss':float(group.unweighted_fitting_log_loss.mean()),
                'mean_coefficient_norm':float(group.coefficient_l2_norm.mean()),
                'minimum_iterations':int(group.iterations_max.min()),'maximum_iterations':int(group.iterations_max.max()),
                'all_converged':True,'max_iterations_limit':int(group.maximum_iterations.max())})
    return pd.DataFrame(outer),pd.DataFrame(inner)


def resume_completed(output):
    path = Path(output)/'readiness.json'
    if not path.is_file():
        raise ValueError('Incomplete readiness output; refusing to overwrite partial evidence')
    receipt = json.loads(path.read_text())
    if not receipt['complete'] or receipt['protocol'] != PROTOCOL:
        raise ValueError('Incomplete or unknown readiness receipt')
    check_bindings(receipt['input_bindings']);check_bindings(receipt['output_bindings'])
    return {**receipt,'resume_verified':True,'new_fits':0,'new_encoder_extractions':0,'new_writes':0}


def generate(speaker_contract,review_manifest,test_receipt,pipeline_doc,output=OUTPUT,resume=False):
    """Generate only after final source/artifact/test receipts have been supplied."""
    out=Path(output)
    if out.exists():
        if not resume:
            raise FileExistsError('Preserving existing readiness evidence; use --resume')
        return resume_completed(out)
    if resume:
        raise FileNotFoundError('No completed readiness receipt to resume')
    old_file_count = preserved_files()
    bindings={str(Path(__file__)):sha256(__file__)}
    bound_file(PRESERVATION, bindings)
    for path in [speaker_contract,review_manifest,test_receipt,pipeline_doc,
                 'src/participant_speech.py','src/checked_inference.py',
                 'src/participant_speech_review.py','configs/participant_speech_comparison.json',
                 'src/best_observed_inference.py','src/common.py','data/metadata/participants.csv']:
        bound_file(path,bindings)
    final_tests=read_json(test_receipt,bindings)
    if final_tests.get('passed') is not True:
        raise ValueError('A passed final test receipt is required before reporting')
    # Patching fitting/extraction entry points makes accidental new computation fail.
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from . import wavlm_accuracy as preserved
    from . import wavlm_features, wavlm_features_complete, semantic_features
    from .encoder_kernel_accuracy import offline_network
    def forbidden(*args,**kwargs):
        raise AssertionError('Readiness verification cannot fit models or run encoders')
    with ExitStack() as stack:
        for cls in [LogisticRegression,PCA,SimpleImputer,StandardScaler]:
            stack.enter_context(patch.object(cls,'fit',forbidden))
        for module,name in [(wavlm_features,'load_encoder'),(wavlm_features_complete,'load_encoder'),
                            (semantic_features,'local_encoder')]:
            stack.enter_context(patch.object(module,name,forbidden))
        stack.enter_context(offline_network())
        x,frame,settings,_,provenance=preserved.load_inputs()
        folds=preserved.nested_folds(frame,settings)
        population=verify_population(frame,folds)
        for path,expected in provenance['bindings'].items():
            if sha256(bound_file(path,bindings))!=expected:
                raise ValueError('Original cached evidence changed')
        contexts={}
        for root in [Path('results/wavlm_accuracy'),Path('results/wavlm_layer_accuracy')]:
            contexts[root.name]=verify_frozen_context(root/'protocol.json',frame,folds,x,bindings)
            for name in ['delivery_manifest.json','verification.json','final/plan.json']:
                record=read_json(root/name,bindings)
                for source,expected in record.get('bindings',{}).items():
                    if sha256(bound_file(source,bindings))!=expected:
                        raise ValueError('Recorded delivery evidence changed')
        layer_path=bound_file('results/wavlm_layer_features/features.npz',bindings)
        with np.load(layer_path,allow_pickle=False) as cache:
            if cache['participant_id'].tolist()!=frame.participant_id.tolist():
                raise ValueError('New cached speech array ownership differs')
            for key in ['final_nonzero','layer6_nonzero']:
                if digest(cache[key].tolist())!=contexts['wavlm_layer_accuracy']['input_hashes'][key]:
                    raise ValueError('Latest cached speech array changed')
        metrics,classes,matrices,predictions=cached_metrics(frame,folds,bindings)
        replay,full_predictions=replay_retained(x,frame,bindings)
        outer,inner=fitting_evidence(bindings)
        fallback_replay = verify_fallback_cache(speaker_contract, x, frame, bindings)
    timing=read_json('results/timing_audit/timing_assessment.json',bindings)
    if timing['participant_speech_isolation_verified'] or set(timing['transcript_schemas'])!={'Start_Time;End_Time;Text;Confidence'}:
        raise ValueError('Historical speaker-annotation assumptions changed; reassess readiness')
    ledger_path=bound_file('results/wavlm_layer_accuracy/experiment_ledger_extended.csv',bindings)
    ledger=pd.read_csv(ledger_path)
    completion_path=Path(speaker_contract).with_name('completion.json')
    speaker_completion=read_json(completion_path,bindings)
    if (speaker_completion.get('participants')!=len(frame)
            or speaker_completion.get('verified_audio_participants')!=0
            or speaker_completion.get('verified_text_participants')!=0):
        raise ValueError('Readiness must be reassessed when verified participant annotations become available')
    latest=read_json('results/wavlm_layer_accuracy/delivery_manifest.json',bindings)
    retained=replay['source_checkpoint']
    if (latest['recommended_saved_model']['sha256']!=retained['sha256']
            or latest['default_replacement_supported']):
        raise ValueError('Expected the unchanged retained default after unsuccessful latest study')
    report=write_report(metrics,classes,matrices,outer,inner,replay,
        {'speaker_contract':speaker_contract,'review_manifest':review_manifest,'test_receipt':test_receipt,'pipeline_doc':pipeline_doc},len(ledger))
    paired_path = bound_file('results/wavlm_layer_accuracy/matched_paired_comparisons.csv', bindings)
    changes_path = bound_file('results/wavlm_layer_accuracy/matched_error_changes.csv', bindings)
    paired = pd.read_csv(paired_path); changes = pd.read_csv(changes_path)
    ci = paired[(paired.model == 'accuracy') & (paired.reference == 'previous_procedure') & (paired.metric == 'accuracy')].iloc[0]
    change = changes[(changes.model == 'accuracy') & (changes.reference == 'previous_procedure')].iloc[0]
    report += f'''\n## Reused paired uncertainty and exact missing resources\n\nThe latest unsuccessful procedure gained {int(change.gained)} and lost {int(change.lost)} repeated correct predictions against the best observed procedure. Its accuracy delta was {100*ci.difference:+.2f} percentage points, with participant-paired95% interval [{100*ci.ci95_low:+.2f}, {100*ci.ci95_high:+.2f}] percentage points. These are cached results, not a new experiment. The original163 people were resampled with all three predictions kept together; the interval is conditional on fitted models and does not correct the long experiment history.\n\nMissing resources are complete human/provider speaker-role and transcript/WAV event-alignment reviews bound to the current raw files. The user confirmed StudentSADD files are not available locally. Approval has not supplied the dataset, and no external cohort was imported. No new controlled comparison, gain/loss count or candidate uncertainty can be measured from unchanged fallback features.\n\nAll {old_file_count} previously existing snapshotted files retain their original hashes and modification times. The all-fallback feature cache is bit-identical to the original inputs for all163 people.\n'''
    check_bindings(bindings)
    preserved_files()
    out.mkdir(parents=True)
    metrics.to_csv(out/'cached_procedure_metrics.csv',index=False)
    classes.to_csv(out/'per_class_precision_recall.csv',index=False)
    predictions.to_csv(out/'cached_oof_predictions.csv',index=False)
    frame[['participant_id','severity_class']].assign(split='train').to_csv(out/'original_training_alignment.csv',index=False)
    outer.to_csv(out/'outer_fitting_evidence.csv',index=False)
    inner.to_csv(out/'inner_fitting_evidence.csv',index=False)
    atomic_json(out/'summed_confusion_matrices.json',{'class_order':CLASS_NAMES,'matrices':matrices,
        'note':'Three predictions per original participant; rows actual, columns predicted. Not additional people.'})
    atomic_json(out/'retained_model_replay.json',replay)
    atomic_json(out/'fallback_replay.json', fallback_replay)
    paired.to_csv(out/'cached_paired_comparisons.csv', index=False)
    changes.to_csv(out/'cached_error_changes.csv', index=False)
    np.savez_compressed(out/'retained_full_fitting_predictions.npz',participant_id=frame.participant_id.to_numpy(str),decision_scores=full_predictions)
    (out/'REPORT.md').write_text(report)
    output_bindings={str(path):sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    receipt={'protocol':PROTOCOL,'complete':True,'status':'blocked_input_annotations',
        'no_new_accuracy_claim':True,'new_fits':0,'new_encoder_extractions':0,'new_validation_test_scores':0,
        'population':population,'retained_checkpoint':retained,
        'retained_inference_command':'.venv/bin/python -m src.checked_inference --output NEW.json --transcript PID_Transcript.csv --audio-csv PID_OpenSMILE2.3.0_mfcc.csv --visual-csv PID_OpenFace2.1.0_Pose_gaze_AUs.csv',
        'cached_original_procedure_accuracy':float(metrics.iloc[0].accuracy),
        'cached_best_observed_procedure_accuracy':float(metrics.iloc[1].accuracy),
        'cached_latest_unsuccessful_procedure_accuracy':float(metrics.iloc[2].accuracy),
        'verified_speaker_annotation_coverage':0,'model_comparison_supported_with_current_annotations':False,
        'bottleneck_assessment':{
            'confirmed_input_limitation':'No source-bound human/provider participant-role and event-alignment annotations for the163 training people',
            'new_implementation_bug_verified':False,
            'insufficient_iterations_supported':False,
            'training_evidence':'Existing logistic solvers converged within budget; fitting performance exceeds inner-heldout performance',
            'weak_features_hypothesis':'Whole-interview aggregation may dilute participant information; benefit of verified participant-only inputs remains unmeasured',
            'limited_data_fact':'163 people with class supports77/36/26/17/7; seven Severe people cannot establish an accuracy ceiling or benefit from more data by themselves',
            'neural_gradient_overfit_probe_needed':False},
        'retained_inference_fallback_available':True,'questionnaire_inputs':False,
        'exclusion_scope':'No questionnaire/label columns or IDs in numeric inference schema; transcript semantic absence of all indirect cues is not proved.',
        'stop_rule':'Stop unsupported model search. Train a participant-input comparison only after source-bound verified intervals/files pass coverage and alignment checks; preserve matched original folds and fallback participants.',
        'future_artifacts':{'speaker_contract':str(speaker_contract),'review_manifest':str(review_manifest),'test_receipt':str(test_receipt),'pipeline_doc':str(pipeline_doc)},
        'participant_input_completion':speaker_completion,
        'fallback_replay': fallback_replay, 'previous_files_and_mtimes_preserved': old_file_count,
        'external_files_available': False, 'new_controlled_comparison_performed': False,
        'final_test_receipt':final_tests,'ledger_rows':len(ledger),
        'input_bindings':bindings,'output_bindings':output_bindings}
    atomic_json(out/'readiness.json',receipt)
    return receipt


def write_report(metrics,classes,matrices,outer,inner,replay,artifacts,ledger_rows):
    text='''# Participant-input readiness — 29 October 2026

**Status: blocked_input_annotations. No new accuracy is claimed and no model is trained by this step.** The retained default remains the preserved original age-free language checkpoint. Existing evidence does not establish 53%, 56% or 60% accuracy.

The next implementation requires verified participant-speech intervals tied to the original files. Speaker identities are not guessed from channels, transcript order, sound level or language. Historical transcript coverage has no speaker field, and the previous audit does not verify participant-only speech or event-level alignment. Local review material is preparation, not ground-truth annotation.

The confirmed limitation is missing identity/alignment evidence. No new implementation bug or failure to converge is verified. Weak whole-interview features are a hypothesis supported by the unresolved attribution, not a proven cause of the current score. The163-person cohort and seven Severe cases are a data limitation, not proof of an accuracy ceiling. There is no basis here for adding epochs or running another architecture search.

## Replayed cached participant-level results

All results below reuse the same 163 original training participants, class supports77/36/26/17/7, five outer/three inner folds and seeds42/137/2026. Each person has three repeated predictions; the489 predictions are not489 independent people. No validation/test cohort is newly scored.

| Cached procedure | Correct /489 | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
'''
    for row in metrics.itertuples():
        text+=f'| {row.procedure} | {row.correct_repeated_predictions} | {100*row.accuracy:.2f}% | {row.macro_f1:.4f} | {100*row.balanced_accuracy:.2f}% |\n'
    text+='\nThese are previously measured selection-procedure scores, not a new evaluation or a per-prediction confidence guarantee. The best-observed procedure selected the original checkpoint during its final inner-only selection; there is no separate saved checkpoint proven to generalize at49.90%. The latest experiment failed to improve that procedure and did not replace the retained model. Existing paired participant intervals remain in the measured experiment reports.\n\n'
    text+='| Class | Unique people | Original precision/recall | Best observed precision/recall | Latest precision/recall |\n|---|---:|---:|---:|---:|\n'
    for c in range(5):
        rows=[classes[(classes.procedure==name)&(classes['class']==c)].iloc[0] for name in STUDIES]
        text+=f'| {CLASS_NAMES[c]} | {rows[0].unique_participants} | '+' | '.join(f'{100*r.precision_pooled:.2f}/{100*r.recall:.2f}' for r in rows)+' |\n'
    text+='\nPrecision and recall use pooled confusion counts; metrics above average complete repetitions. Confusion matrices use rows actual, columns predicted in the listed class order.\n'
    for name,matrix in matrices.items():
        text+=f'\n{name}:\n\n```text\n'+'\n'.join(' '.join(f'{v:3d}' for v in row) for row in matrix)+'\n```\n'
    text+='\n## Training quality from existing diagnostics\n\n| Procedure | Mean outer fitting accuracy | Mean selected inner accuracy |\n|---|---:|---:|\n'
    for row in outer.itertuples():
        text+=f'| {row.procedure} | {100*row.mean_outer_fitting_accuracy:.2f}% | {100*row.mean_inner_selection_accuracy:.2f}% |\n'
    text+='\nInner fitting summaries and convergence evidence are saved as CSV. All checked diagnostic fits converged within their declared limit. These are existing logistic fits, not new neural epochs or new early-stopping checkpoints. Training-versus-inner gaps diagnose capacity; inner scores involved in selection are optimistic and do not substitute for outer evaluation. Repeated and fold-level observations remain dependent.\n\n'
    text+=f'''## Retained model replay and inference contract

The original checkpoint `{replay['source_checkpoint']['path']}` retains SHA256 `{replay['source_checkpoint']['sha256']}`. Its fitting ownership is exactly the original163 participants. A fresh unmodified checkpoint load, an in-memory serialization round trip, the numeric bundle and saved full-fitting predictions reproduce the same decisions. Maximum numeric probability difference: {replay['maximum_numeric_probability_difference']:.3g}. Its full-fitting accuracy is {100*replay['fitting_accuracy']:.2f}%; **this is a fitting diagnostic, not validation accuracy**. There were zero fits, zero encoder extractions and zero new validation/test scores.

The default consumes original semantic/AV512 and fixed language16; the selected projection has48 classifier coordinates. Questionnaire score/answer columns, labels, IDs, age and gender are not numeric inference inputs. PHQ-8 totals are read only to verify the existing training-target mapping. Feature schema and source hashes are checked; they cannot prove that transcript semantics contain no indirect label cues. No questionnaire data is introduced into the future participant-input contract.

The latest ledger contains {ledger_rows} preserved rows. Stop unsupported model/parameter searches until verified inputs provide a specific new evidence-supported comparison. With only seven Severe people, duplicating synthetic training rows cannot create independent clinical evidence or guarantee the requested accuracy.

## Implemented path to the next justified experiment

'''
    for name,path in artifacts.items():
        text+=f'- [{name.replace("_"," ")}]({Path(path).resolve()})\n'
    text+='''
The source-bound interval validator and local review packet are engineering preparation. Until verified human/provider intervals and matching source files are supplied, they authorize no claim of participant-only speech, new accuracy or model superiority. Missing annotations retain the original participant inputs through the documented fallback rather than discarding participants. A future comparison must retain all original participant folds, fitting-only transforms and exact matched references. All label-dependent fitting remains inside the relevant training cohorts.

For current inference use the retained model with a fresh output filename:

```bash
.venv/bin/python -m src.checked_inference \\
  --output /private/tmp/project_v2_retained_new_prediction.json \\
  --transcript /absolute/path/PID_Transcript.csv \\
  --audio-csv /absolute/path/PID_OpenSMILE2.3.0_mfcc.csv \\
  --visual-csv /absolute/path/PID_OpenFace2.1.0_Pose_gaze_AUs.csv
```

Readiness is a data-availability status, not a statement that54–60% is attainable from the currently available inputs. Preserve an independent evaluation cohort before any further model development.
'''
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['speaker-contract','review-manifest','test-receipt','pipeline-doc']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',default=str(OUTPUT));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    result=generate(args.speaker_contract,args.review_manifest,args.test_receipt,args.pipeline_doc,args.output,args.resume)
    print(json.dumps({key:result[key] for key in ['status','new_fits','new_encoder_extractions',
        'cached_original_procedure_accuracy','cached_best_observed_procedure_accuracy','cached_latest_unsuccessful_procedure_accuracy']},indent=2))


if __name__=='__main__':
    main()
