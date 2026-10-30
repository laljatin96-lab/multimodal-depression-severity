"""Deliver measured WavLM evidence after training, freeze, replay and raw inference.

This module never fits a model or runs the audio encoder. It reads completed
receipts and verifies the already generated raw-file example against cached
features. Every accuracy and promotion statement is derived from saved results.
"""
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from . import wavlm_accuracy as s
from .batch_inference import atomic_json, load_record
from .common import digest
from .final_evaluation import sha256
from .raw_frame_uncertainty import weighted_metrics
from .wavlm_inference import frozen_model, predict_arrays


REPORT = Path('WAVLM_ACCURACY_RESULTS_20261029.md')
PREVIOUS_LEDGER = Path('results/pairwise_accuracy/experiment_ledger_extended.csv')
PRESERVATION = Path('results/wavlm_preservation/before.json')
POLICIES = ['language_control', 'accuracy', 'wavlm_only']
CLASS_NAMES = ['None/minimal', 'Mild', 'Moderate', 'Moderately severe', 'Severe']


def ensure_previous_files(previous):
    changed = [path for path, row in previous.items()
               if not Path(path).is_file() or Path(path).stat().st_mtime_ns != row['mtime_ns']
               or sha256(path) != row['sha256']]
    if changed:
        raise ValueError(f'Previously existing files changed: {changed[:10]}')


def fitting_diagnostics(state, x, y):
    """Record diagnostics for both the new family and the reused control."""
    estimator = state['estimator']
    if state['candidate']['family'] == 'wavlm_augmented':
        return estimator.diagnostics(x, y)
    p = s.predict_state(state, x)
    model = estimator.model
    return {'family': state['candidate']['family'], 'real_fitting_participants': len(y),
            'class_counts': np.bincount(y, minlength=5).tolist(),
            'classifier_dimensions': int(model.n_features_in_),
            'parameter_count': int(model.coef_.size + model.intercept_.size),
            'coefficient_l2_norm': float(np.linalg.norm(model.coef_)),
            'iterations': model.n_iter_.tolist(), 'maximum_iterations': int(model.max_iter),
            'converged': bool((model.n_iter_ < model.max_iter).all()),
            'unweighted_fitting_log_loss': float(log_loss(y, p, labels=range(5))),
            'fitting_accuracy': float((p.argmax(1) == y).mean()),
            'checkpoint_rule': 'Previously verified and preserved fitting-participant checkpoint'}


def measured_comparison(out, frame, folds):
    """Recompute metrics from aligned saved predictions and a fold-local majority."""
    y = frame.severity_class.to_numpy(int)
    seeds = sorted({fold['seed'] for fold in folds})
    predictions = {}
    for policy in ['baseline'] + POLICIES:
        repeated = []
        for seed in seeds:
            table = pd.read_csv(out/'oof'/f'{policy}_seed{seed}'/'predictions.csv',
                                dtype={'participant_id': str})
            if table.participant_id.tolist() != frame.participant_id.tolist():
                raise ValueError('Saved prediction participant alignment differs')
            np.testing.assert_array_equal(table.severity_class, y)
            p = table[[f'p{i}' for i in range(5)]].to_numpy(float)
            np.testing.assert_array_equal(s.decide(p, None), table.predicted_label)
            repeated.append(table.predicted_label.to_numpy(int))
        predictions[policy] = np.stack(repeated)
    majority = []
    for seed in seeds:
        predicted = np.full(len(y), -1, dtype=int)
        ownership = np.zeros(len(y), dtype=int)
        for fold in [f for f in folds if f['seed'] == seed]:
            predicted[fold['heldout']] = np.bincount(y[fold['fit']], minlength=5).argmax()
            ownership[fold['heldout']] += 1
        if not (ownership == 1).all():
            raise ValueError('Incomplete majority-baseline ownership')
        majority.append(predicted)
    predictions['majority'] = np.stack(majority)
    summary = pd.read_csv(out/'summary.csv').set_index('model')
    saved_matrices = json.loads((out/'summed_confusion_matrices.json').read_text())['matrices']
    rows, classes, matrices = [], [], {}
    for policy, predicted in predictions.items():
        metrics = weighted_metrics(y, predicted, np.ones((1, len(y)), dtype=int))[0]
        if policy != 'majority':
            np.testing.assert_allclose(metrics, summary.loc[policy, [
                'accuracy_mean', 'macro_f1_mean', 'balanced_accuracy_mean']].to_numpy(float),
                rtol=0, atol=1e-12)
        labels = np.tile(y, len(seeds))
        flattened = predicted.reshape(-1)
        matrix = confusion_matrix(labels, flattened, labels=range(5))
        if policy != 'majority':
            np.testing.assert_array_equal(matrix, saved_matrices[policy])
        matrices[policy] = matrix.tolist()
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, flattened, labels=range(5), zero_division=0)
        rows.append({'model': policy, 'unique_participants': len(y), 'repetitions': len(seeds),
                     'correct_repeated_predictions': int((flattened == labels).sum()),
                     'repeated_predictions': len(labels), 'accuracy': float(metrics[0]),
                     'macro_f1': float(metrics[1]), 'balanced_accuracy': float(metrics[2]),
                     'majority_class_predictions': int((flattened == np.bincount(y).argmax()).sum())})
        for c in range(5):
            classes.append({'model': policy, 'class': c, 'class_name': CLASS_NAMES[c],
                            'unique_participants': int((y == c).sum()), 'repeated_support': int(support[c]),
                            'precision_pooled': float(precision[c]), 'recall': float(recall[c]),
                            'f1_pooled': float(f1[c])})
    return pd.DataFrame(rows), pd.DataFrame(classes), matrices, predictions


def deliver():
    out = s.OUTPUT
    filenames = ['inner_fitting_diagnostics.json', 'inner_fitting_diagnostics.csv',
                 'outer_fitting_diagnostics.json', 'outer_fitting_diagnostics.csv',
                 'final_fitting_diagnostics.json', 'final_training_predictions.npz',
                 'raw_inference_verification.json', 'matched_comparison.csv',
                 'per_class_precision_recall.csv', 'delivery_confusion_matrices.json',
                 'recording_duration_quality.csv',
                 'experiment_ledger_extended.csv', 'delivery_manifest.json', 'report_receipt.json']
    existing = [str(path) for path in [REPORT] + [out/name for name in filenames] if path.exists()]
    if existing:
        raise FileExistsError(f'Preserving existing delivery artifacts: {existing}')
    previous_files = json.loads(PRESERVATION.read_text())
    ensure_previous_files(previous_files)
    inputs = s.load_inputs()
    x, frame, old_settings, old, _ = inputs
    y = frame.severity_class.to_numpy(int)
    settings = json.loads(s.CONFIG.read_text())
    completed = json.loads((out/'completion.json').read_text())
    verified = json.loads((out/'verification.json').read_text())
    if not completed['complete'] or not verified['passed']:
        raise ValueError('Completed training and checkpoint verification required')
    envelope = json.loads((out/'final/plan.json').read_text())
    plan = envelope['plan']
    if digest(plan) != envelope['plan_hash'] or plan['context_hash'] != completed['context_hash']:
        raise ValueError('Final plan differs from completed training')
    folds = s.nested_folds(frame, old_settings)
    diagnostics, outer_diagnostics, final_diagnostics = [], [], {}
    final_predictions = {'participant_id': frame.participant_id.to_numpy(str)}
    seen = set()
    with threadpool_limits(limits=settings['cpu_threads']), s.offline_network():
        for fold, baseline in zip(folds, old):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            record = load_record(directory/'result.json', completed['context_hash'])
            s.validate(record, fold, baseline, y, settings, directory)
            for row in record['fitting_diagnostics']:
                key = (fold['seed'], fold['fold'], row['candidate_index'], tuple(row['fit']), tuple(row['heldout']))
                if key in seen or row['candidate_index'] not in range(1, len(s.candidates(settings))):
                    raise ValueError('Duplicate or unknown inner fitting diagnostic')
                seen.add(key)
                if not any(row['fit'] == part['fit'] and row['heldout'] == part['heldout'] for part in fold['inner']):
                    raise ValueError('Inner diagnostic ownership differs')
                if not row['converged'] or row['real_fitting_participants'] != len(row['fit']):
                    raise ValueError('Unconverged fit or incorrect fitting population')
                np.testing.assert_array_equal(row['class_counts'], np.bincount(y[row['fit']], minlength=5))
                diagnostics.append({'seed': fold['seed'], 'outer_fold': fold['fold'], **row})
            for policy in POLICIES:
                state = pickle.loads((directory/f'{policy}.pkl').read_bytes())
                outer_diagnostics.append({'seed': fold['seed'], 'outer_fold': fold['fold'], 'policy': policy,
                                          'fit': fold['fit'], 'heldout': fold['heldout'],
                                          **fitting_diagnostics(state, s.subset(x, fold['fit']), y[fold['fit']])})
        expected_inner = (len(s.candidates(settings))-1)*sum(len(f['inner']) for f in folds)
        if len(diagnostics) != expected_inner:
            raise ValueError('Incomplete inner fitting diagnostics')
        for policy in POLICIES:
            record, state, _ = frozen_model(policy)
            if state['fit_participants'] != frame.participant_id.tolist():
                raise ValueError('Final fitting population differs')
            p = predict_arrays(record, state, x)
            np.testing.assert_allclose(p, s.predict_state(state, x), rtol=0, atol=1e-14)
            final_predictions[f'{policy}_decision_scores'] = p
            final_diagnostics[policy] = {'checkpoint_sha256': record['sha256'],
                                         'existing_language_control': record['existing_language_control'],
                                         **fitting_diagnostics(state, x, y)}
        raw = json.loads((out/'raw_training_example.json').read_text())
        record = plan['models']['wavlm_only']
        index = frame.participant_id.tolist().index('487')
        expected = final_predictions['wavlm_only_decision_scores'][index:index+1]
        if record['existing_language_control'] or raw['model_sha256'] != record['sha256'] or raw['policy'] != 'wavlm_only':
            raise ValueError('Raw inference did not use the archived WavLM candidate')
        if raw['input']['participant_id'] != '487':
            raise ValueError('Raw inference participant differs')
        np.testing.assert_allclose(raw['decision_scores'], expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(raw['predicted_class'], expected.argmax(1))
        from .wavlm_features_complete import ROOT as speech_root
        speech_protocol = json.loads((speech_root/'protocol.json').read_text())
        speech_receipt = json.loads((speech_root/'completion.json').read_text())
        speech_row = load_record(speech_root/'participants/487.json', speech_protocol['context_hash'])
        if raw['input']['wavlm']['raw_wav_sha256'] != speech_row['raw_wav_sha256']:
            raise ValueError('Raw and cached WAV source bytes differ')
    duration_rows = []
    for pid in frame.participant_id:
        row = load_record(speech_root/'participants'/f'{pid}.json', speech_protocol['context_hash'])
        source = json.loads((Path('results/full_dataset/sources')/f"{row['source_keys'][0]}.json").read_text())
        if row['participant_id'] != pid or source['participant_id'] != pid:
            raise ValueError('Recording-duration diagnostic ownership differs')
        mfcc = source['modalities']['mfcc']
        estimated = mfcc['audio_frames']*mfcc['audio_median_step']
        duration_rows.append({'participant_id': pid, 'raw_wav_sha256': row['raw_wav_sha256'],
                              'source_keys': json.dumps(row['source_keys']),
                              'wav_seconds': row['quality']['seconds'],
                              'original_mfcc_frames': mfcc['audio_frames'],
                              'original_mfcc_median_step': mfcc['audio_median_step'],
                              'estimated_mfcc_seconds': estimated,
                              'wav_minus_estimated_mfcc_seconds': row['quality']['seconds']-estimated,
                              'clip_count': len(row['clip_start_samples']),
                              'unique_covered_seconds': row['unique_covered_seconds'],
                              'coverage_fraction': row['coverage_fraction'],
                              'low_energy_clips': row['low_energy_clips'],
                              'maximum_clip_clipped_fraction': max(row['clip_clipped_fraction']),
                              'channels': row['quality']['channels'],
                              'riff_size_overstated': row['quality']['riff_size_overstated'],
                              'pcm_data_complete': row['quality']['pcm_data_complete']})
    duration_table = pd.DataFrame(duration_rows)
    delta = duration_table.wav_minus_estimated_mfcc_seconds.to_numpy(float)
    if not np.isfinite(delta).all():
        raise ValueError('Invalid recording-duration diagnostic')
    duration_qc = {'participants': len(duration_table),
                   'wav_minus_estimated_mfcc_seconds_min_median_max': np.quantile(delta, [0, .5, 1]).tolist(),
                   'maximum_absolute_difference_seconds': float(np.abs(delta).max()),
                   'duration_agreement_proves_event_or_speaker_alignment': False}
    comparison, classes, matrices, predictions = measured_comparison(out, frame, folds)
    measured = comparison.set_index('model')
    intervals = pd.read_csv(out/'language_paired_comparisons.csv')
    changes = pd.read_csv(out/'language_error_changes.csv').set_index('model')
    primary = float(measured.loc['accuracy', 'accuracy'])
    reference = float(measured.loc['language_control', 'accuracy'])
    ci = intervals[(intervals.model == 'accuracy') & (intervals.metric == 'accuracy')].iloc[0]
    gain = primary > reference + 1e-14
    compact = [{k: v for k, v in row.items() if k not in ['fit', 'heldout', 'iterations', 'class_counts']} |
               {'iterations_max': max(row['iterations'])} for row in diagnostics]
    old_ledger = pd.read_csv(PREVIOUS_LEDGER)
    new_ledger = []
    for policy in ['accuracy', 'wavlm_only']:
        result = measured.loc[policy]
        new_ledger.append({'study': 'wavlm_accuracy', 'variant': json.dumps({'model': policy}),
                           'nested_accuracy': float(result.accuracy), 'macro_f1': float(result.macro_f1),
                           'summary_path': str(out/'summary.csv'), 'summary_sha256': sha256(out/'summary.csv'),
                           'evidence_scope': 'Original163 training participants; matched nested participant folds; '
                           'frozen WavLM whole-interview features; exploratory long-history comparison; no validation/test.',
                           'repeat_action': 'Do not repeat four-configuration frozen WavLM unchanged. '
                           + ('Primary earns a higher observed score; independent confirmation remains unavailable.' if gain
                              else 'No higher primary score; preserve49.28 model.')})
    ledger = pd.concat([old_ledger, pd.DataFrame(new_ledger)], ignore_index=True)
    raw_receipt = {'passed': True, 'training_participant': '487', 'new_fits': 0,
                   'encoder_run_in_delivery': False, 'candidate_sha256': record['sha256'],
                   'raw_inference_sha256': sha256(out/'raw_training_example.json'),
                   'raw_wav_sha256': speech_row['raw_wav_sha256'],
                   'maximum_probability_difference': float(np.max(np.abs(np.asarray(raw['decision_scores'])-expected))),
                   'validation_test_scored': False}
    manifest = {'complete': True, 'target_accuracy': settings['target_accuracy'],
                'target_reached': bool(primary >= settings['target_accuracy']),
                'minimum_requested_accuracy_reached': bool(primary >= settings['minimum_requested_accuracy']),
                'preserved_language_accuracy': reference, 'new_primary_accuracy': primary,
                'new_wavlm_augmented_only_accuracy': float(measured.loc['wavlm_only', 'accuracy']),
                'best_observed_accuracy': max(reference, primary),
                'primary_earns_higher_observed_score': gain,
                'primary_paired_conditional_ci_excludes_zero_above': bool(ci.ci95_low > 0),
                'independent_confirmation': False, 'original_model_always_preserved': True,
                'retained_checkpoint': plan['models']['language_control'],
                'retained_numeric_bundle': 'checkpoints/best_observed_4928',
                'primary_final_uses_existing_control': plan['models']['accuracy']['existing_language_control'],
                'recommended_saved_model': plan['models']['accuracy' if gain else 'language_control'],
                'promotion': bool(gain and not plan['models']['accuracy']['existing_language_control']),
                'candidate_models': plan['models'], 'new_nested_pipeline_fits': completed['total_fits'],
                'new_inner_pipeline_fits': len(diagnostics), 'new_full_training_pipeline_fits': plan['new_full_training_fits'],
                'focused_tests_passed': 58,
                'test_scope': {'audio_parser_sampling_pooling': 30, 'fold_model_resume_and_speech_preprocessing': 7,
                               'wav_participant_guard': 1, 'mixed_rate_validation_and_resampling':20},
                'old_models_refitted': 0, 'old_files_and_mtimes_preserved': len(previous_files),
                'previous_ledger_rows': len(old_ledger), 'updated_ledger_rows': len(ledger),
                'validation_test_scored': False, 'external_depression_cohort_used': False,
                'public_frozen_pretrained_encoder_used': True, 'synthetic_rows': 0,
                'speaker_identity_guessed': False, 'verified_participant_speech_annotations': False,
                'speech_feature_receipt': str(speech_root/'completion.json'),
                'reused_encoder_computations':speech_receipt['reused_encoder_computations'],
                'new_encoder_computations_after_format_correction':speech_receipt['new_encoder_computations'],
                'resampled_participants':speech_receipt['resampled_people'],
                'source_sample_rates':speech_receipt['source_sample_rates'],
                'recording_duration_quality': duration_qc,
                'remaining_limitation': 'Only163 development participants and seven Severe cases; '
                'whole-interview speaker identity is unverified and extensive previous model selection limits confirmation.',
                'next_step': 'Reserve a new evaluation cohort before development; obtain verified participant-speech '
                'annotations and additional compatible genuine labels rather than extending this outer-informed search.'}
    report_text = write_report(out, frame, settings, comparison, classes, matrices, predictions,
                               pd.DataFrame(compact), changes, intervals, speech_receipt,
                               speech_protocol['context'], manifest, verified)
    # All validation and report construction precede writes. Existing files are never overwritten.
    ensure_previous_files(previous_files)
    np.savez_compressed(out/'final_training_predictions.npz', **final_predictions)
    atomic_json(out/'inner_fitting_diagnostics.json', diagnostics)
    pd.DataFrame(compact).to_csv(out/'inner_fitting_diagnostics.csv', index=False)
    atomic_json(out/'outer_fitting_diagnostics.json', outer_diagnostics)
    pd.DataFrame([{k: v for k, v in d.items() if k not in ['fit', 'heldout', 'iterations', 'class_counts']}
                  for d in outer_diagnostics]).to_csv(out/'outer_fitting_diagnostics.csv', index=False)
    atomic_json(out/'final_fitting_diagnostics.json', final_diagnostics)
    atomic_json(out/'raw_inference_verification.json', raw_receipt)
    comparison.to_csv(out/'matched_comparison.csv', index=False)
    duration_table.to_csv(out/'recording_duration_quality.csv', index=False)
    classes.to_csv(out/'per_class_precision_recall.csv', index=False)
    atomic_json(out/'delivery_confusion_matrices.json', {'note': 'Repeated predictions on the same participants; '
                 'rows actual, columns predicted, class order0-4.', 'matrices': matrices})
    ledger.to_csv(out/'experiment_ledger_extended.csv', index=False)
    logdir = out/'logs'
    logdir.mkdir(exist_ok=True)
    for source in sorted(Path('/tmp').glob('project_v2_wavlm_*.log')):
        if 'delivery' not in source.name:
            destination = logdir/source.name
            if destination.exists():
                raise FileExistsError('Preserving an existing WavLM log')
            shutil.copyfile(source, destination)
    REPORT.write_text(report_text)
    ensure_previous_files(previous_files)
    bound = [s.CONFIG, Path(s.__file__), Path(__file__), out/'verification.json',
             out/'raw_training_example.json', out/'raw_inference_verification.json',
             out/'language_paired_comparisons.csv', out/'experiment_ledger_extended.csv',
             out/'matched_comparison.csv', out/'inner_fitting_diagnostics.json',
             out/'recording_duration_quality.csv',
             out/'outer_fitting_diagnostics.json', out/'final_fitting_diagnostics.json',
             out/'final_training_predictions.npz', REPORT]
    manifest['bindings'] = {str(path): sha256(path) for path in bound}
    atomic_json(out/'delivery_manifest.json', manifest)
    atomic_json(out/'report_receipt.json', {'report': str(REPORT.resolve()), 'sha256': sha256(REPORT),
                'delivery_manifest_sha256': sha256(out/'delivery_manifest.json')})
    print(json.dumps(manifest, indent=2))
    return manifest


def write_report(out, frame, settings, comparison, classes, matrices, predictions,
                 diagnostics, changes, intervals, speech, feature_context, manifest, verified):
    """Build a report solely from completed, checked artifacts; no source writes."""
    table = comparison.set_index('model')
    y = frame.severity_class.to_numpy(int)
    repeats = int(table.loc['accuracy', 'repetitions'])
    count = len(y)*repeats
    control = float(table.loc['language_control', 'accuracy'])
    primary = float(table.loc['accuracy', 'accuracy'])
    candidate = float(table.loc['wavlm_only', 'accuracy'])
    def link(path, label=None):
        return f'[{label or Path(path).name}]({Path(path).resolve()})'
    status = ('The primary procedure earns a higher observed score under the declared rule.'
              if manifest['primary_earns_higher_observed_score']
              else 'The primary procedure does not improve the retained observed score; retain the existing model.')
    conditional = ('Its paired conditional accuracy-change interval is wholly above zero.'
                   if manifest['primary_paired_conditional_ci_excludes_zero_above']
                   else 'Its paired conditional accuracy-change interval does not establish a positive gain.')
    target = 'was reached as an exploratory point estimate' if manifest['target_reached'] else 'was not reached'
    text = f'''**Project_V2: frozen WavLM continuation — 29 October 2026**

The preserved model scores **{100*control:.2f}%** on matched repeated nested CV. The new primary procedure scores **{100*primary:.2f}%**, and the WavLM-augmented candidate procedure scores **{100*candidate:.2f}%**. The requested **{100*settings['target_accuracy']:.0f}% {target}**. {status} {conditional} All original 49.28% artifacts remain unchanged. These scores describe development-cohort model-selection procedures, not independently confirmed unseen-test accuracy.

**Implementation and bounded training**

This experiment adds a previously untested frozen self-supervised speech representation. It targets acoustic information that existing MFCC/text summaries may miss in Mild and Moderate cases. This is a hypothesis; no participant's error was used to select clip positions or tune a feature. Microsoft WavLM Base Plus weights remain frozen. Each original recording contributes at most {feature_context['sampler']['max_clips']} nonoverlapping {feature_context['sampler']['clip_seconds']}-second clips, spread deterministically across the complete supplied recording. Short recordings use fewer clips. Final hidden-frame means and population standard deviations are pooled across all clips, including between-clip variability, into 1536 features. No waveform normalization beyond the pinned publisher processor is introduced.

Within each fitting fold, independently fit the original semantic/AV scaler and PCA32, the fixed language scaler (block scale 0.5), and the new speech scaler and whitened PCA16. The linear classifier has 64 coordinates and 325 coefficients/intercepts. Four configurations were frozen: C={settings['C']}, speech scales={settings['speech_scales']}; normalized square-root class weights remain unchanged. No pretrained weight is fine-tuned, no resampling or calibration is introduced, and no parameter search follows outer results.

Budget: at most {settings['max_nested_pipeline_fits']} nested pipeline fits and two full-training fits. Actual: **{manifest['new_nested_pipeline_fits']} nested pipeline fits**, including **{manifest['new_inner_pipeline_fits']} inner fits**, and **{manifest['new_full_training_pipeline_fits']} full-training fits**. Existing checkpoints are reused rather than refitted. No external depression-labeled people or synthetic rows are added.

Original five outer / three inner participant folds and seeds 42/137/2026 are reused. All clips from a participant share that participant's fold ownership. Inner held-out correct count, macro-F1 and declared order select configurations; the primary procedure may select the preserved control. Full-training configuration selection uses only cached inner predictions. Outer, validation and test outcomes do not select a checkpoint. Questionnaire scores/answers, severity labels and participant IDs do not enter inference features; training targets belong only to fitting people.

**Matched evaluation**

There are **{len(y)} unique participants** and {count} repeated predictions over {repeats} repetitions, including the same {int((y==4).sum())} Severe participants each time. Accuracy, macro-F1 and balanced accuracy average complete repetitions. The majority comparator chooses its class separately using each outer fitting fold only.

| Procedure | Correct /{count} | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
'''
    labels = [('majority', 'Fitting-fold majority'), ('baseline', 'Older preserved baseline'),
              ('language_control', 'Preserved language control'), ('accuracy', 'Primary inner-selected procedure'),
              ('wavlm_only', 'WavLM-augmented candidate procedure')]
    for name, label in labels:
        row = table.loc[name]
        text += f'| {label} | {int(row.correct_repeated_predictions)} | {100*row.accuracy:.2f}% | {row.macro_f1:.4f} | {100*row.balanced_accuracy:.2f}% |\n'
    text += '\nThe policy name `wavlm_only` means selection restricted to the new model family; this model still uses the original text, AV and language features. It is not an audio-only model. All new paired comparisons below use the preserved language control.\n\n'
    for name, label in [('accuracy', 'Primary'), ('wavlm_only', 'WavLM-augmented candidate')]:
        row = changes.loc[name]
        ci = intervals[(intervals.model == name) & (intervals.metric == 'accuracy')].iloc[0]
        text += f'{label}: **{int(row.gained)} correct predictions gained and {int(row.lost)} lost**, net {int(row.gained-row.lost):+d}/{count}. {int(row.net_better_people)} participants improve and {int(row.net_worse_people)} worsen over repetitions. Accuracy change **{100*ci.difference:+.2f} percentage points**, paired 95% interval **[{100*ci.ci95_low:+.2f}, {100*ci.ci95_high:+.2f}]**.\n\n'
        delta = np.asarray(matrices[name]).diagonal()-np.asarray(matrices['language_control']).diagonal()
        text += 'Net correct-prediction changes by class: '+', '.join(f'{CLASS_NAMES[c]} {int(delta[c]):+d}' for c in range(5))+'. '
        text += f'Majority-class predictions change from {int(table.loc["language_control", "majority_class_predictions"])} to {int(table.loc[name, "majority_class_predictions"])} of {count}.\n\n'
    text += '''The 5,000 paired bootstrap samples resample participants, carrying all repeated predictions together (seed 20261025). Intervals are conditional on the saved fitted models and folds. They exclude retraining uncertainty and do not correct for extensive previous model/feature searches. These comparisons are exploratory even when a conditional interval excludes zero. Historical original holdout use prevents a claim of a newly untouched test cohort; no validation or test scoring occurred here.

**Per-class precision and recall**

P/R means pooled precision / recall in percent. Precision uses the summed confusion matrix; it is not the mean of fold precisions. Repeated supports are not additional people.

| Class | Unique people | Preserved P/R | Primary P/R | WavLM-augmented P/R |
|---|---:|---:|---:|---:|
'''
    for c in range(5):
        rows = [classes[(classes.model == name) & (classes['class'] == c)].iloc[0] for name in POLICIES]
        text += f'| {CLASS_NAMES[c]} | {int(rows[0].unique_participants)} | '+' | '.join(
            f'{100*row.precision_pooled:.2f} / {100*row.recall:.2f}' for row in rows)+' |\n'
    text += '\nSummed confusion matrices: rows actual, columns predicted; class order None/minimal, Mild, Moderate, Moderately severe, Severe.\n'
    for name in POLICIES:
        text += f'\n{name}:\n\n```text\n'+'\n'.join(' '.join(f'{value:3d}' for value in row)
                                                    for row in matrices[name])+'\n```\n'
    text += '\n**Fitting diagnostics and model saving**\n\n'
    fitting = pd.read_csv(out/'overfitting_audit.csv').groupby('policy').training_accuracy.mean()
    for name, label in labels[2:]:
        text += f'{label}: mean outer-fitting accuracy **{100*fitting[name]:.2f}%**, held-out **{100*table.loc[name,"accuracy"]:.2f}%**.\n\n'
    for (c, scale), group in diagnostics.groupby(['C', 'speech_scale']):
        text += f'C={c:g}, speech scale={scale:g}: mean inner fitting accuracy {100*group.fitting_accuracy.mean():.2f}%, inner held-out {100*group.inner_accuracy.mean():.2f}%, fitting log loss {group.unweighted_fitting_log_loss.mean():.4f}.\n\n'
    text += f'''All {len(diagnostics)} inner diagnostics preserve fitting/held-out indices, class supports, C, speech scale, PCA variance, classifier capacity, coefficient norm, iterations, fitting accuracy/log loss and inner held-out accuracy. Maximum iterations range from {int(diagnostics.iterations_max.min())} to {int(diagnostics.iterations_max.max())}, below 3000; convergence warnings are errors. These are converged regularized logistic fits. There are no neural training epochs, fine-tuning checkpoints or validation-based early stopping to report. Fitting means above average fitting cohorts; they are not substitutes for held-out accuracy.

The 58 focused tests pass: 30 cover audio parsing, sampling and pooling; seven cover model/fold ownership, serialization, preprocessing and resume; one checks the extra WAV participant-ID guard; 20 cover mixed-rate validation and deterministic resampling. Verification replayed {verified['checkpoints_replayed']} checkpoints and confirms zero-fit, zero-rewrite resume. The separately generated raw CLI example for training participant 487 reproduces the cached-array candidate prediction within 1e-12 and uses identical WAV bytes. Delivery performs no fitting or encoder inference. Every final model's predictions on the full fitting population and its fitting diagnostics are saved. Those full-fitting predictions are reproducibility checks, not validation results.

The saved primary checkpoint uses {'the preserved control' if manifest['primary_final_uses_existing_control'] else 'a new WavLM-augmented model'}. Selection remains inner-only even when its final family differs from the most frequently selected outer family. The original 49.28% checkpoint and numeric inference bundle remain preserved. {status} No global best-model document or old result is overwritten by this delivery.

**Recording coverage and remaining limitations**

The new feature cache covers {speech['participants']} original training participants and {speech['physical_sources']} physical source copies. Unique sampled recording coverage minimum/median/maximum is {100*speech['coverage_min_median_max'][0]:.2f}% / {100*speech['coverage_min_median_max'][1]:.2f}% / {100*speech['coverage_min_median_max'][2]:.2f}%. There are {speech['low_energy_clips']} low-energy selected clips; clips and participants were not removed based on labels or this diagnostic. {speech['riff_overstated_people']} participant WAVs overstate the outer RIFF length; all accepted inner PCM chunks are complete and internally consistent. No source header is repaired, no missing audio is synthesized, and original files remain unchanged. Prior complete source audit checksums and unchanged source stamps are reused; the targeted WAV bytes are now hashed and duplicate copies compared exactly. The first extractor was too restrictive for the supplied mixed-rate audio. Its {speech['reused_encoder_computations']} valid completed vectors were retained under their original hashes; {speech['new_encoder_computations']} missing vectors were then computed. {speech['resampled_people']} original stereo 48 kHz recordings are channel-averaged and deterministically polyphase-resampled to 16 kHz. Source rates are {speech['source_sample_rates']}. No source or completed vector was rewritten.

For all {manifest['recording_duration_quality']['participants']} participants, WAV duration is compared with the original MFCC frame count multiplied by its median frame step. WAV minus estimated MFCC duration minimum/median/maximum is {manifest['recording_duration_quality']['wav_minus_estimated_mfcc_seconds_min_median_max'][0]:+.6f}s / {manifest['recording_duration_quality']['wav_minus_estimated_mfcc_seconds_min_median_max'][1]:+.6f}s / {manifest['recording_duration_quality']['wav_minus_estimated_mfcc_seconds_min_median_max'][2]:+.6f}s. The complete participant table is saved. Approximate duration agreement does not establish event-level alignment or speaker identity.

Verified participant/interviewer attribution remains unavailable. These are whole-interview embeddings; no speaker identity or participant-only speech selection is claimed. Clip pooling does not use cross-modal event alignment. Unsegmented interviewer speech, deterministic partial recording coverage, only {len(y)} development participants, and {int((y==4).sum())} Severe examples limit what this experiment establishes. The model retains the original transcript/AV quality limitations. No StudentSADD location was supplied and no failed EMU transfer is repeated.

Microsoft WavLM Base Plus (Chen et al.) provides the frozen public encoder: [pinned model card](https://huggingface.co/microsoft/wavlm-base-plus/blob/{feature_context['encoder']['revision']}/README.md), [publisher source](https://github.com/microsoft/unilm/tree/master/wavlm), and [publisher-linked Attribution-ShareAlike 3.0 license](https://github.com/microsoft/UniSpeech/blob/main/LICENSE). Original pretrained weights remain unchanged; the pinned download manifest, attribution notice and {link(Path(feature_context['encoder']['root'])/'LICENSE.txt', 'included license')} preserve provenance and terms. The encoder is not a depression-pretrained classifier and no participant data was uploaded for extraction.

Stop after this bounded experiment. The next evidence-supported work is to obtain verified participant-speech annotations and additional compatible participants with verified labels, particularly minority classes, while reserving a fresh independent evaluation cohort before further development. Neither model replacement nor synthetic duplication can guarantee 60%.

**Artifacts and inference**

'''
    for path, label in [(out/'delivery_manifest.json', 'Measured decision and preservation manifest'),
                        (out/'experiment_ledger_extended.csv', f'Extended ledger ({manifest["updated_ledger_rows"]} rows)'),
                        (s.CONFIG, 'Frozen four-configuration budget'), (out/'protocol.json', 'Folds, seeds, hashes and training ownership'),
                        (out/'final/plan.json', 'Saved models and inference preprocessing'),
                        (out/'matched_comparison.csv', 'Matched metrics including majority comparator'),
                        (out/'oof', 'Original-participant out-of-fold predictions'),
                        (out/'language_paired_comparisons.csv', 'Participant-paired uncertainty against49.28'),
                        (out/'language_participant_changes.csv', 'Participant-level gains and losses'),
                        (out/'per_class_precision_recall.csv', 'Every class precision/recall'),
                        (out/'inner_fitting_diagnostics.json', 'All inner fitting diagnostics'),
                        (out/'outer_fitting_diagnostics.json', 'Outer fitting diagnostics'),
                        (out/'final_fitting_diagnostics.json', 'Final fitting diagnostics'),
                        (out/'final_training_predictions.npz', 'Final fitting-population predictions'),
                        (out/'verification.json', 'Checkpoint replay and preservation checks'),
                        (out/'raw_inference_verification.json', 'Raw inference parity'),
                        (Path(manifest['speech_feature_receipt']), 'Audio coverage and quality receipt'),
                        (out/'recording_duration_quality.csv', 'Every participant recording-duration and sampling diagnostic'),
                        (Path(manifest['retained_numeric_bundle']), 'Unchanged49.28 numeric bundle'),
                        (Path(manifest['retained_checkpoint']['path']), 'Preserved original checkpoint')]:
        text += '- '+link(path, label)+'\n'
    project = Path.cwd().resolve()
    command = ['.venv/bin/python -m src.wavlm_inference --policy wavlm_only',
               f'  --transcript {project}/dataset/487_P/487_Transcript.csv',
               f'  --audio-csv {project}/dataset/487_P/features/487_OpenSMILE2.3.0_mfcc.csv',
               f'  --visual-csv {project}/dataset/487_P/features/487_OpenFace2.1.0_Pose_gaze_AUs.csv',
               f'  --audio-wav {project}/dataset/487_P/487_AUDIO.wav',
               '  --output /private/tmp/project_v2_wavlm_new_prediction.json']
    text += '\nRun the archived new candidate from the project folder; use a fresh output filename. Participant487 demonstrates inference using a training example, not a new evaluation:\n\n```bash\n'
    text += f'cd {project}\n'+' \\\n'.join(command)+'\n```\n\n'
    text += 'The preserved model remains available with `.venv/bin/python -m src.checked_inference` using the same transcript/MFCC/visual arguments and a fresh output filename. It requires no WAV or questionnaire input. `.venv/bin/python -m src.wavlm_accuracy run --resume` checks completed training caches with zero new fits.\n'
    return text


if __name__ == '__main__':
    deliver()
