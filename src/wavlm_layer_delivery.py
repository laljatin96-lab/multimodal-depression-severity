"""Deliver measured results for the bounded zero-window/middle-layer experiment.

No model fitting or encoder execution occurs here.  The raw-file example must
already exist; its probabilities are compared with the saved fitting cache.
"""
import json
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from . import wavlm_layer_accuracy as s
from .batch_inference import atomic_json, load_record
from .common import digest
from .final_evaluation import sha256
from .raw_frame_uncertainty import weighted_metrics
from .wavlm_layer_inference import frozen_model, predict_arrays
from .wavlm_layer_verify import (CLASS_NAMES, LANGUAGE_ROOT, PRESERVATION,
                                  ensure_previous_files, repeated_predictions)


REPORT = Path('WAVLM_LAYER_RESULTS_20261029.md')
PREVIOUS_LEDGER = Path('results/wavlm_accuracy/experiment_ledger_extended.csv')


def fitting_diagnostics(state, x, y):
    estimator = state['estimator']
    if hasattr(estimator, 'diagnostics'):
        return estimator.diagnostics(x, y)
    p = s.predict_state(state, x); model = estimator.model
    return {'family': state['candidate']['family'], 'real_fitting_participants': len(y),
        'class_counts': np.bincount(y, minlength=5).tolist(),
        'classifier_dimensions': int(model.n_features_in_),
        'parameter_count': int(model.coef_.size+model.intercept_.size),
        'coefficient_l2_norm': float(np.linalg.norm(model.coef_)),
        'iterations': model.n_iter_.tolist(), 'maximum_iterations': int(model.max_iter),
        'converged': bool((model.n_iter_ < model.max_iter).all()),
        'unweighted_fitting_log_loss': float(log_loss(y, p, labels=range(5))),
        'fitting_accuracy': float((p.argmax(1) == y).mean()),
        'checkpoint_rule': 'Previously verified fitting-participant checkpoint; no refitting'}


def measured_comparison(out, frame, folds, policies):
    """Recalculate complete-repetition metrics and pooled per-class precision."""
    y = frame.severity_class.to_numpy(int); seeds = sorted({f['seed'] for f in folds})
    predictions = {name: repeated_predictions(out, name, frame, seeds)[0]
                   for name in ['baseline']+policies}
    predictions['language_control'] = repeated_predictions(LANGUAGE_ROOT, 'without_age', frame, seeds)[0]
    majority = []
    for seed in seeds:
        p = np.full(len(y), -1, int); owners = np.zeros(len(y), int)
        for fold in [f for f in folds if f['seed'] == seed]:
            p[fold['heldout']] = np.bincount(y[fold['fit']], minlength=5).argmax()
            owners[fold['heldout']] += 1
        if not (owners == 1).all():
            raise ValueError('Majority comparator participant ownership differs')
        majority.append(p)
    predictions['majority'] = np.stack(majority)
    saved = pd.read_csv(out/'summary.csv').set_index('model')
    saved_matrices = json.loads((out/'summed_confusion_matrices.json').read_text())['matrices']
    labels = np.tile(y, len(seeds)); rows, classes, matrices = [], [], {}
    for name, pred in predictions.items():
        point = weighted_metrics(y, pred, np.ones((1, len(y)), int))[0]
        matrix = confusion_matrix(labels, pred.reshape(-1), labels=range(5))
        if name in saved.index:
            np.testing.assert_allclose(point, saved.loc[name,
                ['accuracy_mean', 'macro_f1_mean', 'balanced_accuracy_mean']].to_numpy(float), rtol=0, atol=1e-12)
            np.testing.assert_array_equal(matrix, saved_matrices[name])
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, pred.reshape(-1), labels=range(5), zero_division=0)
        matrices[name] = matrix.tolist()
        rows.append({'model': name, 'unique_participants': len(y), 'repetitions': len(seeds),
            'correct_repeated_predictions': int((pred == y).sum()), 'repeated_predictions': pred.size,
            'accuracy': float(point[0]), 'macro_f1': float(point[1]), 'balanced_accuracy': float(point[2]),
            'majority_class_predictions': int((pred == np.bincount(y).argmax()).sum())})
        for c in range(5):
            classes.append({'model': name, 'class': c, 'class_name': CLASS_NAMES[c],
                'unique_participants': int((y == c).sum()), 'repeated_support': int(support[c]),
                'precision_pooled': float(precision[c]), 'recall': float(recall[c]), 'f1_pooled': float(f1[c])})
    return pd.DataFrame(rows), pd.DataFrame(classes), matrices, predictions


def focused_test_evidence(out):
    """Report recorded test evidence without inventing a test count."""
    evidence = []
    for path in sorted((out/'logs').glob('*test*.log')):
        text = path.read_text(errors='replace')
        matches = re.findall(r'(\d+) passed(?:, (\d+) skipped)?', text)
        if matches:
            passed, skipped = matches[-1]
            evidence.append({'path': str(path), 'sha256': sha256(path), 'passed': int(passed),
                'skipped': int(skipped or 0), 'reported_failures': bool(re.search(r'\d+ failed', text))})
    if any(row['reported_failures'] for row in evidence):
        raise ValueError('A supplied focused-test log reports failures')
    return evidence


def collect_diagnostics(out, x, frame, folds, old, settings, completion, plan):
    y = frame.severity_class.to_numpy(int)
    inner, outer, final = [], [], {}
    predictions = {'participant_id': frame.participant_id.to_numpy(str)}
    seen = set()
    with threadpool_limits(limits=settings['cpu_threads']), s.offline_network():
        for fold, baseline in zip(folds, old):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            record = load_record(directory/'result.json', completion['context_hash'])
            s.validate(record, fold, baseline, y, settings, directory)
            for row in record['fitting_diagnostics']:
                key = (fold['seed'], fold['fold'], row['candidate_index'], tuple(row['fit']), tuple(row['heldout']))
                if key in seen or row['candidate_index'] not in range(1, len(s.candidates(settings))):
                    raise ValueError('Unknown or duplicate inner fitting diagnostic')
                seen.add(key)
                if not any(row['fit'] == f['fit'] and row['heldout'] == f['heldout'] for f in fold['inner']):
                    raise ValueError('Inner diagnostic ownership differs')
                if not row['converged'] or row['real_fitting_participants'] != len(row['fit']):
                    raise ValueError('Unconverged or incorrectly owned fitting diagnostic')
                np.testing.assert_array_equal(row['class_counts'], np.bincount(y[row['fit']], minlength=5))
                candidate = s.candidates(settings)[row['candidate_index']]
                for key in ['representation', 'C', 'speech_scale']:
                    if row[key] != candidate[key]:
                        raise ValueError('Inner diagnostic candidate differs')
                inner.append({'seed': fold['seed'], 'outer_fold': fold['fold'], **row})
            for policy in settings['policies']:
                state = pickle.loads((directory/f'{policy}.pkl').read_bytes())
                outer.append({'seed': fold['seed'], 'outer_fold': fold['fold'], 'policy': policy,
                    'fit': fold['fit'], 'heldout': fold['heldout'],
                    **fitting_diagnostics(state, s.subset(x, fold['fit']), y[fold['fit']])})
        expected = (len(s.candidates(settings))-1)*sum(len(f['inner']) for f in folds)
        if len(inner) != expected:
            raise ValueError('Incomplete fitting diagnostics')
        for policy, record in plan['models'].items():
            checked, state, _ = frozen_model(policy)
            if checked != record or state['fit_participants'] != frame.participant_id.tolist():
                raise ValueError('Final model ownership differs')
            p = predict_arrays(record, state, x)
            np.testing.assert_allclose(p, s.predict_state(state, x), rtol=0, atol=1e-14)
            predictions[f'{policy}_decision_scores'] = p
            final[policy] = {'checkpoint_sha256': record['sha256'], 'existing_previous': record['existing_previous'],
                            **fitting_diagnostics(state, x, y)}
    return inner, outer, final, predictions


def raw_parity(out, frame, plan, final_predictions, speech_root):
    policy = 'new_only'; raw = json.loads((out/'raw_training_example.json').read_text())
    record = plan['models'][policy]; i = frame.participant_id.tolist().index('487')
    expected = final_predictions[f'{policy}_decision_scores'][i:i+1]
    if record['existing_previous'] or raw['policy'] != policy or raw['model_sha256'] != record['sha256']:
        raise ValueError('Raw example did not use the saved new candidate')
    if raw['input']['participant_id'] != '487' or raw['questionnaire_inputs']:
        raise ValueError('Raw example participant or inference-input contract differs')
    np.testing.assert_allclose(raw['decision_scores'], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(raw['predicted_class'], expected.argmax(1))
    feature_protocol = json.loads((speech_root/'protocol.json').read_text())
    row = load_record(speech_root/'participants/487.json', feature_protocol['context_hash'])
    if raw['input']['wavlm']['raw_wav_sha256'] != row['raw_wav_sha256']:
        raise ValueError('Raw/cached WAV source differs')
    return {'passed': True, 'training_participant': '487', 'policy': policy, 'new_fits': 0,
        'encoder_run_in_delivery': False, 'model_sha256': record['sha256'],
        'raw_inference_sha256': sha256(out/'raw_training_example.json'),
        'raw_wav_sha256': row['raw_wav_sha256'],
        'maximum_probability_difference': float(np.max(np.abs(np.asarray(raw['decision_scores'])-expected))),
        'validation_test_scored': False}


def deliver():
    from .wavlm_layer_features import ROOT as speech_root
    out = s.OUTPUT
    names = ['inner_fitting_diagnostics.json', 'inner_fitting_diagnostics.csv',
        'outer_fitting_diagnostics.json', 'outer_fitting_diagnostics.csv', 'final_fitting_diagnostics.json',
        'final_training_predictions.npz', 'raw_inference_verification.json', 'matched_comparison.csv',
        'per_class_precision_recall.csv', 'delivery_confusion_matrices.json', 'experiment_ledger_extended.csv',
        'delivery_manifest.json', 'report_receipt.json']
    existing = [str(p) for p in [REPORT]+[out/name for name in names] if p.exists()]
    if existing:
        raise FileExistsError(f'Preserving completed delivery artifacts: {existing}')
    previous_files = json.loads(PRESERVATION.read_text()); ensure_previous_files(previous_files)
    inputs = s.load_inputs(); x, frame, old_settings, old, _ = inputs
    settings = json.loads(s.CONFIG.read_text()); policies = list(settings['policies'])
    completion = json.loads((out/'completion.json').read_text())
    verified = json.loads((out/'verification.json').read_text())
    if not completion['complete'] or not verified['passed']:
        raise ValueError('Completed training and verification required')
    envelope = json.loads((out/'final/plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash'] or plan['context_hash'] != completion['context_hash']:
        raise ValueError('Final plan differs')
    for path, expected in plan['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Final binding changed')
    folds = s.nested_folds(frame, old_settings)
    inner, outer, final, final_predictions = collect_diagnostics(out, x, frame, folds, old, settings, completion, plan)
    raw_receipt = raw_parity(out, frame, plan, final_predictions, speech_root)
    comparison, classes, matrices, predictions = measured_comparison(out, frame, folds, policies)
    table = comparison.set_index('model'); primary = float(table.loc['accuracy', 'accuracy'])
    previous = float(table.loc['previous_procedure', 'accuracy']); original = float(table.loc['language_control', 'accuracy'])
    intervals = pd.read_csv(out/'matched_paired_comparisons.csv')
    changes = pd.read_csv(out/'matched_error_changes.csv')
    ci = intervals[(intervals.model == 'accuracy') & (intervals.reference == 'previous_procedure')
                   & (intervals.metric == 'accuracy')].iloc[0]
    gain = primary > previous+1e-14
    old_ledger = pd.read_csv(PREVIOUS_LEDGER)
    new_ledger = []
    for policy in [p for p in policies if p != 'previous_procedure']:
        row = table.loc[policy]
        new_ledger.append({'study': 'wavlm_layer_accuracy', 'variant': json.dumps({'model': policy}),
            'nested_accuracy': float(row.accuracy), 'macro_f1': float(row.macro_f1),
            'summary_path': str(out/'summary.csv'), 'summary_sha256': sha256(out/'summary.csv'),
            'evidence_scope': 'Same original163 participants; matched nested participant folds; zero-valued-window '
                'ablation and frozen middle-layer WavLM; exploratory long-history comparison; no validation/test.',
            'repeat_action': 'Do not repeat these four frozen configurations unchanged. '
                + ('Primary higher observed procedure score requires independent confirmation.' if gain
                   else 'No higher primary score; preserve previous verified models.')})
    ledger = pd.concat([old_ledger, pd.DataFrame(new_ledger)], ignore_index=True)
    speech_receipt = json.loads((speech_root/'completion.json').read_text())
    tests = focused_test_evidence(out)
    candidate_best = max(policies, key=lambda policy: float(table.loc[policy, 'accuracy']))
    manifest = {'complete': True, 'target_accuracy': settings['target_accuracy'],
        'target_reached': bool(primary >= settings['target_accuracy']),
        'minimum_requested_accuracy_reached': bool(primary > settings['minimum_requested_accuracy']),
        'previous_procedure_accuracy': previous, 'preserved_original_accuracy': original,
        'new_primary_accuracy': primary, 'new_candidate_accuracies': {
            policy: float(table.loc[policy, 'accuracy']) for policy in policies if policy != 'previous_procedure'},
        'best_observed_primary_procedure_accuracy': max(previous, primary),
        'highest_observed_policy': candidate_best,
        'highest_observed_policy_accuracy': float(table.loc[candidate_best, 'accuracy']),
        'highest_observed_policy_used_for_checkpoint_selection': False,
        'primary_earns_higher_observed_score': gain,
        'primary_paired_conditional_ci_excludes_zero_above': bool(ci.ci95_low > 0),
        'independent_confirmation': False, 'original_models_always_preserved': True,
        'retained_numeric_bundle': 'checkpoints/best_observed_4928',
        'primary_final_uses_existing_previous': plan['models']['accuracy']['existing_previous'],
        'recommended_saved_model': plan['models']['accuracy' if gain else 'previous_procedure'],
        'default_replacement_supported': bool(gain and ci.ci95_low > 0 and not plan['models']['accuracy']['existing_previous']),
        'candidate_models': plan['models'], 'new_nested_pipeline_fits': completion['total_fits'],
        'new_inner_pipeline_fits': len(inner), 'new_full_training_pipeline_fits': plan['new_full_training_fits'],
        'focused_test_evidence': tests, 'old_models_refitted': 0,
        'old_files_and_mtimes_preserved': len(previous_files), 'previous_ledger_rows': len(old_ledger),
        'updated_ledger_rows': len(ledger), 'validation_test_scored': False,
        'external_depression_cohort_used': False, 'synthetic_rows': 0, 'speaker_identity_guessed': False,
        'verified_participant_speech_annotations': False, 'speech_feature_receipt': str(speech_root/'completion.json'),
        'remaining_limitation': 'Only163 development people, seven Severe cases, unverified interviewer/participant '
            'speech attribution, and extensive earlier selection. Exact-zero clip removal does not establish speech identity.',
        'next_step': 'Acquire verified participant-speech annotations and additional compatible genuinely labeled '
            'participants while reserving a fresh independent evaluation cohort before further selection.'}
    report = write_report(out, frame, settings, table, classes, matrices, inner, changes, intervals,
                          plan, manifest, verified, speech_receipt)
    # Construct and validate everything before writing new outputs.
    ensure_previous_files(previous_files)
    compact = [{k: v for k, v in row.items() if k not in ['fit', 'heldout', 'class_counts', 'iterations']}
               | {'iterations_max': max(row['iterations'])} for row in inner]
    atomic_json(out/'inner_fitting_diagnostics.json', inner)
    pd.DataFrame(compact).to_csv(out/'inner_fitting_diagnostics.csv', index=False)
    atomic_json(out/'outer_fitting_diagnostics.json', outer)
    pd.DataFrame([{k: v for k, v in row.items() if k not in ['fit', 'heldout', 'class_counts', 'iterations']}
                  for row in outer]).to_csv(out/'outer_fitting_diagnostics.csv', index=False)
    atomic_json(out/'final_fitting_diagnostics.json', final)
    np.savez_compressed(out/'final_training_predictions.npz', **final_predictions)
    atomic_json(out/'raw_inference_verification.json', raw_receipt)
    comparison.to_csv(out/'matched_comparison.csv', index=False)
    classes.to_csv(out/'per_class_precision_recall.csv', index=False)
    atomic_json(out/'delivery_confusion_matrices.json', {'note': 'Rows actual, columns predicted, classes0–4. '
        'Repeated predictions on original people, not independent participants.', 'matrices': matrices})
    ledger.to_csv(out/'experiment_ledger_extended.csv', index=False)
    REPORT.write_text(report)
    ensure_previous_files(previous_files)
    bound = [s.CONFIG, Path(s.__file__), Path(__file__), Path('src/wavlm_layer_verify.py'),
        out/'verification.json', out/'raw_training_example.json', out/'raw_inference_verification.json',
        out/'matched_paired_comparisons.csv', out/'matched_comparison.csv', out/'per_class_precision_recall.csv',
        out/'participant_feature_quality.csv',
        out/'inner_fitting_diagnostics.json', out/'outer_fitting_diagnostics.json',
        out/'final_fitting_diagnostics.json', out/'final_training_predictions.npz',
        out/'experiment_ledger_extended.csv', speech_root/'completion.json', REPORT]
    manifest['bindings'] = {str(path): sha256(path) for path in bound}
    atomic_json(out/'delivery_manifest.json', manifest)
    atomic_json(out/'report_receipt.json', {'report': str(REPORT.resolve()), 'sha256': sha256(REPORT),
        'delivery_manifest_sha256': sha256(out/'delivery_manifest.json')})
    print(json.dumps(manifest, indent=2))
    return manifest


def link(path, label):
    return f'[{label}]({Path(path).resolve()})'


def write_report(out, frame, settings, table, classes, matrices, inner, changes, intervals,
                 plan, manifest, verified, speech):
    y = frame.severity_class.to_numpy(int); n = int(table.loc['accuracy', 'repeated_predictions'])
    policies = list(settings['policies'])
    names = {'majority': 'Fitting-fold majority', 'baseline': 'Older preserved baseline',
        'language_control': 'Retained original language model procedure',
        'previous_procedure': 'Reused previous WavLM selection procedure',
        'accuracy': 'New primary inner-selected procedure', 'new_only': 'New representations only',
        'zero_only': 'Final-layer exact-zero ablation', 'layer6_only': 'Middle-layer exact-zero ablation'}
    improved = manifest['primary_earns_higher_observed_score']
    result = ('The primary procedure has a higher observed score; this remains an exploratory comparison.' if improved
              else 'The primary procedure does not improve the previous observed procedure; prior models are retained.')
    text = f'''# Frozen WavLM layer and exact-zero-window results — 29 October 2026

Primary matched participant-level accuracy: **{100*table.loc['accuracy','accuracy']:.2f}%**. Previous WavLM selection procedure: **{100*table.loc['previous_procedure','accuracy']:.2f}%**. Preserved original language-model procedure: **{100*table.loc['language_control','accuracy']:.2f}%**. {result} The requested above-53% minimum is {'reached' if manifest['minimum_requested_accuracy_reached'] else 'not reached'} and the 56% target is {'reached' if manifest['target_reached'] else 'not reached'}. No score is promised or fabricated.

## Specific opportunity and frozen budget

The previous recording checks found exactly zero-valued selected waveform windows in some original recordings. Passing a zero waveform through a contextual encoder can create nonzero hidden states, so pooling those states may dilute the recording representation. The first ablation omits only exactly zero-valued channel-averaged encoder-input clips from the same deterministic original positions (up to 64), with no energy threshold tuned against labels. The second uses transformer layer 6 with the same omission rule; a middle representation may preserve acoustic cues for Mild/Moderate errors that the final layer misses. This is an explicit hypothesis rather than a claim that either representation has participant-only speech.

All {len(y)} original training participants remain in the evaluation. The model never assigns a channel or segment to an individual, guesses speaker identity, shifts timestamps, invents audio, changes labels or removes difficult participants. Layer 6 and the final hidden layer are frozen public WavLM representations. All previous whole-recording features remain available for the exact matched reference. Zero-window decisions use waveform values only, and encoding uses no depression target.

Four configurations are fixed before fitting: two representations × speech block scales {settings['speech_scales']}, with C={settings['C']}. The prior inner diagnostics motivate keeping the more regularized C=.001; there is no new broad search. Each fitting cohort independently fits the base scaler/PCA32, language scaler with fixed scale.5, speech scaler and whitened PCA16. The logistic model uses 64 coordinates and 325 coefficients/intercepts with normalized square-root class weights. All targets belong to fitting participants, and questionnaire scores/answers, labels and participant IDs are excluded from inference features.

The cap is {settings['max_nested_pipeline_fits']} nested pipeline fits and {settings['max_full_training_fits']} full-training fits. Actual: **{manifest['new_nested_pipeline_fits']} nested fits**, comprising **{manifest['new_inner_pipeline_fits']} inner fits**, plus **{manifest['new_full_training_pipeline_fits']} full-training fits**. No previous model is refitted. The original five outer / three inner participant folds and seeds 42/137/2026 are reused. Preprocessing and tuning belong entirely to fitting/inner-training people. There is no resampling, calibration, pretrained-weight fine-tuning, outer-result tuning, validation or test scoring. The fixed stopping rule is enforced.

## Matched participant-level evaluation

There are **{len(y)} unique participants**, class supports {np.bincount(y,minlength=5).tolist()}, and {n} repeated predictions. Repeats are not additional independent participants. Accuracy, macro-F1 and balanced accuracy average complete repetitions; the majority comparator chooses its class from each outer fitting cohort only.

| Procedure | Correct /{n} | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
'''
    for name in ['majority', 'baseline', 'language_control']+policies:
        row = table.loc[name]
        text += f'| {names[name]} | {int(row.correct_repeated_predictions)} | {100*row.accuracy:.2f}% | {row.macro_f1:.4f} | {100*row.balanced_accuracy:.2f}% |\n'
    text += '\nEncoder references: [WavLM paper](https://arxiv.org/abs/2110.13900) and [Transformers WavLM documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/wavlm). These describe the pretrained encoder and hidden-state interface; they do not establish that layer six improves depression classification in this cohort. The saved prefix-equivalence probe verifies the implemented layer indexing.\n\n'
    text += 'The primary policy can select the exact previous procedure or a new configuration using inner held-out correct count, macro-F1 and declared order. `new_only` restricts that selection to the four new configurations; `zero_only` and `layer6_only` restrict it to the corresponding two-scale ablation. Every new model still uses original semantic/AV and age-free language inputs; these are not audio-only classifiers.\n\n'
    text += '## Correct predictions gained/lost and paired uncertainty\n\n'
    for reference in ['previous_procedure', 'language_control']:
        text += f'Compared with **{names[reference]}**:\n\n'
        for name in [p for p in policies if p != reference]:
            row = changes[(changes.model == name) & (changes.reference == reference)].iloc[0]
            ci = intervals[(intervals.model == name) & (intervals.reference == reference)
                           & (intervals.metric == 'accuracy')].iloc[0]
            text += f'- {names[name]}: **{int(row.gained)} gained / {int(row.lost)} lost**, net {int(row.gained-row.lost):+d}/{n}; {int(row.net_better_people)} participants improve and {int(row.net_worse_people)} worsen across repetitions. Accuracy change **{100*ci.difference:+.2f} pp**, paired 95% interval **[{100*ci.ci95_low:+.2f}, {100*ci.ci95_high:+.2f}] pp**. Majority-class predictions {int(row.reference_majority_predictions)} → {int(row.majority_predictions)}.\n'
        text += '\n'
    text += '''The 5,000 bootstrap samples resample the 163 people, carrying all three predictions for a selected person together (seed 20261025). Intervals are conditional on saved fitted models and folds; they exclude retraining uncertainty and do not correct for the long history of feature/model comparisons. All results are exploratory, including an interval that may exclude zero. Historically reused holdouts do not establish a fresh untouched test cohort; no new validation/test results are reported here. Full intervals for accuracy, macro-F1 and balanced accuracy, per-participant changes and every prediction transition are saved.

## Every class's precision and recall

P/R is pooled precision / recall in percent from summed confusion matrices. Precision is not the mean of fold precisions; repeated support does not increase the number of people.

| Class | Unique people | Original P/R | Previous procedure P/R | Primary P/R | New-only P/R | Zero-only P/R | Layer6-only P/R |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    class_policies = ['language_control', 'previous_procedure', 'accuracy', 'new_only', 'zero_only', 'layer6_only']
    for c in range(5):
        rows = [classes[(classes.model == name) & (classes['class'] == c)].iloc[0] for name in class_policies]
        text += f'| {CLASS_NAMES[c]} | {int(rows[0].unique_participants)} | '+' | '.join(
            f'{100*r.precision_pooled:.2f} / {100*r.recall:.2f}' for r in rows)+' |\n'
    text += '\nAgainst the previous procedure, net correct-prediction changes by class for the primary are: '
    delta = np.asarray(matrices['accuracy']).diagonal()-np.asarray(matrices['previous_procedure']).diagonal()
    text += ', '.join(f'{CLASS_NAMES[c]} {int(delta[c]):+d}' for c in range(5))+'.\n\n'
    text += 'Summed confusion matrices: rows actual, columns predicted, class order None/minimal, Mild, Moderate, Moderately severe, Severe.\n'
    for name in class_policies:
        text += f'\n{name}:\n\n```text\n'+'\n'.join(' '.join(f'{v:3d}' for v in row) for row in matrices[name])+'\n```\n'
    text += '\n## Fitting quality, final selection and reproduction\n\n'
    fitting = pd.read_csv(out/'overfitting_audit.csv').groupby('policy').training_accuracy.mean()
    for name in policies:
        text += f'{names[name]}: mean outer-fitting accuracy **{100*fitting[name]:.2f}%**, outer held-out **{100*table.loc[name,"accuracy"]:.2f}%**.\n\n'
    diagnostics = pd.DataFrame(inner)
    for (representation, scale), group in diagnostics.groupby(['representation', 'speech_scale']):
        text += f'{representation}, scale{scale:g}: mean inner fitting accuracy {100*group.fitting_accuracy.mean():.2f}%, inner held-out {100*group.inner_accuracy.mean():.2f}%, fitting log loss {group.unweighted_fitting_log_loss.mean():.4f}.\n\n'
    iterations = [max(row['iterations']) for row in inner]
    text += f'''All {len(inner)} inner diagnostics record class supports, fitting/held-out indices, C, representation, block scale, PCA variance, model capacity, coefficient norm, iterations, fitting accuracy/log loss and held-out accuracy. Maximum iterations range from {min(iterations)} to {max(iterations)}, below3000; convergence warnings raise errors. These are converged regularized logistic fits, so there are no neural training epochs or early-stopping checkpoints. Means of fitting cohorts describe fitting quality rather than validation accuracy.

Verification replayed **{verified['checkpoints_replayed']}** outer/final checkpoints, independently checked scaler/PCA statistics and weights against fitting people, and confirmed zero-fit, zero-rewrite completed resume. The training-participant487 raw-file CLI example reproduces cached candidate predictions within1e-12 using identical WAV bytes. The delivery runs no encoder and performs no fitting. Final model predictions on all fitting people are saved strictly for reproducibility; they are not validation scores.

'''
    tests = manifest['focused_test_evidence']
    if tests:
        text += 'Recorded focused-test evidence: '+', '.join(f'{row["passed"]} passed in `{row["path"]}`' for row in tests)+'.\n\n'
    else:
        text += 'Focused-test execution evidence is kept in run logs; no unverified test count is claimed by this report.\n\n'
    for policy, record in plan['models'].items():
        text += f'Final `{policy}`: '+('reuses the previous saved checkpoint' if record['existing_previous']
            else f'saves a newly fitted {record["candidate"]["representation"]} model')+f'; `{record["path"]}`.\n\n'
    text += '''The previous49.90% number describes a nested selection procedure. Its full-training, inner-only selection retained the original49.28% checkpoint; it does not represent a separate saved checkpoint with49.90% independently measured generalization accuracy. The same distinction applies to the new procedure: final selection uses cached inner predictions only, never the outer score. A better outer ablation is not silently promoted by selecting its outer performance. No old model or global best-model document is overwritten.

## Input coverage and limitations

The new extraction completion receipt is reproduced below; source and per-participant provenance remain in the feature cache. It records original training people only, deterministic clip ownership, exact-zero removal and feature hashes. Exactly zero-valued averaged input does not prove silence in each original stereo channel. Nonzero clips can still contain interviewer speech, music, noise or low signal. No verified speaker annotations are available, and sampling/recording-duration agreement does not prove cross-modal event alignment.

```json
'''
    compact_speech = {key: value for key, value in speech.items()
                      if key not in ['participant_records', 'bindings', 'new_participant_records']}
    text += json.dumps(compact_speech, indent=2)+'\n```\n\n'
    text += f'''The remaining limitations include {len(y)} development people, only{int((y==4).sum())} Severe cases, mixed input quality and extensive prior search. No additional compatible labeled cohort or StudentSADD location is available. The failed EMU transfer was not repeated. Synthetic duplication cannot add independent minority participants. Neither changing the model name nor changing evaluation folds can establish53–56% accuracy.

This bounded experiment stops at its declared budget. The evidence-supported next step is verified participant-speech annotations and additional compatible genuinely labeled people, particularly Mild/Moderate/Severe, with a new independent cohort reserved before further development.

## Saved artifacts and inference

'''
    for path, label in [(out/'delivery_manifest.json', 'Measured model decision and preservation'),
        (out/'experiment_ledger_extended.csv', f'Extended experiment ledger ({manifest["updated_ledger_rows"]} rows)'),
        (s.CONFIG, 'Frozen four-configuration budget'), (out/'protocol.json', 'Folds, seeds, ownership and source hashes'),
        (out/'final/plan.json', 'Final models and preprocessing contract'),
        (out/'matched_comparison.csv', 'Matched metrics including majority comparator'),
        (out/'oof', 'Original-participant out-of-fold predictions'),
        (out/'matched_paired_comparisons.csv', 'Participant-paired uncertainty against both references'),
        (out/'matched_participant_changes.csv', 'Participant gains and losses'),
        (out/'matched_prediction_changes.csv', 'Every repeated prediction transition'),
        (out/'per_class_precision_recall.csv', 'Every class precision/recall'),
        (out/'inner_fitting_diagnostics.json', 'All inner fitting diagnostics'),
        (out/'outer_fitting_diagnostics.json', 'Outer fitting diagnostics'),
        (out/'final_fitting_diagnostics.json', 'Final fitting diagnostics'),
        (out/'final_training_predictions.npz', 'Final fitting-population predictions'),
        (out/'verification.json', 'Replay, preservation and resume verification'),
        (out/'raw_inference_verification.json', 'Raw-file inference parity'),
        (out/'participant_feature_quality.csv', 'Every participant clip-retention and coverage check'),
        (Path(manifest['speech_feature_receipt']), 'New audio feature coverage and provenance'),
        (Path(manifest['retained_numeric_bundle']), 'Unchanged original49.28 numeric bundle')]:
        text += '- '+link(path, label)+'\n'
    project = Path.cwd().resolve()
    command = ['.venv/bin/python -m src.wavlm_layer_inference --policy new_only',
        f'  --transcript {project}/dataset/487_P/487_Transcript.csv',
        f'  --audio-csv {project}/dataset/487_P/features/487_OpenSMILE2.3.0_mfcc.csv',
        f'  --visual-csv {project}/dataset/487_P/features/487_OpenFace2.1.0_Pose_gaze_AUs.csv',
        f'  --audio-wav {project}/dataset/487_P/487_AUDIO.wav',
        '  --output /private/tmp/project_v2_wavlm_layer_new_prediction.json']
    text += '\nRun the archived candidate from the project folder with a fresh output filename. Participant487 is a training inference demonstration, not a new evaluation:\n\n```bash\n'
    text += f'cd {project}\n'+' \\\n'.join(command)+'\n```\n\n'
    text += 'The retained original model remains available through `.venv/bin/python -m src.checked_inference` with transcript/MFCC/visual inputs. `.venv/bin/python -m src.wavlm_layer_accuracy run --resume` validates completed caches without fitting.\n'
    return text


if __name__ == '__main__':
    deliver()
