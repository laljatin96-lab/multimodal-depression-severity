"""Replay frozen layer/zero-clip experiments and analyze paired participant errors.

This command performs no fitting or encoder extraction.  Every resample keeps
all repeated predictions for the same participant together.
"""
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import wavlm_layer_accuracy as s
from .batch_inference import atomic_json, load_record
from .common import digest
from .final_evaluation import sha256
from .raw_frame_uncertainty import participant_counts, weighted_metrics


PRESERVATION = Path('results/wavlm_layer_preservation/before.json')
PREVIOUS_ROOT = Path('results/wavlm_accuracy')
LANGUAGE_ROOT = Path('results/age_language_accuracy')
CLASS_NAMES = ['None/minimal', 'Mild', 'Moderate', 'Moderately severe', 'Severe']


def ensure_previous_files(previous):
    changed = [path for path, row in previous.items()
               if not Path(path).is_file()
               or Path(path).stat().st_mtime_ns != row['mtime_ns']
               or sha256(path) != row['sha256']]
    if changed:
        raise ValueError(f'Previously existing files changed: {changed[:10]}')


def speech_key(model):
    """The declared representation determines the only speech block to inspect."""
    return getattr(model, 'speech_key', model.candidate.get('representation', 'speech'))


def inspect_model(model, x, y):
    """Verify fitting-population statistics independently of stored diagnostics."""
    y = np.asarray(y, int)
    if model.real_count != len(y):
        raise ValueError('Model fitting population differs')
    np.testing.assert_array_equal(model.class_counts, np.bincount(y, minlength=5))
    if hasattr(model, 'speech_pca'):
        key = speech_key(model)
        if key not in x or np.asarray(x[key]).shape != (len(y), 1536):
            raise ValueError('Declared speech representation is absent or misaligned')
        if model.base_pca.n_samples_ != len(y) or model.speech_pca.n_samples_ != len(y):
            raise ValueError('PCA fitting ownership differs')
        for scaler, values in [(model.base_scaler, x['semantic_av']),
                               (model.language_scaler, x['language']),
                               (model.speech_scaler, x[key])]:
            np.testing.assert_allclose(scaler.mean_, np.asarray(values).mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.base_pca.mean_,
            model.base_scaler.transform(x['semantic_av']).mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.speech_pca.mean_,
            model.speech_scaler.transform(x[key]).mean(0), rtol=0, atol=1e-12)
        if not model.speech_pca.whiten or not (model.speech_pca.explained_variance_ > 0).all():
            raise ValueError('Speech projection whitening/variance differs')
        transformed = model.transform(x)
        if transformed.shape != (len(y), 64) or model.model.n_features_in_ != 64:
            raise ValueError('Expected 64 regularized classifier coordinates')
    else:
        projection = model.projection
        if projection.real_count != len(y) or projection.pca.n_samples_ != len(y):
            raise ValueError('Language-control preprocessing ownership differs')
        np.testing.assert_allclose(projection.base_scaler.mean_, x['semantic_av'].mean(0), rtol=0, atol=1e-12)
        # The historical projection stores unused age alternatives as well.
        # This retained checkpoint selects only the language block at inference.
        key = model.candidate['feature_set']
        if key != 'language':
            raise ValueError('Expected the preserved age-free language checkpoint')
        for key in [key]:
            scaler = projection.scalers[key]
            np.testing.assert_allclose(scaler.mean_,
                projection.imputers[key].transform(x[key]).mean(0), rtol=0, atol=1e-12)
    power = .5 if hasattr(model, 'speech_pca') else model.candidate['weight_power']
    expected = (len(y)/(5*np.bincount(y, minlength=5)[y]))**power
    expected *= len(y)/expected.sum()
    np.testing.assert_allclose(model.weights, expected, rtol=0, atol=1e-12)
    if not (model.model.n_iter_ < model.model.max_iter).all():
        raise ValueError('Model did not converge within its declared limit')


def repeated_predictions(root, policy, frame, seeds):
    y = frame.severity_class.to_numpy(int)
    labels, scores = [], []
    for seed in seeds:
        table = pd.read_csv(root/'oof'/f'{policy}_seed{seed}'/'predictions.csv',
                            dtype={'participant_id': str})
        if table.participant_id.tolist() != frame.participant_id.tolist():
            raise ValueError('Prediction participant alignment differs')
        np.testing.assert_array_equal(table.severity_class, y)
        if 'seed' in table and not table.seed.eq(seed).all():
            raise ValueError('Prediction repetition differs')
        if not table.split.eq('train_nested_oof').all():
            raise ValueError('Predictions must be original-training out-of-fold outputs')
        p = table[[f'p{i}' for i in range(5)]].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('Invalid saved probabilities')
        np.testing.assert_allclose(p.sum(1), 1, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(p.argmax(1), table.predicted_label)
        labels.append(table.predicted_label.to_numpy(int)); scores.append(p)
    return np.stack(labels), np.stack(scores)


def feature_ownership(frame, x, out):
    """Check retained clip sets and cache/row equality without decoding or encoding."""
    from .wavlm_layer_features import ROOT as feature_root
    from .wavlm_features_complete import ROOT as previous_feature_root
    protocol = json.loads((feature_root/'protocol.json').read_text())
    previous_protocol = json.loads((previous_feature_root/'protocol.json').read_text())
    receipt = json.loads((feature_root/'completion.json').read_text())
    ids = frame.participant_id.tolist()
    if (protocol['context']['ids'] != ids or receipt['participants'] != len(ids)
            or receipt['dropped_participants'] != 0 or receipt['targets_used']
            or receipt['validation_test_processed']):
        raise ValueError('New representation population or target ownership differs')
    rows = []
    for i, pid in enumerate(ids):
        row = load_record(feature_root/'participants'/f'{pid}.json', protocol['context_hash'])
        old = load_record(previous_feature_root/'participants'/f'{pid}.json', previous_protocol['context_hash'])
        starts = np.asarray(old['clip_start_samples'], int)
        zero = np.asarray(old['clip_rms']) == 0
        if (row['participant_id'] != pid or row['source_keys'] != old['source_keys']
                or row['raw_wav_sha256'] != old['raw_wav_sha256'] or row['targets_used']
                or row['alignment_used'] or row['clip_samples'] != old['clip_samples']
                or row['clip_start_sample_rate'] != 16000):
            raise ValueError('Feature source/target/clip ownership differs')
        np.testing.assert_array_equal(row['original_clip_start_samples'], starts)
        np.testing.assert_array_equal(row['clip_start_samples'], starts[~zero])
        np.testing.assert_array_equal(row['removed_clip_start_samples'], starts[zero])
        if not len(row['clip_start_samples']) or row['exact_zero_clips_removed'] != int(zero.sum()):
            raise ValueError('Incorrect clip retention count')
        if row['final_features_reused'] != bool((~zero).all()):
            raise ValueError('Incorrect final-vector reuse ownership')
        for key in ['final_nonzero', 'layer6_nonzero']:
            np.testing.assert_array_equal(row[key], x[key][i])
        if row['final_features_reused']:
            np.testing.assert_array_equal(row['final_nonzero'], x['speech'][i])
        rows.append({'participant_id': pid, 'source_keys': json.dumps(row['source_keys']),
            'raw_wav_sha256': row['raw_wav_sha256'], 'original_sampled_clips': len(starts),
            'retained_sampled_clips': len(row['clip_start_samples']),
            'exact_zero_clips_removed': row['exact_zero_clips_removed'],
            'final_features_reused': row['final_features_reused'],
            'unique_covered_seconds': row['unique_covered_seconds'],
            'coverage_fraction': row['coverage_fraction'], 'wav_seconds': row['quality']['seconds']})
    table = pd.DataFrame(rows)
    expected = {'original_sampled_clips': int(table.original_sampled_clips.sum()),
        'retained_sampled_clips': int(table.retained_sampled_clips.sum()),
        'exact_zero_clips_removed': int(table.exact_zero_clips_removed.sum()),
        'participants_with_zero_clips': int((table.exact_zero_clips_removed > 0).sum()),
        'final_vectors_reused': int(table.final_features_reused.sum()),
        'final_vectors_recomputed': int((~table.final_features_reused).sum()),
        'new_layer6_vectors': len(ids)}
    for key, value in expected.items():
        if receipt[key] != value:
            raise ValueError(f'Feature completion count differs: {key}')
    probe = json.loads((feature_root/'layer_probe.json').read_text())
    if (probe['context_hash'] != protocol['context_hash'] or probe['full_hidden_state_index'] != 6
            or probe['full_hidden_state_count'] != 13 or probe['prefix_hidden_state_count'] != 7
            or probe['maximum_absolute_difference'] > probe['absolute_tolerance'] or probe['targets_used']):
        raise ValueError('Middle-layer prefix inference equivalence was not established')
    table.to_csv(out/'participant_feature_quality.csv', index=False)
    return {'passed': True, 'participants': len(ids), 'dropped_participants': 0,
        'original_sampling_positions_preserved': True, 'cache_matches_participant_records': True,
        'middle_layer_prefix_parity': probe, **expected}


def matched_comparisons(out, frame, policies, seeds):
    """Compare with the previous procedure and retained original checkpoint rule."""
    y = frame.severity_class.to_numpy(int)
    predictions, probabilities = {}, {}
    for name in policies:
        predictions[name], probabilities[name] = repeated_predictions(out, name, frame, seeds)
    previous, previous_scores = repeated_predictions(PREVIOUS_ROOT, 'accuracy', frame, seeds)
    np.testing.assert_array_equal(predictions['previous_procedure'], previous)
    np.testing.assert_allclose(probabilities['previous_procedure'], previous_scores, rtol=0, atol=1e-14)
    predictions['language_control'], probabilities['language_control'] = repeated_predictions(
        LANGUAGE_ROOT, 'without_age', frame, seeds)
    counts = participant_counts(len(y), 5000, 20261025)
    point_counts = np.ones((1, len(y)), int)
    boot = {name: weighted_metrics(y, pred, counts) for name, pred in predictions.items()}
    point = {name: weighted_metrics(y, pred, point_counts)[0] for name, pred in predictions.items()}
    intervals, changes, classes, people, transitions, draws = [], [], [], [], [], {}
    for reference in ['previous_procedure', 'language_control']:
        old = predictions[reference]; old_good = old == y
        for name in policies:
            if name == reference:
                continue
            pred = predictions[name]; good = pred == y
            gained, lost = good & ~old_good, old_good & ~good
            difference = boot[name]-boot[reference]
            draws[f'{name}_minus_{reference}'] = difference
            limits = np.percentile(difference, [2.5, 97.5], axis=0)
            for i, metric in enumerate(['accuracy', 'macro_f1', 'balanced_accuracy']):
                intervals.append({'model': name, 'reference': reference, 'metric': metric,
                    'reference_value': float(point[reference][i]), 'candidate_value': float(point[name][i]),
                    'difference': float(point[name][i]-point[reference][i]),
                    'ci95_low': float(limits[0, i]), 'ci95_high': float(limits[1, i])})
            changes.append({'model': name, 'reference': reference, 'correct': int(good.sum()),
                'reference_correct': int(old_good.sum()), 'gained': int(gained.sum()), 'lost': int(lost.sum()),
                'unique_people_with_gain': int(gained.any(0).sum()), 'unique_people_with_loss': int(lost.any(0).sum()),
                'net_better_people': int((good.sum(0) > old_good.sum(0)).sum()),
                'net_worse_people': int((good.sum(0) < old_good.sum(0)).sum()),
                'majority_predictions': int((pred == np.bincount(y).argmax()).sum()),
                'reference_majority_predictions': int((old == np.bincount(y).argmax()).sum())})
            for c in range(5):
                mask = y == c
                classes.append({'model': name, 'reference': reference, 'class': c, 'class_name': CLASS_NAMES[c],
                    'participants': int(mask.sum()), 'reference_correct': int(old_good[:, mask].sum()),
                    'correct': int(good[:, mask].sum()), 'reference_recall': float(old_good[:, mask].mean()),
                    'recall': float(good[:, mask].mean()), 'gained': int(gained[:, mask].sum()),
                    'lost': int(lost[:, mask].sum())})
            for i, pid in enumerate(frame.participant_id):
                people.append({'model': name, 'reference': reference, 'participant_id': pid, 'class': int(y[i]),
                    'reference_correct_repeats': int(old_good[:, i].sum()), 'correct_repeats': int(good[:, i].sum()),
                    'gained': int(gained[:, i].sum()), 'lost': int(lost[:, i].sum())})
                for j, seed in enumerate(seeds):
                    transitions.append({'model': name, 'reference': reference, 'participant_id': pid,
                        'class': int(y[i]), 'seed': seed, 'reference_prediction': int(old[j, i]),
                        'prediction': int(pred[j, i]), 'gained': bool(gained[j, i]), 'lost': bool(lost[j, i])})
    for filename, rows in [('matched_paired_comparisons', intervals), ('matched_error_changes', changes),
                           ('matched_class_changes', classes), ('matched_participant_changes', people),
                           ('matched_prediction_changes', transitions)]:
        pd.DataFrame(rows).to_csv(out/f'{filename}.csv', index=False)
    np.savez_compressed(out/'matched_paired_bootstrap.npz', participant_counts=counts, **draws)
    return changes


def verify():
    out = s.OUTPUT
    if (out/'verification.json').exists():
        raise FileExistsError('Preserving completed verification')
    previous_files = json.loads(PRESERVATION.read_text())
    ensure_previous_files(previous_files)
    inputs = s.load_inputs(); x, frame, old_settings, old, _ = inputs
    settings = json.loads(s.CONFIG.read_text()); y = frame.severity_class.to_numpy(int)
    completion = json.loads((out/'completion.json').read_text())
    envelope = json.loads((out/'final/plan.json').read_text()); plan = envelope['plan']
    if not completion['complete'] or digest(plan) != envelope['plan_hash'] or plan['context_hash'] != completion['context_hash']:
        raise ValueError('Final plan differs from completed study')
    for path, expected in plan['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Final binding changed')
    before = {str(p): [sha256(p), p.stat().st_mtime_ns] for p in out.rglob('*')
              if p.is_file() and p.name != '.batch.lock'}
    def forbidden(*args, **kwargs):
        raise AssertionError('No fitting or prediction on a completed resume')
    with patch.object(s, 'job', forbidden), patch.object(s, 'predict_state', forbidden):
        if s.run(inputs, settings, resume=True)['new_fits'] != 0:
            raise ValueError('Completed resume performed new fits')
    if before != {path: [sha256(path), Path(path).stat().st_mtime_ns] for path in before}:
        raise ValueError('Completed resume rewrote evidence')
    folds = s.nested_folds(frame, old_settings)
    records, rows, replayed = [], [], 0
    with threadpool_limits(limits=settings['cpu_threads']), s.offline_network():
        for fold, baseline in zip(folds, old):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            record = load_record(directory/'result.json', completion['context_hash'])
            s.validate(record, fold, baseline, y, settings, directory); records.append(record)
            for policy, selected in record['selected'].items():
                state = pickle.loads((directory/f'{policy}.pkl').read_bytes())
                if state['fit'] != fold['fit'] or state['heldout'] != fold['heldout']:
                    raise ValueError('Checkpoint participant ownership differs')
                inspect_model(state['estimator'], s.subset(x, fold['fit']), y[fold['fit']])
                p = s.predict_state(state, s.subset(x, fold['heldout']))
                np.testing.assert_allclose(p, selected['probabilities'], rtol=0, atol=1e-14)
                fit_accuracy = float((s.predict_state(state, s.subset(x, fold['fit'])).argmax(1) == y[fold['fit']]).mean())
                if fit_accuracy != selected['real_training_accuracy']:
                    raise ValueError('Recorded fitting accuracy differs')
                rows.append({'policy': policy, 'seed': fold['seed'], 'fold': fold['fold'],
                    'training_accuracy': fit_accuracy,
                    'inner_accuracy': record['options'][selected['selection']['index']]['accuracy'],
                    'outer_accuracy': s.score(y[fold['heldout']], p.argmax(1))['accuracy']})
                replayed += 1
        if s.training_choices(frame, records, old, settings) != json.loads((out/'final/training_selection.json').read_text()):
            raise ValueError('Final inner-only training selection differs')
        from .wavlm_layer_inference import frozen_model, predict_arrays
        for policy, record in plan['models'].items():
            checked, state, _ = frozen_model(policy)
            if checked != record or state['fit_participants'] != frame.participant_id.tolist():
                raise ValueError('Final model ownership or manifest differs')
            inspect_model(state['estimator'], x, y)
            np.testing.assert_allclose(predict_arrays(checked, state, x), s.predict_state(state, x), rtol=0, atol=1e-14)
            replayed += 1
    pd.DataFrame(rows).to_csv(out/'overfitting_audit.csv', index=False)
    feature_receipt = feature_ownership(frame, x, out)
    from .conditional_hierarchy_verify import error_analysis
    original_changes = error_analysis(out, frame, list(settings['policies']))
    changes = matched_comparisons(out, frame, list(settings['policies']), sorted({f['seed'] for f in folds}))
    summary = pd.read_csv(out/'summary.csv').set_index('model')
    primary, previous = float(summary.loc['accuracy', 'accuracy_mean']), float(summary.loc['previous_procedure', 'accuracy_mean'])
    intervals = pd.read_csv(out/'matched_paired_comparisons.csv')
    ci = intervals[(intervals.model == 'accuracy') & (intervals.reference == 'previous_procedure')
                   & (intervals.metric == 'accuracy')].iloc[0]
    atomic_json(out/'best_valid_model.json', {'previous_procedure_accuracy': previous, 'primary_accuracy': primary,
        'primary_observed_gain': primary > previous, 'primary_conditional_ci_excludes_zero_above': bool(ci.ci95_low > 0),
        'candidate_models': plan['models'], 'original_overwritten': False, 'independent_confirmation': False,
        'target_reached': primary >= settings['target_accuracy'], 'validation_test_used_for_selection': False})
    ensure_previous_files(previous_files)
    receipt = {'passed': True, 'checkpoints_replayed': replayed, 'zero_fit_zero_rewrite_resume': True,
        'previous_files_and_mtimes_preserved': len(previous_files),
        'new_nested_pipeline_fits': completion['total_fits'], 'new_full_training_pipeline_fits': plan['new_full_training_fits'],
        'old_models_refitted': 0, 'bootstrap_repetitions': 5000, 'bootstrap_seed': 20261025,
        'bootstrap_unit': 'Participant with all repeated predictions, not independent recording chunks or repeats',
        'bootstrap_includes_retraining_uncertainty': False, 'independent_confirmation': False,
        'validation_test_scored': False, 'feature_ownership': feature_receipt,
        'error_changes': changes, 'original_baseline_error_changes': original_changes}
    atomic_json(out/'verification.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
