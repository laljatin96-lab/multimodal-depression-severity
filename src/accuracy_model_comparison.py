"""Accuracy-first nested model comparison on official training participants only."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest, write_json
from .crossvalidated_baselines import fit
from .evaluate import metrics
from .exploratory_baselines import load_audited_sequences, pool_modalities
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .raw_frame_comparison import load_inputs as verified_raw_inputs, training_inputs
from .raw_frame_uncertainty import participant_counts, weighted_metrics, METRICS

PROTOCOL = 'post-final-accuracy-first-nested-svm-logistic-v1'
DIMENSIONS = {'cached_visual': 50, 'cached_audio_visual': 128, 'raw_audio_visual': 128}
FAMILIES = ['logistic_regression', 'rbf_svm']


def name_for(family, feature):
    return ('LR' if family == 'logistic_regression' else 'RBF-SVM')+'-'+feature


def label_metrics(y, predictions):
    predictions = np.asarray(predictions)
    if predictions.shape != np.asarray(y).shape or not np.isfinite(predictions).all() or not np.equal(predictions, predictions.astype(int)).all() or (predictions < 0).any() or (predictions > 4).any():
        raise ValueError('Expected one integer class 0–4 for every participant')
    # Adapt hard classes to the existing metric implementation only. These
    # one-hot arrays are never saved or described as model probabilities.
    return metrics(y, np.eye(5)[predictions.astype(int)])


def select_training_features(cached, raw, manifest):
    a, training = training_inputs(cached, manifest)
    b, same = training_inputs(raw, manifest)
    pd.testing.assert_frame_equal(training, same)
    np.testing.assert_array_equal(a[:, 78:], b[:, 78:])
    return {'cached_visual': a[:, 78:], 'cached_audio_visual': a, 'raw_audio_visual': b}, training


def load_inputs(settings):
    path = Path(settings['source_comparison_config'])
    source_settings = json.loads(path.read_text())
    raw, training, old_folds, _, coverage, bindings, source_context = verified_raw_inputs(source_settings)
    sequences, _ = load_audited_sequences(source_context['feature_config'], coverage)
    cached = np.stack([pool_modalities(sequences[pid]) for pid in coverage.participant_id])
    cached = cached[coverage.split.eq('train').to_numpy()]
    np.testing.assert_array_equal(cached[:, 78:], raw[:, 78:])
    expected_ids = coverage.loc[coverage.split.eq('train'), 'participant_id'].tolist()
    if expected_ids != training.participant_id.tolist() or nested_folds(training, settings) != old_folds:
        raise ValueError('Training participant order or folds differ from the existing study')
    matrices = {'cached_visual': cached[:, 78:], 'cached_audio_visual': cached, 'raw_audio_visual': raw}
    bindings[str(path)] = sha256(path)
    provenance = {'input_bindings': bindings, 'source_context_sha256': digest(source_context),
                  'coverage': coverage.to_dict('records'), 'sources': len(source_context['sources']),
                  'archives': sum(s['kind'] == 'archive' for s in source_context['sources']),
                  'folders': sum(s['kind'] == 'directory' for s in source_context['sources'])}
    return matrices, training, provenance, source_context['feature_config']['dataset']


def validate_settings(settings):
    if not settings['feature_inputs'] or len(set(settings['feature_inputs'])) != len(settings['feature_inputs']) or not set(settings['feature_inputs']) <= DIMENSIONS.keys():
        raise ValueError('Unknown or repeated feature input')
    if set(settings['models']) != set(FAMILIES):
        raise ValueError('Both matched logistic and RBF-SVM families are required')
    for family, options in settings['models'].items():
        grid = options['C_candidates']
        if not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all() or min(grid) <= 0 or options['class_weight'] is not None:
            raise ValueError('Use unique positive finite C values and ordinary class weights')
        if options['max_iter'] <= 0:
            raise ValueError('Fit iteration bounds must be positive')
    svm = settings['models']['rbf_svm']
    if svm['probability'] or svm['gamma'] != 'scale' or not svm['break_ties']:
        raise ValueError('Declared SVM uses scale gamma, explicit OVR tie handling and no probability fitting')
    if len(settings['models'][FAMILIES[0]]['C_candidates']) != len(settings['models'][FAMILIES[1]]['C_candidates']):
        raise ValueError('Matched families require equal candidate budgets')
    if not 1 <= settings['budget']['cpu_threads'] <= 2:
        raise ValueError('Use one or two CPU threads')


def make_estimator(family, c, settings):
    options = settings['models'][family]
    if family == 'logistic_regression':
        estimator = LogisticRegression(C=c, solver='lbfgs', max_iter=options['max_iter'], class_weight=None,
                                       random_state=settings['fit_seed'])
    elif family == 'rbf_svm':
        # The default disables probability fitting. Passing probability=False
        # explicitly is deprecated in sklearn 1.9; verify the fitted state below.
        estimator = SVC(C=c, kernel='rbf', gamma='scale', class_weight=None,
                        decision_function_shape='ovr', break_ties=True, max_iter=options['max_iter'],
                        cache_size=128, random_state=settings['fit_seed'])
    else:
        raise ValueError('Unknown estimator family')
    return make_pipeline(StandardScaler(), estimator)


def fit_checked(x, y, family, c, settings):
    estimator = make_estimator(family, c, settings)
    fit(estimator, x, y)  # Convergence warnings are errors, never silently accepted.
    np.testing.assert_array_equal(estimator.classes_, np.arange(5))
    if family == 'rbf_svm' and (estimator.named_steps['svc'].probA_.size or estimator.named_steps['svc'].probB_.size):
        raise ValueError('Unexpected probability calibration in SVM fit')
    scaler = estimator.named_steps['standardscaler']
    np.testing.assert_allclose(scaler.mean_, x.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, x.var(0), rtol=1e-12, atol=1e-12)
    return estimator


def decisions(estimator, x):
    predicted = estimator.predict(x).astype(int)
    scores = np.asarray(estimator.decision_function(x), dtype=float)
    if scores.shape != (len(x), 5) or not np.isfinite(scores).all():
        raise ValueError('Expected finite five-class decision scores')
    np.testing.assert_array_equal(predicted, scores.argmax(1))
    return predicted, scores


def choose_candidate(candidates):
    return max(candidates, key=lambda r: (r['inner_correct'], r['inner_macro_f1'], -r['C']))


def run_job(x, y, fold, family, feature, settings, directory):
    started = time.monotonic()
    outer_fit, heldout = np.asarray(fold['fit']), np.asarray(fold['heldout'])
    candidates, checks = [], []
    for c in sorted(settings['models'][family]['C_candidates']):
        inner_predictions = np.full(len(y), -1, dtype=int)
        for i, partition in enumerate(fold['inner']):
            rows, validation = np.asarray(partition['fit']), np.asarray(partition['heldout'])
            estimator = fit_checked(x[rows], y[rows], family, c, settings)
            inner_predictions[validation], _ = decisions(estimator, x[validation])
            checks.append({'C': c, 'inner_fold': i, 'fit_count': len(rows),
                           'fit_class_counts': np.bincount(y[rows], minlength=5).tolist(), 'scaler_fit_rows_verified': True})
        predicted = inner_predictions[outer_fit]
        report = label_metrics(y[outer_fit], predicted)
        candidates.append({'C': c, 'inner_predictions': predicted.tolist(), 'inner_correct': int((predicted == y[outer_fit]).sum()),
                           'inner_accuracy': report['accuracy'], 'inner_macro_f1': report['macro_f1']})
    selected = choose_candidate(candidates)
    estimator = fit_checked(x[outer_fit], y[outer_fit], family, selected['C'], settings)
    predicted, scores = decisions(estimator, x[heldout])
    path = Path(directory)/'model.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pkl.tmp')
    state = {'estimator': estimator, 'purpose': PROTOCOL, 'family': family, 'feature_input': feature,
             'feature_count': x.shape[1], 'selected_C': selected['C'], 'fit': fold['fit'], 'heldout': fold['heldout'],
             'selection': settings['selection'], 'scores_are_probabilities': False}
    with temporary.open('wb') as handle:
        pickle.dump(state, handle)
    temporary.replace(path)
    with path.open('rb') as handle:
        restored = pickle.load(handle)  # Only the model just produced locally.
    replay, replay_scores = decisions(restored['estimator'], x[heldout])
    np.testing.assert_array_equal(predicted, replay)
    np.testing.assert_array_equal(scores, replay_scores)
    return {'model': name_for(family, feature), 'family': family, 'feature_input': feature,
            'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
            'selected_C': selected['C'], 'inner_candidates': candidates, 'preprocessing_checks': checks,
            'outer_predictions': predicted.tolist(), 'outer_decision_scores': scores.tolist(),
            'decision_score_kind': 'logit' if family == 'logistic_regression' else 'uncalibrated_ovr_score',
            'fits': len(candidates)*len(fold['inner'])+1, 'seconds': time.monotonic()-started,
            'model_sha256': sha256(path), 'outer_scaler_verified': True, 'checkpoint_prediction_parity': True}


def validate_job(record, fold, family, feature, settings, directory, y):
    expected = {'model': name_for(family, feature), 'family': family, 'feature_input': feature,
                'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
                'fits': len(settings['models'][family]['C_candidates'])*len(fold['inner'])+1,
                'decision_score_kind': 'logit' if family == 'logistic_regression' else 'uncalibrated_ovr_score',
                'outer_scaler_verified': True, 'checkpoint_prediction_parity': True}
    if any(record.get(k) != v for k, v in expected.items()) or sha256(Path(directory)/'model.pkl') != record['model_sha256']:
        raise ValueError('Saved job identity/checkpoint changed')
    candidates = record['inner_candidates']
    if [c['C'] for c in candidates] != sorted(settings['models'][family]['C_candidates']):
        raise ValueError('Saved candidate grid changed')
    for candidate in candidates:
        predictions = np.asarray(candidate['inner_predictions'])
        report = label_metrics(y[fold['fit']], predictions)
        if (candidate['inner_correct'] != int((predictions == y[fold['fit']]).sum())
                or abs(candidate['inner_accuracy']-report['accuracy']) > 1e-12
                or abs(candidate['inner_macro_f1']-report['macro_f1']) > 1e-12):
            raise ValueError('Saved inner scores disagree with predictions')
    if record['selected_C'] != choose_candidate(candidates)['C']:
        raise ValueError('Saved selection violates accuracy-first rule')
    expected_checks = [{'C': c, 'inner_fold': i, 'fit_count': len(p['fit']),
                        'fit_class_counts': np.bincount(y[p['fit']], minlength=5).tolist(), 'scaler_fit_rows_verified': True}
                       for c in sorted(settings['models'][family]['C_candidates']) for i, p in enumerate(fold['inner'])]
    if record['preprocessing_checks'] != expected_checks:
        raise ValueError('Saved preprocessing ownership/class counts changed')
    scores = np.asarray(record['outer_decision_scores'])
    if scores.shape != (len(fold['heldout']), 5) or not np.isfinite(scores).all():
        raise ValueError('Invalid saved decision scores')
    label_metrics(y[fold['heldout']], record['outer_predictions'])
    np.testing.assert_array_equal(record['outer_predictions'], scores.argmax(1))


def write_prediction_report(training, predictions, scores, family, feature, seed, out):
    out.mkdir(parents=True, exist_ok=True)
    model = 'majority' if family == 'majority' else name_for(family, feature)
    report = label_metrics(training.severity_class, predictions)
    report['group_metrics'] = {str(g): label_metrics(rows.severity_class, predictions[rows.index])
                               for g, rows in training.groupby('gender', dropna=False)}
    report.update({'model': model, 'seed': seed, 'split': 'train_nested_oof', 'protocol': PROTOCOL})
    frame = training.copy()
    frame['predicted_label'] = predictions
    frame['model'], frame['seed'], frame['split'] = model, seed, 'train_nested_oof'
    if scores is not None:
        for i in range(5):
            frame[f'decision_score_{i}'] = scores[:, i]
        frame['decision_score_kind'] = 'logit' if family == 'logistic_regression' else 'uncalibrated_ovr_score'
    frame.to_csv(out/'predictions.csv', index=False)
    write_json(out/'metrics.json', report)
    pd.DataFrame(report['per_class']).to_csv(out/'per_class.csv', index=False)
    pd.DataFrame(report['confusion_matrix']).to_csv(out/'confusion_matrix.csv', index=False)
    return report


def summarize(training, records, folds, settings, out, provenance):
    rows, by_model = [], {}
    definitions = [(f, x) for x in settings['feature_inputs'] for f in FAMILIES]+[('majority', '')]
    for family, feature in definitions:
        model = 'majority' if family == 'majority' else name_for(family, feature)
        repeats = []
        for seed in settings['fold_seeds']:
            predicted = np.full(len(training), -1, dtype=int)
            scores, ownership = np.full((len(training), 5), np.nan), np.zeros(len(training), int)
            if family == 'majority':
                for fold in [f for f in folds if f['seed'] == seed]:
                    predicted[fold['heldout']] = np.bincount(training.iloc[fold['fit']].severity_class, minlength=5).argmax()
                    ownership[fold['heldout']] += 1
            else:
                for record in records:
                    if record['model'] == model and record['seed'] == seed:
                        predicted[record['heldout']] = record['outer_predictions']
                        scores[record['heldout']] = record['outer_decision_scores']
                        ownership[record['heldout']] += 1
            if not (ownership == 1).all() or (family != 'majority' and not np.isfinite(scores).all()):
                raise ValueError('OOF predictions must cover each training participant exactly once per partition')
            report = write_prediction_report(training, predicted, None if family == 'majority' else scores, family, feature, seed,
                                             out/'oof'/f'{model}_seed{seed}')
            repeats.append(predicted)
            rows.append({'model': model, 'family': family, 'feature_input': feature, 'seed': seed,
                         **{k: report[k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']},
                         'predicted_class_counts': json.dumps(np.bincount(predicted, minlength=5).tolist()),
                         'severe_correct': int(((predicted == 4) & training.severity_class.eq(4).to_numpy()).sum())})
        by_model[model] = np.stack(repeats)
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'repeat_scores.csv', index=False)
    summary = scores.groupby(['model', 'family', 'feature_input'], sort=False, dropna=False)[METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    pd.DataFrame([{k: r[k] for k in ['model', 'family', 'feature_input', 'seed', 'fold', 'selected_C', 'fits', 'seconds']}
                  for r in records]).to_csv(out/'outer_selections.csv', index=False)
    counts = participant_counts(len(training), settings['bootstrap_repetitions'], settings['bootstrap_seed'])
    y = training.severity_class.to_numpy()
    bootstrap = {name: weighted_metrics(y, p, counts) for name, p in by_model.items()}
    points = {name: weighted_metrics(y, p, np.ones((1, len(y)), int))[0] for name, p in by_model.items()}
    pairs = [('majority', name) for name in by_model if name != 'majority']
    pairs += [(name_for('logistic_regression', f), name_for('rbf_svm', f)) for f in settings['feature_inputs']]
    comparisons = []
    for base, candidate in pairs:
        ci = np.percentile(bootstrap[candidate]-bootstrap[base], [2.5, 97.5], axis=0)
        for i, metric in enumerate(METRICS):
            comparisons.append({'baseline': base, 'candidate': candidate, 'metric': metric,
                                'mean_difference': points[candidate][i]-points[base][i], 'ci95_low': ci[0, i], 'ci95_high': ci[1, i]})
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv', index=False)
    plot_summary(summary, out/'comparison.png')
    lines = ['# Accuracy-first nonlinear model comparison', '',
             f"{len(training)} official training participants; {len(folds)} outer partitions. "
             'C selection prioritizes inner accuracy, then macro-F1, then smaller C. All scalers and models fit only their own training folds.', '',
             '| Model/input | Accuracy | Macro-F1 | Balanced accuracy |', '|---|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.model} | {row.accuracy_mean:.4f} ± {row.accuracy_std:.4f} | {row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f} | {row.balanced_accuracy_mean:.4f} ± {row.balanced_accuracy_std:.4f} |')
    lines += ['', 'All models use the same folds and equal candidate counts. SVM gamma is scale; no probability calibration is fitted. '
              'Saved SVM decision scores are uncalibrated ranking scores, not probabilities. The reported majority model is learned separately inside each outer training fold.', '',
              'Accuracy-first selection can favor the common class. Inspect per-class recall, severe_correct and predicted_class_counts alongside accuracy. '
              'Earlier studies selected C by macro-F1, so their scores are contextual rather than matched controls for this experiment.', '',
              'This is exploratory development after the original final test was seen. No validation/test scoring, full-training refit or reference-model replacement occurred. '
              'Partition SD is not independent replication uncertainty. Paired bootstrap intervals condition on saved fitted models, share participant draws across partitions, and are unadjusted for multiple comparisons.', '',
              f"Dataset coverage: {len(provenance.get('coverage', []))} participants and {provenance.get('sources', 'unspecified')} physical sources. "
              'Existing source data, caches, original models/results and experiment log are preserved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    labels = [name.replace('cached_audio_visual', 'cached AV').replace('raw_audio_visual', 'raw AV').replace('cached_visual', 'visual')
              for name in summary.model]
    for ax, metric in zip(axes, ['accuracy', 'macro_f1']):
        ax.barh(np.arange(len(summary)), summary[metric+'_mean'], xerr=summary[metric+'_std'].fillna(0), capsize=3,
                color=['#bb713d' if f == 'rbf_svm' else '#6e93a7' if f == 'logistic_regression' else '#999999' for f in summary.family])
        ax.set_yticks(np.arange(len(summary)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, min(1, max(.6, (summary[metric+'_mean']+summary[metric+'_std'].fillna(0)).max()+.05)))
        ax.set_xlabel('Accuracy' if metric == 'accuracy' else 'Five-class macro-F1')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Accuracy-first nested comparison: training participants only\nBars: partition mean; error bars: partition SD')
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run_experiment(matrices, training, settings, output, provenance=None, resume=False, max_new_jobs=None, verify_inputs=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing experiment; use --resume with unchanged inputs')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires an existing protocol')
    validate_settings(settings)
    if max_new_jobs is not None and max_new_jobs < 1:
        raise ValueError('Pilot job bound must be positive')
    training = training.reset_index(drop=True)
    y = training.severity_class.to_numpy()
    if training.participant_id.duplicated().any() or not np.isfinite(y).all() or not np.equal(y, y.astype(int)).all() or (y < 0).any() or (y > 4).any():
        raise ValueError('Invalid training participant IDs or classes')
    y = y.astype(int)
    for feature in settings['feature_inputs']:
        x = np.asarray(matrices[feature], dtype=float)
        if x.shape != (len(training), DIMENSIONS[feature]) or not np.isfinite(x).all():
            raise ValueError('Invalid training feature dimensions/values')
    folds = nested_folds(training, settings)
    jobs = [(fold, family, feature) for fold in folds for feature in settings['feature_inputs'] for family in FAMILIES]
    expected_fits = sum(len(settings['models'][family]['C_candidates'])*len(fold['inner'])+1 for fold, family, _ in jobs)
    if len(jobs) != settings['budget']['outer_jobs'] or expected_fits != settings['budget']['total_estimator_fits']:
        raise ValueError('Declared compute budget does not match the nested experiment')
    context = {'protocol': PROTOCOL, 'settings': settings, 'training_participants': training.to_dict('records'), 'folds': folds,
               'feature_hashes': {f: hashlib.sha256(np.asarray(matrices[f], dtype=np.float64).tobytes()).hexdigest() for f in settings['feature_inputs']},
               'provenance': provenance or {}, 'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
               'implementation_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                    ['accuracy_model_comparison.py', 'raw_frame_comparison.py', 'raw_frame_features.py', 'raw_frame_uncertainty.py',
                     'modality_diagnostics.py', 'exploratory_baselines.py', 'crossvalidated_baselines.py', 'batch_inference.py', 'evaluate.py', 'common.py']},
               'validation_test_scoring': False, 'final_refit': False}
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['budget']['cpu_threads']):
        envelope = {'context': context, 'context_hash': context_hash}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != envelope:
                raise ValueError('Experiment inputs/code/protocol changed; choose a new output')
        else:
            atomic_json(out/'protocol.json', envelope)
            atomic_json(out/'folds.json', folds)
            if provenance and provenance.get('coverage'):
                pd.DataFrame(provenance['coverage']).to_csv(out/'participant_coverage.csv', index=False)
        records, reused, new = [], 0, 0
        for fold, family, feature in jobs:
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/name_for(family, feature)
            path = directory/'result.json'
            if path.exists():
                record = load_record(path, context_hash)
                reused += 1
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                record = run_job(matrices[feature], y, fold, family, feature, settings, directory)
                save_record(path, record, context_hash)
                new += 1
            validate_job(record, fold, family, feature, settings, directory, y)
            records.append(record)
            if len(records) % 6 == 0 or len(records) == len(jobs):
                print(f"Accuracy comparison: {len(records)}/{len(jobs)} jobs; {new} new, {reused} reused", flush=True)
        if verify_inputs:
            verify_inputs()
        complete = len(records) == len(jobs)
        if complete:
            summarize(training, records, folds, settings, out, provenance or {})
        completion = {'complete': complete, 'context_hash': context_hash, 'training_participants': len(training),
                      'expected_outer_jobs': len(jobs), 'completed_outer_jobs': len(records), 'completed_fits': sum(r['fits'] for r in records),
                      'new_jobs_this_invocation': new, 'reused_jobs_this_invocation': reused,
                      'new_fits_this_invocation': new*(len(settings['models'][FAMILIES[0]]['C_candidates'])*settings['inner_folds']+1),
                      'fit_seconds_total': sum(r['seconds'] for r in records), 'validation_test_scoring': False,
                      'full_training_refit': False, 'frozen_reference_replaced': False}
        atomic_json(out/'completion.json', completion)
        return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/accuracy_model_comparison.json')
    parser.add_argument('--output', default='results/accuracy_model_comparison')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int, help='Bound a pilot; completed jobs are reusable with --resume')
    args = parser.parse_args()
    if Path(args.output).exists() and not args.resume:
        parser.error('Preserving existing output; use --resume for unchanged inputs')
    settings = json.loads(Path(args.config).read_text())
    matrices, training, provenance, dataset = load_inputs(settings)
    source_settings = json.loads(Path(settings['source_comparison_config']).read_text())
    if any(Path(args.output).resolve().is_relative_to(Path(p).resolve()) for p in
           [dataset, source_settings['raw_features'], source_settings['cached_diagnostic']]):
        parser.error('Output must be outside existing data/experiment directories')
    def verify_inputs():
        current, labels, checked, _ = load_inputs(settings)
        if checked != provenance or not labels.equals(training) or any(not np.array_equal(current[k], matrices[k]) for k in matrices):
            raise ValueError('Source data changed during the experiment')
    result = run_experiment(matrices, training, settings, args.output, provenance, args.resume, args.max_new_jobs, verify_inputs)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
