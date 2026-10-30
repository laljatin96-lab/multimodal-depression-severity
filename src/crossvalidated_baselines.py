"""Select logistic regularization within training; evaluate frozen choices on validation."""
import argparse
import json
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import digest, encode_gender, joined_manifest, load_config, write_json
from .evaluate import metrics, write_report
from .exploratory_baselines import load_summary_matrix
from .report_independent import paired_mean_bootstrap
from .train import record_result

PROTOCOL = 'exploratory-training-cv-logistic-v1'
VARIANTS = [(False, False), (False, True), (True, False), (True, True)]


def variant_name(gender, balanced):
    return 'CV-LR-AV' + ('-gender' if gender else '') + ('-balanced' if balanced else '')


def make_estimator(c, balanced, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, class_weight='balanced' if balanced else None, max_iter=2000,
        solver='lbfgs', random_state=seed))


def fit(estimator, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(x, y)


def five_probabilities(estimator, x):
    probabilities = np.zeros((len(x), 5))
    probabilities[:, estimator.classes_.astype(int)] = estimator.predict_proba(x)
    return probabilities


def training_folds(training, folds, seeds):
    counts = np.bincount(training.severity_class.to_numpy(), minlength=5)
    if folds < 2 or (counts < folds).any():
        raise ValueError('Each of the five training classes must support the requested folds')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Specify unique nonempty fold seeds')
    result = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (fit_indices, heldout) in enumerate(splitter.split(training, training.severity_class)):
            result.append((seed, fold, fit_indices, heldout))
    return result


def select_regularization(features, training, settings, folds, balanced, output):
    """Only training rows enter this function; OOF scores choose C."""
    out = Path(output)
    out.mkdir(parents=True)
    y = training.severity_class.to_numpy()
    grid = sorted(settings['C_candidates'])
    if not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all() or min(grid) <= 0:
        raise ValueError('C candidates must be finite, positive and unique')
    fold_scores, repeat_scores, scaler_checks, oof_records = [], [], [], []
    candidate_scores = []
    for c in grid:
        oof = {seed: np.full((len(training), 5), np.nan) for seed in settings['fold_seeds']}
        for seed, fold, fit_indices, heldout in folds:
            estimator = make_estimator(c, balanced, settings['fit_seed'])
            fit(estimator, features[fit_indices], y[fit_indices])
            scaler = estimator.named_steps['standardscaler']
            np.testing.assert_allclose(scaler.mean_, features[fit_indices].mean(0), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(scaler.var_, features[fit_indices].var(0), rtol=1e-12, atol=1e-12)
            probabilities = five_probabilities(estimator, features[heldout])
            oof[seed][heldout] = probabilities
            result = metrics(y[heldout], probabilities)
            fold_scores.append({'C': c, 'fold_seed': seed, 'fold': fold,
                                **{key: result[key] for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']}})
            scaler_checks.append({'C': c, 'fold_seed': seed, 'fold': fold,
                                  'fit_count': len(fit_indices), 'heldout_count': len(heldout),
                                  'scaler_mean_matches_fold_training': True, 'scaler_variance_matches_fold_training': True,
                                  'fit_class_counts': np.bincount(y[fit_indices], minlength=5).tolist()})
        for seed, probabilities in oof.items():
            if not np.isfinite(probabilities).all():
                raise ValueError('Incomplete OOF prediction coverage')
            result = metrics(y, probabilities)
            repeat_scores.append({'C': c, 'fold_seed': seed,
                                  **{key: result[key] for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']}})
            for index, row in training.iterrows():
                oof_records.append({'participant_id': row.participant_id, 'C': c, 'fold_seed': seed,
                                    'severity_class': int(row.severity_class), 'predicted_label': int(probabilities[index].argmax()),
                                    **{f'p{i}': float(probabilities[index, i]) for i in range(5)}})
        scores = [row['macro_f1'] for row in repeat_scores if row['C'] == c]
        candidate_scores.append({'C': c, 'mean_oof_macro_f1': float(np.mean(scores)),
                                 'sd_across_fold_partitions': float(np.std(scores, ddof=1)) if len(scores) > 1 else None})
    # Sorted grid plus first-max gives smaller C on exact ties.
    selected = max(candidate_scores, key=lambda row: row['mean_oof_macro_f1'])
    pd.DataFrame(fold_scores).to_csv(out/'fold_scores.csv', index=False)
    pd.DataFrame(repeat_scores).to_csv(out/'repeat_scores.csv', index=False)
    pd.DataFrame(candidate_scores).to_csv(out/'candidate_scores.csv', index=False)
    pd.DataFrame(oof_records).to_csv(out/'oof_predictions.csv', index=False)
    write_json(out/'fold_preprocessing_checks.json', scaler_checks)
    write_json(out/'selection.json', {**selected, 'selection_data': 'official training participants only',
                                    'rule': 'Maximum mean pooled OOF five-class macro-F1 across fold partitions; ties smaller C'})
    return selected


def run_experiment(matrix, manifest, feature_config, settings, output, source_provenance=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f'Preserving existing experiment: {out}')
    if len(matrix) != len(manifest) or not np.isfinite(matrix).all() or manifest.participant_id.duplicated().any():
        raise ValueError('Invalid participant summaries')
    if not set(manifest.split) <= {'train', 'validation', 'test'}:
        raise ValueError('Unexpected split')
    train_mask, validation_mask = manifest.split.eq('train').to_numpy(), manifest.split.eq('validation').to_numpy()
    training = manifest[train_mask].reset_index(drop=True)
    validation = manifest[validation_mask].reset_index(drop=True)
    if not len(validation):
        raise ValueError('Validation participants required')
    folds = training_folds(training, settings['folds'], settings['fold_seeds'])
    out.mkdir(parents=True)
    started = time.monotonic()
    protocol = {'protocol': PROTOCOL, 'settings': settings, 'feature_config': feature_config,
                'manifest_hash': digest(manifest.to_dict('records')), 'source_provenance': source_provenance,
                'variants': VARIANTS, 'test_evaluated': False,
                'features': 'Existing per-modality observed-bin mean and population standard deviation; optional original recorded gender',
                'selection': 'Training-only repeated stratified folds; mean pooled out-of-fold fixed-five-class macro-F1; tie smaller C',
                'final_fit': 'Each selected estimator refits all official training participants once, then evaluates validation',
                'limitations': ['Experiment was motivated by earlier validation results; this is not a fresh validation holdout',
                               'Selected OOF scores have selection optimism; not nested CV',
                               'Partition repeat SD is not independent training-seed uncertainty',
                               'Unverified timing origins, cached overlap crop and 0.5s bins, and unsegmented audio remain',
                               'Sparse validation classes and groups; no clinical or fairness claim']}
    write_json(out/'protocol.json', protocol)
    manifest[['participant_id', 'split']].to_csv(out/'participant_coverage.csv', index=False)
    fold_membership = []
    for seed, fold, fit_indices, heldout in folds:
        if set(fit_indices) & set(heldout) or len(fit_indices)+len(heldout) != len(training):
            raise ValueError('Training fold overlap/coverage error')
        fold_membership.extend({'fold_seed': seed, 'fold': fold, 'participant_id': training.iloc[i].participant_id,
                                'role': role} for role, indices in [('fit', fit_indices), ('heldout', heldout)] for i in indices)
    pd.DataFrame(fold_membership).to_csv(out/'fold_membership.csv', index=False)
    genders = np.stack([encode_gender(value, feature_config['gender_categories']) for value in manifest.gender])
    selected_models, selections = {}, {}
    # Complete every C choice before any validation prediction/score is made.
    for gender, balanced in VARIANTS:
        name = variant_name(gender, balanced)
        features = np.column_stack([matrix, genders]) if gender else matrix
        selection = select_regularization(features[train_mask], training, settings, folds, balanced, out/name/'cv')
        estimator = make_estimator(selection['C'], balanced, settings['fit_seed'])
        fit(estimator, features[train_mask], training.severity_class.to_numpy())
        np.testing.assert_allclose(estimator.named_steps['standardscaler'].mean_, features[train_mask].mean(0), rtol=1e-12, atol=1e-12)
        state = {'protocol': PROTOCOL, 'estimator': estimator, 'gender_input': gender, 'balanced': balanced,
                 'gender_categories': feature_config['gender_categories'], 'fit_participant_ids': training.participant_id.tolist(),
                 'selection': selection, 'provenance_hash': digest(protocol)}
        with (out/name/'model.pkl').open('wb') as handle:
            pickle.dump(state, handle)
        selected_models[name] = (estimator, features)
        selections[name] = selection
        print(f"{name}: selected C={selection['C']:g} using training OOF macro-F1={selection['mean_oof_macro_f1']:.4f}", flush=True)
    write_json(out/'frozen_selections.json', selections)
    rows, validation_predictions, verification = [], {}, []
    for name, (estimator, features) in selected_models.items():
        train_report = write_report(training, five_probabilities(estimator, features[train_mask]), out/name/'train', name, settings['fit_seed'], 'train')
        probabilities = five_probabilities(estimator, features[validation_mask])
        report = write_report(validation, probabilities, out/name/'validation', name, settings['fit_seed'], 'validation')
        with (out/name/'model.pkl').open('rb') as handle:
            restored = pickle.load(handle)
        np.testing.assert_array_equal(five_probabilities(restored['estimator'], features[validation_mask]), probabilities)
        verification.append({'model': name, 'reloaded_predictions_identical': True, 'final_scaler_matches_train': True})
        validation_predictions[name] = probabilities.argmax(1)[None, :]
        rows.append({'model': name, 'run': str(out/name), 'seed': settings['fit_seed'], **selections[name],
                     'train_macro_f1': train_report['macro_f1'],
                     **{key: report[key] for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']}})
        print(f"{name}: validation macro-F1={report['macro_f1']:.4f}, accuracy={report['accuracy']:.4f}", flush=True)
    selected = pd.DataFrame(rows)
    selected.to_csv(out/'validation_summary.csv', index=False)
    comparisons = {}
    for baseline, candidate in [('CV-LR-AV', 'CV-LR-AV-gender'), ('CV-LR-AV-balanced', 'CV-LR-AV-gender-balanced'),
                                ('CV-LR-AV', 'CV-LR-AV-balanced'), ('CV-LR-AV-gender', 'CV-LR-AV-gender-balanced')]:
        comparisons[candidate+' minus '+baseline] = paired_mean_bootstrap(
            validation.severity_class.to_numpy(), validation_predictions[baseline], validation_predictions[candidate],
            repetitions=settings['bootstrap_repetitions'], seed=settings['fit_seed'])
    write_json(out/'paired_validation_bootstrap.json', {
        'comparisons': comparisons, 'repetitions': settings['bootstrap_repetitions'],
        'limitations': 'Conditional on fitted models; participant bootstrap; prior validation exposure and model-selection optimism are not corrected',
    })
    write_json(out/'verification.json', {'final_models': verification,
                                        'fold_fits_verified': len(VARIANTS)*len(folds)*len(settings['C_candidates']),
                                        'test_evaluated': False})
    write_json(out/'completion.json', {'complete': True, 'total_participants': len(manifest), 'fit_participants': len(training),
                                       'validation_participants': len(validation), 'reserved_test_participants': int(manifest.split.eq('test').sum()),
                                       'final_fits': len(rows), 'cv_fits': len(VARIANTS)*len(folds)*len(settings['C_candidates']),
                                       'test_evaluated': False, 'seconds': time.monotonic()-started})
    write_summary(out, selected, comparisons, len(training), len(validation))
    return selected


def write_summary(out, selected, comparisons, train_count, val_count):
    lines = ['# Logistic baselines with training-only regularization selection', '',
             f'The same four AV/gender/class-weight variants use {train_count} official training participants. Regularization is selected with five-class out-of-fold macro-F1, then each frozen model is evaluated on {val_count} validation participants. Test participants remain reserved.', '',
             'See protocol.json for the fixed candidate grid, fold count/seeds, and motivation. All variants share the same folds and tuning budget. Scaling and balanced class weights are fitted inside each training fold. Repeats change fold partitions; they are not independent replications of the final deterministic logistic fit.', '',
             '| Model | Selected C | Mean training OOF macro-F1 | In-sample train macro-F1 | Validation macro-F1 | Validation accuracy |',
             '|---|---:|---:|---:|---:|---:|']
    for row in selected.itertuples():
        lines.append(f'| {row.model} | {row.C:g} | {row.mean_oof_macro_f1:.4f} | {row.train_macro_f1:.4f} | {row.macro_f1:.4f} | {row.accuracy:.4f} |')
    lines += ['', '## Matched validation comparisons', '', '| Comparison | Macro-F1 difference | Paired participant bootstrap 95% interval |', '|---|---:|---|']
    for name, report in comparisons.items():
        value = report['macro_f1']; lo, hi = value['paired_participant_ci95']
        lines.append(f"| {name} | {value['mean_seed_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '## Interpretation limits', '',
              'This experiment was motivated by previously observed validation behavior; moving C selection into training does not erase that prior exposure. Selected OOF scores are optimistic without an outer cross-validation loop. Bootstrap intervals condition on fitted models and do not correct selection. Do not treat fold-partition SD as independent seed uncertainty.', '',
              'The original cache crop, 0.5-second averaging, unverified timestamp origins, and possible interviewer speech remain. Sparse classes and recorded-gender subgroups limit conclusions. Predictions are research classifications of questionnaire-derived targets, not clinical diagnoses.', '',
              'Every supplied source remains represented by the original audit, with duplicate copies counted once per participant. No participant was excluded. Fold membership, all candidate OOF predictions/scores, preprocessing checks, frozen selections, saved models, validation subgroup/confusion reports, and reload checks are stored alongside this report.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/crossvalidated_baselines.json')
    parser.add_argument('--output', default='results/crossvalidated_baselines')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Choose a new output path to preserve previous results')
    settings = load_config(args.config)
    config = load_config(settings['feature_config'])
    manifest = joined_manifest(config)
    matrix, sources = load_summary_matrix(config, manifest)
    selected = run_experiment(matrix, manifest, config, settings, args.output, sources)
    for row in selected.itertuples():
        report = json.loads((Path(row.run)/'validation/metrics.json').read_text())
        record_result(row.run, {**report, 'model': 'EXPLORATORY-'+report['model']}, {**config, 'seed': settings['fit_seed']})


if __name__ == '__main__':
    main()
