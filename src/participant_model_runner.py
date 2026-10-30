"""One fixed regularized model, typed inputs and a source-evidence training gate."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .final_evaluation import sha256
from .participant_model_data import (FeatureBatch, TrainingTargets, SCHEMA, build_store,
    read_store, verify_repetitions)
from .raw_frame_uncertainty import participant_counts, weighted_metrics

ROOT = Path('results/participant_model_v1')
CONFIG = Path('configs/participant_speech_comparison.json')
PARAMETER_KEYS = {'base_mean', 'base_scale', 'pca_mean', 'pca_components', 'language_imputer',
    'language_mean', 'language_scale', 'language_weight', 'coef', 'intercept'}


@dataclass(frozen=True)
class ModelSpec:
    components: int = 32
    C: float = .001
    language_scale: float = .5
    class_weight_power: float = .5
    fit_seed: int = 42
    max_iterations: int = 3000

    def __post_init__(self):
        if (type(self.components) is not int or not 1 <= self.components <= 32
                or not np.isfinite(self.C) or self.C <= 0
                or not np.isfinite(self.language_scale) or self.language_scale <= 0
                or self.class_weight_power != .5 or self.max_iterations != 3000):
            raise ValueError('Invalid declared model capacity or regularization')


def validate_parameters(parameters):
    if set(parameters) != PARAMETER_KEYS:
        raise ValueError('Unexpected numeric model parameters')
    components = parameters['pca_components'].shape
    if len(components) != 2 or components[1] != 512 or not 1 <= components[0] <= 32:
        raise ValueError('Invalid fitting projection dimensions')
    shapes = {'base_mean': (512,), 'base_scale': (512,), 'pca_mean': (512,),
        'language_imputer': (16,), 'language_mean': (16,), 'language_scale': (16,),
        'language_weight': (), 'coef': (5, components[0]+16), 'intercept': (5,)}
    if any(parameters[k].shape != shape for k, shape in shapes.items()) or any(not np.isfinite(p).all() for p in parameters.values()):
        raise ValueError('Invalid finite numeric inference parameter schema')
    if (parameters['base_scale'] <= 0).any() or (parameters['language_scale'] <= 0).any() or parameters['language_weight'] <= 0:
        raise ValueError('Invalid model scaling')


def predict_parameters(parameters, features):
    if not isinstance(features, FeatureBatch):
        raise TypeError('Inference accepts only a target-free FeatureBatch')
    validate_parameters(parameters)
    base = (features.semantic_av-parameters['base_mean'])/parameters['base_scale']
    base = (base-parameters['pca_mean'])@parameters['pca_components'].T
    language = parameters['language_weight']*(features.language-parameters['language_mean'])/parameters['language_scale']
    return softmax(np.column_stack([base, language])@parameters['coef'].T+parameters['intercept'], axis=1)


class RegularizedParticipantModel:
    """Frozen input embeddings, fitting-only transforms and one small convex head."""
    def __init__(self, spec=ModelSpec()):
        self.spec = spec

    def fit(self, features, targets):
        if not isinstance(features, FeatureBatch) or not isinstance(targets, TrainingTargets):
            raise TypeError('Explicit separated feature and training-target objects required')
        if features.participant_ids != targets.participant_ids:
            raise ValueError('Fitting feature and target IDs must match in exact order')
        y = targets.severity_class
        if set(y) != set(range(5)) or len(y) <= self.spec.components:
            raise ValueError('Five real fitting classes and sufficient fitting participants required')
        self.fit_ids = features.participant_ids; self.counts = np.bincount(y, minlength=5)
        self.base_scaler = StandardScaler().fit(features.semantic_av)
        base = self.base_scaler.transform(features.semantic_av)
        self.projection = PCA(n_components=self.spec.components, svd_solver='full').fit(base)
        self.language_scaler = StandardScaler().fit(features.language)
        self.weights = (len(y)/(5*self.counts[y]))**self.spec.class_weight_power
        self.weights *= len(y)/self.weights.sum()
        z = np.column_stack([self.projection.transform(base), self.spec.language_scale*self.language_scaler.transform(features.language)])
        self.estimator = LogisticRegression(C=self.spec.C, solver='lbfgs', max_iter=self.spec.max_iterations, random_state=self.spec.fit_seed)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.estimator.fit(z, y, sample_weight=self.weights)
        self.parameters = {'base_mean': self.base_scaler.mean_.copy(), 'base_scale': self.base_scaler.scale_.copy(),
            'pca_mean': self.projection.mean_.copy(), 'pca_components': self.projection.components_.copy(),
            'language_imputer': np.median(features.language, axis=0),
            'language_mean': self.language_scaler.mean_.copy(), 'language_scale': self.language_scaler.scale_.copy(),
            'language_weight': np.asarray(self.spec.language_scale), 'coef': self.estimator.coef_.copy(),
            'intercept': self.estimator.intercept_.copy()}
        np.testing.assert_allclose(predict_parameters(self.parameters, features), self.estimator.predict_proba(z), rtol=0, atol=1e-14)
        return self

    def predict_proba(self, features):
        return predict_parameters(self.parameters, features)

    def diagnostics(self, features, targets):
        if features.participant_ids != self.fit_ids or targets.participant_ids != self.fit_ids:
            raise ValueError('Fitting diagnostics require the actual fitting cohort')
        p = self.predict_proba(features)
        return {'fit_participants': len(self.fit_ids), 'class_supports': self.counts.tolist(),
            'classifier_dimensions': int(self.estimator.n_features_in_),
            'parameter_count': int(self.estimator.coef_.size+self.estimator.intercept_.size),
            'coefficient_l2_norm': float(np.linalg.norm(self.estimator.coef_)),
            'iterations': self.estimator.n_iter_.tolist(), 'converged': bool((self.estimator.n_iter_ < self.spec.max_iterations).all()),
            'fitting_accuracy': float((p.argmax(1) == targets.severity_class).mean()),
            'fitting_log_loss': float(log_loss(targets.severity_class, p, labels=range(5))), 'spec': asdict(self.spec)}


def save_model(root, parameters, metadata):
    root = Path(root); validate_parameters(parameters)
    if root.exists():
        raise FileExistsError('Preserving existing numeric model')
    root.mkdir(parents=True)
    np.savez_compressed(root/'weights.npz', **parameters)
    contract = {'format': 'typed-participant-numeric-v1', 'feature_schema': SCHEMA,
        'weights_sha256': sha256(root/'weights.npz'), 'metadata': metadata,
        'implementation_sha256': sha256(__file__), 'data_implementation_sha256': sha256('src/participant_model_data.py')}
    atomic_json(root/'model.json', {'contract': contract, 'contract_hash': digest(contract)})
    return {'path': str(root), 'model_json_sha256': sha256(root/'model.json'), 'weights_sha256': contract['weights_sha256']}


def load_model(root):
    root = Path(root); envelope = json.loads((root/'model.json').read_text()); contract = envelope['contract']
    if (digest(contract) != envelope['contract_hash'] or contract['feature_schema'] != SCHEMA
            or contract['format'] != 'typed-participant-numeric-v1'
            or sha256(root/'weights.npz') != contract['weights_sha256']
            or sha256(__file__) != contract['implementation_sha256']
            or sha256('src/participant_model_data.py') != contract['data_implementation_sha256']):
        raise ValueError('Numeric model or extraction schema binding changed')
    with np.load(root/'weights.npz', allow_pickle=False) as archive:
        parameters = {k: archive[k].copy() for k in archive.files}
    validate_parameters(parameters)
    return parameters, contract


def export_retained(store, output):
    from .best_observed_inference import load, predict_features
    features, _, _, _ = read_store(store); contract, parameters = load()
    if not np.array_equal(np.asarray(contract['classes']), np.arange(5)):
        raise ValueError('Retained class mapping differs')
    expected = predict_features(parameters, features.semantic_av[:, :384], features.semantic_av[:, 384:], features.language)
    record = save_model(output, parameters, {'kind': 'retained_original_checkpoint_export',
        'source_model': contract['source_model'], 'observed_cv_procedure_accuracy': contract['observed_nested_procedure_accuracy'],
        'independent_confirmation': False, 'new_fits': 0,
        'input_contract': 'Original whole-interview feature pipeline; this retained export does not claim verified participant-only inputs'})
    loaded, _ = load_model(output); actual = predict_parameters(loaded, features)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    receipt = {'passed': True, 'original_participants': len(features.participant_ids), 'new_fits': 0,
        'maximum_probability_difference': float(np.abs(actual-expected).max()),
        'all_decisions_match': bool(np.array_equal(actual.argmax(1), expected.argmax(1))),
        'validation_test_scored': False, 'new_accuracy_claim': False, 'artifact': record}
    atomic_json(Path(output)/'verification.json', receipt)
    return receipt


def evaluate_fold(features, targets, fold, spec=ModelSpec()):
    """The only fitting path: one fixed candidate with strictly local transforms."""
    inner = []
    for part in fold.inner:
        train, labels = features.select(part.fit_ids), targets.select(part.fit_ids)
        model = RegularizedParticipantModel(spec).fit(train, labels)
        p = model.predict_proba(features.select(part.heldout_ids))
        inner.append({'fit_ids': list(part.fit_ids), 'heldout_ids': list(part.heldout_ids),
            'heldout_accuracy': float((p.argmax(1) == targets.select(part.heldout_ids).severity_class).mean()),
            **model.diagnostics(train, labels)})
    train, labels = features.select(fold.fit_ids), targets.select(fold.fit_ids)
    model = RegularizedParticipantModel(spec).fit(train, labels)
    p = model.predict_proba(features.select(fold.heldout_ids))
    return model, {'fold': fold.as_dict(), 'inner_diagnostics': inner,
        'outer_fitting_diagnostics': model.diagnostics(train, labels), 'outer_probabilities': p.tolist(),
        'new_fits': len(fold.inner)+1, 'fixed_configuration': True, 'outer_results_used_for_tuning': False}


def summarize_candidate(features, targets, folds, records, reference_file, out, promotion):
    import pandas as pd
    ids = features.participant_ids; lookup = {pid: i for i, pid in enumerate(ids)}
    seeds = verify_repetitions(folds, ids); candidate = []; probabilities = []
    for seed in seeds:
        p = np.full((len(ids), 5), np.nan); owners = np.zeros(len(ids), int)
        for row in records:
            if row['fold']['seed'] != seed:
                continue
            positions = [lookup[pid] for pid in row['fold']['heldout_ids']]
            p[positions] = row['outer_probabilities']; owners[positions] += 1
        if not (owners == 1).all() or not np.isfinite(p).all():
            raise ValueError('Incomplete original-participant out-of-fold predictions')
        candidate.append(p.argmax(1)); probabilities.append(p)
    predictions = {'candidate': np.stack(candidate)}
    with np.load(reference_file, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['participant_id'], ids); np.testing.assert_array_equal(archive['seeds'], seeds)
        for key in ['original', 'best_procedure']:
            predictions[key] = archive[key].argmax(2)
    y = targets.severity_class; counts = participant_counts(len(ids), 5000, 20261025)
    point = {key: weighted_metrics(y, p, np.ones((1, len(ids)), int))[0] for key, p in predictions.items()}
    boot = {key: weighted_metrics(y, p, counts) for key, p in predictions.items()}
    metrics, classes, matrices, changes, intervals = [], [], {}, [], []
    for key, prediction in predictions.items():
        matrix = confusion_matrix(np.tile(y, len(seeds)), prediction.reshape(-1), labels=range(5)); matrices[key] = matrix.tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(np.tile(y, len(seeds)), prediction.reshape(-1), labels=range(5), zero_division=0)
        metrics.append({'model': key, 'accuracy': float(point[key][0]), 'macro_f1': float(point[key][1]),
            'balanced_accuracy': float(point[key][2]), 'correct_repeated_predictions': int((prediction == y).sum()),
            'majority_predictions': int((prediction == np.bincount(y).argmax()).sum())})
        classes.extend({'model': key, 'class': c, 'unique_people': int((y == c).sum()),
            'precision': float(precision[c]), 'recall': float(recall[c]), 'f1': float(f1[c])} for c in range(5))
    for reference in ['original', 'best_procedure']:
        a, b = predictions['candidate'] == y, predictions[reference] == y
        gained, lost = a & ~b, b & ~a
        changes.append({'reference': reference, 'gained': int(gained.sum()), 'lost': int(lost.sum()),
            'unique_people_with_gain': int(gained.any(0).sum()), 'unique_people_with_loss': int(lost.any(0).sum()),
            'net_better_people': int((a.sum(0) > b.sum(0)).sum()), 'net_worse_people': int((a.sum(0) < b.sum(0)).sum())})
        delta = boot['candidate']-boot[reference]; limits = np.percentile(delta, [2.5, 97.5], axis=0)
        intervals.extend({'reference': reference, 'metric': metric, 'difference': float(point['candidate'][i]-point[reference][i]),
            'ci95_low': float(limits[0, i]), 'ci95_high': float(limits[1, i])} for i, metric in enumerate(['accuracy', 'macro_f1', 'balanced_accuracy']))
    accuracy_ci = next(row for row in intervals if row['reference'] == 'best_procedure' and row['metric'] == 'accuracy')
    severe = next(row['recall'] for row in classes if row['model'] == 'candidate' and row['class'] == 4)
    eligible = bool(point['candidate'][0] > promotion['accuracy_strictly_above'] and accuracy_ci['ci95_low'] > 0
        and point['candidate'][1] >= point['best_procedure'][1] and point['candidate'][2] >= point['best_procedure'][2] and severe > 0)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False); pd.DataFrame(classes).to_csv(out/'per_class.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_uncertainty.csv', index=False); pd.DataFrame(changes).to_csv(out/'gains_losses.csv', index=False)
    atomic_json(out/'confusion_matrices.json', matrices)
    np.savez_compressed(out/'oof_predictions.npz', participant_id=np.asarray(ids), seeds=np.asarray(seeds),
        candidate_probabilities=np.stack(probabilities), **predictions)
    return {'metrics': metrics, 'paired_intervals': intervals, 'gains_losses': changes,
        'exploratory_statistical_criteria_met': eligible, 'default_replaced': False,
        'bootstrap_unit': 'participant with all repeated predictions', 'bootstrap_repetitions': 5000,
        'independent_confirmation': False, 'promotion_pending': 'Raw verified-input inference parity and independent confirmation are still required'}


def train(store, output, resume=False):
    store, output = Path(store), Path(output)
    features, targets, folds, manifest = read_store(store)
    settings = json.loads(CONFIG.read_text()); spec = ModelSpec()
    if (settings['candidate_count'] != 1 or settings['max_inner_fits'] != 45 or settings['max_outer_fits'] != 15
            or settings['max_full_training_fits'] != 1 or settings['candidate']['C'] != spec.C
            or settings['candidate']['block_scale'] != spec.language_scale
            or settings['candidate']['weight_power'] != spec.class_weight_power
            or settings['candidate']['base_pca_components'] != spec.components):
        raise ValueError('The one-candidate fixed model budget changed')
    context = {'store_manifest_sha256': sha256(store/'manifest.json'), 'configuration_sha256': sha256(CONFIG),
        'implementation_sha256': sha256(__file__), 'data_implementation_sha256': sha256('src/participant_model_data.py'),
        'spec': asdict(spec), 'folds': [f.as_dict() for f in folds], 'training_allowed': manifest['training_allowed']}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}
    if output.exists():
        if not resume:
            raise FileExistsError('Preserving previous run; use --resume')
        if json.loads((output/'protocol.json').read_text()) != envelope:
            raise ValueError('Training context changed')
        if (output/'completion.json').exists():
            receipt = json.loads((output/'completion.json').read_text())
            for path, expected in receipt['artifacts'].items():
                if sha256(path) != expected:
                    raise ValueError('Completed run artifact changed')
            return {**receipt, 'new_fits_this_call': 0, 'rewritten_files': 0}
    elif resume:
        raise FileNotFoundError('No run to resume')
    else:
        output.mkdir(parents=True); atomic_json(output/'protocol.json', envelope)
    if not manifest['training_allowed']:
        receipt = {'status': 'blocked_verified_input_gate', 'complete': True, 'new_fits_this_call': 0,
            'total_real_data_fits': 0, 'new_accuracy_claim': False, 'default_replaced': False,
            'blockers': manifest['training_blockers'], 'artifacts': {str(output/'protocol.json'): sha256(output/'protocol.json')}}
        atomic_json(output/'completion.json', receipt)
        return receipt
    if len(folds) != 15 or sum(len(f.inner) for f in folds) != 45 or len(features.participant_ids) != 163:
        raise ValueError('Original bounded participant protocol changed')
    records = []; new_fits = 0
    with output_lock(output), threadpool_limits(limits=2):
        for fold in folds:
            directory = output/'jobs'/f'seed{fold.seed}_fold{fold.fold}'; result_path = directory/'result.json'
            if result_path.exists():
                record = load_record(result_path, ch)
                if record['fold'] != fold.as_dict():
                    raise ValueError('Cached fitting ownership changed')
                parameters, _ = load_model(directory/'model')
                np.testing.assert_allclose(predict_parameters(parameters, features.select(fold.heldout_ids)), record['outer_probabilities'], rtol=0, atol=1e-14)
            else:
                if directory.exists():
                    raise ValueError('Incomplete job artifacts are preserved; automatic refitting is forbidden')
                model, record = evaluate_fold(features, targets, fold, spec); directory.mkdir(parents=True)
                save_model(directory/'model', model.parameters, {'fit_ids': list(fold.fit_ids), 'heldout_ids': list(fold.heldout_ids), 'spec': asdict(spec)})
                parameters, _ = load_model(directory/'model')
                np.testing.assert_allclose(predict_parameters(parameters, features.select(fold.heldout_ids)), record['outer_probabilities'], rtol=0, atol=1e-14)
                save_record(result_path, record, ch); new_fits += record['new_fits']
            records.append(record)
        summary = summarize_candidate(features, targets, folds, records, store/'reference_predictions.npz', output, settings['promotion'])
        final = output/'candidate'
        if not final.exists():
            model = RegularizedParticipantModel(spec).fit(features, targets)
            save_model(final, model.parameters, {'fit_ids': list(features.participant_ids), 'spec': asdict(spec),
                'status': 'exploratory_candidate', 'required_input': 'Verified participant-only transcript features with the recorded source contract and fallback'})
            atomic_json(final/'fitting_diagnostics.json', model.diagnostics(features, targets)); new_fits += 1
        elif not (final/'fitting_diagnostics.json').is_file():
            raise ValueError('Incomplete full-fit artifact preserved; no automatic refit')
        parameters, _ = load_model(final); p = predict_parameters(parameters, features)
        np.savez_compressed(output/'candidate_fitting_predictions.npz', participant_id=np.asarray(features.participant_ids), probabilities=p)
        receipt = {'status': 'completed_exploratory_comparison', 'complete': True, 'new_fits_this_call': new_fits,
            'total_real_data_fits': 61, 'new_accuracy_claim': True, **summary,
            'validation_test_scored': False, 'artifacts': {str(p): sha256(p) for p in output.rglob('*') if p.is_file() and p.name != '.batch.lock'}}
        atomic_json(output/'completion.json', receipt)
    return receipt


def predict(model_root, feature_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserving existing predictions')
    features = FeatureBatch.load(feature_path); parameters, contract = load_model(model_root)
    probabilities = predict_parameters(parameters, features)
    atomic_json(output, {'participant_ids': list(features.participant_ids), 'class_order': SCHEMA['classes'],
        'decision_scores': probabilities.tolist(), 'predicted_class': probabilities.argmax(1).tolist(),
        'model_sha256': sha256(Path(model_root)/'model.json'), 'features_sha256': sha256(feature_path),
        'model_kind': contract['metadata'].get('kind', 'exploratory_candidate'),
        'questionnaire_inputs': False, 'new_accuracy_claim': False, 'scores_are_calibrated': False})
    return {'output': str(output), 'participants': len(features.participant_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'export-retained', 'train', 'predict'])
    parser.add_argument('--store', default=str(ROOT/'store')); parser.add_argument('--input-cache', default='results/participant_speech')
    parser.add_argument('--output'); parser.add_argument('--model'); parser.add_argument('--features'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = build_store(args.store, args.input_cache, args.resume)
    elif args.action == 'export-retained':
        result = export_retained(args.store, args.output or ROOT/'retained')
    elif args.action == 'train':
        result = train(args.store, args.output or ROOT/'controlled_run', args.resume)
    else:
        if not all([args.model, args.features, args.output]):
            parser.error('Prediction needs --model, --features and --output')
        result = predict(args.model, args.features, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
