"""Local, label-free inference with the frozen AV logistic reference.

Export reads a trusted project checkpoint once. Prediction loads only numeric
NPZ parameters plus JSON, and never reads questionnaire or gender metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import VERSION, digest, write_json
from .exploratory_baselines import pool_modalities
from .full_dataset import MAX_SELECTED_BYTES, SOURCE_PATTERN, read_source, source_stamp
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS, align

FORMAT = 'portable-av-logistic-v1'
SETTINGS = ['audio_kind', 'audio_time_offset', 'visual_time_offset', 'confidence_threshold', 'bin_seconds', 'minimum_window_steps']
LABELS = ['No significant depressive symptoms', 'Mild', 'Moderate', 'Moderately severe', 'Severe']
DEFAULT_BUNDLE = 'checkpoints/reference_av_portable'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_parameters(parameters):
    shapes = {'classes': (5,), 'mean': (128,), 'scale': (128,), 'coefficients': (5, 128), 'intercept': (5,)}
    if set(parameters) != set(shapes):
        raise ValueError('Unexpected portable model arrays')
    for name, shape in shapes.items():
        value = parameters[name]
        if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError(f'Invalid portable parameter: {name}')
    if not np.array_equal(parameters['classes'], np.arange(5)) or (parameters['scale'] <= 0).any():
        raise ValueError('Expected five ordered classes and positive scaler values')


def parameters_from_estimator(estimator):
    if list(estimator.named_steps) != ['standardscaler', 'logisticregression']:
        raise ValueError('Only the frozen standard-scaler/logistic pipeline is supported')
    scaler, classifier = estimator.named_steps.values()
    if not scaler.with_mean or not scaler.with_std or classifier.solver != 'lbfgs':
        raise ValueError('Unsupported scaler or multiclass solver')
    arrays = {'classes': np.asarray(classifier.classes_, dtype=np.int64),
              'mean': np.asarray(scaler.mean_, dtype=np.float64),
              'scale': np.asarray(scaler.scale_, dtype=np.float64),
              'coefficients': np.asarray(classifier.coef_, dtype=np.float64),
              'intercept': np.asarray(classifier.intercept_, dtype=np.float64)}
    validate_parameters(arrays)
    return arrays


def probabilities(parameters, summaries):
    validate_parameters(parameters)
    values = np.asarray(summaries, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 128 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite participant summaries with 128 features')
    with np.errstate(over='raise', invalid='raise'):
        logits = ((values-parameters['mean'])/parameters['scale']) @ parameters['coefficients'].T + parameters['intercept']
        exp = np.exp(logits-logits.max(axis=1, keepdims=True))
        result = exp/exp.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite portable predictions')
    return result


def validate_sequence(sequence, preprocessing):
    expected = {'audio', 'visual', 'audio_mask', 'visual_mask', 'timestamps'}
    if set(sequence) != expected:
        raise ValueError('Expected audio/visual arrays, observation masks and timestamps')
    times = sequence['timestamps']
    if times.ndim != 1 or times.dtype.kind not in 'fiu' or len(times) < preprocessing['minimum_window_steps']:
        raise ValueError('Invalid timestamp grid length/type')
    if not np.isfinite(times).all() or times[0] < 0 or not np.allclose(np.diff(times), preprocessing['bin_seconds'], rtol=1e-8, atol=1e-8):
        raise ValueError('Timestamp grid does not match the frozen preprocessing')
    for name, columns in [('audio', AUDIO_COLUMNS), ('visual', VISUAL_COLUMNS)]:
        values, mask = sequence[name], sequence[name+'_mask']
        if values.shape != (len(times), len(columns)) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name} feature dimensions/values')
        if mask.shape != (len(times),) or mask.dtype != bool or not mask.any():
            raise ValueError(f'Invalid or empty {name} observation mask')


def load_bundle(path):
    root = Path(path)
    envelope = json.loads((root/'model.json').read_text())
    contract = envelope['contract']
    if envelope['contract_sha256'] != digest(contract) or contract['format'] != FORMAT:
        raise ValueError('Portable model contract integrity mismatch')
    if contract['audio_columns'] != AUDIO_COLUMNS or contract['visual_columns'] != VISUAL_COLUMNS or contract['feature_version'] != VERSION:
        raise ValueError('Portable model feature schema mismatch')
    if contract['weights_sha256'] != sha256(root/'weights.npz'):
        raise ValueError('Portable model weights checksum mismatch')
    with np.load(root/'weights.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_parameters(arrays)
    return contract, arrays


def export_reference(plan_path, output):
    import pickle
    from .common import joined_manifest
    from .exploratory_baselines import load_audited_sequences
    from .final_evaluation import verify_plan
    out = Path(output)
    if out.exists():
        raise FileExistsError(f'Preserving existing portable model: {out}')
    plan = verify_plan(plan_path)
    reference = plan['reference_model_selected_before_test']
    candidates = [item for item in plan['models'] if item['model'] == reference]
    if len(candidates) != 1 or candidates[0]['kind'] != 'summary':
        raise ValueError('Expected one frozen summary reference model')
    item = candidates[0]
    with Path(item['artifact']).open('rb') as handle:
        state = pickle.load(handle)  # Verified trusted local export input only.
    if state['gender_input'] or state.get('gender_only'):
        raise ValueError('Label-free reference export requires AV-only input')
    config = plan['feature_config']
    manifest = joined_manifest(config)
    sequences, sources = load_audited_sequences(config, manifest)
    if digest(manifest.to_dict('records')) != plan['manifest_hash'] or sources != plan['source_provenance']:
        raise ValueError('Frozen reference dataset changed')
    if state['fit_participant_ids'] != manifest.loc[manifest.split.eq('train'), 'participant_id'].tolist():
        raise ValueError('Reference fit participant provenance mismatch')
    validation = manifest[manifest.split.eq('validation')].reset_index(drop=True)
    summaries = np.stack([pool_modalities(sequences[pid]) for pid in validation.participant_id])
    arrays = parameters_from_estimator(state['estimator'])
    portable = probabilities(arrays, summaries)
    original = state['estimator'].predict_proba(summaries)
    saved = pd.read_csv(item['validation_predictions'], dtype={'participant_id': str})
    if saved.participant_id.tolist() != validation.participant_id.tolist():
        raise ValueError('Validation participant order changed')
    np.testing.assert_allclose(portable, original, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(portable, saved[[f'p{i}' for i in range(5)]], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(portable.argmax(1), original.argmax(1))
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'weights.npz', **arrays)
    contract = {'format': FORMAT, 'model': reference, 'feature_version': VERSION,
                'audio_columns': AUDIO_COLUMNS, 'visual_columns': VISUAL_COLUMNS,
                'preprocessing': {key: config[key] for key in SETTINGS},
                'summary_order': ['audio_mean', 'audio_population_std', 'visual_mean', 'visual_population_std'],
                'summary_features': 128, 'class_labels': LABELS,
                'weights_sha256': sha256(out/'weights.npz'), 'source_model_sha256': sha256(item['artifact']),
                'frozen_plan_sha256': digest(plan), 'training_participants': len(state['fit_participant_ids']),
                'requires_questionnaire': False, 'requires_gender': False,
                'limitations': ['Exploratory questionnaire-class prediction; not a clinical diagnosis',
                               'Probabilities are not calibrated clinical certainty',
                               'Requires supplied MFCC/OpenFace features; no WAV/video feature extractor is included',
                               'Original common-interval crop and 0.5-second averaging retained',
                               'Timestamp origins and participant-only speech remain unverified',
                               'Original final test already evaluated; this export provides no new generalization evidence']}
    write_json(out/'model.json', {'contract': contract, 'contract_sha256': digest(contract)})
    _, restored = load_bundle(out)
    np.testing.assert_array_equal(probabilities(restored, summaries), portable)
    write_json(out/'verification.json', {'validation_participants': len(validation),
                                        'all_audited_participants_accounted_for': len(manifest),
                                        'max_probability_difference_from_original': float(np.max(np.abs(portable-original))),
                                        'all_predicted_classes_match': True, 'test_inference_performed': False,
                                        'models_fitted': 0, 'numeric_arrays_only': True})
    (out/'README.md').write_text(
        '# Local AV reference model\n\n'
        'Numeric export of the model selected before final test evaluation. Keep this research model local. '
        'The bundle contains numeric scaler/classifier parameters and a feature contract; it contains no interview arrays, participant identifiers, or questionnaire records.\n\n'
        'Use `python -m src.reference_inference predict --bundle '+str(out)+' --source <interview_archive_or_folder> --output <new_result.json>`, '
        'or provide `--audio-csv` and `--visual-csv`, or a prepared `--cache` with its matching JSON sidecar. '
        'No labels or gender are required. Prediction loads NPZ arrays with object loading disabled.\n\n'
        'The input must contain the fixed MFCC/OpenFace feature schema. Plain audio/video files are insufficient. '
        'The existing weak final performance and preprocessing limits remain; this is not a clinical service.\n')
    return {'bundle': str(out), 'validation_predictions_verified': len(validation), 'test_inference_performed': False}


def read_cache(path, contract):
    path = Path(path)
    sidecar = path.with_suffix('.json')
    info = json.loads(sidecar.read_text())
    if info['version'] != contract['feature_version'] or info['audio_columns'] != contract['audio_columns'] or info['visual_columns'] != contract['visual_columns']:
        raise ValueError('Cached feature schema differs from the frozen model')
    if any(info['settings'].get(key) != value for key, value in contract['preprocessing'].items()):
        raise ValueError('Cached preprocessing differs from the frozen model')
    with np.load(path, allow_pickle=False) as data:
        sequence = {key: data[key] for key in data.files}
    validate_sequence(sequence, contract['preprocessing'])
    return sequence, {'kind': 'cache_with_declared_schema', 'record_id': str(info['participant_id']),
                      'files': {str(path): sha256(path), str(sidecar): sha256(sidecar)}}


def read_features(contract, source=None, audio_csv=None, visual_csv=None):
    if source is not None:
        path = Path(source)
        match = SOURCE_PATTERN.fullmatch(path.name)
        if not match or not path.exists():
            raise ValueError('Source must be a numeric participant archive/folder such as 300_P.zip')
        stamp = source_stamp(path)
        pid = match[1]
        blobs, _, checksum = read_source({'path': path, 'participant_id': pid})
        if source_stamp(path) != stamp:
            raise ValueError('Source changed while being read')
        names = [f'{pid}_OpenSMILE2.3.0_mfcc.csv', f'{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv']
        if not all(name in blobs for name in names):
            raise ValueError('Source is missing the required MFCC/OpenFace pair')
        audio, visual = [blobs[name] for name in names]
        provenance = {'kind': 'fully_read_source', 'record_id': pid, 'source': str(path), 'source_sha256': checksum,
                      'selected_feature_sha256': {name: hashlib.sha256(blobs[name]).hexdigest() for name in names}}
    else:
        if audio_csv is None or visual_csv is None:
            raise ValueError('Both --audio-csv and --visual-csv are required')
        paths = [Path(audio_csv), Path(visual_csv)]
        if any(path.stat().st_size > MAX_SELECTED_BYTES for path in paths):
            raise ValueError('Selected CSV exceeds 1 GiB bound')
        audio, visual = [path.read_bytes() for path in paths]
        provenance = {'kind': 'explicit_feature_pair', 'record_id': None,
                      'files': {str(path): hashlib.sha256(blob).hexdigest() for path, blob in zip(paths, [audio, visual])}}
    # Exactly the same transformation as the reference's cached features.
    sequence, quality = align(audio, visual, contract['preprocessing'])
    validate_sequence(sequence, contract['preprocessing'])
    return sequence, {**provenance, 'raw_feature_quality': quality}


def prediction_record(contract, parameters, sequence, provenance):
    validate_sequence(sequence, contract['preprocessing'])
    probs = probabilities(parameters, pool_modalities(sequence)[None, :])[0]
    predicted = int(probs.argmax())
    return {'format': FORMAT, 'model': contract['model'], 'record_id': provenance['record_id'],
            'predicted_class': predicted, 'predicted_questionnaire_category': LABELS[predicted],
            'probabilities': probs.tolist(), 'class_order': list(range(5)),
            'input': provenance, 'model_weights_sha256': contract['weights_sha256'],
            'observations': {'bins': len(sequence['timestamps']),
                             'audio_observed_fraction': float(sequence['audio_mask'].mean()),
                             'visual_observed_fraction': float(sequence['visual_mask'].mean())},
            'limitations': contract['limitations']}


def predict(bundle, output, cache=None, source=None, audio_csv=None, visual_csv=None):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f'Preserving existing output: {destination}')
    if sum(value is not None for value in [cache, source, audio_csv]) != 1 or (visual_csv is not None and audio_csv is None):
        raise ValueError('Choose exactly one cache, source, or explicit feature pair')
    contract, parameters = load_bundle(bundle)
    sequence, provenance = read_cache(cache, contract) if cache is not None else read_features(contract, source, audio_csv, visual_csv)
    report = prediction_record(contract, parameters, sequence, provenance)
    write_json(destination, report)
    # Keep participant-level predictions in the requested local file.
    return {'saved': str(destination), 'model': contract['model'], 'records': 1, 'model_fitted': False}


def validation_demo(bundle, plan_path, output):
    from .common import joined_manifest
    from .exploratory_baselines import load_audited_sequences
    from .final_evaluation import verify_plan
    out = Path(output)
    if out.exists():
        raise FileExistsError(f'Preserving existing demonstration: {out}')
    contract, parameters = load_bundle(bundle)
    plan = verify_plan(plan_path)
    if contract['frozen_plan_sha256'] != digest(plan):
        raise ValueError('Portable model and frozen plan differ')
    manifest = joined_manifest(plan['feature_config'])
    _, sources = load_audited_sequences(plan['feature_config'], manifest)
    if sources != plan['source_provenance'] or digest(manifest.to_dict('records')) != plan['manifest_hash']:
        raise ValueError('Frozen demonstration dataset changed')
    validation = manifest[manifest.split.eq('validation')].reset_index(drop=True)
    item = next(row for row in plan['models'] if row['model'] == contract['model'])
    saved = pd.read_csv(item['validation_predictions'], dtype={'participant_id': str})
    if saved.participant_id.tolist() != validation.participant_id.tolist():
        raise ValueError('Stored validation order changed')
    records = []
    for pid in validation.participant_id:
        sequence, provenance = read_cache(sources[pid]['cache'], contract)
        records.append(prediction_record(contract, parameters, sequence, provenance))
    actual = np.array([row['probabilities'] for row in records])
    expected = saved[[f'p{i}' for i in range(5)]].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual.argmax(1), saved.predicted_label)
    out.mkdir(parents=True, exist_ok=False)
    for record in records:
        write_json(out/f"{record['record_id']}.json", record)
    pd.DataFrame([{'participant_id': row['record_id'], 'split': 'validation', 'predicted_class': row['predicted_class'],
                   **{f'p{i}': value for i, value in enumerate(row['probabilities'])}} for row in records]).to_csv(out/'predictions.csv', index=False)
    report = {'complete': True, 'model': contract['model'], 'validation_participants': len(records),
              'all_audited_participants_accounted_for': len(manifest), 'max_probability_difference': float(np.max(np.abs(actual-expected))),
              'predicted_classes_match_saved_validation': True, 'test_inference_performed': False, 'models_fitted': 0}
    write_json(out/'verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    export = commands.add_parser('export', help='Export the trusted frozen reference without fitting')
    export.add_argument('--plan', default='results/final_evaluation_plan.json')
    export.add_argument('--output', default=DEFAULT_BUNDLE)
    inference = commands.add_parser('predict', help='Predict from feature inputs without labels or gender')
    inference.add_argument('--bundle', default=DEFAULT_BUNDLE)
    inputs = inference.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--cache', help='Prepared NPZ with matching JSON sidecar')
    inputs.add_argument('--source', help='Participant .zip/.tar/.tar.gz/.tgz archive or extracted folder')
    inputs.add_argument('--audio-csv', help='MFCC CSV; also provide --visual-csv')
    inference.add_argument('--visual-csv', help='OpenFace CSV paired with --audio-csv')
    inference.add_argument('--output', required=True)
    demo = commands.add_parser('validation-demo', help='Replay all validation participants and check saved probabilities')
    demo.add_argument('--bundle', default=DEFAULT_BUNDLE)
    demo.add_argument('--plan', default='results/final_evaluation_plan.json')
    demo.add_argument('--output', default='results/reference_inference/validation')
    args = parser.parse_args()
    if args.action == 'export':
        result = export_reference(args.plan, args.output)
    elif args.action == 'predict':
        result = predict(args.bundle, args.output, args.cache, args.source, args.audio_csv, args.visual_csv)
    else:
        result = validation_demo(args.bundle, args.plan, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
