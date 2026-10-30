"""Numeric inference bundle for the retained semantic/AV accuracy model.

Unlike the historical AV-only reference, this model requires transcript features
as well as the original MFCC/OpenFace summaries. No questionnaire is an input.
"""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json
from .common import digest
from .final_evaluation import sha256

BUNDLE = Path('checkpoints/best_accuracy_20261027')
FORMAT = 'best-semantic-av-numeric-v1'


def numeric_parameters(model):
    if model.pca_ is None or model.n_features_in_ != 512 or not np.array_equal(model.classes_, np.arange(5)):
        raise ValueError('Expected retained five-class semantic/AV PCA model')
    return {'scaler_mean': model.scaler_.mean_.copy(), 'scaler_scale': model.scaler_.scale_.copy(),
            'pca_mean': model.pca_.mean_.copy(), 'pca_components': model.pca_.components_.copy(),
            'coef': model.model_.coef_.copy(), 'intercept': model.model_.intercept_.copy()}


def validate_parameters(parameters):
    shapes = {'scaler_mean': (512,), 'scaler_scale': (512,), 'pca_mean': (512,),
              'pca_components': (32, 512), 'coef': (5, 32), 'intercept': (5,)}
    if set(parameters) != set(shapes) or any(parameters[k].shape != shape or not np.isfinite(parameters[k]).all() for k, shape in shapes.items()):
        raise ValueError('Invalid numeric model schema')
    if (parameters['scaler_scale'] <= 0).any():
        raise ValueError('Invalid scaler scale')


def predict_features(parameters, semantic, av):
    validate_parameters(parameters)
    semantic, av = np.asarray(semantic, float), np.asarray(av, float)
    if semantic.ndim != 2 or semantic.shape[1] != 384 or av.shape != (len(semantic), 128) or not len(semantic):
        raise ValueError('Expected aligned N x 384 semantic and N x 128 AV features')
    if not np.isfinite(semantic).all() or not np.isfinite(av).all() or not np.allclose(np.linalg.norm(semantic, axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError('Finite features and normalized MiniLM document embeddings required')
    x = np.column_stack([semantic, av])
    z = (x-parameters['scaler_mean'])/parameters['scaler_scale']
    z = (z-parameters['pca_mean']) @ parameters['pca_components'].T
    return softmax(z @ parameters['coef'].T+parameters['intercept'], axis=1)


def load_bundle(root=BUNDLE):
    root = Path(root); envelope = json.loads((root/'model.json').read_text()); contract = envelope['contract']
    if digest(contract) != envelope['contract_hash'] or contract['format'] != FORMAT or sha256(root/'weights.npz') != contract['weights_sha256']:
        raise ValueError('Bundle integrity mismatch')
    with np.load(root/'weights.npz', allow_pickle=False) as archive:
        parameters = {k: archive[k].copy() for k in archive.files}
    validate_parameters(parameters)
    return contract, parameters


def export(root=BUNDLE):
    from .severe_case_diagnosis import load_baseline
    from .encoder_kernel_accuracy import offline_network
    root = Path(root)
    if root.exists():
        raise FileExistsError('Preserving existing bundle')
    manifest = json.loads(Path('results/decision_offset_accuracy/best_valid_model.json').read_text())
    if manifest['replacement']:
        raise ValueError('Exporter expects the retained original baseline; do not silently export another family')
    retained = manifest['best_model']
    if sha256(retained['path']) != retained['sha256']:
        raise ValueError('Retained source model changed')
    x, frame, _, _, _ = load_baseline()
    state = pickle.loads(Path(retained['path']).read_bytes()); model = state['estimator']
    if state['fit_participants'] != frame.participant_id.tolist():
        raise ValueError('Full-training checkpoint ownership changed')
    parameters = numeric_parameters(model)
    with threadpool_limits(limits=2), offline_network():
        actual = predict_features(parameters, x['semantic'], x['semantic_av'][:, 384:])
        original = model.predict_proba(x['semantic_av'])
    np.testing.assert_allclose(actual, original, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.argmax(1), original.argmax(1))
    schema = json.loads(Path('results/temporal_summary_comparison/feature_schema.json').read_text())['cached_av']
    encoder_protocol = json.loads(Path('results/semantic_features/protocol.json').read_text())['context']
    plan = json.loads(Path('results/final_evaluation_plan.json').read_text())['plan']
    feature_config = plan['feature_config']
    config = {k: feature_config[k] for k in ['audio_kind','audio_time_offset','visual_time_offset','bin_seconds',
              'minimum_window_steps','confidence_threshold','timestamp_origins_verified','allow_unsegmented']}
    contract = {'format': FORMAT, 'source_model': retained, 'classes': [0, 1, 2, 3, 4],
                'class_names': ['None/minimal', 'Mild', 'Moderate', 'Moderately severe', 'Severe'],
                'semantic_dimensions': 384, 'av_feature_names': schema, 'preprocessing': config,
                'semantic_pooling': '254 content tokens/chunk; masked token mean and L2 normalization; token-count-weighted chunk mean and final L2 normalization',
                'encoder': encoder_protocol['encoder'], 'encoder_root': encoder_protocol['model_root'],
                'extraction_code_sha256': {str(Path('src')/n): sha256(Path('src')/n) for n in
                    ['semantic_features.py','transcript_corpus.py','prepare_features.py','exploratory_baselines.py']},
                'fit_participants': len(frame), 'questionnaire_required': False, 'gender_required': False,
                'nested_selection_procedure_accuracy': manifest['baseline_nested_accuracy'],
                'nested_accuracy_is_not_this_checkpoint_test_accuracy': True,
                'independent_confirmation': False, 'test_scored_by_export': False,
                'limitations': ['Uncalibrated research scores', 'Speaker identity and event timing remain unverified',
                    'Previously exposed evaluation population', 'Feature declarations cannot prove external feature provenance']}
    root.mkdir(parents=True); np.savez_compressed(root/'weights.npz', **parameters)
    contract['weights_sha256'] = sha256(root/'weights.npz')
    atomic_json(root/'model.json', {'contract': contract, 'contract_hash': digest(contract)})
    _, restored = load_bundle(root)
    np.testing.assert_allclose(actual, predict_features(restored, x['semantic'], x['semantic_av'][:, 384:]), rtol=0, atol=1e-12)
    result = {'source_model_sha256': retained['sha256'], 'training_rows_replayed': len(frame),
              'max_probability_difference': float(np.max(np.abs(actual-original))), 'all_decisions_match': True,
              'models_fitted': 0, 'test_inference': False, 'numeric_arrays_only': True}
    atomic_json(root/'verification.json', result)
    return result


def extract_inputs(contract, transcript, audio_csv, visual_csv):
    import torch
    from .encoder_kernel_accuracy import offline_network
    from .semantic_features import local_encoder, encode_document
    from .transcript_corpus import clean_transcript
    from .prepare_features import align
    from .exploratory_baselines import pool_modalities
    for p, h in contract['extraction_code_sha256'].items():
        if sha256(p) != h:
            raise ValueError('Feature extraction code changed')
    encoder_root = Path(contract['encoder_root'])
    for name, expected in contract['encoder']['files'].items():
        if sha256(encoder_root/name) != expected['sha256']:
            raise ValueError('Frozen text encoder changed')
    paths = [Path(transcript), Path(audio_csv), Path(visual_csv)]
    blobs = [p.read_bytes() for p in paths]
    text = clean_transcript(blobs[0])['text']
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2), offline_network():
        tokenizer, encoder = local_encoder(encoder_root)
        semantic = np.asarray(encode_document(text, tokenizer, encoder)['embedding'])[None, :]
        sequence, _ = align(blobs[1], blobs[2], contract['preprocessing'])
        av = pool_modalities(sequence)[None, :]
    return semantic, av, {'input_hashes': {str(p): sha256(p) for p in paths}, 'mode': 'original_feature_extraction'}


def predict(root, output, features=None, transcript=None, audio_csv=None, visual_csv=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserving existing predictions')
    contract, parameters = load_bundle(root)
    raw = [transcript, audio_csv, visual_csv]
    if features is not None:
        if any(p is not None for p in raw):
            raise ValueError('Choose prepared features or the three source CSVs')
        with np.load(features, allow_pickle=False) as archive:
            if set(archive.files) != {'semantic', 'av', 'contract_hash'} or archive['contract_hash'].shape != () or str(archive['contract_hash']) != digest(contract):
                raise ValueError('Prepared features must declare this exact feature contract')
            semantic, av = archive['semantic'].copy(), archive['av'].copy()
        provenance = {'mode': 'declared_precomputed_features', 'sha256': sha256(features)}
    elif all(p is not None for p in raw):
        semantic, av, provenance = extract_inputs(contract, *raw)
    else:
        raise ValueError('Provide --features or --transcript, --audio-csv and --visual-csv together')
    with threadpool_limits(limits=2):
        p = predict_features(parameters, semantic, av)
    result = {'format': FORMAT, 'model_sha256': contract['source_model']['sha256'], 'contract_hash': digest(contract),
              'class_order': contract['classes'], 'predicted_class': p.argmax(1).tolist(), 'decision_scores': p.tolist(),
              'score_interpretation': 'Uncalibrated model outputs; no correctness or diagnosis claim', 'input': provenance}
    atomic_json(output, result)
    return {'output': str(output), 'participants': len(p), 'model_sha256': result['model_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'predict']); parser.add_argument('--bundle', default=str(BUNDLE))
    parser.add_argument('--output'); parser.add_argument('--features'); parser.add_argument('--transcript')
    parser.add_argument('--audio-csv'); parser.add_argument('--visual-csv'); args = parser.parse_args()
    if args.action == 'export':
        result = export(args.bundle)
    else:
        if not args.output:
            parser.error('--output is required for prediction')
        result = predict(args.bundle, args.output, args.features, args.transcript, args.audio_csv, args.visual_csv)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
