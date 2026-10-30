"""Numeric export and inference for the preserved 49.28% age-free language candidate."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from .age_language_accuracy import LANGUAGE_NAMES, language_features
from .batch_inference import atomic_json
from .best_accuracy_inference import load_bundle as original_bundle, extract_inputs
from .common import digest
from .final_evaluation import sha256
from .transcript_corpus import clean_transcript

BUNDLE = Path('checkpoints/best_observed_4928')
FORMAT = 'preserved-age-free-language-numeric-v1'
SHAPES = {'base_mean': (512,), 'base_scale': (512,), 'pca_mean': (512,), 'pca_components': (32, 512),
    'language_imputer': (16,), 'language_mean': (16,), 'language_scale': (16,),
    'language_weight': (), 'coef': (5, 48), 'intercept': (5,)}


def validate(parameters):
    if set(parameters) != set(SHAPES) or any(parameters[k].shape != shape or not np.isfinite(parameters[k]).all() for k, shape in SHAPES.items()):
        raise ValueError('Invalid numeric model schema')
    if (parameters['base_scale'] <= 0).any() or (parameters['language_scale'] <= 0).any():
        raise ValueError('Invalid scaling parameters')


def predict_features(parameters, semantic, av, language):
    validate(parameters)
    semantic, av, language = np.asarray(semantic, float), np.asarray(av, float), np.asarray(language, float)
    if (semantic.ndim != 2 or semantic.shape[1] != 384 or not len(semantic)
            or av.shape != (len(semantic), 128) or language.shape != (len(semantic), 16)):
        raise ValueError('Expected aligned MiniLM384, original AV128 and fixed language16 features')
    if (not np.isfinite(semantic).all() or not np.isfinite(av).all() or np.isinf(language).any()
            or not np.allclose(np.linalg.norm(semantic, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Finite AV and normalized MiniLM features required')
    base = (np.column_stack([semantic, av])-parameters['base_mean'])/parameters['base_scale']
    base = (base-parameters['pca_mean'])@parameters['pca_components'].T
    language = np.where(np.isnan(language), parameters['language_imputer'], language)
    language = parameters['language_weight']*(language-parameters['language_mean'])/parameters['language_scale']
    return softmax(np.column_stack([base, language])@parameters['coef'].T+parameters['intercept'], axis=1)


def load(root=BUNDLE):
    root = Path(root); envelope = json.loads((root/'model.json').read_text()); contract = envelope['contract']
    if digest(contract) != envelope['contract_hash'] or contract['format'] != FORMAT or sha256(root/'weights.npz') != contract['weights_sha256']:
        raise ValueError('Numeric bundle integrity mismatch')
    if contract['language_names'] != LANGUAGE_NAMES:
        raise ValueError('Language feature schema changed')
    for p, expected in contract['additional_code_hashes'].items():
        if sha256(p) != expected:
            raise ValueError('Language extraction or inference implementation changed')
    with np.load(root/'weights.npz', allow_pickle=False) as archive:
        parameters = {k: archive[k].copy() for k in archive.files}
    validate(parameters)
    return contract, parameters


def export(root=BUNDLE):
    from .age_language_finalize import frozen_plan
    from .severe_case_diagnosis import load_baseline
    root = Path(root)
    if root.exists():
        raise FileExistsError('Preserving the existing best-observed bundle')
    record = frozen_plan()[0]['models']['without_age']
    state = pickle.loads(Path(record['path']).read_bytes()); m = state['estimator']; p = m.projection
    if state['candidate']['feature_set'] != 'language' or not np.array_equal(m.classes_, np.arange(5)):
        raise ValueError('Expected the original age-free five-class checkpoint')
    parameters = {'base_mean': p.base_scaler.mean_, 'base_scale': p.base_scaler.scale_,
        'pca_mean': p.pca.mean_, 'pca_components': p.pca.components_,
        'language_imputer': p.imputers['language'].statistics_, 'language_mean': p.scalers['language'].mean_,
        'language_scale': p.scalers['language'].scale_, 'language_weight': np.asarray(m.candidate['block_scale']),
        'coef': m.model.coef_, 'intercept': m.model.intercept_}
    validate(parameters)
    x, frame, _, _, _ = load_baseline()
    if state['fit_participants'] != frame.participant_id.tolist():
        raise ValueError('Original fitting ownership differs')
    corpus = {r['participant_id']: r for r in json.loads(Path('results/transcript_corpus/participants.json').read_text())}
    language = np.stack([language_features(corpus[pid]['text']) for pid in frame.participant_id])
    with threadpool_limits(limits=2):
        actual = predict_features(parameters, x['semantic'], x['semantic_av'][:, 384:], language)
        expected = m.predict_proba({'base': x['semantic_av'], 'language': language})
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    base_contract, _ = original_bundle(); root.mkdir(parents=True)
    np.savez_compressed(root/'weights.npz', **parameters)
    contract = {'format': FORMAT, 'weights_sha256': sha256(root/'weights.npz'), 'source_model': record,
        'original_feature_contract': base_contract, 'language_names': LANGUAGE_NAMES,
        'classes': [0, 1, 2, 3, 4], 'class_names': ['None/minimal', 'Mild', 'Moderate', 'Moderately severe', 'Severe'],
        'fit_participants': len(frame), 'questionnaire_required': False, 'age_required': False,
        'observed_nested_procedure_accuracy': 241/489, 'previously_reused_validation_checkpoint_accuracy': 23/56,
        'observed_gain_is_not_statistically_supported': True, 'independent_confirmation': False,
        'source_checkpoint_unchanged': True, 'export_refits': 0,
        'additional_code_hashes': {p: sha256(p) for p in ['src/age_language_accuracy.py', 'src/best_observed_inference.py']}}
    atomic_json(root/'model.json', {'contract': contract, 'contract_hash': digest(contract)})
    receipt = {'passed': True, 'original_training_participants_checked': len(frame),
        'maximum_probability_difference': float(np.max(np.abs(actual-expected))), 'all_decisions_match': True,
        'new_fits': 0, 'validation_test_scored': False, 'source_model_sha256': record['sha256']}
    atomic_json(root/'verification.json', receipt)
    return receipt


def predict(root, output, transcript, audio_csv, visual_csv):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserving existing prediction output')
    contract, parameters = load(root)
    semantic, av, provenance = extract_inputs(contract['original_feature_contract'], transcript, audio_csv, visual_csv)
    text = clean_transcript(Path(transcript).read_bytes())['text']
    p = predict_features(parameters, semantic, av, language_features(text)[None, :])
    atomic_json(output, {'model_sha256': contract['source_model']['sha256'], 'contract_hash': digest(contract),
        'predicted_class': p.argmax(1).tolist(), 'class_order': contract['classes'], 'decision_scores': p.tolist(),
        'score_interpretation': 'Uncalibrated research outputs; 49.28% is prior nested-procedure accuracy, not a per-prediction confidence guarantee',
        'input': provenance})
    return {'output': str(output), 'participants': len(p), 'source_model_sha256': contract['source_model']['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['export', 'predict'])
    parser.add_argument('--bundle', default=str(BUNDLE)); parser.add_argument('--output')
    parser.add_argument('--transcript'); parser.add_argument('--audio-csv'); parser.add_argument('--visual-csv')
    args = parser.parse_args()
    if args.action == 'export':
        result = export(args.bundle)
    else:
        if not all([args.output, args.transcript, args.audio_csv, args.visual_csv]):
            parser.error('Prediction requires --output and all three source CSVs')
        result = predict(args.bundle, args.output, args.transcript, args.audio_csv, args.visual_csv)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
