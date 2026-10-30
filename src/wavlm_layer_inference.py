"""Raw-file inference for archived zero-window/middle-layer WavLM procedures."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

from .age_language_accuracy import language_features
from .batch_inference import atomic_json
from .best_accuracy_inference import load_bundle, extract_inputs
from .common import digest
from .final_evaluation import sha256
from .transcript_corpus import clean_transcript

ROOT = Path('results/wavlm_layer_accuracy')
POLICIES = ['previous_procedure', 'accuracy', 'new_only', 'zero_only', 'layer6_only']


def frozen_model(policy):
    envelope = json.loads((ROOT/'final/plan.json').read_text()); plan = envelope['plan']
    keys = ['semantic_av', 'language', 'speech', 'final_nonzero', 'layer6_nonzero']
    if digest(plan) != envelope['plan_hash'] or plan['inference_feature_keys'] != keys:
        raise ValueError('Frozen model plan or feature schema changed')
    record = plan['models'][policy]
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Frozen model changed')
    for path, expected in plan['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen model/extraction binding changed')
    contract, _ = load_bundle()
    if digest(contract) != plan['feature_contract_hash']:
        raise ValueError('Original feature contract changed')
    from .wavlm_layer_features import encoder_contract, runtime_contract
    if runtime_contract() != plan['speech_runtime'] or encoder_contract() != plan['speech_encoder']:
        raise ValueError('Frozen audio runtime/encoder changed')
    return record, pickle.loads(Path(record['path']).read_bytes()), contract


def speech_requirement(state):
    family = state['candidate']['family']
    if family == 'previous_procedure':
        family = state['source_candidate']['family']
    if family == 'wavlm_layer':
        return state['candidate']['representation']
    if family in {'language_control', 'block_logistic'}:
        return None
    if family == 'wavlm_augmented':
        return 'speech'
    raise ValueError('Unknown saved classifier family')


def predict_arrays(record, state, x):
    from .wavlm_layer_accuracy import predict_state
    base, language = np.asarray(x['semantic_av']), np.asarray(x['language'])
    if base.ndim != 2 or base.shape[1] != 512 or language.shape != (len(base), 16) or not len(base):
        raise ValueError('Original semantic/AV and language features required')
    if not np.isfinite(base).all() or not np.isfinite(language).all():
        raise ValueError('Invalid input features')
    key = speech_requirement(state)
    if key is not None:
        speech = np.asarray(x[key])
        if speech.shape != (len(base), 1536) or not np.isfinite(speech).all():
            raise ValueError('Expected aligned finite WavLM1536 features')
    return predict_state(state, x)


def predict(policy, output, transcript, audio_csv, visual_csv, audio_wav=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserving existing predictions')
    from .checked_inference import validate_identity
    participant = validate_identity(transcript, audio_csv, visual_csv)
    if audio_wav is not None and Path(audio_wav).name != participant+'_AUDIO.wav':
        raise ValueError('Raw WAV participant ID differs')
    record, state, contract = frozen_model(policy); key = speech_requirement(state)
    if key is not None and audio_wav is None:
        raise ValueError('WavLM candidate inference also requires --audio-wav')
    semantic, av, provenance = extract_inputs(contract, transcript, audio_csv, visual_csv)
    text = clean_transcript(Path(transcript).read_bytes())['text']
    x = {'semantic_av': np.column_stack([semantic, av]), 'language': language_features(text)[None, :]}
    if key is not None:
        from .encoder_kernel_accuracy import offline_network
        with threadpool_limits(limits=2), offline_network():
            if key == 'speech':
                from .wavlm_features_complete import load_encoder, encode_blob
                processor, encoder = load_encoder()
                speech = encode_blob(Path(audio_wav).read_bytes(), processor, encoder)
            else:
                from .wavlm_layer_features import load_encoder, encode_blob
                processor, encoder = load_encoder(key)
                speech = encode_blob(Path(audio_wav).read_bytes(), key, processor, encoder)
        x[key] = np.asarray(speech['features'])[None, :]
        provenance['wavlm'] = {k:v for k,v in speech.items() if k != 'features'}
    provenance['participant_id'] = participant
    with threadpool_limits(limits=2):
        p = predict_arrays(record, state, x)
    atomic_json(output, {'model_sha256': record['sha256'], 'policy': policy,
        'class_order': [0, 1, 2, 3, 4], 'predicted_class': p.argmax(1).tolist(), 'decision_scores': p.tolist(),
        'feature_contract_hash': digest(contract), 'input': provenance,
        'score_interpretation': 'Uncalibrated exploratory model outputs', 'questionnaire_inputs': False})
    return {'output': str(output), 'participants': len(p), 'model_sha256': record['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    parser.add_argument('--transcript', required=True); parser.add_argument('--audio-csv', required=True)
    parser.add_argument('--visual-csv', required=True); parser.add_argument('--audio-wav')
    parser.add_argument('--output', required=True); args = parser.parse_args()
    print(json.dumps(predict(args.policy, args.output, args.transcript, args.audio_csv, args.visual_csv, args.audio_wav), indent=2))


if __name__ == '__main__':
    main()
