"""Raw-file inference for frozen pairwise candidates or the unchanged language control."""
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

ROOT = Path('results/pairwise_accuracy')


def frozen_model(policy):
    envelope = json.loads((ROOT/'final/plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash'] or plan['inference_feature_keys'] != ['semantic_av', 'language']:
        raise ValueError('Frozen model plan or feature schema changed')
    record = plan['models'][policy]
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Frozen model changed')
    for path, expected in plan['bindings'].items():
        if Path(path).parent == Path('src') and sha256(path) != expected:
            raise ValueError('Frozen model/extraction implementation changed')
    contract, _ = load_bundle()
    if digest(contract) != plan['feature_contract_hash']:
        raise ValueError('Original feature contract changed')
    return record, pickle.loads(Path(record['path']).read_bytes()), contract


def predict_arrays(record, state, x):
    base, language = np.asarray(x['semantic_av']), np.asarray(x['language'])
    if base.ndim != 2 or base.shape[1] != 512 or language.shape != (len(base), 16) or not len(base):
        raise ValueError('Original semantic/AV and language features required')
    if not np.isfinite(base).all() or not np.isfinite(language).all():
        raise ValueError('Invalid input features')
    if record['existing_language_control']:
        return state['estimator'].predict_proba({'base': base, 'language': language})
    return state['estimator'].predict_proba(x)


def predict(policy, output, transcript, audio_csv, visual_csv):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserving existing predictions')
    from .checked_inference import validate_identity
    participant_id=validate_identity(transcript,audio_csv,visual_csv)
    record, state, contract = frozen_model(policy)
    semantic, av, provenance = extract_inputs(contract, transcript, audio_csv, visual_csv)
    text = clean_transcript(Path(transcript).read_bytes())['text']
    x = {'semantic_av': np.column_stack([semantic, av]), 'language': language_features(text)[None, :]}
    with threadpool_limits(limits=2):
        p = predict_arrays(record, state, x)
    provenance['participant_id']=participant_id
    atomic_json(output, {'model_sha256': record['sha256'], 'policy': policy,
        'class_order': [0, 1, 2, 3, 4], 'predicted_class': p.argmax(1).tolist(), 'decision_scores': p.tolist(),
        'feature_contract_hash': digest(contract), 'input': provenance,
        'score_interpretation': 'Uncalibrated exploratory model outputs', 'questionnaire_inputs': False})
    return {'output': str(output), 'participants': len(p), 'model_sha256': record['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['language_control', 'accuracy', 'pairwise_only'], required=True)
    parser.add_argument('--transcript', required=True); parser.add_argument('--audio-csv', required=True)
    parser.add_argument('--visual-csv', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    print(json.dumps(predict(args.policy, args.output, args.transcript, args.audio_csv, args.visual_csv), indent=2))


if __name__ == '__main__':
    main()
