"""Inference for frozen robust-AV research models, with original raw feature extraction."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json
from .best_accuracy_inference import load_bundle, extract_inputs
from .common import digest
from .final_evaluation import sha256

ROOT = Path('results/robust_av_accuracy')


def frozen_model(policy='accuracy'):
    envelope = json.loads((ROOT/'final/plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Frozen plan changed')
    record = plan['models'][policy]
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Frozen checkpoint changed')
    for path, expected in plan['bindings'].items():
        if Path(path).parent == Path('src') and sha256(path) != expected:
            raise ValueError('Frozen model implementation changed')
    state = pickle.loads(Path(record['path']).read_bytes())
    contract, _ = load_bundle()
    if digest(contract) != plan['feature_contract_hash']:
        raise ValueError('Original input feature contract changed')
    return record, state, contract


def predict_arrays(record, state, semantic, av):
    semantic, av = np.asarray(semantic, float), np.asarray(av, float)
    if (semantic.ndim != 2 or semantic.shape[1] != 384 or av.shape != (len(semantic), 128)
            or not len(semantic) or not np.isfinite(semantic).all() or not np.isfinite(av).all()
            or not np.allclose(np.linalg.norm(semantic, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Normalized MiniLM and aligned original AV features required')
    base = np.column_stack([semantic, av])
    if record['existing_baseline']:
        values = {'semantic': semantic, 'semantic_av': base}[state['candidate']['feature']]
    else:
        if state['candidate']['family'] != 'robust_av':
            raise ValueError('Wrong frozen model family')
        values = base
    return state['estimator'].predict_proba(values)


def predict(policy, output, transcript, audio_csv, visual_csv):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserving existing predictions')
    record, state, contract = frozen_model(policy)
    semantic, av, provenance = extract_inputs(contract, transcript, audio_csv, visual_csv)
    with threadpool_limits(limits=2):
        p = predict_arrays(record, state, semantic, av)
    atomic_json(output, {'model_sha256': record['sha256'], 'policy': policy,
        'class_order': [0, 1, 2, 3, 4], 'predicted_class': p.argmax(1).tolist(), 'decision_scores': p.tolist(),
        'feature_contract_hash': digest(contract), 'input': provenance,
        'score_interpretation': 'Uncalibrated research model scores', 'questionnaire_inputs': False})
    return {'output': str(output), 'participants': len(p), 'model_sha256': record['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['accuracy', 'robust_only'], default='accuracy')
    parser.add_argument('--transcript', required=True); parser.add_argument('--audio-csv', required=True)
    parser.add_argument('--visual-csv', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.policy, args.output, args.transcript, args.audio_csv, args.visual_csv), indent=2))


if __name__ == '__main__':
    main()
