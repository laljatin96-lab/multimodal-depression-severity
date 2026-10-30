"""Questionnaire-free raw-file inference for frozen language and hierarchy candidates."""
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

STUDIES = {'age-language': Path('results/age_language_accuracy'),
           'hierarchy': Path('results/conditional_hierarchy_accuracy')}


def frozen_model(study, policy):
    root = STUDIES[study]; envelope = json.loads((root/'final/plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Frozen plan changed')
    record = plan['models'][policy]
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Frozen checkpoint changed')
    # Inference needs the model implementation, not historical questionnaire or dataset files.
    for path, expected in plan['bindings'].items():
        if Path(path).parent == Path('src') and sha256(path) != expected:
            raise ValueError('Frozen model implementation changed')
    state = pickle.loads(Path(record['path']).read_bytes())
    contract, _ = load_bundle()
    if study == 'hierarchy' and digest(contract) != plan['feature_contract_hash']:
        raise ValueError('Feature contract changed')
    if study == 'age-language' and not record['existing_baseline'] and state['candidate']['feature_set'] != 'language':
        raise ValueError('This command supports only the age-free candidate')
    return record, state, contract


def predict_arrays(record, state, semantic, av, cleaned_text=None):
    semantic, av = np.asarray(semantic, float), np.asarray(av, float)
    if (semantic.ndim != 2 or semantic.shape[1] != 384 or av.shape != (len(semantic), 128)
            or not len(semantic) or not np.isfinite(semantic).all() or not np.isfinite(av).all()
            or not np.allclose(np.linalg.norm(semantic, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Normalized MiniLM and aligned original AV features required')
    base = np.column_stack([semantic, av])
    if record['existing_baseline']:
        features = {'semantic': semantic, 'semantic_av': base}
        p = state['estimator'].predict_proba(features[state['candidate']['feature']])
    elif state['candidate']['family'] == 'block_logistic':
        from .age_language_accuracy import language_features
        if cleaned_text is None or len(cleaned_text) != len(base):
            raise ValueError('Aligned cleaned transcript text required')
        p = state['estimator'].predict_proba({'base': base, 'language': np.stack([language_features(t) for t in cleaned_text])})
    elif state['candidate']['family'] == 'conditional_hierarchy':
        p = state['estimator'].predict_proba(base)
    else:
        raise ValueError('Unsupported frozen family')
    return p


def predict(study, policy, output, transcript, audio_csv, visual_csv):
    from .transcript_corpus import clean_transcript
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserving existing predictions')
    record, state, contract = frozen_model(study, policy)
    semantic, av, provenance = extract_inputs(contract, transcript, audio_csv, visual_csv)
    text = clean_transcript(Path(transcript).read_bytes())['text']
    with threadpool_limits(limits=2):
        p = predict_arrays(record, state, semantic, av, [text])
    atomic_json(output, {'model_sha256': record['sha256'], 'study': study, 'policy': policy,
        'class_order': [0, 1, 2, 3, 4], 'predicted_class': p.argmax(1).tolist(), 'decision_scores': p.tolist(),
        'score_interpretation': 'Uncalibrated research model outputs', 'feature_contract_hash': digest(contract),
        'input': provenance, 'questionnaire_inputs': False})
    return {'output': str(output), 'participants': len(p), 'model_sha256': record['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=list(STUDIES), required=True); parser.add_argument('--policy', required=True)
    parser.add_argument('--transcript', required=True); parser.add_argument('--audio-csv', required=True)
    parser.add_argument('--visual-csv', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.study, args.policy, args.output, args.transcript, args.audio_csv, args.visual_csv), indent=2))


if __name__ == '__main__':
    main()
