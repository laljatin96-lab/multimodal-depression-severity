"""Check all distinct saved inference paths against fitting-cache predictions."""
import json
from pathlib import Path

import numpy as np

from .batch_inference import atomic_json
from .final_evaluation import sha256
from .wavlm_layer_verify import PRESERVATION, ensure_previous_files


def verify():
    root = Path('results/wavlm_layer_accuracy')
    output = root/'supplemental_inference_verification.json'
    if output.exists():
        raise FileExistsError('Preserving supplemental verification')
    manifest = json.loads((root/'delivery_manifest.json').read_text())
    for path, expected in manifest['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Completed delivery evidence changed')
    plan = json.loads((root/'final/plan.json').read_text())['plan']; rows = []
    with np.load(root/'final_training_predictions.npz', allow_pickle=False) as cache:
        i = cache['participant_id'].tolist().index('487')
        for policy, filename in [('new_only', 'raw_training_example.json'),
                                 ('layer6_only', 'raw_layer6_training_example.json'),
                                 ('accuracy', 'raw_retained_training_example.json')]:
            path = root/filename; raw = json.loads(path.read_text())
            expected = cache[f'{policy}_decision_scores'][i:i+1]
            if (raw['policy'] != policy or raw['input']['participant_id'] != '487'
                    or raw['questionnaire_inputs'] or raw['model_sha256'] != plan['models'][policy]['sha256']):
                raise ValueError('Raw inference ownership or saved model differs')
            np.testing.assert_allclose(raw['decision_scores'], expected, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(raw['predicted_class'], expected.argmax(1))
            rows.append({'policy': policy, 'model_sha256': raw['model_sha256'],
                'input_output_path': str(path), 'input_output_sha256': sha256(path),
                'maximum_probability_difference': float(np.abs(np.asarray(raw['decision_scores'])-expected).max())})
    previous = json.loads(PRESERVATION.read_text()); ensure_previous_files(previous)
    receipt = {'passed': True, 'raw_paths': rows, 'new_fits': 0, 'new_encoder_runs_in_this_command': 0,
        'validation_test_scored': False, 'preserved_files_and_mtimes': len(previous),
        'source_sha256': sha256(__file__), 'delivery_manifest_sha256': sha256(root/'delivery_manifest.json')}
    atomic_json(output, receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
