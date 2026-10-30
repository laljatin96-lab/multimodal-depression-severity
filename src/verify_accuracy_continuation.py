"""Verify both accuracy continuation studies without refitting completed jobs."""
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import baseline_averaging_accuracy as averaging
from . import semantic_severe_repair as baseline
from . import supervised_accuracy_finalize as supervised
from .batch_inference import atomic_json, load_record
from .final_evaluation import sha256
from .severe_case_diagnosis import load_baseline


def verify():
    first = supervised.verify()
    inputs = load_baseline(); x, frame, old_settings, old_records, _ = inputs
    settings = json.loads(Path('configs/baseline_averaging_accuracy.json').read_text()); out = averaging.OUTPUT
    completion = json.loads((out/'completion.json').read_text())
    envelope = json.loads((out/'final'/'plan.json').read_text()); plan = envelope['plan']
    paths = list((out/'jobs').glob('*/*'))+list((out/'final').glob('*'))+list((out/'validation').glob('*/result.json'))
    before = {str(p): [sha256(p), p.stat().st_mtime_ns] for p in paths if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('Completed averaging resume cannot fit, infer or deserialize')
    with patch.object(averaging, 'fit_members', forbidden), patch.object(averaging, 'predict_state', forbidden), patch.object(pickle, 'load', forbidden):
        resumed = averaging.run(inputs, settings, resume=True)
        averaging.evaluate()
    if resumed['new_fits'] or before != {p: [sha256(p), Path(p).stat().st_mtime_ns] for p in before}:
        raise ValueError('Averaging resume changed frozen jobs/models')
    replayed = receipts = 0; fitting_scores = []
    with threadpool_limits(limits=2), averaging.offline_network():
        for fold, old in zip(averaging.nested_folds(frame, old_settings), old_records):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            r = load_record(directory/'result.json', completion['context_hash'])
            averaging.validate(r, fold, old, settings, directory)
            for name, result in r['selected'].items():
                with (directory/f'{name}.pkl').open('rb') as f:
                    state = pickle.load(f)
                if state['fit'] != fold['fit'] or state['heldout'] != fold['heldout']:
                    raise ValueError('Ensemble participant ownership changed')
                if [m['candidate_index'] for m in state['members']] != result['selection']['member_indices']:
                    raise ValueError('Serialized ensemble membership changed')
                p = averaging.predict_state(state, x, fold['heldout'])
                np.testing.assert_allclose(p, result['probabilities'], rtol=0, atol=1e-14); replayed += 1
                for member in state['members']:
                    model = member['estimator']; candidate = member['candidate']
                    baseline.validate_receipt(baseline.fit_receipt(model, candidate), candidate,
                                              frame.severity_class.to_numpy(int)[fold['fit']], old_settings)
                    receipts += 1
                fit_accuracy = float((averaging.predict_state(state, x, fold['fit']).argmax(1) == frame.severity_class.to_numpy(int)[fold['fit']]).mean())
                if fit_accuracy != result['real_training_accuracy']:
                    raise ValueError('Ensemble fitting accuracy changed')
                fitting_scores.append({'policy': name, 'seed': fold['seed'], 'fold': fold['fold'], 'training_accuracy': fit_accuracy,
                                       'outer_accuracy': averaging.score(frame.severity_class.to_numpy(int)[fold['heldout']], p.argmax(1))['accuracy']})
        vx, vf = supervised.validation_inputs(plan)
        for name, r in plan['models'].items():
            with Path(r['path']).open('rb') as f:
                state = pickle.load(f)
            if state['fit_participants'] != frame.participant_id.tolist():
                raise ValueError('Final model training population changed')
            p = averaging.predict_state(state, vx)
            saved = load_record(out/'validation'/name/'result.json', envelope['plan_hash'])
            np.testing.assert_allclose(p, saved['probabilities'], rtol=0, atol=1e-14); replayed += 1
    pd.DataFrame(fitting_scores).to_csv(out/'overfitting_audit.csv', index=False)
    summary = pd.read_csv(out/'summary.csv').set_index('model')
    original = json.loads((supervised.OUTPUT/'best_valid_model.json').read_text())['best_model']
    improved = bool(summary.loc['accuracy', 'accuracy_mean'] > summary.loc['baseline', 'accuracy_mean'])
    manifest = {'retained': 'top_five_exploratory' if improved else 'original_verified_49.08_baseline',
                'replacement': improved, 'reference_model_overwritten': False,
                'primary_policy': settings['primary'], 'primary_nested_accuracy': float(summary.loc['accuracy', 'accuracy_mean']),
                'baseline_nested_accuracy': float(summary.loc['baseline', 'accuracy_mean']),
                'best_model': plan['models']['accuracy'] if improved else original,
                'research_models': plan['models'], 'selected_using_validation': False,
                'independent_confirmation': False,
                'target_53_percent_reached': bool(summary.loc['accuracy', 'accuracy_mean'] >= settings['target_accuracy'])}
    atomic_json(out/'best_valid_model.json', manifest)
    prior = json.loads(Path('results/supervised_accuracy_preservation/before.json').read_text())
    changed = [p for p,v in prior.items() if not Path(p).exists() or Path(p).stat().st_mtime_ns != v['mtime_ns'] or sha256(p) != v['sha256']]
    if changed:
        raise ValueError(f'Pre-existing files changed: {changed}')
    result = {'passed': True, 'supervised_study': first, 'averaging_checkpoints_replayed': replayed,
              'averaging_fitting_receipts_checked': receipts, 'zero_fit_averaging_resume': True,
              'averaging_saved_records_preserved_on_resume': len(before), 'previous_files_and_mtimes_preserved': len(prior),
              'total_checkpoints_replayed': replayed+first['checkpoints_replayed'],
              'total_new_classifier_fits': completion['fits']+plan['new_full_training_fits']+first['new_classifier_fits'],
              'validation_participants': len(vf), 'test_scored': False,
              'automated_tests': first['automated_tests']}
    atomic_json(out/'verification.json', result)
    atomic_json(Path('results/supervised_accuracy_preservation/combined_verification.json'), result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
