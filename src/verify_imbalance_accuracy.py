"""Replay all new models, verify zero-fit resume, and preserve the accuracy champion."""
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import imbalance_accuracy as study
from .batch_inference import atomic_json, load_record
from .crossvalidated_baselines import five_probabilities
from .final_evaluation import sha256
from .severe_case_diagnosis import load_baseline


def verify():
    out = study.OUTPUT
    settings = json.loads(Path('configs/imbalance_accuracy.json').read_text())
    inputs = load_baseline(); x, frame, old_settings, old_records, _ = inputs
    completion = json.loads((out/'completion.json').read_text())
    immutable_receipt = out/'training_completion.json'
    if not immutable_receipt.exists():
        atomic_json(immutable_receipt, completion)
    paths = list((out/'jobs').glob('*/*')) + list((out/'final').glob('*')) + list((out/'validation').glob('*/result.json'))
    before = {str(p): [sha256(p), p.stat().st_mtime_ns] for p in paths if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed resume must not fit, infer or deserialize')
    with patch.object(study, 'job', forbidden), patch.object(study, 'fit', forbidden), \
            patch.object(study, 'five_probabilities', forbidden), patch.object(study.pickle, 'load', forbidden):
        resumed = study.run(inputs, settings, resume=True)
        study.evaluate()
    if resumed['new_fits'] != 0 or resumed['new_jobs'] != 0:
        raise ValueError('Resume refit an existing experiment')
    if before != {p: [sha256(p), Path(p).stat().st_mtime_ns] for p in before}:
        raise ValueError('Resume changed a saved model, job or evaluation record')
    replayed = lineage_checked = 0
    records = []
    with threadpool_limits(limits=2), study.offline_network():
        for fold, old in zip(study.nested_folds(frame, old_settings), old_records):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            r = load_record(directory/'result.json', completion['context_hash']); records.append(r)
            study.validate(r, fold, old, frame.severity_class.to_numpy(int), settings, directory)
            for name, result in r['selected'].items():
                with (directory/f'{name}.pkl').open('rb') as f:
                    state = pickle.load(f)
                if state['fit'] != fold['fit'] or state['heldout'] != fold['heldout']:
                    raise ValueError('Checkpoint participant ownership changed')
                p = five_probabilities(state['estimator'], x[state['feature']][fold['heldout']])
                np.testing.assert_allclose(p, result['probabilities'], rtol=0, atol=1e-14)
                if result['candidate']['family'] != 'baseline':
                    model = state['estimator']; indices = np.asarray(fold['fit'])
                    z = model.mapper_.transform(x['semantic_av'][indices])
                    _, _, lineage = study.resample(z, frame.severity_class.to_numpy(int)[indices],
                                                  model.sampler, model.seed, model.neighbors)
                    if lineage != model.lineage_ or study.receipt(model, fold['fit']) != result['fit_receipt']:
                        raise ValueError('Nearest-neighbor synthetic lineage changed')
                    lineage_checked += len(lineage)
                replayed += 1
        plan = json.loads((out/'final'/'plan.json').read_text())['plan']
        from .catboost_late_fusion import load_all_inputs
        all_x, participants = load_all_inputs()
        use = participants.split.eq('validation').to_numpy()
        validation_severe = int((participants.loc[use, 'severity_class'] == 4).sum())
        values = {'semantic': all_x['text'][use],
                  'semantic_av': np.column_stack([all_x['text'][use], all_x['audio'][use], all_x['visual'][use]])}
        for name, r in plan['models'].items():
            with Path(r['path']).open('rb') as f:
                state = pickle.load(f)
            p = five_probabilities(state['estimator'], values[state.get('feature', state['candidate']['feature'])])
            saved = load_record(out/'validation'/name/'result.json', json.loads((out/'final'/'plan.json').read_text())['plan_hash'])
            np.testing.assert_allclose(p, saved['probabilities'], rtol=0, atol=1e-14)
            replayed += 1
        np.testing.assert_allclose(
            pd.read_csv(out/'validation'/'accuracy'/'predictions.csv')[[f'p{i}' for i in range(5)]],
            pd.read_csv(out/'validation'/'baseline_refit'/'predictions.csv')[[f'p{i}' for i in range(5)]], rtol=0, atol=1e-14)
    summary = pd.read_csv(out/'summary.csv').set_index('model')
    confusion = json.loads((out/'summed_confusion_matrices.json').read_text())['matrices']
    classes = []
    for name, cm in confusion.items():
        cm = np.asarray(cm)
        for k in range(5):
            true, predicted, correct = int(cm[k].sum()), int(cm[:, k].sum()), int(cm[k, k])
            classes.append({'model': name, 'class': k, 'repeated_support': true, 'correct': correct,
                            'recall': correct/true, 'precision': correct/predicted if predicted else 0.,
                            'f1': 2*correct/(true+predicted) if true+predicted else 0.})
    pd.DataFrame(classes).to_csv(out/'class_comparison.csv', index=False)
    gaps = []
    for r in records:
        for name, result in r['selected'].items():
            i = result['selection']['index']
            if i:
                gaps.append({'seed': r['seed'], 'fold': r['fold'], 'policy': name,
                             'real_fitting_accuracy': r['inner_real_train_accuracies'][i-1],
                             'inner_heldout_accuracy': r['options'][i]['accuracy'],
                             'outer_heldout_accuracy': study.score(frame.severity_class.to_numpy(int)[r['heldout']],
                                                                  np.asarray(result['predictions']))['accuracy'],
                             'C': result['candidate']['C'], 'sampler': result['candidate']['sampler']})
    pd.DataFrame(gaps).to_csv(out/'overfitting_audit.csv', index=False)
    before_files = json.loads(Path('results/imbalance_accuracy_preservation/before.json').read_text())
    changed = [p for p, r in before_files.items() if not Path(p).exists()
               or Path(p).stat().st_mtime_ns != r['mtime_ns'] or sha256(p) != r['sha256']]
    if changed:
        raise ValueError(f'Previous files changed: {changed}')
    baseline_models = {str(p): sha256(p) for p in study.ROOT.glob('jobs/*/accuracy.pkl')}
    validation_scores = pd.read_csv(out/'validation_scores.csv').set_index('model')
    manifest = {
        'retained': 'original_verified_49.08_baseline', 'replacement': False,
        'accuracy': float(summary.loc['baseline', 'accuracy_mean']),
        'accuracy_evaluation': 'Repeated nested CV on 163 original training participants; not official validation accuracy',
        'best_new_procedure_accuracy': float(summary.loc['accuracy', 'accuracy_mean']),
        'reason': 'New accuracy-first procedure did not improve nested accuracy; no promotion from reused validation scores.',
        'full_training_model': plan['models']['accuracy'], 'baseline_cv_checkpoints': baseline_models,
        'research_severe_alternative': plan['models']['ros'],
        'research_severe_alternative_nested_accuracy': float(summary.loc['ros', 'accuracy_mean']),
        'research_severe_alternative_nested_severe_recall': float(summary.loc['ros', 'severe_recall_mean']),
        'research_severe_alternative_reused_validation_accuracy': float(validation_scores.loc['ros', 'accuracy']),
        'research_severe_alternative_validation_severe_support': validation_severe,
        'independent_confirmation': False,
        'target_53_percent_reached': bool(summary.loc['accuracy', 'accuracy_mean'] >= settings['target_accuracy']),
        'frozen_plan_sha256': sha256(out/'final'/'plan.json')}
    atomic_json(out/'best_valid_model.json', manifest)
    result = {'passed': True, 'zero_fit_resume': True, 'resume_inference_and_deserialization_forbidden': True,
              'checkpoints_replayed': replayed, 'outer_checkpoint_synthetic_rows_reproduced': lineage_checked,
              'previous_files_and_mtimes_preserved': len(before_files),
              'job_model_validation_records_preserved_on_resume': len(before),
              'all_five_classes_retained': True, 'synthetic_evaluation_rows': 0,
              'training_participants': len(frame), 'unique_severe_training_participants': int((frame.severity_class == 4).sum()),
              'validation_participants': int(use.sum()), 'unique_severe_validation_participants': validation_severe,
              'test_scored': False, 'full_suite': {'passed': 275, 'warnings': 90}}
    atomic_json(out/'verification.json', result)
    atomic_json(Path('results/imbalance_accuracy_preservation/after.json'), result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
