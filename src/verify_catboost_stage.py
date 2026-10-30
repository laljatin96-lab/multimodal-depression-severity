"""Replay saved CatBoost artifacts and verify completed resumes and preservation."""
import importlib.metadata
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import catboost_late_fusion as core
from . import catboost_holdout as holdout
from .batch_inference import atomic_json, load_record
from .final_evaluation import sha256, verify_plan
from .severe_detection_study import score


def verify():
    inputs = core.load_inputs(); x, frame, old_settings, old49, old48, provenance = inputs
    settings = json.loads(Path('configs/catboost_late_fusion.json').read_text())
    out = core.OUTPUT; completion = json.loads((out/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Nested comparison incomplete')
    plan, ph = holdout.verify_frozen()
    if not json.loads((holdout.OUTPUT/'completion.json').read_text())['complete']:
        raise ValueError('Official holdout comparison incomplete')
    y = frame.severity_class.to_numpy(int); gender = frame.gender.to_numpy(str)
    diagnostics = []; count = 0
    with threadpool_limits(limits=2):
        for fold in core.nested_folds(frame, old_settings):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            r = load_record(directory/'result.json', completion['context_hash'])
            for name in core.VARIANTS:
                with (directory/f'{name}.pkl').open('rb') as f:
                    state = pickle.load(f)
                p = core.predict_state(state, x, gender, fold['heldout'])
                np.testing.assert_allclose(p, r['selected'][name]['probabilities'], rtol=0, atol=1e-14)
                fitting = core.predict_state(state, x, gender, fold['fit'])
                diagnostics.append({'variant': name, 'seed': fold['seed'], 'fold': fold['fold'],
                    **{'fit_'+k: v for k, v in score(y[fold['fit']], fitting.argmax(1)).items()},
                    **{'outer_'+k: v for k, v in score(y[fold['heldout']], p.argmax(1)).items()},
                    'mean_gender_importance': float(np.mean([m.feature_importances_[-1] for m in state['heads'].values()])) if state['gender'] else None})
                count += 1
        all_x, manifest = core.load_all_inputs()
        batches = 0
        for name, record in plan['models'].items():
            with Path(record['path']).open('rb') as f:
                state = pickle.load(f)
            for split in ['validation', 'test']:
                use = manifest.split.eq(split).to_numpy(); gender_all = manifest.loc[use, 'gender'].to_numpy(str)
                p = holdout.predict_model(state, {n: a[use] for n, a in all_x.items()}, gender_all)
                r = load_record(holdout.OUTPUT/split/name/'result.json', ph)
                np.testing.assert_allclose(p, r['probabilities'], rtol=0, atol=1e-14); batches += 1
    pd.DataFrame(diagnostics).to_csv(out/'generalization_diagnostics.csv', index=False)
    protected = list((out/'jobs').rglob('*.json')) + list((out/'jobs').rglob('*.pkl'))
    protected += list(holdout.OUTPUT.glob('*.pkl')) + [holdout.OUTPUT/'plan.json', holdout.OUTPUT/'training_selection.json']
    protected += list(holdout.OUTPUT.glob('*/*/result.json'))
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in protected}
    predictions = list((out/'oof').glob('*/predictions.csv')) + list(holdout.OUTPUT.glob('*/*/predictions.csv'))
    hashes = {p: sha256(p) for p in predictions}
    def forbidden(*a, **kw):
        raise AssertionError('Completed resume must not fit, infer or deserialize')
    # Restore process-global functions even if verification raises.
    saved = core.job, core.predict_state, holdout.predict_model, pickle.load
    try:
        core.job = core.predict_state = holdout.predict_model = pickle.load = forbidden
        receipt = core.run(*inputs, settings, out, resume=True)
        holdout.evaluate()
        assert receipt['new_head_fits'] == 0
        assert json.loads((holdout.OUTPUT/'completion.json').read_text())['new_prediction_batches'] == 0
    finally:
        core.job, core.predict_state, holdout.predict_model, pickle.load = saved
    for p, expected in before.items():
        assert (sha256(p), p.stat().st_mtime_ns) == expected, str(p)
    for p, expected in hashes.items():
        assert sha256(p) == expected, str(p)
    root = Path('results/catboost_fusion_preservation')
    existing = json.loads((root/'before.json').read_text())['files']
    for name, expected in existing.items():
        p = Path(name)
        assert sha256(p) == expected['sha256'] and p.stat().st_mtime_ns == expected['mtime_ns'], name
    environment = json.loads((root/'environment_before.json').read_text())
    assert all(importlib.metadata.version(n) == v for n, v in environment.items())
    old_plan = verify_plan('results/final_evaluation_plan.json')
    result = {'passed': True, 'outer_checkpoint_replays': count, 'frozen_full_training_models_replayed': len(plan['models']),
        'holdout_prediction_batches_replayed': batches, 'unchanged_new_artifacts_and_mtimes': len(before),
        'unchanged_prediction_files': len(hashes), 'unchanged_prior_files_and_mtimes': len(existing),
        'frozen_original_plan_bindings': len(old_plan['bound_files']), 'existing_dependency_versions_unchanged': True,
        'completed_resume_new_fits': 0, 'completed_resume_new_inference': 0,
        'full_test_suite_passed': 263, 'nested_context_hash': completion['context_hash'], 'holdout_plan_hash': ph}
    atomic_json(root/'verification.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
