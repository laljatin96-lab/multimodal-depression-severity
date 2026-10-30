"""Read-only checkpoint replay, paired error changes and preservation verification."""
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import robust_av_accuracy as s
from .batch_inference import atomic_json, load_record
from .common import digest
from .final_evaluation import sha256
from .raw_frame_uncertainty import participant_counts, weighted_metrics


def inspect_model(model, x, y):
    if model.real_count != len(y) or model.transformer.real_count != len(y) or model.pca.n_samples_ != len(y):
        raise ValueError('Model fitting population differs')
    np.testing.assert_array_equal(model.class_counts, np.bincount(y, minlength=5))
    transform = model.transformer; av = x[:, 384:]
    if transform.mode == 'rank_gaussian':
        expected = np.nanpercentile(av, transform.quantiles.references_*100, axis=0)
        np.testing.assert_allclose(transform.quantiles.quantiles_, np.maximum.accumulate(expected), rtol=0, atol=1e-10)
        if transform.endpoint != .5/len(y):
            raise ValueError('Rank endpoint differs')
    else:
        np.testing.assert_allclose(transform.median, np.median(av, axis=0), rtol=0, atol=1e-12)
        q = np.quantile(av, [.25, .75], axis=0); iqr = q[1]-q[0]; std = av.std(axis=0)
        expected = np.where(iqr > 1e-12, iqr, np.where(std > 1e-12, std, 1.))
        np.testing.assert_allclose(transform.scale, expected, rtol=0, atol=1e-12)
    transformed = transform.transform(x)
    np.testing.assert_array_equal(transformed[:, :384], x[:, :384])
    np.testing.assert_allclose(model.scaler.mean_, transformed.mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.pca.mean_, model.scaler.transform(transformed).mean(0), rtol=0, atol=1e-12)


# Reuse the already verified participant-paired analysis, writing only to the new directory.
from .conditional_hierarchy_verify import error_analysis


def verify():
    out = s.OUTPUT
    if (out/'verification.json').exists():
        raise FileExistsError('Preserving completed verification')
    inputs = s.load_baseline(); x, frame, old_settings, old, _ = inputs
    settings = json.loads(s.CONFIG.read_text()); y = frame.severity_class.to_numpy(int)
    completion = json.loads((out/'completion.json').read_text())
    envelope = json.loads((out/'final/plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash'] or plan['context_hash'] != completion['context_hash']:
        raise ValueError('Final plan differs')
    for path, expected in plan['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Final binding changed')
    before = {str(p): [sha256(p), p.stat().st_mtime_ns] for p in out.rglob('*') if p.is_file() and p.name != '.batch.lock'}
    def forbidden(*args, **kwargs):
        raise AssertionError('No refitting or inference on completed resume')
    with patch.object(s, 'job', forbidden), patch.object(s, 'predict_state', forbidden):
        assert s.run(inputs, settings, resume=True)['new_fits'] == 0
    if before != {p: [sha256(p), Path(p).stat().st_mtime_ns] for p in before}:
        raise ValueError('Resume rewrote evidence')
    records = []; rows = []; replayed = 0
    with threadpool_limits(limits=2), s.offline_network():
        for fold, baseline in zip(s.nested_folds(frame, old_settings), old):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            r = load_record(directory/'result.json', completion['context_hash'])
            s.validate(r, fold, baseline, y, settings, directory); records.append(r)
            for name, selected in r['selected'].items():
                state = pickle.loads((directory/f'{name}.pkl').read_bytes())
                if state['fit'] != fold['fit'] or state['heldout'] != fold['heldout']:
                    raise ValueError('Checkpoint ownership differs')
                if state['candidate']['family'] != 'baseline':
                    inspect_model(state['estimator'], x['semantic_av'][fold['fit']], y[fold['fit']])
                p = s.predict_state(state, {k: v[fold['heldout']] for k, v in x.items()})
                np.testing.assert_allclose(p, selected['probabilities'], rtol=0, atol=1e-14); replayed += 1
                fit_acc = float((s.predict_state(state, {k: v[fold['fit']] for k, v in x.items()}).argmax(1) == y[fold['fit']]).mean())
                if fit_acc != selected['real_training_accuracy']:
                    raise ValueError('Training accuracy differs')
                rows.append({'policy': name, 'seed': fold['seed'], 'fold': fold['fold'], 'training_accuracy': fit_acc,
                    'inner_accuracy': r['options'][selected['selection']['index']]['accuracy'],
                    'outer_accuracy': s.score(y[fold['heldout']], p.argmax(1))['accuracy']})
        selection = s.training_choices(frame, records, old, settings)
        if selection != json.loads((out/'final/training_selection.json').read_text()):
            raise ValueError('Final training selection changed')
        from .robust_av_inference import frozen_model, predict_arrays
        for name, record in plan['models'].items():
            checked, state, _ = frozen_model(name)
            if state['fit_participants'] != frame.participant_id.tolist():
                raise ValueError('Final ownership differs')
            if not record['existing_baseline']:
                inspect_model(state['estimator'], x['semantic_av'], y)
                expected = s.predict_state(state, x)
            else:
                expected = state['estimator'].predict_proba(x[state['candidate']['feature']])
            np.testing.assert_allclose(predict_arrays(checked, state, x['semantic'], x['semantic_av'][:, 384:]), expected, rtol=0, atol=1e-14)
            replayed += 1
    pd.DataFrame(rows).to_csv(out/'overfitting_audit.csv', index=False)
    totals = error_analysis(out, frame, list(settings['policies']))
    summary = pd.read_csv(out/'summary.csv').set_index('model')
    previous = json.loads(Path('results/project_accuracy_audit_20261027/best_model.json').read_text())['best_model']
    primary = float(summary.loc['accuracy', 'accuracy_mean']); baseline = float(summary.loc['baseline', 'accuracy_mean'])
    intervals = pd.read_csv(out/'paired_comparisons.csv')
    ci = intervals[(intervals.model == 'accuracy') & (intervals.metric == 'accuracy')].iloc[0]
    manifest = {'baseline_accuracy': baseline, 'primary_accuracy': primary,
        'primary_observed_gain': primary > baseline, 'primary_pointwise_interval_excludes_zero_above': bool(ci.ci95_low > 0),
        'independent_confirmation': False, 'retained_verified_baseline': previous,
        'candidate_models': plan['models'], 'target_reached': primary >= settings['target_accuracy'],
        'original_overwritten': False, 'validation_or_test_used_for_selection': False}
    atomic_json(out/'best_valid_model.json', manifest)
    previous_files = json.loads(Path('results/robust_av_preservation/before.json').read_text())
    changed = [p for p, v in previous_files.items() if not Path(p).is_file() or Path(p).stat().st_mtime_ns != v['mtime_ns'] or sha256(p) != v['sha256']]
    if changed:
        raise ValueError(f'Previously existing files changed: {changed[:10]}')
    receipt = {'passed': True, 'checkpoints_replayed': replayed, 'zero_fit_zero_rewrite_resume': True,
        'previous_files_and_mtimes_preserved': len(previous_files), 'new_nested_pipeline_fits': completion['total_fits'],
        'new_full_training_pipeline_fits': plan['new_full_training_fits'], 'old_models_refitted': 0,
        'bootstrap_repetitions': 5000, 'bootstrap_unit': 'participant with all three repeated predictions',
        'validation_test_scored': False, 'error_changes': totals}
    atomic_json(out/'verification.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
