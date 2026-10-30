"""Read-only checkpoint replay, paired error changes and preservation verification."""
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import quadratic_accuracy as s
from .batch_inference import atomic_json, load_record
from .common import digest
from .final_evaluation import sha256
from .raw_frame_uncertainty import participant_counts, weighted_metrics


def inspect_model(model, x, y):
    if model.real_count != len(y) or model.base_pca.n_samples_ != len(y):
        raise ValueError('Model fitting population differs')
    np.testing.assert_array_equal(model.class_counts, np.bincount(y, minlength=5))
    np.testing.assert_allclose(model.base_scaler.mean_, x['semantic_av'].mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.language_scaler.mean_, x['language'].mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.output_scaler.mean_, model.unscaled(x).mean(0), rtol=0, atol=1e-12)
    z = model.transform(x)
    np.testing.assert_allclose(model.means_, np.stack([z[y == c].mean(0) for c in range(5)]), rtol=0, atol=1e-12)
    residuals = z-model.means_[y]
    np.testing.assert_allclose(model.pooled_, residuals.T @ residuals / len(y), rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(model.covariances_).min() > 0


# Reuse the already verified participant-paired analysis, writing only to the new directory.
from .conditional_hierarchy_verify import error_analysis


def matched_language_comparisons(out, frame):
    seeds = [42,137,2026]; y = frame.severity_class.to_numpy(int); predictions = {}
    for name in ['language_control','accuracy','quadratic_only']:
        tables = [pd.read_csv(out/'oof'/f'{name}_seed{seed}/predictions.csv', dtype={'participant_id':str}) for seed in seeds]
        for table, seed in zip(tables, seeds):
            if table.participant_id.tolist() != frame.participant_id.tolist() or not np.array_equal(table.severity_class, y):
                raise ValueError('Language-paired participant alignment differs')
            if name == 'language_control':
                old = pd.read_csv(s.LANGUAGE_ROOT/'oof'/f'without_age_seed{seed}/predictions.csv', dtype={'participant_id':str})
                np.testing.assert_allclose(table[[f'p{i}' for i in range(5)]], old[[f'p{i}' for i in range(5)]], rtol=0, atol=1e-14)
        predictions[name] = np.stack([t.predicted_label.to_numpy(int) for t in tables])
    counts = participant_counts(len(y), 5000, 20261025)
    point_counts = np.ones((1,len(y)),int)
    boot = {name:weighted_metrics(y,p,counts) for name,p in predictions.items()}
    point = {name:weighted_metrics(y,p,point_counts)[0] for name,p in predictions.items()}
    reference = predictions['language_control']; old_good = reference == y
    intervals,changes,classes,people = [],[],[],[]
    for name in ['accuracy','quadratic_only']:
        p=predictions[name]; good=p==y; gained=good & ~old_good; lost=old_good & ~good
        limits=np.percentile(boot[name]-boot['language_control'],[2.5,97.5],axis=0)
        for i,metric in enumerate(['accuracy','macro_f1','balanced_accuracy']):
            intervals.append({'model':name,'reference':'language_control','metric':metric,
                'difference':float(point[name][i]-point['language_control'][i]),'ci95_low':float(limits[0,i]),'ci95_high':float(limits[1,i])})
        changes.append({'model':name,'reference':'language_control','gained':int(gained.sum()),'lost':int(lost.sum()),
            'correct':int(good.sum()),'reference_correct':int(old_good.sum()),'net_better_people':int((good.sum(0)>old_good.sum(0)).sum()),
            'net_worse_people':int((good.sum(0)<old_good.sum(0)).sum())})
        for c in range(5):
            mask=y==c
            classes.append({'model':name,'class':c,'participants':int(mask.sum()),'reference_correct':int(old_good[:,mask].sum()),
                'correct':int(good[:,mask].sum()),'reference_recall':float(old_good[:,mask].mean()),'recall':float(good[:,mask].mean()),
                'gained':int(gained[:,mask].sum()),'lost':int(lost[:,mask].sum())})
        for i,pid in enumerate(frame.participant_id):
            people.append({'model':name,'participant_id':pid,'class':int(y[i]),'reference_correct_repeats':int(old_good[:,i].sum()),
                'correct_repeats':int(good[:,i].sum()),'gained':int(gained[:,i].sum()),'lost':int(lost[:,i].sum())})
    for name,rows in [('language_paired_comparisons',intervals),('language_error_changes',changes),
                       ('language_per_class',classes),('language_participant_changes',people)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    np.savez_compressed(out/'language_paired_bootstrap.npz',participant_counts=counts,
        **{name:boot[name]-boot['language_control'] for name in ['accuracy','quadratic_only']})


def verify():
    out = s.OUTPUT
    if (out/'verification.json').exists():
        raise FileExistsError('Preserving completed verification')
    inputs = s.load_inputs(); x, frame, old_settings, old, _ = inputs
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
                if state['candidate']['family'] == 'regularized_quadratic':
                    inspect_model(state['estimator'], s.subset(x, fold['fit']), y[fold['fit']])
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
        from .quadratic_inference import frozen_model, predict_arrays
        for name, record in plan['models'].items():
            checked, state, _ = frozen_model(name)
            if state['fit_participants'] != frame.participant_id.tolist():
                raise ValueError('Final ownership differs')
            if not record['existing_language_control']:
                inspect_model(state['estimator'], x, y)
                expected = s.predict_state(state, x)
            else:
                expected = s.predict_state(state, x)
            np.testing.assert_allclose(predict_arrays(checked, state, x), expected, rtol=0, atol=1e-14)
            replayed += 1
    pd.DataFrame(rows).to_csv(out/'overfitting_audit.csv', index=False)
    totals = error_analysis(out, frame, list(settings['policies']))
    summary = pd.read_csv(out/'summary.csv').set_index('model')
    previous = plan['models']['language_control']
    primary = float(summary.loc['accuracy', 'accuracy_mean']); baseline = float(summary.loc['language_control', 'accuracy_mean'])
    matched_language_comparisons(out, frame)
    intervals = pd.read_csv(out/'language_paired_comparisons.csv')
    ci = intervals[(intervals.model == 'accuracy') & (intervals.metric == 'accuracy')].iloc[0]
    manifest = {'preserved_language_accuracy': baseline, 'primary_accuracy': primary,
        'primary_observed_gain': primary > baseline, 'primary_pointwise_interval_excludes_zero_above': bool(ci.ci95_low > 0),
        'independent_confirmation': False, 'retained_best_observed_language_model': previous,
        'candidate_models': plan['models'], 'target_reached': primary >= settings['target_accuracy'],
        'original_overwritten': False, 'validation_or_test_used_for_selection': False}
    atomic_json(out/'best_valid_model.json', manifest)
    previous_files = json.loads(Path('results/quadratic_accuracy_preservation/before.json').read_text())
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
