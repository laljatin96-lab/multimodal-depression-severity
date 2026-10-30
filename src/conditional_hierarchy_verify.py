"""Read-only checkpoint replay, paired error changes and preservation verification."""
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import conditional_hierarchy_accuracy as s
from .batch_inference import atomic_json, load_record
from .common import digest
from .final_evaluation import sha256
from .raw_frame_uncertainty import participant_counts, weighted_metrics


def inspect_model(model, x, y):
    if model.real_count != len(y):
        raise ValueError('Model fitting population differs')
    np.testing.assert_array_equal(model.class_counts, np.bincount(y, minlength=5))
    boundary = model.candidate['boundary']
    nodes = [(model.gate, np.arange(len(y)), (y > boundary).astype(int))]
    nodes += [(h, np.flatnonzero((y > boundary) == g), y[(y > boundary) == g]) for g, h in model.heads.items() if h is not None]
    for node, positions, labels in nodes:
        np.testing.assert_array_equal(node.positions, positions)
        np.testing.assert_array_equal(node.classes_, np.unique(labels))
        np.testing.assert_allclose(node.scaler.mean_, x[positions].mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(node.pca.mean_, node.scaler.transform(x[positions]).mean(0), rtol=0, atol=1e-12)
        if node.real_count != len(positions) or node.pca.n_samples_ != len(positions):
            raise ValueError('Conditional preprocessing ownership differs')


def error_analysis(out, frame, models):
    seeds = [42, 137, 2026]; y = frame.severity_class.to_numpy(int)
    predictions = {}; probabilities = {}
    for name in ['baseline']+models:
        tables = [pd.read_csv(out/'oof'/f'{name}_seed{seed}/predictions.csv', dtype={'participant_id': str}) for seed in seeds]
        for table in tables:
            if table.participant_id.tolist() != frame.participant_id.tolist() or not np.array_equal(table.severity_class, y):
                raise ValueError('Paired participant alignment differs')
        predictions[name] = np.stack([t.predicted_label.to_numpy(int) for t in tables])
        probabilities[name] = np.stack([t[[f'p{i}' for i in range(5)]].to_numpy() for t in tables])
    counts = participant_counts(len(y), 5000, 20261025)
    boot = {n: weighted_metrics(y, p, counts) for n, p in predictions.items()}
    np.savez_compressed(out/'paired_bootstrap.npz', participant_counts=counts,
        **{n: value-boot['baseline'] for n, value in boot.items() if n != 'baseline'})
    transitions, people, classes, totals, opportunities = [], [], [], [], []
    baseline = predictions['baseline']; good_base = baseline == y
    for name in models:
        pred = predictions[name]; good = pred == y; gained = good & ~good_base; lost = good_base & ~good
        totals.append({'model': name, 'correct': int(good.sum()), 'gained': int(gained.sum()), 'lost': int(lost.sum()),
            'unique_people_with_gain': int(gained.any(0).sum()), 'unique_people_with_loss': int(lost.any(0).sum()),
            'net_improved_people': int((good.sum(0) > good_base.sum(0)).sum()),
            'net_worsened_people': int((good.sum(0) < good_base.sum(0)).sum()),
            'majority_predictions': int((pred == 0).sum()), 'baseline_majority_predictions': int((baseline == 0).sum())})
        for c in range(5):
            mask = y == c
            classes.append({'model': name, 'class': c, 'unique_participants': int(mask.sum()),
                'baseline_correct': int(good_base[:, mask].sum()), 'candidate_correct': int(good[:, mask].sum()),
                'baseline_recall': float(good_base[:, mask].mean()), 'candidate_recall': float(good[:, mask].mean()),
                'gained': int(gained[:, mask].sum()), 'lost': int(lost[:, mask].sum())})
        for i, pid in enumerate(frame.participant_id):
            people.append({'model': name, 'participant_id': pid, 'class': int(y[i]),
                'baseline_correct_repeats': int(good_base[:, i].sum()), 'candidate_correct_repeats': int(good[:, i].sum())})
            for j, seed in enumerate(seeds):
                transitions.append({'model': name, 'participant_id': pid, 'class': int(y[i]), 'seed': seed,
                    'baseline': int(baseline[j, i]), 'candidate': int(pred[j, i]), 'gained': bool(gained[j, i]), 'lost': bool(lost[j, i])})
    for j, seed in enumerate(seeds):
        ranks = np.argsort(-probabilities['baseline'][j], axis=1)
        for i, pid in enumerate(frame.participant_id):
            if y[i] in [1, 2] and not good_base[j, i]:
                opportunities.append({'participant_id': pid, 'seed': seed, 'class': int(y[i]),
                    'baseline_prediction': int(baseline[j, i]), 'true_class_rank': int(np.flatnonzero(ranks[i] == y[i])[0])+1,
                    'baseline_true_class_probability': float(probabilities['baseline'][j, i, y[i]]),
                    **{f'{name}_prediction': int(predictions[name][j, i]) for name in models}})
    for name, rows in [('paired_error_changes', transitions), ('participant_net_changes', people),
                       ('per_class_comparison', classes), ('error_change_totals', totals), ('targeted_mild_moderate_errors', opportunities)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False)
    # Independently verify the stored pointwise intervals without refitting anything.
    saved = pd.read_csv(out/'paired_comparisons.csv')
    for row in saved.itertuples():
        metric = ['accuracy', 'macro_f1', 'balanced_accuracy'].index(row.metric)
        expected = np.percentile(boot[row.model][:, metric]-boot['baseline'][:, metric], [2.5, 97.5])
        np.testing.assert_allclose([row.ci95_low, row.ci95_high], expected, rtol=0, atol=1e-14)
    return totals


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
        from .research_accuracy_inference import frozen_model, predict_arrays
        for name, record in plan['models'].items():
            checked, state, _ = frozen_model('hierarchy', name)
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
        'candidate_models': plan['models'], 'target_reached': primary >= .56,
        'original_overwritten': False, 'validation_or_test_used_for_selection': False}
    atomic_json(out/'best_valid_model.json', manifest)
    previous_files = json.loads(Path('results/conditional_hierarchy_preservation/before.json').read_text())
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
