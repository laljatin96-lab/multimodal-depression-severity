"""Four-parameter, inner-only decision correction of the existing best model."""
import argparse
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .chunk_severe_study import load_inputs
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .evaluate import write_report
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import ROOT
from .severe_detection_study import decide, score, summarize

OUTPUT = Path('results/decision_offset_accuracy')
PROTOCOL = 'inner-only-four-class-decision-offsets-v1'


def adjust(p, offsets):
    p = np.asarray(p, float); offsets = np.asarray(offsets, float)
    decide(p, None)
    if offsets.shape != (5,) or not np.isfinite(offsets).all() or offsets[0] != 0:
        raise ValueError('Four finite offsets relative to class zero required')
    return softmax(np.log(p.clip(1e-15))+offsets, axis=1)


def fit_offsets(p, y, settings):
    p, y = np.asarray(p, float), np.asarray(y)
    decide(p, None)
    if y.shape != (len(p),) or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
        raise ValueError('Real fitting labels in all five classes required')
    logp = np.log(p.clip(1e-15)); bias = np.zeros(5)
    def key(b):
        correct = int(((logp+b).argmax(1) == y).sum())
        norm = float(b @ b)
        return correct-settings['penalty']*norm, -norm
    for _ in range(settings['passes']):
        previous = bias.copy()
        for label in range(1, 5):
            best = bias.copy()
            for value in settings['grid']:
                candidate = bias.copy(); candidate[label] = value
                if key(candidate) > key(best):
                    best = candidate
            bias = best
        if np.array_equal(previous, bias):
            break
    return bias


def predict_state(state, x):
    return adjust(state['estimator'].predict_proba(x[state['feature']]), state['offsets'])


def job(frame, fold, old, settings, directory):
    y = frame.severity_class.to_numpy(int)
    bias = fit_offsets(old['cached_inner_probabilities'], y[fold['fit']], settings)
    source = ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
    with source.open('rb') as f:
        original = pickle.load(f)
    directory.mkdir(parents=True, exist_ok=True); selected = {}
    for name, shrink in settings['policies'].items():
        offsets = bias*shrink
        p = adjust(old['policies']['accuracy']['probabilities'], offsets)
        candidate = {'family': 'decision_offset', 'shrinkage': shrink}
        state = {'protocol': PROTOCOL, 'candidate': candidate, 'estimator': original['estimator'],
                 'feature': old['policies']['accuracy']['candidate']['feature'], 'offsets': offsets,
                 'fit': fold['fit'], 'heldout': fold['heldout'], 'baseline_sha256': sha256(source)}
        path = directory/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
        selected[name] = {'selection': {'offsets': offsets.tolist()}, 'candidate': candidate, 'threshold': None,
                          'probabilities': p.tolist(), 'predictions': p.argmax(1).tolist(),
                          'checkpoint_sha256': sha256(path)}
    return {**{k: fold[k] for k in ['seed', 'fold', 'fit', 'heldout']}, 'raw_offsets': bias.tolist(),
            'selected': selected, 'baseline_models_fitted': 0, 'offset_fits': 1}


def validate(r, fold, old, frame, settings, directory):
    if any(r[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']) or set(r['selected']) != set(settings['policies']):
        raise ValueError('Ownership/policies changed')
    bias = np.asarray(r['raw_offsets'], float)
    if bias.shape != (5,) or bias[0] != 0 or any(v not in settings['grid'] for v in bias):
        raise ValueError('Invalid offsets')
    for name, result in r['selected'].items():
        offsets = bias*settings['policies'][name]
        p = adjust(old['policies']['accuracy']['probabilities'], offsets)
        if result['selection'] != {'offsets': offsets.tolist()} or result['probabilities'] != p.tolist() or result['predictions'] != p.argmax(1).tolist():
            raise ValueError('Saved correction changed')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')


def run(inputs, settings, output=OUTPUT, resume=False):
    x, frame, old_settings, old_records, provenance = inputs; out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing results')
    folds = nested_folds(frame, old_settings)
    if len(folds) != settings['expected_jobs'] or len(folds) != len(old_records):
        raise ValueError('Incomplete folds')
    for fold, old in zip(folds, old_records):
        if any(fold[k] != old[k] for k in ['seed', 'fold', 'fit', 'heldout']) or set(fold['fit']) & set(fold['heldout']):
            raise ValueError('Participant ownership/leakage')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
               'baseline_records_hash': digest(old_records), 'provenance': provenance, 'source_sha256': sha256(__file__)}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}; out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=2), offline_network():
        path = out/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Frozen protocol changed')
        elif resume:
            raise ValueError('Missing resume protocol')
        else:
            atomic_json(path, envelope)
        records = []; new = 0
        for fold, old in zip(folds, old_records):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; path = directory/'result.json'
            if path.exists():
                r = load_record(path, ch)
            else:
                r = job(frame, fold, old, settings, directory); save_record(path, r, ch); new += 1
            validate(r, fold, old, frame, settings, directory); records.append(r)
        targets = summarize(frame, records, old_records, settings, out)
        for p, h in provenance['bindings'].items():
            if sha256(p) != h:
                raise ValueError('Existing input changed')
        result = {'complete': True, 'context_hash': ch, 'new_offset_fits': new, 'offset_fits': len(records),
                  'base_models_refitted': 0, 'targets': targets, 'validation_test_scored': False}
        atomic_json(out/'completion.json', result)
    return result


def final_offsets(frame, old, settings):
    n = len(frame); p = np.zeros((n, 5)); counts = np.zeros(n, int)
    for r in old:
        p[r['fit']] += np.asarray(r['cached_inner_probabilities']); counts[r['fit']] += 1
    if not (counts == counts[0]).all() or not counts[0]:
        raise ValueError('Unequal training probability coverage')
    p /= counts[:, None]
    return fit_offsets(p, frame.severity_class.to_numpy(int), settings), p, counts


def freeze(inputs, settings):
    x, frame, old_settings, old, provenance = inputs
    completion = run(inputs, settings, resume=True)
    directory = OUTPUT/'final'
    if directory.exists():
        raise FileExistsError('Already frozen')
    bias, p, counts = final_offsets(frame, old, settings)
    baseline = json.loads(Path('results/supervised_accuracy/best_valid_model.json').read_text())['best_model']
    if sha256(baseline['path']) != baseline['sha256']:
        raise ValueError('Original full-training model changed')
    baseline_plan = json.loads(Path('results/catboost_holdout/plan.json').read_text())['plan']
    if baseline_plan['training_ids'] != frame.participant_id.tolist():
        raise ValueError('Original training participants changed')
    with Path(baseline['path']).open('rb') as f:
        original = pickle.load(f)
    directory.mkdir(); models = {}
    atomic_json(directory/'training_offsets.json', {'raw_offsets': bias.tolist(), 'participant_ids': frame.participant_id.tolist(),
                'inner_probability_mean': p.tolist(), 'predictions_per_participant': counts.tolist()})
    for name, shrink in settings['policies'].items():
        state = {'protocol': PROTOCOL, 'candidate': {'family': 'decision_offset', 'shrinkage': shrink},
                 'estimator': original['estimator'], 'feature': baseline['candidate']['feature'], 'offsets': bias*shrink,
                 'fit_participants': frame.participant_id.tolist(), 'baseline_sha256': baseline['sha256']}
        path = directory/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
        models[name] = {'path': str(path), 'sha256': sha256(path), 'offsets': state['offsets'].tolist()}
    bindings = {str(p): sha256(p) for p in [Path(__file__), Path('configs/decision_offset_accuracy.json'), OUTPUT/'protocol.json',
                directory/'training_offsets.json', Path('results/semantic_features/embeddings.json'),
                Path('results/temporal_summary_comparison/participant_features.npz'), Path('data/metadata/participants.csv')]}
    plan = {'protocol': PROTOCOL, 'training_ids': frame.participant_id.tolist(), 'models': models,
            'context_hash': completion['context_hash'], 'bindings': bindings, 'base_models_refitted': 0,
            'primary': settings['primary'], 'independent_confirmation': False, 'validation_selected_model': False, 'test_scored': False}
    atomic_json(directory/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return plan


def frozen_plan():
    envelope = json.loads((OUTPUT/'final'/'plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Plan changed')
    for p, h in plan['bindings'].items():
        if sha256(p) != h:
            raise ValueError('Bound final source changed')
    for record in plan['models'].values():
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Final checkpoint changed')
    return plan, envelope['plan_hash']


def evaluate():
    from .supervised_accuracy_finalize import validation_inputs
    plan, ph = frozen_plan(); x, frame = validation_inputs(plan); rows = []
    with output_lock(OUTPUT), threadpool_limits(limits=2), offline_network():
        for name, record in plan['models'].items():
            directory = OUTPUT/'validation'/name; path = directory/'result.json'
            if path.exists():
                r = load_record(path, ph)
                if r['participant_ids'] != frame.participant_id.tolist() or r['model_sha256'] != record['sha256']:
                    raise ValueError('Validation ownership changed')
                p = np.asarray(r['probabilities'])
            else:
                state = pickle.loads(Path(record['path']).read_bytes()); p = predict_state(state, x)
                save_record(path, {'participant_ids': frame.participant_id.tolist(), 'model_sha256': record['sha256'], 'probabilities': p.tolist()}, ph)
            write_report(frame, p, directory, name, 42, 'reused_official_validation')
            rows.append({'model': name, **score(frame.severity_class.to_numpy(int), p.argmax(1))})
        pd.DataFrame(rows).to_csv(OUTPUT/'validation_scores.csv', index=False)
    return rows


def verify(inputs, settings):
    from .supervised_accuracy_finalize import validation_inputs
    x, frame, old_settings, old, provenance = inputs; completion = json.loads((OUTPUT/'completion.json').read_text())
    plan, ph = frozen_plan(); vx, vf = validation_inputs(plan)
    paths = list((OUTPUT/'jobs').glob('*/*'))+list((OUTPUT/'final').glob('*'))+list((OUTPUT/'validation').glob('*/result.json'))
    before = {str(p): [sha256(p), p.stat().st_mtime_ns] for p in paths if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('Completed resume cannot fit or load models')
    with patch('src.decision_offset_accuracy.job', forbidden), patch('src.decision_offset_accuracy.fit_offsets', forbidden), patch.object(pickle, 'load', forbidden), patch.object(pickle, 'loads', forbidden):
        result = run(inputs, settings, resume=True); evaluate()
    if result['new_offset_fits'] or before != {p: [sha256(p), Path(p).stat().st_mtime_ns] for p in before}:
        raise ValueError('Completed resume mutated artifacts')
    replayed = 0
    with threadpool_limits(limits=2), offline_network():
        for fold, original in zip(nested_folds(frame, old_settings), old):
            directory = OUTPUT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            r = load_record(directory/'result.json', completion['context_hash'])
            np.testing.assert_array_equal(fit_offsets(original['cached_inner_probabilities'], frame.severity_class.to_numpy(int)[fold['fit']], settings), r['raw_offsets'])
            for name, selected in r['selected'].items():
                state = pickle.loads((directory/f'{name}.pkl').read_bytes())
                if state['fit'] != fold['fit'] or state['heldout'] != fold['heldout']:
                    raise ValueError('Saved participant ownership changed')
                p = predict_state(state, {k: v[fold['heldout']] for k, v in x.items()})
                np.testing.assert_allclose(p, selected['probabilities'], rtol=0, atol=1e-14); replayed += 1
        for name, record in plan['models'].items():
            state = pickle.loads(Path(record['path']).read_bytes()); p = predict_state(state, vx)
            saved = load_record(OUTPUT/'validation'/name/'result.json', ph)
            np.testing.assert_allclose(p, saved['probabilities'], rtol=0, atol=1e-14); replayed += 1
    bias, p, counts = final_offsets(frame, old, settings)
    if json.loads((OUTPUT/'final'/'training_offsets.json').read_text())['raw_offsets'] != bias.tolist():
        raise ValueError('Final training offsets changed')
    summary = pd.read_csv(OUTPUT/'summary.csv').set_index('model')
    original = json.loads(Path('results/supervised_accuracy/best_valid_model.json').read_text())['best_model']
    improved = bool(summary.loc['accuracy', 'accuracy_mean'] > summary.loc['baseline', 'accuracy_mean'])
    manifest = {'retained': 'decision_offset_exploratory' if improved else 'original_verified_49.08_baseline',
                'replacement': improved, 'original_overwritten': False,
                'baseline_nested_accuracy': float(summary.loc['baseline', 'accuracy_mean']),
                'new_nested_accuracy': float(summary.loc['accuracy', 'accuracy_mean']),
                'best_model': plan['models']['accuracy'] if improved else original,
                'new_research_models': plan['models'], 'selected_using_validation': False, 'independent_confirmation': False,
                'target_53_percent_reached': bool(summary.loc['accuracy', 'accuracy_mean'] >= settings['target_accuracy'])}
    atomic_json(OUTPUT/'best_valid_model.json', manifest)
    prior = json.loads(Path('results/multitask_attention_preservation/before.json').read_text())
    changed = [p for p, v in prior.items() if not Path(p).exists() or Path(p).stat().st_mtime_ns != v['mtime_ns'] or sha256(p) != v['sha256']]
    if changed:
        raise ValueError(f'Pre-existing files changed: {changed}')
    result = {'passed': True, 'checkpoints_replayed': replayed, 'zero_fit_resume': True,
              'previous_files_and_mtimes_preserved': len(prior), 'base_models_refitted': 0, 'offset_fits': 16,
              'validation_participants': len(vf), 'test_scored': False, 'synthetic_rows': 0}
    atomic_json(OUTPUT/'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'freeze', 'evaluate', 'verify']); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); settings = json.loads(Path('configs/decision_offset_accuracy.json').read_text())
    if args.action == 'evaluate':
        result = evaluate()
    else:
        inputs = load_inputs()
        result = run(inputs, settings, resume=args.resume) if args.action == 'run' else globals()[args.action](inputs, settings)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    from .decision_offset_accuracy import main as module_main
    module_main()
