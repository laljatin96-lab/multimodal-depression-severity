"""Fixed top-K averaging of the verified baseline family using inner-only ranking."""
import argparse
import json
from pathlib import Path
import pickle
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .evaluate import write_report
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline
from .semantic_severe_repair import estimator_for
from .severe_detection_study import score, summarize, decide

PROTOCOL = 'fixed-top-five-baseline-probability-average-v1'
OUTPUT = Path('results/baseline_averaging_accuracy')


def rank(rows):
    eligible = [i for i, r in enumerate(rows) if r['candidate']['family'] == 'logistic']
    return sorted(eligible, key=lambda i: (-rows[i]['correct'], -rows[i]['macro_f1'], i))


def predict_state(state, x, indices=None):
    if indices is None:
        indices = np.arange(len(next(iter(x.values()))))
    values = [five_probabilities(m['estimator'], x[m['candidate']['feature']][indices]) for m in state['members']]
    p = np.mean(values, axis=0); decide(p, None)
    return p


def fit_members(x, y, indices, rows, chosen, old_settings):
    members = []
    for i in chosen:
        c = rows[i]['candidate']; model = estimator_for(c, old_settings)
        fit(model, x[c['feature']][indices], y[indices])
        members.append({'estimator': model, 'candidate': c, 'candidate_index': i})
    return members


def save_checkpoint(path, state, x, indices):
    p = predict_state(state, x, indices)
    with path.open('wb') as f:
        pickle.dump(state, f)
    with path.open('rb') as f:
        restored = pickle.load(f)
    np.testing.assert_allclose(p, predict_state(restored, x, indices), rtol=0, atol=1e-14)
    return p


def job(x, frame, fold, old, old_settings, settings, directory):
    start = time.monotonic(); y = frame.severity_class.to_numpy(int)
    ranking = rank(old['candidates']); maximum = max(settings['policies'].values())
    if len(ranking) < maximum:
        raise ValueError('Insufficient candidate ensemble members')
    members = fit_members(x, y, fold['fit'], old['candidates'], ranking[:maximum], old_settings)
    directory.mkdir(parents=True, exist_ok=True); selected = {}
    for name, k in settings['policies'].items():
        state = {'protocol': PROTOCOL, 'members': members[:k], 'fit': fold['fit'], 'heldout': fold['heldout']}
        path = directory/f'{name}.pkl'; p = save_checkpoint(path, state, x, fold['heldout'])
        selected[name] = {'selection': {'member_indices': ranking[:k]}, 'candidate': {'family': 'uniform_average', 'members': k},
                          'threshold': None, 'probabilities': p.tolist(), 'predictions': p.argmax(1).tolist(),
                          'checkpoint_sha256': sha256(path),
                          'real_training_accuracy': float((predict_state(state, x, fold['fit']).argmax(1) == y[fold['fit']]).mean())}
    return {**{k: fold[k] for k in ['seed', 'fold', 'fit', 'heldout']}, 'ranking': ranking,
            'selected': selected, 'fits': maximum, 'seconds': time.monotonic()-start}


def validate(r, fold, old, settings, directory):
    if any(r[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']) or r['ranking'] != rank(old['candidates']):
        raise ValueError('Fold or inner ranking changed')
    if set(r['selected']) != set(settings['policies']) or r['fits'] != max(settings['policies'].values()):
        raise ValueError('Ensemble policy or fit budget changed')
    for name, result in r['selected'].items():
        k = settings['policies'][name]
        if result['selection'] != {'member_indices': r['ranking'][:k]} or result['candidate'] != {'family': 'uniform_average', 'members': k}:
            raise ValueError('Ensemble membership changed')
        p = np.asarray(result['probabilities'])
        if p.shape != (len(fold['heldout']), 5) or decide(p, None).tolist() != result['predictions']:
            raise ValueError('Invalid held-out predictions')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')


def run(inputs, settings, output=OUTPUT, resume=False):
    x, frame, old_settings, old_records, provenance = inputs; out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing experiment')
    folds = nested_folds(frame, old_settings)
    if (len(folds) != settings['expected_jobs'] or len(old_records) != len(folds) or frame.participant_id.duplicated().any()
            or settings['primary'] not in settings['policies']
            or any(not isinstance(k, int) or k < 1 for k in settings['policies'].values())):
        raise ValueError('Invalid participant coverage')
    for f, old in zip(folds, old_records):
        if any(f[k] != old[k] for k in ['seed', 'fold', 'fit', 'heldout']) or set(f['fit']) & set(f['heldout']):
            raise ValueError('Baseline fold mismatch/leakage')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
               'old_records_hash': digest(old_records), 'provenance': provenance, 'old_settings': old_settings,
               'features': {n: digest(v.tolist()) for n,v in x.items()}, 'code_sha256': sha256(__file__),
               'support_sha256': {n: sha256(Path(__file__).with_name(n)) for n in
                  ['semantic_severe_repair.py','severe_case_diagnosis.py','severe_detection_study.py','evaluate.py','raw_frame_uncertainty.py']}}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        path = out/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Inputs/code changed')
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
                r = job(x, frame, fold, old, old_settings, settings, directory); save_record(path, r, ch); new += 1
            validate(r, fold, old, settings, directory); records.append(r)
            print(f'Baseline averaging: {len(records)}/{len(folds)} jobs; {new} new', flush=True)
        targets = summarize(frame, records, old_records, settings, out)
        for p, h in provenance.get('bindings', {}).items():
            if sha256(p) != h:
                raise ValueError('Original baseline changed')
        result = {'complete': True, 'context_hash': ch, 'new_jobs': new, 'new_fits': new*max(settings['policies'].values()),
                  'fits': sum(r['fits'] for r in records), 'targets': targets,
                  'primary': settings['primary'], 'validation_test_scored': False}
        atomic_json(out/'completion.json', result)
    return result


def freeze(inputs, settings):
    x, frame, old_settings, old_records, _ = inputs
    completion = run(inputs, settings, resume=True)
    if (OUTPUT/'final').exists():
        raise FileExistsError('Already frozen')
    y = frame.severity_class.to_numpy(int); labels = np.concatenate([y[r['fit']] for r in old_records])
    owners = np.zeros(len(y), int)
    for r in old_records:
        owners[r['fit']] += 1
    if not (owners == owners[0]).all():
        raise ValueError('Unequal training selection contributions')
    rows = [{'candidate': c['candidate'], **score(labels, np.concatenate([r['candidates'][i]['predictions'] for r in old_records]))}
            for i,c in enumerate(old_records[0]['candidates'])]
    ranking = rank(rows); directory = OUTPUT/'final'; directory.mkdir()
    with output_lock(OUTPUT), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        atomic_json(directory/'training_selection.json', {'rows': rows, 'ranking': ranking,
                    'note': 'Training-only tuning scores; not evaluation estimates.'})
        members = fit_members(x, y, list(range(len(y))), rows, ranking[:max(settings['policies'].values())], old_settings)
        models = {}
        for name, k in settings['policies'].items():
            state = {'protocol': PROTOCOL, 'members': members[:k], 'fit_participants': frame.participant_id.tolist()}
            path = directory/f'{name}.pkl'; save_checkpoint(path, state, x, None)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'member_indices': ranking[:k]}
        bindings = {str(p): sha256(p) for p in [Path(__file__), OUTPUT/'protocol.json', directory/'training_selection.json',
                    Path('configs/baseline_averaging_accuracy.json'), Path('results/semantic_features/embeddings.json'),
                    Path('results/temporal_summary_comparison/participant_features.npz')]}
        plan = {'protocol': PROTOCOL, 'models': models, 'bindings': bindings, 'training_ids': frame.participant_id.tolist(),
                'primary': settings['primary'], 'new_full_training_fits': len(members),
                'validation_previously_exposed': True, 'validation_may_select_K': False, 'test_scoring': False,
                'context_hash': completion['context_hash']}
        atomic_json(directory/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return {'frozen': True, 'models': models}


def evaluate():
    from .supervised_accuracy_finalize import validation_inputs
    envelope = json.loads((OUTPUT/'final'/'plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Frozen plan changed')
    for p,h in plan['bindings'].items():
        if sha256(p) != h:
            raise ValueError('Frozen input changed')
    x, frame = validation_inputs(plan); rows = []
    with output_lock(OUTPUT), threadpool_limits(limits=2), offline_network():
        for name,r in plan['models'].items():
            if sha256(r['path']) != r['sha256']:
                raise ValueError('Frozen model changed')
            directory = OUTPUT/'validation'/name; path = directory/'result.json'
            if path.exists():
                saved = load_record(path, envelope['plan_hash'])
                if saved['participant_ids'] != frame.participant_id.tolist():
                    raise ValueError('Validation ownership changed')
                p = np.asarray(saved['probabilities'])
            else:
                with Path(r['path']).open('rb') as f:
                    state = pickle.load(f)
                p = predict_state(state, x); directory.mkdir(parents=True)
                save_record(path, {'participant_ids': frame.participant_id.tolist(), 'probabilities': p.tolist()}, envelope['plan_hash'])
            decide(p, None)
            write_report(frame, p, directory, name, 42, 'reused_official_validation')
            rows.append({'model': name, **score(frame.severity_class.to_numpy(int), decide(p, None))})
        pd.DataFrame(rows).to_csv(OUTPUT/'validation_scores.csv', index=False)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', action='store_true'); p.add_argument('--freeze', action='store_true'); p.add_argument('--evaluate', action='store_true')
    args = p.parse_args(); settings = json.loads(Path('configs/baseline_averaging_accuracy.json').read_text())
    result = evaluate() if args.evaluate else (freeze(load_baseline(), settings) if args.freeze else run(load_baseline(), settings, resume=args.resume))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
