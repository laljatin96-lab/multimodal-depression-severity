"""Freeze training-selected CatBoost/baseline refits, then score reused official holdouts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import catboost_late_fusion as core
from . import semantic_accuracy_target as old48_module
from . import semantic_severe_repair as old49_module
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .evaluate import write_report, metrics
from .final_evaluation import sha256
from .severe_detection_study import score

PROTOCOL = 'frozen-catboost-reused-official-holdouts-v1'
OUTPUT = Path('results/catboost_holdout')


def pooled_training_choices(frame, records, old49, old48, settings):
    """Only cached inner predictions choose final refits; no outer predictions are read."""
    y = frame.severity_class.to_numpy(int)
    ownership = np.zeros(len(y), int)
    labels = []
    for r in records:
        ownership[r['fit']] += 1; labels.extend(y[r['fit']])
    if not (ownership == ownership[0]).all() or not ownership[0]:
        raise ValueError('Unequal training participant tuning contributions')
    labels = np.asarray(labels)
    selections, all_options = {}, {}
    for name in core.VARIANTS:
        p = np.concatenate([r['inner_probabilities'][name] for r in records], axis=2)
        rows = core.options(p, labels, settings); i = core.choose(rows)
        selections[name] = rows[i]; all_options[name] = rows
    for name, source in [('baseline49_refit', old49), ('baseline48_refit', old48)]:
        if len(source) != len(records) or any(r['fit'] != s['fit'] for r, s in zip(records, source)):
            raise ValueError('Baseline tuning participants differ')
        candidates = [v['candidate'] for v in source[0]['candidates']]
        if any([v['candidate'] for v in r['candidates']] != candidates for r in source):
            raise ValueError('Baseline candidate order changed')
        rows = []
        for i, candidate in enumerate(candidates):
            predictions = np.concatenate([r['candidates'][i]['predictions'] for r in source])
            rows.append({'candidate': candidate, **score(labels, predictions)})
        selections[name] = rows[core.choose(rows)]; all_options[name] = rows
    names = list(core.VARIANTS)
    primary = max(names, key=lambda n: (selections[n]['correct'], selections[n]['macro_f1'], -names.index(n)))
    return {'choices': selections, 'options': all_options, 'primary_catboost': primary,
            'inner_predictions_per_training_participant': int(ownership[0]),
            'note': 'Training-only selection estimates are optimistic; not performance claims.'}


def baseline_matrix(x, feature):
    if feature == 'semantic':
        return x['text']
    if feature == 'semantic_av':
        return np.column_stack([x['text'], x['audio'], x['visual']])
    raise ValueError('Unknown baseline feature schema')


def predict_model(state, x, gender):
    if state['family'] == 'catboost':
        return core.predict_state(state, x, gender, list(range(len(gender))))
    return five_probabilities(state['estimator'], baseline_matrix(x, state['candidate']['feature']))


def freeze(output=OUTPUT):
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserving frozen refits; use evaluate or report for saved artifacts')
    settings = json.loads(Path('configs/catboost_late_fusion.json').read_text())
    inputs = core.load_inputs()
    x, frame, old_settings, old49, old48, provenance = inputs
    receipt = core.run(*inputs, settings, core.OUTPUT, resume=True)
    if not receipt['complete']:
        raise ValueError('Complete nested study required before refitting')
    records = [load_record(core.OUTPUT/'jobs'/f"seed{f['seed']}_fold{f['fold']}"/'result.json', receipt['context_hash'])
               for f in core.nested_folds(frame, old_settings)]
    selection = pooled_training_choices(frame, records, old49, old48, settings)
    y = frame.severity_class.to_numpy(int); gender = frame.gender.to_numpy(str); indices = list(range(len(y)))
    source_bindings = {str(p): sha256(p) for p in (core.OUTPUT/'jobs').rglob('*') if p.is_file()}
    source_bindings.update(provenance['bindings'])
    source_bindings.update({str(p): sha256(p) for p in [Path(__file__), Path(core.__file__),
        Path('configs/catboost_late_fusion.json'), core.OUTPUT/'protocol.json']})
    out.mkdir(parents=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        atomic_json(out/'training_selection.json', selection)
        mapper = core.FeatureMap(settings['text_components']).fit(x, indices)
        z = mapper.transform(x, indices); models = {}
        for name, enabled in core.VARIANTS.items():
            option = selection['choices'][name]; profile = core.profiles(settings)[option['profile_index']]
            heads = {m: core.fit_head(z[m], y, gender, enabled, profile, settings) for m in core.MODALITIES}
            state = {'protocol': PROTOCOL, 'family': 'catboost', 'feature_map': mapper, 'heads': heads,
                     'profile': profile, 'weights': option['weights'], 'gender': enabled, 'fit': indices,
                     'fit_participants': frame.participant_id.tolist()}
            path = out/f'{name}.pkl'; core.save_state(path, state)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'profile': profile, 'weights': option['weights']}
        original_settings = json.loads(Path('configs/semantic_accuracy_target.json').read_text())
        for name, module, config in [('baseline49_refit', old49_module, old_settings), ('baseline48_refit', old48_module, original_settings)]:
            candidate = selection['choices'][name]['candidate']; model = module.estimator_for(candidate, config)
            fit(model, baseline_matrix(x, candidate['feature']), y)
            state = {'protocol': PROTOCOL, 'family': 'baseline', 'estimator': model, 'candidate': candidate,
                     'fit_participants': frame.participant_id.tolist()}
            path = out/f'{name}.pkl'; core.save_state(path, state)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': candidate}
        for name, record in models.items():
            with Path(record['path']).open('rb') as f:
                restored = pickle.load(f)
            p = predict_model(restored, x, gender); core.decide(p, None)
            atomic_json(out/f'{name}_training_replay.json', {'probabilities_sha256': digest(p.tolist()), 'participants': len(frame)})
        source_bindings[str(out/'training_selection.json')] = sha256(out/'training_selection.json')
        plan = {'protocol': PROTOCOL, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'nested_context_hash': receipt['context_hash'], 'models': models,
                'primary_catboost': selection['primary_catboost'], 'majority_class': int(np.bincount(y).argmax()),
                'training_ids': frame.participant_id.tolist(), 'source_bindings': source_bindings,
                'selection_sha256': digest(selection), 'catboost_head_refits': 6, 'baseline_refits': 2,
                'holdout_policy': settings['holdout_plan'], 'prior_holdout_exposure': True,
                'reference_replaced': False, 'independent_confirmation': False}
        atomic_json(out/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    print(json.dumps({'frozen': True, 'primary_catboost': plan['primary_catboost'], 'models': 4, 'new_head_fits': 6, 'new_baseline_fits': 2}, indent=2))
    return plan


def verify_frozen(output=OUTPUT):
    out = Path(output); envelope = json.loads((out/'plan.json').read_text()); plan = envelope['plan']
    if plan['protocol'] != PROTOCOL or digest(plan) != envelope['plan_hash']:
        raise ValueError('Frozen holdout plan changed')
    for path, expected in plan['source_bindings'].items():
        if sha256(path) != expected:
            raise ValueError(f'Frozen source changed: {path}')
    for record in plan['models'].values():
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Frozen model changed')
    return plan, envelope['plan_hash']


def evaluate(output=OUTPUT):
    out = Path(output); plan, ph = verify_frozen(out)
    x, manifest = core.load_all_inputs()
    if manifest.participant_id.duplicated().any() or manifest.loc[manifest.split == 'train', 'participant_id'].tolist() != plan['training_ids']:
        raise ValueError('Original split participants changed')
    train_ids = set(plan['training_ids']); rows = []; new = 0
    with output_lock(out), threadpool_limits(limits=2), offline_network():
        for split in ['validation', 'test']:
            use = manifest.split.eq(split).to_numpy(); frame = manifest.loc[use].reset_index(drop=True)
            if not len(frame) or train_ids & set(frame.participant_id):
                raise ValueError('Missing or overlapping held-out participants')
            values = {n: a[use] for n, a in x.items()}; gender = frame.gender.to_numpy(str)
            for name in [*plan['models'], 'majority']:
                directory = out/split/name; path = directory/'result.json'
                if path.exists():
                    r = load_record(path, ph)
                    if r['participant_ids'] != frame.participant_id.tolist() or r['split'] != split or r['model'] != name:
                        raise ValueError('Evaluation ownership changed')
                    p = np.asarray(r['probabilities'])
                else:
                    if name == 'majority':
                        p = np.tile(np.eye(5)[plan['majority_class']], (len(frame), 1))
                    else:
                        with Path(plan['models'][name]['path']).open('rb') as f:
                            state = pickle.load(f)
                        p = predict_model(state, values, gender)
                    core.decide(p, None)
                    r = {'participant_ids': frame.participant_id.tolist(), 'split': split, 'model': name,
                         'probabilities': p.tolist(), 'metrics': metrics(frame.severity_class, p)}
                    save_record(path, r, ph); new += 1
                core.decide(p, None)
                if p.shape != (len(frame), 5) or metrics(frame.severity_class, p) != r['metrics']:
                    raise ValueError('Saved evaluation metrics changed')
                report = write_report(frame, p, directory, name, 42, f'reused_official_{split}')
                rows.append({'split': split, 'model': name, **score(frame.severity_class.to_numpy(int), p.argmax(1))})
        verify_frozen(out)
        table = pd.DataFrame(rows); table.to_csv(out/'summary.csv', index=False)
        atomic_json(out/'completion.json', {'complete': True, 'plan_hash': ph, 'new_prediction_batches': new,
                    'fits_during_evaluation': 0, 'primary_catboost': plan['primary_catboost'],
                    'prior_holdout_exposure': True, 'independent_confirmation': False, 'reference_replaced': False})
    print(table[['split', 'model', 'accuracy', 'macro_f1', 'balanced_accuracy', 'severe_recall']].to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'evaluate'])
    args = parser.parse_args()
    (freeze if args.command == 'freeze' else evaluate)()


if __name__ == '__main__':
    from .catboost_holdout import main as module_main
    module_main()
