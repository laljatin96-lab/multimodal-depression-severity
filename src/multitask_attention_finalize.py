"""Freeze attention models using inner training predictions; audit reused validation."""
import argparse
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import multitask_attention_accuracy as s
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .evaluate import write_report
from .final_evaluation import sha256

OUTPUT = s.OUTPUT


def training_choices(frame, records, old, settings):
    """Only inner predictions and fitting labels participate in selection."""
    y = frame.severity_class.to_numpy(int)
    if len(records) != len(old) or not records:
        raise ValueError('Incomplete selection records')
    owners = np.zeros(len(y), int)
    for r, baseline in zip(records, old):
        if any(r[k] != baseline[k] for k in ['seed', 'fold', 'fit', 'heldout']):
            raise ValueError('Selection populations differ')
        owners[r['fit']] += 1
    if not (owners == owners[0]).all() or not owners[0]:
        raise ValueError('Unequal participant contribution')
    labels = np.concatenate([y[r['fit']] for r in records])
    # The baseline is its already evaluated selection procedure, with cached
    # inner decisions. Its existing full-training refit is retained unchanged.
    base = np.concatenate([r['candidates'][r['policies']['accuracy']['selection']['index']]['predictions'] for r in old])
    inner = np.concatenate([np.asarray(r['inner_probabilities']) for r in records], axis=1)
    if inner.shape != (len(s.candidates(settings))-1, len(labels), 5):
        raise ValueError('Incomplete inner predictions')
    rows = [{'candidate': {'family': 'baseline'}, **s.score(labels, base)}]
    rows += [{'candidate': c, **s.score(labels, s.decide(p, None))} for c, p in zip(s.candidates(settings)[1:], inner)]
    return {'rows': rows, 'choices': {name: s.choose(rows, name) for name in settings['policies']},
            'inner_predictions_per_participant': int(owners[0]),
            'note': 'Optimistic tuning scores only; performance is reported separately from outer predictions.'}


def get_records(inputs, settings):
    completion = json.loads((OUTPUT/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Complete all nested jobs first')
    frame, old_settings = inputs[3:5]
    records = [load_record(OUTPUT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'result.json', completion['context_hash'])
               for fold in s.nested_folds(frame, old_settings)]
    return completion, records


def freeze():
    settings = json.loads(Path('configs/multitask_attention_accuracy.json').read_text())
    inputs = s.load_inputs(); data, clinical, x, frame, _, old, _ = inputs
    completion, records = get_records(inputs, settings)
    directory = OUTPUT/'final'
    if directory.exists():
        raise FileExistsError('Final models already frozen')
    # Validates hashes and cached records; a complete resume performs zero fits.
    if s.run(inputs, settings, resume=True)['new_fits']:
        raise ValueError('Unexpected incomplete search')
    selection = training_choices(frame, records, old, settings)
    original = json.loads(Path('results/supervised_accuracy/best_valid_model.json').read_text())['best_model']
    if sha256(original['path']) != original['sha256']:
        raise ValueError('Original baseline changed')
    original_plan = json.loads(Path('results/catboost_holdout/plan.json').read_text())['plan']
    if original_plan['training_ids'] != frame.participant_id.tolist():
        raise ValueError('Baseline training participants differ')
    directory.mkdir(); models = {}; fitted = {}; y = frame.severity_class.to_numpy(int)
    with output_lock(OUTPUT), threadpool_limits(limits=settings['cpu_threads']), s.offline_network():
        atomic_json(directory/'training_selection.json', selection)
        projection = s.Projection().fit(data, settings)
        for name, i in selection['choices'].items():
            if not i:
                models[name] = {**original, 'existing_baseline': True}
                continue
            if i not in fitted:
                candidate = s.candidates(settings)[i]
                model = s.AttentionClassifier(candidate, settings).fit(data, y, clinical, projection)
                fitted[i] = {'protocol': s.PROTOCOL, 'estimator': model, 'candidate': candidate,
                             'fit_participants': frame.participant_id.tolist()}
            state = fitted[i]; path = directory/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
            restored = pickle.loads(path.read_bytes())
            np.testing.assert_allclose(restored['estimator'].predict_proba(data), state['estimator'].predict_proba(data), rtol=0, atol=1e-7)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': state['candidate'], 'existing_baseline': False}
        models['baseline_refit'] = {**original, 'existing_baseline': True}
        paths = [OUTPUT/'protocol.json', directory/'training_selection.json', Path(__file__), Path(s.__file__),
                 Path('configs/multitask_attention_accuracy.json'), Path('data/metadata/participants.csv'),
                 Path('results/chunk_semantic_features/features.json'), Path('results/semantic_features/embeddings.json'),
                 Path('results/temporal_summary_comparison/participant_features.npz')]
        plan = {'protocol': s.PROTOCOL, 'training_ids': frame.participant_id.tolist(), 'models': models,
                'context_hash': completion['context_hash'], 'bindings': {str(p): sha256(p) for p in paths},
                'new_full_training_fits': len(fitted), 'primary': 'accuracy',
                'validation_previously_exposed': True, 'validation_may_select_models': False,
                'test_scoring': False, 'questionnaire_inputs_at_inference': False,
                'independent_confirmation': False}
        atomic_json(directory/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return {'frozen': True, 'fits': len(fitted), 'models': models}


def frozen_plan():
    envelope = json.loads((OUTPUT/'final'/'plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Frozen plan changed')
    for path, h in plan['bindings'].items():
        if sha256(path) != h:
            raise ValueError(f'Frozen input changed: {path}')
    for record in plan['models'].values():
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Frozen model changed')
    return plan, envelope['plan_hash']


def validation_inputs(plan):
    from .catboost_late_fusion import load_all_inputs
    all_x, participants = load_all_inputs()
    if participants.loc[participants.split.eq('train'), 'participant_id'].tolist() != plan['training_ids']:
        raise ValueError('Training ownership changed')
    use = participants.split.eq('validation').to_numpy(); frame = participants.loc[use].reset_index(drop=True)
    if len(frame) != 56 or frame.participant_id.duplicated().any() or set(frame.participant_id) & set(plan['training_ids']):
        raise ValueError('Validation ownership/leakage')
    # No symptom file is accessed during validation inference.
    chunks = json.loads(Path('results/chunk_semantic_features/features.json').read_text())
    by_id = {r['participant_id']: r for r in chunks}
    selected = [by_id[p] for p in frame.participant_id]
    if len(by_id) != len(chunks) or any(r['split'] != 'validation' for r in selected):
        raise ValueError('Validation chunk ownership changed')
    av = np.column_stack([all_x['audio'][use], all_x['visual'][use]])
    data = {'vectors': [np.asarray(r['vectors'], float) for r in selected],
            'lengths': [r['lengths'] for r in selected], 'av': av}
    x = {'semantic': all_x['text'][use], 'semantic_av': np.column_stack([all_x['text'][use], av])}
    return data, x, frame


def predict_saved(record, data, x):
    with Path(record['path']).open('rb') as f:
        state = pickle.load(f)
    if record['existing_baseline']:
        return state['estimator'].predict_proba(x[state['candidate']['feature']])
    return state['estimator'].predict_proba(data)


def evaluate():
    plan, ph = frozen_plan(); data, x, frame = validation_inputs(plan); rows = []
    torch.set_num_threads(2)
    with output_lock(OUTPUT), threadpool_limits(limits=2), s.offline_network():
        for name, record in plan['models'].items():
            directory = OUTPUT/'validation'/name; path = directory/'result.json'
            if path.exists():
                r = load_record(path, ph)
                if r['participant_ids'] != frame.participant_id.tolist() or r['model_sha256'] != record['sha256']:
                    raise ValueError('Validation identity changed')
                p = np.asarray(r['probabilities'])
            else:
                p = predict_saved(record, data, x); s.decide(p, None)
                save_record(path, {'participant_ids': frame.participant_id.tolist(), 'model_sha256': record['sha256'],
                                  'probabilities': p.tolist()}, ph)
            write_report(frame, p, directory, name, 42, 'reused_official_validation')
            rows.append({'model': name, **s.score(frame.severity_class.to_numpy(int), p.argmax(1))})
        pd.DataFrame(rows).to_csv(OUTPUT/'validation_scores.csv', index=False)
        atomic_json(OUTPUT/'validation_receipt.json', {'participants': len(frame), 'plan_hash': ph,
            'original_participants_only': True, 'synthetic_evaluation_rows': 0, 'questionnaire_inputs': False,
            'validation_selected_model': False, 'prior_holdout_exposure': True, 'independent_confirmation': False, 'test_scored': False})
    return rows


def verify():
    settings = json.loads(Path('configs/multitask_attention_accuracy.json').read_text())
    inputs = s.load_inputs(); data, clinical, x, frame, old_settings, old, _ = inputs
    completion, records = get_records(inputs, settings)
    plan, ph = frozen_plan(); vd, vx, vf = validation_inputs(plan)
    paths = list((OUTPUT/'jobs').glob('*/*'))+list((OUTPUT/'final').glob('*'))+list((OUTPUT/'validation').glob('*/result.json'))
    before = {str(p): [sha256(p), p.stat().st_mtime_ns] for p in paths if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('No fitting/inference during completed resume')
    with patch.object(s, 'job', forbidden), patch.object(s, 'predict_state', forbidden), patch.object(pickle, 'load', forbidden):
        resumed = s.run(inputs, settings, resume=True)
        evaluate()
    if resumed['new_fits'] or before != {p: [sha256(p), Path(p).stat().st_mtime_ns] for p in before}:
        raise ValueError('Completed resume changed saved artifacts')
    selection = training_choices(frame, records, old, settings)
    if selection != json.loads((OUTPUT/'final'/'training_selection.json').read_text()):
        raise ValueError('Final selection changed')
    rows = []; replayed = 0
    with threadpool_limits(limits=2), s.offline_network():
        for fold, r in zip(s.nested_folds(frame, old_settings), records):
            for name, result in r['selected'].items():
                path = OUTPUT/'jobs'/f"seed{r['seed']}_fold{r['fold']}"/f'{name}.pkl'
                state = pickle.loads(path.read_bytes())
                if state['fit'] != fold['fit'] or state['heldout'] != fold['heldout']:
                    raise ValueError('Saved fit ownership changed')
                if state['candidate']['family'] != 'baseline':
                    model = state['estimator']
                    if model.real_count != len(fold['fit']) or model.projection.real_count != len(fold['fit']):
                        raise ValueError('Fitting count mismatch')
                    projection = s.Projection().fit(s.subset(data, fold['fit']), settings)
                    for a, b in zip(model.projection.text+model.projection.av, projection.text+projection.av):
                        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
                p = s.predict_state(state, s.subset(data, fold['heldout']), s.subset(x, fold['heldout']))
                np.testing.assert_allclose(p, result['probabilities'], rtol=0, atol=1e-7); replayed += 1
                tp = s.predict_state(state, s.subset(data, fold['fit']), s.subset(x, fold['fit']))
                fit_accuracy = float((tp.argmax(1) == frame.severity_class.to_numpy(int)[fold['fit']]).mean())
                if fit_accuracy != result['real_training_accuracy']:
                    raise ValueError('Fitting accuracy changed')
                rows.append({'policy': name, 'seed': r['seed'], 'fold': r['fold'], 'training_accuracy': fit_accuracy,
                             'inner_accuracy': r['options'][result['selection']['index']]['accuracy'],
                             'outer_accuracy': s.score(frame.severity_class.to_numpy(int)[fold['heldout']], p.argmax(1))['accuracy']})
        for name, record in plan['models'].items():
            p = predict_saved(record, vd, vx)
            saved = load_record(OUTPUT/'validation'/name/'result.json', ph)
            np.testing.assert_allclose(p, saved['probabilities'], rtol=0, atol=1e-7); replayed += 1
    pd.DataFrame(rows).to_csv(OUTPUT/'overfitting_audit.csv', index=False)
    summary = pd.read_csv(OUTPUT/'summary.csv').set_index('model')
    original = plan['models']['baseline_refit']
    improved = bool(summary.loc['accuracy', 'accuracy_mean'] > summary.loc['baseline', 'accuracy_mean'])
    manifest = {'retained': 'attention_selection_exploratory' if improved else 'original_verified_49.08_baseline',
                'replacement': improved, 'reference_model_overwritten': False,
                'baseline_nested_accuracy': float(summary.loc['baseline', 'accuracy_mean']),
                'new_nested_accuracy': float(summary.loc['accuracy', 'accuracy_mean']),
                'best_model': plan['models']['accuracy'] if improved else original,
                'new_research_models': plan['models'], 'selected_using_validation': False,
                'independent_confirmation': False, 'target_53_percent_reached': bool(summary.loc['accuracy', 'accuracy_mean'] >= .53)}
    atomic_json(OUTPUT/'best_valid_model.json', manifest)
    prior = json.loads(Path('results/multitask_attention_preservation/before.json').read_text())
    changed = [p for p, v in prior.items() if not Path(p).exists() or Path(p).stat().st_mtime_ns != v['mtime_ns'] or sha256(p) != v['sha256']]
    if changed:
        raise ValueError(f'Pre-existing files changed: {changed}')
    result = {'passed': True, 'checkpoints_replayed': replayed, 'saved_artifacts_preserved_on_resume': len(before),
              'previous_files_and_mtimes_preserved': len(prior), 'zero_fit_resume': True,
              'new_network_fits': completion['total_fits']+plan['new_full_training_fits'], 'old_models_refitted': 0,
              'validation_participants': len(vf), 'test_scored': False}
    atomic_json(OUTPUT/'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'evaluate', 'verify']); args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))


if __name__ == '__main__':
    main()
