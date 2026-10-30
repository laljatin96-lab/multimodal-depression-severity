"""Training-only final refits and audited inference for the PLS/ridge study."""
import argparse
import json
from pathlib import Path
import pickle
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import supervised_accuracy as study
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .evaluate import write_report
from .final_evaluation import sha256
from .semantic_severe_repair import estimator_for as baseline_estimator
from .severe_detection_study import score

OUTPUT = Path('results/supervised_accuracy')


def training_choices(frame, records, old_records, settings):
    """Select configurations from inner predictions; never inspect outer outcomes."""
    y = frame.severity_class.to_numpy(int)
    if len(records) != len(old_records) or not records:
        raise ValueError('Missing training selection records')
    for r, old in zip(records, old_records):
        if any(r[k] != old[k] for k in ['seed', 'fold', 'fit', 'heldout']):
            raise ValueError('Training selection fold mismatch')
    owners = np.zeros(len(y), int)
    for r in records:
        owners[r['fit']] += 1
    if not (owners == owners[0]).all() or not owners[0]:
        raise ValueError('Unequal tuning contribution')
    labels = np.concatenate([y[r['fit']] for r in records])
    baseline_candidates = [r['candidate'] for r in old_records[0]['candidates']]
    if any([r['candidate'] for r in old['candidates']] != baseline_candidates for old in old_records):
        raise ValueError('Baseline candidates changed')
    baseline_rows = []
    for i, c in enumerate(baseline_candidates):
        pred = np.concatenate([r['candidates'][i]['predictions'] for r in old_records])
        baseline_rows.append({'candidate': c, **score(labels, pred)})
    baseline_choice = baseline_rows[study.choose(baseline_rows)]
    new_inner = np.concatenate([np.asarray(r['inner_probabilities']) for r in records], axis=1)
    search = study.candidates(settings)
    if new_inner.shape != (len(search)-1, len(labels), 5):
        raise ValueError('Missing inner prediction coverage')
    rows = [{'candidate': search[0], **{k:v for k,v in baseline_choice.items() if k != 'candidate'}}]
    rows.extend({'candidate': c, **score(labels, study.decide(p, None))} for c, p in zip(search[1:], new_inner))
    return {'rows': rows, 'baseline_choice': baseline_choice,
            'choices': {'primary': study.choose(rows), 'best_new': 1+study.choose(rows[1:])},
            'inner_predictions_per_participant': int(owners[0]),
            'note': 'Training-only selection scores are optimistic; they are not performance estimates.'}


def get_records(inputs, settings):
    completion = json.loads((OUTPUT/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Complete all nested jobs first')
    _, frame, old_settings, _, _ = inputs
    records = [load_record(OUTPUT/'jobs'/f"seed{f['seed']}_fold{f['fold']}"/'result.json', completion['context_hash'])
               for f in study.nested_folds(frame, old_settings)]
    return completion, records


def freeze():
    settings = json.loads(Path('configs/supervised_accuracy.json').read_text())
    inputs = study.load_inputs(); x, frame, old_settings, old_records, _ = inputs
    completion, records = get_records(inputs, settings)
    if (OUTPUT/'final').exists():
        raise FileExistsError('Final models are already frozen')
    selection = training_choices(frame, records, old_records, settings)
    # Validate the complete frozen training context before fitting final models.
    study.run(*inputs, settings, OUTPUT, resume=True)
    directory = OUTPUT/'final'; directory.mkdir()
    models = {}; fitted = {}; search = study.candidates(settings); y = frame.severity_class.to_numpy(int)
    with output_lock(OUTPUT), threadpool_limits(limits=settings['cpu_threads']), study.offline_network():
        atomic_json(directory/'training_selection.json', selection)
        for name, i in selection['choices'].items():
            if i not in fitted:
                candidate = search[i] if i else selection['baseline_choice']['candidate']
                feature = candidate['feature']
                model = study.estimator(candidate, settings) if i else baseline_estimator(candidate, old_settings)
                fit(model, x[feature], y)
                fitted[i] = {'protocol': study.PROTOCOL, 'estimator': model, 'candidate': candidate,
                             'feature': feature, 'fit_participants': frame.participant_id.tolist()}
            state = fitted[i]; path = directory/f'{name}.pkl'
            with path.open('wb') as f:
                pickle.dump(state, f)
            with path.open('rb') as f:
                restored = pickle.load(f)
            np.testing.assert_allclose(five_probabilities(restored['estimator'], x[state['feature']]),
                                       five_probabilities(state['estimator'], x[state['feature']]), rtol=0, atol=1e-14)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': state['candidate'],
                            'score_kind': 'uncalibrated_softmax_ridge' if state['candidate']['family'] == 'ridge' else 'logistic_probability'}
        previous = json.loads(Path('results/catboost_holdout/plan.json').read_text())['plan']
        baseline = previous['models']['baseline49_refit']
        if sha256(baseline['path']) != baseline['sha256'] or previous['training_ids'] != frame.participant_id.tolist():
            raise ValueError('Original full-training baseline mismatch')
        models['baseline_refit'] = baseline
        paths = [OUTPUT/'protocol.json', directory/'training_selection.json', Path(__file__), Path(study.__file__),
                 Path('configs/supervised_accuracy.json'), Path('data/metadata/participants.csv'),
                 Path('results/semantic_features/embeddings.json'), Path('results/mpnet_features/embeddings.json'),
                 Path('results/temporal_summary_comparison/participant_features.npz')]
        plan = {'protocol': study.PROTOCOL, 'training_ids': frame.participant_id.tolist(), 'models': models,
                'context_hash': completion['context_hash'], 'bindings': {str(p): sha256(p) for p in paths},
                'new_full_training_fits': len(fitted), 'primary': 'primary',
                'validation_previously_exposed': True, 'validation_may_select_models': False,
                'test_scoring': False, 'independent_confirmation': False}
        atomic_json(directory/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return {'frozen': True, 'new_full_training_fits': len(fitted), 'models': models}


def frozen_plan():
    envelope = json.loads((OUTPUT/'final'/'plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Frozen plan changed')
    for p, h in plan['bindings'].items():
        if sha256(p) != h:
            raise ValueError(f'Frozen input changed: {p}')
    for r in plan['models'].values():
        if sha256(r['path']) != r['sha256']:
            raise ValueError('Frozen model changed')
    return plan, envelope['plan_hash']


def validation_inputs(plan):
    from .catboost_late_fusion import load_all_inputs
    all_x, participants = load_all_inputs()
    if participants.loc[participants.split.eq('train'), 'participant_id'].tolist() != plan['training_ids']:
        raise ValueError('Original training population changed')
    use = participants.split.eq('validation').to_numpy()
    frame = participants.loc[use].reset_index(drop=True)
    if not len(frame) or frame.participant_id.duplicated().any() or set(frame.participant_id) & set(plan['training_ids']):
        raise ValueError('Validation participant leakage')
    mpnet = json.loads(Path('results/mpnet_features/embeddings.json').read_text())
    if [r['participant_id'] for r in mpnet] != participants.participant_id.tolist():
        raise ValueError('MPNet participant alignment changed')
    larger = np.asarray([r['embedding'] for r in mpnet], float)[use]
    av = np.column_stack([all_x['audio'][use], all_x['visual'][use]])
    x = {'semantic': all_x['text'][use], 'semantic_av': np.column_stack([all_x['text'][use], av])}
    x.update(minilm=x['semantic'], minilm_av=x['semantic_av'], mpnet=larger, mpnet_av=np.column_stack([larger, av]))
    return x, frame


def evaluate():
    plan, ph = frozen_plan(); x, frame = validation_inputs(plan); rows = []
    with output_lock(OUTPUT), threadpool_limits(limits=2), study.offline_network():
        for name, record in plan['models'].items():
            directory = OUTPUT/'validation'/name; path = directory/'result.json'
            if path.exists():
                result = load_record(path, ph)
                if result['participant_ids'] != frame.participant_id.tolist() or result['model_sha256'] != record['sha256']:
                    raise ValueError('Validation ownership changed')
                p = np.asarray(result['probabilities'])
            else:
                with Path(record['path']).open('rb') as f:
                    state = pickle.load(f)
                p = five_probabilities(state['estimator'], x[state.get('feature', state['candidate']['feature'])])
                study.decide(p, None)
                directory.mkdir(parents=True)
                save_record(path, {'participant_ids': frame.participant_id.tolist(), 'model_sha256': record['sha256'],
                                  'probabilities': p.tolist()}, ph)
            study.decide(p, None)
            write_report(frame, p, directory, name, 42, 'reused_official_validation')
            rows.append({'model': name, **score(frame.severity_class.to_numpy(int), p.argmax(1))})
        pd.DataFrame(rows).to_csv(OUTPUT/'validation_scores.csv', index=False)
        atomic_json(OUTPUT/'validation_receipt.json', {'participants': len(frame), 'plan_hash': ph,
                    'original_participants_only': True, 'synthetic_evaluation_rows': 0,
                    'validation_selected_model': False, 'prior_holdout_exposure': True,
                    'independent_confirmation': False, 'test_scored': False})
    return rows


def verify():
    settings = json.loads(Path('configs/supervised_accuracy.json').read_text())
    inputs = study.load_inputs(); x, frame, old_settings, old_records, _ = inputs
    completion, records = get_records(inputs, settings)
    if not (OUTPUT/'training_completion.json').exists():
        atomic_json(OUTPUT/'training_completion.json', completion)
    paths = list((OUTPUT/'jobs').glob('*/*')) + list((OUTPUT/'final').glob('*')) + list((OUTPUT/'validation').glob('*/result.json'))
    before = {str(p): [sha256(p), p.stat().st_mtime_ns] for p in paths if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('Completed resume cannot fit, infer or deserialize')
    with patch.object(study, 'job', forbidden), patch.object(study, 'five_probabilities', forbidden), patch.object(pickle, 'load', forbidden):
        resumed = study.run(*inputs, settings, OUTPUT, resume=True)
        evaluate()
    if resumed['new_fits'] != 0 or before != {p: [sha256(p), Path(p).stat().st_mtime_ns] for p in before}:
        raise ValueError('Resume mutated models/jobs or refitted')
    replayed = 0; overfit = []
    with threadpool_limits(limits=2), study.offline_network():
        for r in records:
            result = r['selected']['accuracy']; path = OUTPUT/'jobs'/f"seed{r['seed']}_fold{r['fold']}"/'accuracy.pkl'
            with path.open('rb') as f:
                state = pickle.load(f)
            if state['fit'] != r['fit'] or state['heldout'] != r['heldout']:
                raise ValueError('Saved participant ownership changed')
            p = five_probabilities(state['estimator'], x[state['feature']][r['heldout']])
            np.testing.assert_allclose(p, result['probabilities'], rtol=0, atol=1e-14); replayed += 1
            train_accuracy = float((state['estimator'].predict(x[state['feature']][r['fit']]) == frame.severity_class.to_numpy(int)[r['fit']]).mean())
            if train_accuracy != r['real_training_accuracy']:
                raise ValueError('Saved fitting accuracy changed')
            overfit.append({'seed': r['seed'], 'fold': r['fold'], **result['candidate'],
                            'training_accuracy': train_accuracy,
                            'inner_accuracy': r['options'][result['selection']['index']]['accuracy'],
                            'outer_accuracy': score(frame.severity_class.to_numpy(int)[r['heldout']], p.argmax(1))['accuracy']})
        plan, ph = frozen_plan(); vx, vf = validation_inputs(plan)
        for name, record in plan['models'].items():
            with Path(record['path']).open('rb') as f:
                state = pickle.load(f)
            p = five_probabilities(state['estimator'], vx[state.get('feature', state['candidate']['feature'])])
            saved = load_record(OUTPUT/'validation'/name/'result.json', ph)
            np.testing.assert_allclose(p, saved['probabilities'], rtol=0, atol=1e-14); replayed += 1
    pd.DataFrame(overfit).to_csv(OUTPUT/'overfitting_audit.csv', index=False)
    summary = pd.read_csv(OUTPUT/'summary.csv').set_index('model')
    classes = []
    matrices = json.loads((OUTPUT/'summed_confusion_matrices.json').read_text())['matrices']
    for name, cm in matrices.items():
        cm = np.asarray(cm)
        for k in range(5):
            actual, predicted, correct = int(cm[k].sum()), int(cm[:, k].sum()), int(cm[k,k])
            classes.append({'model': name, 'class': k, 'repeated_support': actual, 'correct': correct,
                            'recall': correct/actual, 'precision': correct/predicted if predicted else 0.,
                            'f1': 2*correct/(actual+predicted) if actual+predicted else 0.})
    pd.DataFrame(classes).to_csv(OUTPUT/'class_comparison.csv', index=False)
    improved = bool(summary.loc['accuracy', 'accuracy_mean'] > summary.loc['baseline', 'accuracy_mean'])
    chosen = plan['models']['primary'] if improved else plan['models']['baseline_refit']
    manifest = {'retained': 'new_exploratory_procedure' if improved else 'original_verified_49.08_baseline',
                'replacement': improved, 'reference_model_overwritten': False,
                'baseline_nested_accuracy': float(summary.loc['baseline', 'accuracy_mean']),
                'new_nested_accuracy': float(summary.loc['accuracy', 'accuracy_mean']),
                'evaluation': 'Same 163 training participants in three repeated nested-CV partitions; exploratory.',
                'best_model': chosen, 'training_selected_primary': plan['models']['primary'],
                'training_selected_new_candidate': plan['models']['best_new'],
                'selection_used_validation': False, 'independent_confirmation': False,
                'target_53_percent_reached': bool(summary.loc['accuracy', 'accuracy_mean'] >= settings['target_accuracy']),
                'final_plan_sha256': sha256(OUTPUT/'final'/'plan.json')}
    atomic_json(OUTPUT/'best_valid_model.json', manifest)
    preserved = json.loads(Path('results/supervised_accuracy_preservation/before.json').read_text())
    changed = [p for p, v in preserved.items() if not Path(p).exists() or Path(p).stat().st_mtime_ns != v['mtime_ns'] or sha256(p) != v['sha256']]
    if changed:
        raise ValueError(f'Previous files changed: {changed}')
    suite = ET.parse(OUTPUT/'tests.xml').getroot().find('testsuite')
    counts = {k: int(suite.attrib[k]) for k in ['tests', 'failures', 'errors', 'skipped']}
    if counts['failures'] or counts['errors']:
        raise ValueError('Automated tests did not pass')
    result = {'passed': True, 'checkpoints_replayed': replayed, 'zero_fit_resume': True,
              'saved_records_and_models_preserved_on_resume': len(before), 'previous_files_and_mtimes_preserved': len(preserved),
              'training_participants': len(frame), 'unique_severe_training_participants': int(frame.severity_class.eq(4).sum()),
              'validation_participants': len(vf), 'synthetic_rows': 0, 'test_scored': False,
              'automated_tests': counts, 'new_classifier_fits': completion['pipeline_fits']+plan['new_full_training_fits']}
    atomic_json(OUTPUT/'verification.json', result)
    atomic_json(Path('results/supervised_accuracy_preservation/after.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'evaluate', 'verify'])
    args = parser.parse_args()
    print(json.dumps({'freeze': freeze, 'evaluate': evaluate, 'verify': verify}[args.action](), indent=2))


if __name__ == '__main__':
    main()
