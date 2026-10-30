"""Bounded zero-window and middle-layer WavLM comparison on frozen folds."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from . import wavlm_accuracy as previous
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_detection_study import decide, score, summarize

OUTPUT = Path('results/wavlm_layer_accuracy')
CONFIG = Path('configs/wavlm_layer_accuracy.json')
PREVIOUS_ROOT = Path('results/wavlm_accuracy')
PROTOCOL = 'training-only-zero-window-middle-layer-wavlm-v1'


class WavLMLayerClassifier(previous.WavLMClassifier):
    """Identical small head; select one declared frozen speech representation."""
    def speech_inputs(self, x):
        key = self.candidate['representation']
        if key not in {'final_nonzero', 'layer6_nonzero'}:
            raise ValueError('Unknown frozen speech representation')
        return {**x, 'speech': x[key]}

    def fit(self, x, y):
        return super().fit(self.speech_inputs(x), y)

    def transform(self, x):
        return super().transform(self.speech_inputs(x))

    def diagnostics(self, x, y):
        return {**super().diagnostics(x, y), 'representation': self.candidate['representation']}


def candidates(settings):
    if settings['C'] != [.001] or set(settings['representations']) != {'final_nonzero', 'layer6_nonzero'}:
        raise ValueError('Frozen two-representation regularization budget changed')
    return [{'family': 'previous_procedure'}] + [
        {'family': 'wavlm_layer', 'representation': representation, 'C': .001, 'speech_scale': scale}
        for representation in settings['representations'] for scale in settings['speech_scales']]


def previous_inner(old):
    """Recover the previous procedure's chosen inner prediction without fitting."""
    record = old['previous_procedure']
    i = record['selected']['accuracy']['selection']['index']
    if i:
        p = np.asarray(record['inner_probabilities'])[i-1]
    else:
        control = old['language_control']
        j = control['selected']['without_age']['selection']['index']
        p = np.asarray(control['inner_probabilities'])[j-1]
    if p.shape != (len(record['fit']), 5):
        raise ValueError('Previous procedure inner coverage changed')
    decide(p, None)
    return p


def options(inner, old, y, settings):
    predictions = [decide(previous_inner(old), None)] + [decide(p, None) for p in inner]
    return [{'candidate': c, **score(y, p)} for c, p in zip(candidates(settings), predictions)]


def choose(rows, policy):
    if policy == 'previous_procedure':
        return 0
    if policy == 'accuracy':
        allowed = range(len(rows))
    elif policy == 'new_only':
        allowed = range(1, len(rows))
    elif policy in {'zero_only', 'layer6_only'}:
        representation = 'final_nonzero' if policy == 'zero_only' else 'layer6_nonzero'
        allowed = [i for i, r in enumerate(rows) if r['candidate'].get('representation') == representation]
    else:
        raise ValueError('Unknown policy')
    return max(allowed, key=lambda i: (rows[i]['correct'], rows[i]['macro_f1'], -i))


def predict_state(state, x):
    family = state['candidate']['family']
    if family == 'wavlm_layer':
        return state['estimator'].predict_proba(x)
    if family == 'previous_procedure':
        state = {**state, 'candidate': state['source_candidate']}
    return previous.predict_state(state, x)


subset = previous.subset


def job(x, frame, fold, old, settings, out):
    started = time.monotonic(); y = frame.severity_class.to_numpy(int); search = candidates(settings)
    inner = np.full((len(search)-1, len(y), 5), np.nan); diagnostics = []
    for part in fold['inner']:
        for i, c in enumerate(search[1:]):
            model = WavLMLayerClassifier(c, settings).fit(subset(x, part['fit']), y[part['fit']])
            inner[i, part['heldout']] = model.predict_proba(subset(x, part['heldout']))
            diagnostics.append({'candidate_index': i+1, 'fit': part['fit'], 'heldout': part['heldout'],
                'inner_accuracy': float((inner[i, part['heldout']].argmax(1) == y[part['heldout']]).mean()),
                **model.diagnostics(subset(x, part['fit']), y[part['fit']])})
    inner = inner[:, fold['fit']]; rows = options(inner, old, y[fold['fit']], settings)
    out.mkdir(parents=True, exist_ok=True); selected, fitted = {}, {}
    for name in settings['policies']:
        i = choose(rows, name); c = search[i]
        if i not in fitted:
            if i:
                model = WavLMLayerClassifier(c, settings).fit(subset(x, fold['fit']), y[fold['fit']])
                state = {'estimator': model, 'candidate': c, 'feature': 'semantic_av'}
            else:
                source = PREVIOUS_ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
                if sha256(source) != old['previous_procedure']['selected']['accuracy']['checkpoint_sha256']:
                    raise ValueError('Previous outer checkpoint changed')
                original = pickle.loads(source.read_bytes())
                state = {'estimator': original['estimator'], 'candidate': c,
                         'source_candidate': original['candidate']}
            fitted[i] = {**state, 'protocol': PROTOCOL, 'fit': fold['fit'], 'heldout': fold['heldout']}
        state = fitted[i]; heldout = subset(x, fold['heldout']); p = predict_state(state, heldout)
        if not i:
            np.testing.assert_allclose(p, old['previous_procedure']['selected']['accuracy']['probabilities'], rtol=0, atol=1e-14)
        path = out/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
        np.testing.assert_allclose(p, predict_state(pickle.loads(path.read_bytes()), heldout), rtol=0, atol=1e-14)
        training = subset(x, fold['fit'])
        selected[name] = {'selection': {'index': i}, 'candidate': c, 'threshold': None,
            'probabilities': p.tolist(), 'predictions': p.argmax(1).tolist(), 'checkpoint_sha256': sha256(path),
            'real_training_accuracy': float((predict_state(state, training).argmax(1) == y[fold['fit']]).mean())}
    count = sum(i > 0 for i in fitted)
    if count > len(settings['representations']):
        raise ValueError('Selected outer fitting budget exceeded')
    return {**{k: fold[k] for k in ['seed', 'fold', 'fit', 'heldout']}, 'inner_probabilities': inner.tolist(),
        'options': rows, 'selected': selected, 'inner_receipts': fold['inner'], 'fitting_diagnostics': diagnostics,
        'fits': (len(search)-1)*len(fold['inner'])+count, 'seconds': time.monotonic()-started}


def validate(record, fold, old, y, settings, directory):
    if (any(record[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout'])
            or record['inner_receipts'] != fold['inner'] or set(record['selected']) != set(settings['policies'])):
        raise ValueError('Fold or training ownership differs')
    if np.asarray(record['inner_probabilities']).shape != (len(candidates(settings))-1, len(fold['fit']), 5):
        raise ValueError('Incomplete inner coverage')
    rows = options(record['inner_probabilities'], old, y[fold['fit']], settings)
    if rows != record['options']:
        raise ValueError('Inner options changed')
    indices = set()
    for name, result in record['selected'].items():
        i = choose(rows, name); indices.add(i)
        if result['selection'] != {'index': i} or result['candidate'] != rows[i]['candidate']:
            raise ValueError('Selection changed')
        if np.asarray(result['probabilities']).shape != (len(fold['heldout']), 5) or decide(result['probabilities'], None).tolist() != result['predictions']:
            raise ValueError('Invalid probabilities')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
    if record['fits'] != (len(candidates(settings))-1)*len(fold['inner'])+sum(i > 0 for i in indices):
        raise ValueError('Recorded fit count changed')


def run(inputs, settings, output=OUTPUT, resume=False, max_new_jobs=None):
    x, frame, old_settings, old, provenance = inputs; out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing study')
    folds = nested_folds(frame, old_settings); y = frame.severity_class.to_numpy(int)
    if len(folds) != settings['expected_jobs'] or len(old) != len(folds) or frame.participant_id.duplicated().any():
        raise ValueError('Population or folds changed')
    for fold, baseline in zip(folds, old):
        if any(fold[k] != baseline[k] for k in ['seed', 'fold', 'fit', 'heldout']) or set(fold['fit']) & set(fold['heldout']):
            raise ValueError('Outer participant ownership differs')
        for part in fold['inner']:
            if set(part['fit']) & set(part['heldout']) or set(part['fit']+part['heldout']) != set(fold['fit']):
                raise ValueError('Inner participant ownership differs')
    maximum = sum((len(candidates(settings))-1)*len(f['inner'])+len(settings['representations']) for f in folds)
    if maximum > settings['max_nested_pipeline_fits']:
        raise ValueError('Declared fit budget exceeded')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
        'provenance': provenance, 'source_sha256': sha256(__file__), 'input_hashes': {k: digest(v.tolist()) for k,v in x.items()},
        'old_records_hash': digest(old), 'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'support_hashes': {str(Path('src')/name): sha256(Path('src')/name) for name in
            ['severe_case_diagnosis.py', 'severe_detection_study.py', 'raw_frame_uncertainty.py', 'batch_inference.py',
             'wavlm_accuracy.py', 'wavlm_layer_inference.py', 'age_language_accuracy.py', 'wavlm_features.py',
             'wavlm_features_complete.py', 'wavlm_layer_features.py', 'checked_inference.py']}}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        path = out/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Frozen inputs or code changed')
        elif resume:
            raise ValueError('Missing frozen protocol')
        else:
            atomic_json(path, envelope)
        records = []; new = fits = 0
        for fold, baseline in zip(folds, old):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; path = directory/'result.json'
            if path.exists():
                record = load_record(path, ch)
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                record = job(x, frame, fold, baseline, settings, directory)
                save_record(path, record, ch); new += 1; fits += record['fits']
            validate(record, fold, baseline, y, settings, directory); records.append(record)
            print(f"Zero-window/middle-layer WavLM: {len(records)}/{len(folds)} jobs; {record['seconds']:.1f}s/job", flush=True)
        complete = len(records) == len(folds)
        if complete and not new and (out/'completion.json').exists():
            result = json.loads((out/'completion.json').read_text())
            return {**result, 'new_jobs': 0, 'new_fits': 0}
        targets = summarize(frame, records, old, settings, out) if complete else None
        result = {'complete': complete, 'context_hash': ch, 'completed_jobs': len(records),
            'new_jobs': new, 'new_fits': fits, 'total_fits': sum(r['fits'] for r in records),
            'targets': targets, 'old_models_refitted': 0, 'synthetic_rows': 0, 'validation_test_scored': False}
        atomic_json(out/'completion.json', result)
    return result


def training_choices(frame, records, old, settings):
    y = frame.severity_class.to_numpy(int); owners = np.zeros(len(y), int)
    if not records or len(records) != len(old):
        raise ValueError('Incomplete final selection')
    for record, baseline in zip(records, old):
        if any(record[k] != baseline[k] for k in ['seed', 'fold', 'fit', 'heldout']):
            raise ValueError('Final selection ownership differs')
        owners[record['fit']] += 1
    if not (owners == owners[0]).all() or not owners[0]:
        raise ValueError('Unequal final selection contribution')
    labels = np.concatenate([y[r['fit']] for r in records])
    baseline = np.concatenate([decide(previous_inner(r), None) for r in old])
    inner = np.concatenate([np.asarray(r['inner_probabilities']) for r in records], axis=1)
    rows = [{'candidate': {'family': 'previous_procedure'}, **score(labels, baseline)}]
    rows += [{'candidate': c, **score(labels, decide(p, None))} for c, p in zip(candidates(settings)[1:], inner)]
    return {'rows': rows, 'choices': {n: choose(rows, n) for n in settings['policies']},
            'inner_predictions_per_participant': int(owners[0]), 'outer_results_used': False}


def freeze(inputs, settings, out=OUTPUT):
    from .best_accuracy_inference import load_bundle
    from .wavlm_layer_features import encoder_contract, runtime_contract
    x, frame, old_settings, old, _ = inputs; out = Path(out)
    completion = json.loads((out/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Study incomplete')
    records = [load_record(out/'jobs'/f"seed{f['seed']}_fold{f['fold']}"/'result.json', completion['context_hash'])
               for f in nested_folds(frame, old_settings)]
    final = out/'final'
    if final.exists():
        raise FileExistsError('Preserving frozen final model')
    selection = training_choices(frame, records, old, settings)
    envelope = json.loads((PREVIOUS_ROOT/'final/plan.json').read_text()); plan = envelope['plan']
    if digest(plan) != envelope['plan_hash']:
        raise ValueError('Previous final plan changed')
    reference = plan['models']['accuracy']
    if sha256(reference['path']) != reference['sha256']:
        raise ValueError('Previous final checkpoint changed')
    if pickle.loads(Path(reference['path']).read_bytes())['fit_participants'] != frame.participant_id.tolist():
        raise ValueError('Previous fitting participants changed')
    choices = set(selection['choices'].values())-{0}
    if len(choices) > settings['max_full_training_fits']:
        raise ValueError('Declared full fitting budget exceeded')
    contract, _ = load_bundle(); final.mkdir(); models, fitted = {}, {}
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        atomic_json(final/'training_selection.json', selection)
        for name, i in selection['choices'].items():
            if not i:
                models[name] = {**reference, 'existing_previous': True}
                continue
            if i not in fitted:
                c = candidates(settings)[i]
                estimator = WavLMLayerClassifier(c, settings).fit(x, frame.severity_class.to_numpy(int))
                fitted[i] = {'protocol': PROTOCOL, 'estimator': estimator, 'candidate': c, 'feature': 'semantic_av',
                             'fit_participants': frame.participant_id.tolist()}
            path = final/f'{name}.pkl'; path.write_bytes(pickle.dumps(fitted[i]))
            np.testing.assert_allclose(predict_state(fitted[i], x), predict_state(pickle.loads(path.read_bytes()), x), rtol=0, atol=1e-14)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': fitted[i]['candidate'], 'existing_previous': False}
        bindings = {str(p): sha256(p) for p in [Path(__file__), CONFIG, out/'protocol.json', final/'training_selection.json',
            PREVIOUS_ROOT/'final/plan.json', Path('src/wavlm_layer_inference.py'), Path('src/wavlm_layer_features.py'),
            Path('src/wavlm_accuracy.py'), Path('src/age_language_accuracy.py'), Path('src/wavlm_features.py'),
            Path('src/wavlm_features_complete.py'), Path('src/checked_inference.py')]}
        new_plan = {'speech_encoder': encoder_contract(), 'speech_runtime': runtime_contract(), 'protocol': PROTOCOL,
            'context_hash': completion['context_hash'], 'models': models, 'bindings': bindings,
            'fit_participants': frame.participant_id.tolist(), 'feature_contract': contract,
            'feature_contract_hash': digest(contract),
            'inference_feature_keys': ['semantic_av', 'language', 'speech', 'final_nonzero', 'layer6_nonzero'],
            'new_full_training_fits': len(fitted), 'validation_test_scored': False,
            'selection_uses_outer_results': False, 'independent_confirmation': False}
        atomic_json(final/'plan.json', {'plan': new_plan, 'plan_hash': digest(new_plan)})
    return new_plan


def load_inputs():
    from .wavlm_layer_features import load_training, ROOT as feature_root
    x, frame, old_settings, old, provenance = previous.load_inputs()
    envelope = json.loads((PREVIOUS_ROOT/'protocol.json').read_text()); context = envelope['context']
    old_config = json.loads(previous.CONFIG.read_text())
    if (digest(context) != envelope['context_hash'] or context['source_sha256'] != sha256(previous.__file__)
            or context['settings'] != old_config or context['old_records_hash'] != digest(old)
            or context['input_hashes'] != {k: digest(v.tolist()) for k,v in x.items()}
            or context['provenance'] != provenance
            or context['training'] != frame.to_dict('records')
            or context['folds'] != nested_folds(frame, old_settings)):
        raise ValueError('Previous matched procedure changed')
    for path, expected in context['support_hashes'].items():
        if sha256(path) != expected:
            raise ValueError('Previous support implementation changed')
    augmented = []
    for fold, record in zip(context['folds'], old):
        directory = PREVIOUS_ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
        previous_record = load_record(directory/'result.json', envelope['context_hash'])
        previous.validate(previous_record, fold, record, frame.severity_class.to_numpy(int), old_config, directory)
        augmented.append({**record, 'previous_procedure': previous_record})
    features, speech_provenance = load_training(frame.participant_id.tolist())
    if set(features) != {'final_nonzero', 'layer6_nonzero'}:
        raise ValueError('Frozen new representation schema changed')
    x.update(features); bindings = dict(provenance['bindings'])
    for path in [Path('src/wavlm_layer_features.py'), PREVIOUS_ROOT/'protocol.json', PREVIOUS_ROOT/'completion.json',
                 feature_root/'protocol.json', feature_root/'completion.json', feature_root/'features.npz']:
        bindings[str(path)] = sha256(path)
    return x, frame, old_settings, augmented, {**provenance, 'bindings': bindings, 'layer_features': speech_provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'freeze']); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int); args = parser.parse_args()
    settings = json.loads(CONFIG.read_text()); inputs = load_inputs()
    result = (run(inputs, settings, resume=args.resume, max_new_jobs=args.max_new_jobs)
              if args.action == 'run' else freeze(inputs, settings))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    from .wavlm_layer_accuracy import main as module_main
    module_main()
