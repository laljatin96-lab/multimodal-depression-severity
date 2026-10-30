"""Nested external-cohort text transfer with matched no-external ablation and preserved 49.28% control."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import decide, score, summarize

OUTPUT = Path('results/external_transfer_accuracy')
CONFIG = Path('configs/external_transfer_accuracy.json')
PROTOCOL = 'training-only-emu-domain-text-transfer-v1'
LANGUAGE_ROOT = Path('results/age_language_accuracy')


_external_cache = None


def external_inputs():
    global _external_cache
    if _external_cache is None:
        from .external_emu import load_verified
        _external_cache = load_verified()
    return _external_cache


class ExternalTransferClassifier:
    """Shared text features plus an interview-specific text/AV component."""
    def __init__(self, candidate, settings):
        self.candidate, self.settings = dict(candidate), dict(settings)

    def fit(self, x, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        y=np.asarray(y)
        if y.shape != (len(x['semantic_av']),) or not np.array_equal(y,y.astype(int)) or set(y)!=set(range(5)):
            raise ValueError('Five classes and real fitting participants required')
        if not np.isfinite(x['semantic_av']).all() or not np.isfinite(x['language']).all():
            raise ValueError('Invalid fitting features')
        y=y.astype(int); self.real_count=len(y); self.class_counts=np.bincount(y,minlength=5)
        self.classes_=np.arange(5); dim=self.settings['semantic_dimensions']
        if self.candidate['external_mass']:
            source,source_y,source_ids,_=external_inputs()
            if source['semantic'].shape[1]!=dim or set(source_y)!=set(range(5)):
                raise ValueError('External feature/target contract differs')
            semantic=np.vstack([x['semantic_av'][:,:dim],source['semantic']])
            language=np.vstack([x['language'],source['language']])
        else:
            source={'semantic':np.empty((0,dim)), 'language':np.empty((0,x['language'].shape[1]))}
            source_y=np.array([],int);source_ids=[]
            semantic=x['semantic_av'][:,:dim]; language=x['language']
        self.external_count=len(source_y); self.external_participants=list(source_ids)
        self.external_class_counts=np.bincount(source_y,minlength=5)
        self.shared_scaler=StandardScaler().fit(semantic)
        self.shared_pca=PCA(n_components=self.settings['text_components'],svd_solver='full').fit(self.shared_scaler.transform(semantic))
        self.language_scaler=StandardScaler().fit(language)
        # No AV values are fabricated for EMU. This entire branch belongs to original interviews.
        av=x['semantic_av'][:,dim:]
        self.av_scaler=StandardScaler().fit(av)
        self.av_pca=PCA(n_components=self.settings['av_components'],svd_solver='full').fit(self.av_scaler.transform(av))
        own=self.transform(x)
        weights=np.sqrt(len(y)/(5*self.class_counts[y]));weights*=len(y)/weights.sum()
        if self.external_count:
            common=self.common_text(source['semantic'],source['language'])
            # Zeroes route source examples only through the shared head, not fake AV measurements.
            external=np.column_stack([common,np.zeros((len(source_y),common.shape[1]+self.settings['av_components'])),np.zeros(len(source_y))])
            own=np.vstack([own,external])
            source_weights=np.sqrt(len(source_y)/(5*self.external_class_counts[source_y]))
            source_weights*=self.candidate['external_mass']*len(y)/source_weights.sum()
            weights=np.r_[weights,source_weights]
        weights*=len(y)/weights.sum()
        self.domain_weight_sums=[float(weights[:len(y)].sum()),float(weights[len(y):].sum())]
        labels=np.r_[y,source_y]
        self.total_fitting_rows=len(labels)
        self.model=LogisticRegression(C=self.candidate['C'],solver='lbfgs',max_iter=3000,random_state=self.settings['fit_seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            self.model.fit(own,labels,sample_weight=weights)
        return self

    def common_text(self, semantic, language):
        return np.column_stack([self.shared_pca.transform(self.shared_scaler.transform(semantic)),
                                self.settings['language_scale']*self.language_scaler.transform(language)])

    def transform(self,x):
        dim=self.settings['semantic_dimensions']
        common=self.common_text(x['semantic_av'][:,:dim],x['language'])
        av=self.av_pca.transform(self.av_scaler.transform(x['semantic_av'][:,dim:]))
        return np.column_stack([common,self.candidate['adaptation_scale']*common,av,np.ones(len(common))])

    def predict_proba(self,x):
        p=self.model.predict_proba(self.transform(x));decide(p,None)
        return p


def candidates(settings):
    return [{'family':'language_control'}]+[{'family':'external_text_transfer','external_mass':mass,
        'adaptation_scale':scale,'C':c} for mass in settings['external_masses']
        for scale in settings['adaptation_scales'] for c in settings['C']]


def options(inner, old, y, settings):
    control = old['language_control']; i = control['selected']['without_age']['selection']['index']
    predictions = [decide(np.asarray(control['inner_probabilities'])[i-1], None)]+[decide(p, None) for p in inner]
    return [{'candidate': c, **score(y, p)} for c, p in zip(candidates(settings), predictions)]


def choose(rows, policy):
    if policy == 'language_control': return 0
    if policy not in {'accuracy','transfer_only','no_external'}: raise ValueError('Unknown policy')
    allowed = [i for i,r in enumerate(rows) if (i == 0 and policy == 'accuracy') or
               (i > 0 and ((r['candidate']['external_mass'] > 0 and policy in {'accuracy','transfer_only'}) or
                            (r['candidate']['external_mass'] == 0 and policy == 'no_external')))]
    return max(allowed,key=lambda i:(rows[i]['correct'],rows[i]['macro_f1'],-i))


def predict_state(state, x):
    if state['candidate']['family'] in {'language_control', 'block_logistic'}:
        return state['estimator'].predict_proba({'base': x['semantic_av'], 'language': x['language']})
    return state['estimator'].predict_proba(x)


def subset(x, indices):
    return {k: np.asarray(v)[indices] for k, v in x.items()}


def job(x, frame, fold, old, settings, out):
    started = time.monotonic(); y = frame.severity_class.to_numpy(int); search = candidates(settings)
    inner = np.full((len(search)-1, len(y), 5), np.nan)
    for part in fold['inner']:
        for i, c in enumerate(search[1:]):
            model = ExternalTransferClassifier(c, settings).fit(subset(x, part['fit']), y[part['fit']])
            inner[i, part['heldout']] = model.predict_proba(subset(x, part['heldout']))
    inner = inner[:, fold['fit']]; rows = options(inner, old, y[fold['fit']], settings)
    out.mkdir(parents=True, exist_ok=True); selected, fitted = {}, {}
    for name in settings['policies']:
        i = choose(rows, name); c = search[i]
        if i not in fitted:
            if i:
                model = ExternalTransferClassifier(c, settings).fit(subset(x, fold['fit']), y[fold['fit']])
                state = {'estimator': model, 'candidate': c, 'feature': 'semantic_av'}
            else:
                source = LANGUAGE_ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'without_age.pkl'
                original = pickle.loads(source.read_bytes())
                state = {'estimator': original['estimator'], 'candidate': c,
                         'source_candidate': original['candidate']}
            fitted[i] = {**state, 'protocol': PROTOCOL, 'fit': fold['fit'], 'heldout': fold['heldout']}
        state = fitted[i]; heldout = {k: v[fold['heldout']] for k, v in x.items()}
        p = predict_state(state, heldout)
        if not i:
            np.testing.assert_allclose(p, old['language_control']['selected']['without_age']['probabilities'], rtol=0, atol=1e-14)
        path = out/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
        np.testing.assert_allclose(p, predict_state(pickle.loads(path.read_bytes()), heldout), rtol=0, atol=1e-14)
        training = {k: v[fold['fit']] for k, v in x.items()}
        selected[name] = {'selection': {'index': i}, 'candidate': c, 'threshold': None,
            'probabilities': p.tolist(), 'predictions': p.argmax(1).tolist(), 'checkpoint_sha256': sha256(path),
            'real_training_accuracy': float((predict_state(state, training).argmax(1) == y[fold['fit']]).mean())}
    return {**{k: fold[k] for k in ['seed', 'fold', 'fit', 'heldout']}, 'inner_probabilities': inner.tolist(),
        'options': rows, 'selected': selected, 'inner_receipts': fold['inner'],
        'fits': (len(search)-1)*len(fold['inner'])+sum(i > 0 for i in fitted), 'seconds': time.monotonic()-started}


def validate(record, fold, old, y, settings, directory):
    if (any(record[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout'])
            or record['inner_receipts'] != fold['inner'] or set(record['selected']) != set(settings['policies'])):
        raise ValueError('Fold or training ownership differs')
    if np.asarray(record['inner_probabilities']).shape != (len(candidates(settings))-1, len(fold['fit']), 5):
        raise ValueError('Incomplete inner coverage')
    rows = options(record['inner_probabilities'], old, y[fold['fit']], settings)
    if rows != record['options']:
        raise ValueError('Inner options changed')
    for name, result in record['selected'].items():
        i = choose(rows, name)
        if result['selection'] != {'index': i} or result['candidate'] != rows[i]['candidate']:
            raise ValueError('Selection changed')
        if np.asarray(result['probabilities']).shape != (len(fold['heldout']), 5) or decide(result['probabilities'], None).tolist() != result['predictions']:
            raise ValueError('Invalid probabilities')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')


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
    maximum = sum((len(candidates(settings))-1)*len(f['inner'])+len(settings['policies']) for f in folds)
    if maximum > settings['max_nested_pipeline_fits']:
        raise ValueError('Declared fit budget exceeded')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
        'provenance': provenance, 'source_sha256': sha256(__file__), 'input_hashes': {k: digest(v.tolist()) for k,v in x.items()},
        'old_records_hash': digest(old), 'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'support_hashes': {str(Path('src')/name): sha256(Path('src')/name) for name in
            ['severe_case_diagnosis.py', 'severe_detection_study.py', 'raw_frame_uncertainty.py', 'batch_inference.py', 'external_transfer_inference.py', 'external_emu.py', 'age_language_accuracy.py']}}
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
            print(f"External text transfer: {len(records)}/{len(folds)} jobs; {record['seconds']:.1f}s/job", flush=True)
        complete = len(records) == len(folds)
        # A completed resume validates caches and returns without rewriting evidence.
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
    baseline = np.concatenate([decide(np.asarray(r['language_control']['inner_probabilities'])[r['language_control']['selected']['without_age']['selection']['index']-1], None) for r in old])
    inner = np.concatenate([np.asarray(r['inner_probabilities']) for r in records], axis=1)
    rows = [{'candidate': {'family': 'language_control'}, **score(labels, baseline)}]
    rows += [{'candidate': c, **score(labels, decide(p, None))} for c, p in zip(candidates(settings)[1:], inner)]
    return {'rows': rows, 'choices': {n: choose(rows, n) for n in settings['policies']},
            'inner_predictions_per_participant': int(owners[0]), 'outer_results_used': False}


def freeze(inputs, settings, out=OUTPUT):
    from .best_accuracy_inference import load_bundle
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
    from .age_language_finalize import frozen_plan
    previous = frozen_plan()[0]['models']['without_age']
    if sha256(previous['path']) != previous['sha256']:
        raise ValueError('Baseline checkpoint changed')
    original = pickle.loads(Path(previous['path']).read_bytes())
    if original['fit_participants'] != frame.participant_id.tolist():
        raise ValueError('Baseline fitting participants changed')
    contract, _ = load_bundle(); final.mkdir(); models, fitted = {}, {}
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        atomic_json(final/'training_selection.json', selection)
        for name, i in selection['choices'].items():
            if not i:
                models[name] = {**previous, 'existing_language_control': True}
                continue
            if i not in fitted:
                c = candidates(settings)[i]
                estimator = ExternalTransferClassifier(c, settings).fit(x, frame.severity_class.to_numpy(int))
                fitted[i] = {'protocol': PROTOCOL, 'estimator': estimator, 'candidate': c, 'feature': 'semantic_av',
                             'fit_participants': frame.participant_id.tolist()}
            path = final/f'{name}.pkl'; path.write_bytes(pickle.dumps(fitted[i]))
            np.testing.assert_allclose(predict_state(fitted[i], x), predict_state(pickle.loads(path.read_bytes()), x), rtol=0, atol=1e-14)
            models[name] = {'path': str(path), 'sha256': sha256(path), 'candidate': fitted[i]['candidate'], 'existing_language_control': False}
        bindings = {str(p): sha256(p) for p in [Path(__file__), CONFIG, out/'protocol.json', final/'training_selection.json',
            Path('src/external_transfer_inference.py'), Path('src/external_emu.py'), Path('src/age_language_accuracy.py')]}
        plan = {'protocol': PROTOCOL, 'context_hash': completion['context_hash'], 'models': models,
            'bindings': bindings, 'fit_participants': frame.participant_id.tolist(), 'feature_contract': contract,
            'feature_contract_hash': digest(contract), 'inference_feature_keys': ['semantic_av', 'language'], 'new_full_training_fits': len(fitted),
            'external_training_participants': external_inputs()[2], 'validation_test_scored': False, 'selection_uses_outer_results': False, 'independent_confirmation': False}
        atomic_json(final/'plan.json', {'plan': plan, 'plan_hash': digest(plan)})
    return plan


def load_inputs():
    global _external_cache
    from .external_emu import load_verified
    _external_cache = load_verified()
    from . import age_language_accuracy as language
    from .age_language_finalize import frozen_plan
    x, frame, old_settings, original, provenance = load_baseline()
    done = json.loads((LANGUAGE_ROOT/'completion.json').read_text())
    env = json.loads((LANGUAGE_ROOT/'protocol.json').read_text())
    if not done['complete'] or digest(env['context']) != env['context_hash'] or done['context_hash'] != env['context_hash']:
        raise ValueError('Incomplete or changed 49.28% control')
    frozen_plan()  # Read-only verification; never run the old study again.
    settings = json.loads(Path('configs/age_language_accuracy.json').read_text()); old = []
    bindings = dict(provenance['bindings'])
    for fold, baseline in zip(nested_folds(frame, old_settings), original):
        directory = LANGUAGE_ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
        r = load_record(directory/'result.json', done['context_hash'])
        language.validate(r, fold, baseline, frame.severity_class.to_numpy(int), settings, directory)
        old.append({**baseline, 'language_control': r})
        for path in [directory/'result.json', directory/'without_age.pkl']:
            bindings[str(path)] = sha256(path)
    docs = json.loads(Path('results/transcript_corpus/participants.json').read_text())
    text = {r['participant_id']: r for r in docs}
    if any(text[p]['split'] != 'train' for p in frame.participant_id):
        raise ValueError('Language participant split differs')
    x['language'] = np.stack([language.language_features(text[p]['text']) for p in frame.participant_id])
    x = {k: x[k] for k in ['semantic_av', 'language']}
    bindings['results/transcript_corpus/participants.json'] = sha256('results/transcript_corpus/participants.json')
    bindings.update(_external_cache[3])
    return x, frame, old_settings, old, {**provenance, 'bindings': bindings, 'language_control_context': done['context_hash']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'freeze']); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int); args = parser.parse_args()
    settings = json.loads(CONFIG.read_text()); inputs = load_inputs()
    result = (run(inputs, settings, resume=args.resume, max_new_jobs=args.max_new_jobs)
              if args.action == 'run' else freeze(inputs, settings))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    from .external_transfer_accuracy import main as module_main
    module_main()
