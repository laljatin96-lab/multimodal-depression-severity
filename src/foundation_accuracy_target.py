"""Nested tabular-foundation and shrinkage-model strategy selection toward a declared accuracy target."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import os
import socket
import torch
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .accuracy_model_comparison import label_metrics
from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest, write_json
from .crossvalidated_baselines import fit, five_probabilities
from .evaluate import write_report
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .raw_frame_uncertainty import METRICS, participant_counts, weighted_metrics
from . import semantic_accuracy_target as previous
from .download_tabular_foundation import MODEL, WEIGHTS, FILES

PROTOCOL = 'post-final-nested-foundation-strategy-target55-v1'


@contextmanager
def offline_network():
    """Fail closed if any library attempts an outgoing connection during modeling."""
    names = [(socket.socket, 'connect'), (socket.socket, 'connect_ex'),
             (socket.socket, 'sendto'), (socket, 'create_connection'), (socket, 'getaddrinfo')]
    originals = [(owner, name, getattr(owner, name)) for owner, name in names]
    def forbidden(*args, **kwargs):
        raise RuntimeError('Network access forbidden during confidential model processing')
    try:
        for owner, name, _ in originals:
            setattr(owner, name, forbidden)
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def foundation_class():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TABPFN_NO_BROWSER'] = '1'
    os.environ['SKB_DATA_DIRECTORY'] = str(Path('results/tabular_runtime/skrub').resolve())
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier


def candidates(settings):
    result = previous.candidates(settings)
    result += [{'family': 'tabpfn_v2', 'feature': f, 'components': n}
               for f in settings['feature_inputs'] for n in settings['foundation_components']]
    result += [{'family': 'lda', 'feature': f, 'covariance': c}
               for f in settings['feature_inputs'] for c in settings['lda_covariance']]
    return result


def estimator_for(candidate, settings):
    if candidate['family'] in ['majority', 'logistic']:
        return previous.estimator_for(candidate, settings)
    if candidate['family'] == 'tabpfn_v2':
        classifier = foundation_class()(n_estimators=settings['foundation_estimators'],
            model_path=str(Path(settings['foundation_directory'])/WEIGHTS),
            device='cpu', random_state=settings['fit_seed'], n_preprocessing_jobs=1,
            inference_precision=torch.float32, balance_probabilities=False,
            enable_torch_compile=False, fit_mode='fit_preprocessors', tuning_config=None)
        return make_pipeline(StandardScaler(), PCA(n_components=candidate['components'], svd_solver='full'), classifier)
    if candidate['family'] == 'lda':
        kwargs = {'shrinkage': 'auto'} if candidate['covariance'] == 'ledoit_wolf' else {'covariance_estimator': OAS()}
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', **kwargs))
    raise ValueError('Unknown classifier family')


def chosen_index(records):
    return max(range(len(records)), key=lambda i: (records[i]['correct'], records[i]['macro_f1'], -i))


def run_job(matrices, y, fold, settings, directory):
    start = time.monotonic()
    search = candidates(settings)
    scores = []
    for candidate in search:
        x = matrices[candidate['feature']]
        prediction = np.full(len(y), -1, int)
        for inner in fold['inner']:
            estimator = estimator_for(candidate, settings)
            fit(estimator, x[inner['fit']], y[inner['fit']])
            prediction[inner['heldout']] = estimator.predict(x[inner['heldout']])
        predicted = prediction[fold['fit']]
        metric = label_metrics(y[fold['fit']], predicted)
        scores.append({'candidate': candidate, 'predictions': predicted.tolist(),
                       'correct': int((predicted == y[fold['fit']]).sum()),
                       'accuracy': metric['accuracy'], 'macro_f1': metric['macro_f1']})
    choice = chosen_index(scores)
    candidate = search[choice]
    x = matrices[candidate['feature']]
    estimator = estimator_for(candidate, settings)
    fit(estimator, x[fold['fit']], y[fold['fit']])
    p = five_probabilities(estimator, x[fold['heldout']])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'model.pkl'
    temporary = path.with_suffix('.pkl.tmp')
    state = {'estimator': estimator, 'candidate': candidate, 'protocol': PROTOCOL,
             'fit': fold['fit'], 'heldout': fold['heldout'], 'feature_count': x.shape[1]}
    with temporary.open('wb') as handle:
        pickle.dump(state, handle)
    temporary.replace(path)
    with path.open('rb') as handle:
        restored = pickle.load(handle)
    np.testing.assert_allclose(p, five_probabilities(restored['estimator'], x[fold['heldout']]), rtol=0, atol=1e-14)
    return {'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
            'candidates': scores, 'selected_index': choice, 'selected_candidate': candidate, 'probabilities': p.tolist(),
            'fit_class_counts': np.bincount(y[fold['fit']], minlength=5).tolist(), 'checkpoint_parity': True,
            'model_sha256': sha256(path), 'fits': len(search)*len(fold['inner'])+1, 'seconds': time.monotonic()-start}


def validate_job(record, fold, settings, directory, y):
    for key in ['seed', 'fold', 'fit', 'heldout']:
        if record[key] != fold[key]:
            raise ValueError('Job fold identity changed')
    search = candidates(settings)
    if ([r['candidate'] for r in record['candidates']] != search or
            record['model_sha256'] != sha256(directory/'model.pkl') or not record['checkpoint_parity'] or
            record['fits'] != len(search)*len(fold['inner'])+1 or
            record['fit_class_counts'] != np.bincount(y[fold['fit']], minlength=5).tolist()):
        raise ValueError('Candidate/checkpoint identity changed')
    for result in record['candidates']:
        prediction = np.asarray(result['predictions'])
        metric = label_metrics(y[fold['fit']], prediction)
        if result['correct'] != int((prediction == y[fold['fit']]).sum()) or any(not np.isfinite(result[k]) or abs(result[k]-metric[k]) > 1e-12 for k in ['accuracy', 'macro_f1']):
            raise ValueError('Inner selection metrics changed')
    choice = chosen_index(record['candidates'])
    if choice != record['selected_index'] or search[choice] != record['selected_candidate']:
        raise ValueError('Saved candidate selection changed')
    p = np.asarray(record['probabilities'])
    if p.shape != (len(fold['heldout']), 5) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid outer probabilities')


def load_inputs(settings):
    source_settings = json.loads(Path(settings['source_config']).read_text())
    matrices, training, provenance = previous.load_inputs(source_settings)
    root = Path(settings['source_directory'])
    envelope = json.loads((root/'protocol.json').read_text()); context = envelope['context']
    expected = {'protocol': previous.PROTOCOL, 'settings': source_settings,
                'training': training.to_dict('records'), 'folds': nested_folds(training, source_settings),
                'provenance': provenance,
                'feature_hashes': {k: digest(np.asarray(matrices[k]).tolist()) for k in source_settings['feature_inputs']},
                'code_sha256': sha256(previous.__file__),
                'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    if context != expected or digest(context) != envelope['context_hash'] or not json.loads((root/'completion.json').read_text())['complete']:
        raise ValueError('Previous semantic experiment identity changed or incomplete')
    if nested_folds(training, settings) != context['folds']:
        raise ValueError('Retain existing training participants and folds')
    reference, bindings = [], {}
    y = training.severity_class.to_numpy(int)
    for fold in context['folds']:
        directory = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
        record = load_record(directory/'result.json', envelope['context_hash'])
        previous.validate_job(record, fold, source_settings, directory, y)
        reference.append({k: record[k] for k in ['seed','fold','fit','heldout','probabilities']})
        for name in ['result.json','model.pkl']:
            bindings[str(directory/name)] = sha256(directory/name)
    model_root = Path(settings['foundation_directory'])
    receipt = json.loads((model_root/'download_receipt.json').read_text())
    if receipt['model'] != MODEL or set(receipt['files']) != set(FILES):
        raise ValueError('Unexpected TabPFN checkpoint receipt')
    for name, item in receipt['files'].items():
        if sha256(model_root/name) != item['sha256']:
            raise ValueError('Foundation checkpoint asset changed')
    return matrices, training, {'source_context_hash': envelope['context_hash'], 'source_provenance': provenance,
        'reference_records': reference, 'reference_bindings': bindings, 'foundation': receipt,
        'source_config_sha256': sha256(settings['source_config']),
        'previous_code_sha256': sha256(previous.__file__), 'local_network_guard': True}


def summarize(training, records, settings, out, provenance):
    y, rows, all_predictions = training.severity_class.to_numpy(int), [], {}
    names = ['selected_strategy', 'majority'] + (['previous_semantic'] if provenance.get('reference_records') else [])
    for name in names:
        repeated = []
        for seed in settings['fold_seeds']:
            p, owners = np.full((len(y), 5), np.nan), np.zeros(len(y), int)
            source_records = provenance['reference_records'] if name == 'previous_semantic' else records
            for r in source_records:
                if r['seed'] != seed:
                    continue
                p[r['heldout']] = r['probabilities'] if name != 'majority' else np.eye(5)[np.bincount(y[r['fit']], minlength=5).argmax()]
                owners[r['heldout']] += 1
            if not (owners == 1).all() or not np.isfinite(p).all():
                raise ValueError('Incomplete participant OOF coverage')
            metric = write_report(training, p, out/'oof'/f'{name}_seed{seed}', name, seed, 'train_nested_oof')
            repeated.append(p.argmax(1))
            rows.append({'model': name, 'seed': seed, **{k: metric[k] for k in METRICS},
                         'severe_correct': int(((p.argmax(1) == 4) & (y == 4)).sum()),
                         'predicted_class_counts': json.dumps(np.bincount(p.argmax(1), minlength=5).tolist())})
        all_predictions[name] = np.stack(repeated)
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'repeat_scores.csv', index=False)
    summary = scores.groupby('model')[METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'summary.csv')
    counts = participant_counts(len(y), settings['bootstrap_repetitions'], settings['bootstrap_seed'])
    boot = {n: weighted_metrics(y, p, counts) for n, p in all_predictions.items()}
    comparisons = []
    for reference in names[1:]:
        ci = np.percentile(boot['selected_strategy']-boot[reference], [2.5,97.5], axis=0)
        point = summary.loc['selected_strategy']-summary.loc[reference]
        comparisons += [{'reference': reference, 'metric': k, 'difference': point[k+'_mean'],
                         'ci95_low': ci[0,i], 'ci95_high': ci[1,i]} for i,k in enumerate(METRICS)]
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv', index=False)
    pd.DataFrame([{'seed': r['seed'], 'fold': r['fold'], **r['selected_candidate']} for r in records]).to_csv(out/'outer_choices.csv', index=False)
    actual = float(summary.loc['selected_strategy', 'accuracy_mean'])
    target = {'target_accuracy': settings['target_accuracy'], 'observed_nested_accuracy': actual,
              'internal_target_reached': actual >= settings['target_accuracy'], 'independent_confirmation': False,
              'majority_accuracy': float(summary.loc['majority', 'accuracy_mean']),
              'previous_semantic_accuracy': float(summary.loc['previous_semantic', 'accuracy_mean']) if 'previous_semantic' in names else None}
    write_json(out/'target_status.json', target)
    (out/'REPORT.md').write_text(f"# Nested foundation-model strategy selection\n\nTarget: {settings['target_accuracy']:.1%}; observed mean training OOF accuracy: {actual:.2%}; majority: {target['majority_accuracy']:.2%}. Internal target reached: {target['internal_target_reached']}.\n\nModel family, feature input and settings were selected inside each outer training fold using pooled inner predictions. Only selected strategies were scored on outer folds. All candidates, choices, class/group reports and checkpoint checks are saved.\n\nAll supplied sources remain accounted for; only official training participants were fitted/scored. This is post-final exploratory development, not independent test confirmation. Intervals condition on fitted models and repeated partitions share participants. No test/validation scoring, full-training deployment refit or reference replacement occurred.\n")
    return target


def run_experiment(matrices, training, provenance, settings, output, resume=False, max_new_jobs=None, verify_inputs=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing experiment; use --resume')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires protocol')
    if not 0 < settings['target_accuracy'] <= 1 or not 1 <= settings['cpu_threads'] <= 2 or settings['max_iter'] < 1:
        raise ValueError('Invalid target/thread/iteration settings')
    training = training.reset_index(drop=True)
    y = training.severity_class.to_numpy(int)
    if training.participant_id.duplicated().any() or not np.array_equal(y, training.severity_class) or set(y) != set(range(5)):
        raise ValueError('Invalid unique training participants/classes')
    for feature in settings['feature_inputs']:
        if len(matrices[feature]) != len(y) or not np.isfinite(matrices[feature]).all():
            raise ValueError('Invalid training feature matrix')
    folds = nested_folds(training, settings)
    fits_per_job = len(candidates(settings))*settings['inner_folds']+1
    if len(folds) != settings['expected_jobs'] or len(folds)*fits_per_job != settings['expected_fits'] or (max_new_jobs is not None and max_new_jobs < 1):
        raise ValueError('Declared budget mismatch')
    context = {'protocol': PROTOCOL, 'settings': settings, 'training': training.to_dict('records'), 'folds': folds,
               'provenance': provenance, 'feature_hashes': {k: digest(np.asarray(matrices[k]).tolist()) for k in settings['feature_inputs']},
               'code_sha256': sha256(__file__), 'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__,
               'tabpfn': importlib.metadata.version('tabpfn'), 'skrub': importlib.metadata.version('skrub')},
               'support_code_sha256': {'previous': sha256(previous.__file__), 'download': sha256(Path(__file__).with_name('download_tabular_foundation.py'))}}
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings['cpu_threads'])
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        envelope = {'context': context, 'context_hash': context_hash}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != envelope:
                raise ValueError('Inputs/code/protocol changed')
        else:
            atomic_json(out/'protocol.json', envelope)
        records, new, reused = [], 0, 0
        for fold in folds:
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            path = directory/'result.json'
            if path.exists():
                record = load_record(path, context_hash)
                reused += 1
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                record = run_job(matrices, y, fold, settings, directory)
                save_record(path, record, context_hash)
                new += 1
            validate_job(record, fold, settings, directory, y)
            records.append(record)
            print(f'Foundation strategy: {len(records)}/{len(folds)} jobs; {new} new, {reused} reused', flush=True)
        if verify_inputs:
            verify_inputs()
        complete = len(records) == len(folds)
        target = summarize(training, records, settings, out, provenance) if complete else None
        result = {'complete': complete, 'context_hash': context_hash, 'completed_jobs': len(records), 'expected_jobs': len(folds),
                  'completed_fits': len(records)*fits_per_job, 'new_jobs': new, 'reused_jobs': reused, 'new_fits': new*fits_per_job,
                  'fit_seconds': sum(r['seconds'] for r in records), 'target': target, 'validation_test_scoring': False, 'reference_replaced': False}
        atomic_json(out/'completion.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/foundation_accuracy_target.json')
    parser.add_argument('--output', default='results/foundation_accuracy_target')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    if any(Path(args.output).resolve().is_relative_to(Path(p).resolve()) for p in ['data','dataset','checkpoints',settings['source_directory'],'results/transcript_corpus','results/semantic_features']):
        parser.error('Output must be outside original inputs')
    matrices, training, provenance = load_inputs(settings)
    def check():
        x, frame, source = load_inputs(settings)
        if source != provenance or not frame.equals(training) or any(not np.array_equal(x[k], matrices[k]) for k in matrices):
            raise ValueError('Inputs changed during experiment')
    print(json.dumps(run_experiment(matrices, training, provenance, settings, args.output, args.resume, args.max_new_jobs, check), indent=2))


if __name__ == '__main__':
    main()
