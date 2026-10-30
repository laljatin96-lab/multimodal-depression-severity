"""Nested local interview-text and AV strategy selection toward a declared accuracy target."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from threadpoolctl import threadpool_limits

from .accuracy_model_comparison import label_metrics
from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest, write_json
from .crossvalidated_baselines import fit, five_probabilities
from .evaluate import write_report
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .raw_frame_uncertainty import METRICS, participant_counts, weighted_metrics
from .ensemble_accuracy_target import load_inputs as numeric_inputs
from .transcript_corpus import load_verified as verified_corpus

PROTOCOL = 'post-final-local-text-av-strategy-target55-v1'


def candidates(settings):
    return [{'family': 'majority', 'feature': 'av'}]+[
        {'family': 'logistic', 'feature': feature, 'C': c}
        for feature in settings['feature_inputs'] for c in settings['C_candidates'][feature]]


def estimator_for(candidate, settings):
    if candidate['family'] == 'majority':
        return DummyClassifier(strategy='most_frequent')
    feature = candidate['feature']
    numeric = [f'av{i}' for i in range(128)]
    if feature == 'av':
        transform = ColumnTransformer([('av', StandardScaler(), numeric)], remainder='drop')
    else:
        options = settings['text']
        char = feature == 'text_char'
        vectorizer = TfidfVectorizer(analyzer='char_wb' if char else 'word',
                        ngram_range=tuple(options['char_ngrams'] if char else options['word_ngrams']),
                        min_df=options['min_df'], max_df=options['max_df'],
                        max_features=options['char_max_features'] if char else options['word_max_features'],
                        sublinear_tf=options['sublinear_tf'], norm='l2', lowercase=True,
                        token_pattern=None if char else r"(?u)\b[a-z][a-z']*\b")
        transformers = [('text', vectorizer, 'text')]
        weights = {'text': 1.0}
        if feature == 'text_av':
            transformers.append(('av', make_pipeline(StandardScaler(), Normalizer(norm='l2')), numeric))
            weights['av'] = settings['fusion_av_weight']
        transform = ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0, transformer_weights=weights)
    return make_pipeline(transform, LogisticRegression(C=candidate['C'], solver='lbfgs', max_iter=settings['max_iter'],
                                                       random_state=settings['fit_seed'], class_weight=None))


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
            fit(estimator, x.iloc[inner['fit']], y[inner['fit']])
            prediction[inner['heldout']] = estimator.predict(x.iloc[inner['heldout']])
        predicted = prediction[fold['fit']]
        metric = label_metrics(y[fold['fit']], predicted)
        scores.append({'candidate': candidate, 'predictions': predicted.tolist(),
                       'correct': int((predicted == y[fold['fit']]).sum()),
                       'accuracy': metric['accuracy'], 'macro_f1': metric['macro_f1']})
    choice = chosen_index(scores)
    candidate = search[choice]
    x = matrices[candidate['feature']]
    estimator = estimator_for(candidate, settings)
    fit(estimator, x.iloc[fold['fit']], y[fold['fit']])
    p = five_probabilities(estimator, x.iloc[fold['heldout']])
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
    np.testing.assert_allclose(p, five_probabilities(restored['estimator'], x.iloc[fold['heldout']]), rtol=0, atol=1e-14)
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
    numeric_config = json.loads(Path(settings['numeric_config']).read_text())
    matrices, training, numeric_provenance = numeric_inputs(numeric_config)
    documents, corpus_bindings = verified_corpus(settings['frozen_plan'], settings['corpus_directory'])
    if [{k:r[k] for k in ['participant_id', 'split']} for r in documents] != numeric_provenance['coverage']:
        raise ValueError('Text/AV participant coverage or splits differ')
    by_id = {r['participant_id']: r['text'] for r in documents}
    numeric = pd.DataFrame(matrices['cached_av'], columns=[f'av{i}' for i in range(128)])
    text = pd.DataFrame({'text': [by_id[pid] for pid in training.participant_id]})
    inputs = {'text_word': text, 'text_char': text, 'text_av': pd.concat([text, numeric], axis=1), 'av': numeric}
    return inputs, training, {'numeric_provenance': numeric_provenance, 'corpus_bindings': corpus_bindings,
                             'numeric_config_sha256': sha256(settings['numeric_config']),
                             'text_processor_sha256': sha256(Path(__file__).with_name('transcript_corpus.py'))}


def summarize(training, records, settings, out):
    y, rows, all_predictions = training.severity_class.to_numpy(int), [], {}
    for name in ['selected_strategy', 'majority']:
        repeated = []
        for seed in settings['fold_seeds']:
            p, owners = np.full((len(y), 5), np.nan), np.zeros(len(y), int)
            for r in records:
                if r['seed'] != seed:
                    continue
                p[r['heldout']] = r['probabilities'] if name == 'selected_strategy' else np.eye(5)[np.bincount(y[r['fit']], minlength=5).argmax()]
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
    ci = np.percentile(boot['selected_strategy']-boot['majority'], [2.5, 97.5], axis=0)
    point = summary.loc['selected_strategy']-summary.loc['majority']
    pd.DataFrame([{'metric': k, 'difference': point[k+'_mean'], 'ci95_low': ci[0,i], 'ci95_high': ci[1,i]} for i,k in enumerate(METRICS)]).to_csv(out/'paired_comparisons.csv', index=False)
    pd.DataFrame([{'seed': r['seed'], 'fold': r['fold'], **r['selected_candidate']} for r in records]).to_csv(out/'outer_choices.csv', index=False)
    actual = float(summary.loc['selected_strategy', 'accuracy_mean'])
    target = {'target_accuracy': settings['target_accuracy'], 'observed_nested_accuracy': actual,
              'internal_target_reached': actual >= settings['target_accuracy'], 'independent_confirmation': False,
              'majority_accuracy': float(summary.loc['majority', 'accuracy_mean'])}
    write_json(out/'target_status.json', target)
    (out/'REPORT.md').write_text(f"# Nested interview-text strategy selection\n\nTarget: {settings['target_accuracy']:.1%}; observed mean training OOF accuracy: {actual:.2%}; majority: {target['majority_accuracy']:.2%}. Internal target reached: {target['internal_target_reached']}.\n\nInterview-text representation, fusion and regularization were selected inside each outer training fold using pooled inner predictions. Only selected strategies were scored on outer folds. All candidates, choices, class/group reports and checkpoint checks are saved.\n\nAll supplied sources remain accounted for; only official training participants were fitted/scored. This is post-final exploratory development, not independent test confirmation. Intervals condition on fitted models and repeated partitions share participants. No test/validation scoring, full-training deployment refit or reference replacement occurred.\n")
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
        if len(matrices[feature]) != len(y) or not np.isfinite(matrices[feature].select_dtypes(include='number')).all().all():
            raise ValueError('Invalid training feature matrix')
    folds = nested_folds(training, settings)
    fits_per_job = len(candidates(settings))*settings['inner_folds']+1
    if len(folds) != settings['expected_jobs'] or len(folds)*fits_per_job != settings['expected_fits'] or (max_new_jobs is not None and max_new_jobs < 1):
        raise ValueError('Declared budget mismatch')
    context = {'protocol': PROTOCOL, 'settings': settings, 'training': training.to_dict('records'), 'folds': folds,
               'provenance': provenance, 'feature_hashes': {k: digest(matrices[k].to_dict('list')) for k in settings['feature_inputs']},
               'code_sha256': sha256(__file__), 'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']):
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
            print(f'Text strategy: {len(records)}/{len(folds)} jobs; {new} new, {reused} reused', flush=True)
        if verify_inputs:
            verify_inputs()
        complete = len(records) == len(folds)
        target = summarize(training, records, settings, out) if complete else None
        result = {'complete': complete, 'context_hash': context_hash, 'completed_jobs': len(records), 'expected_jobs': len(folds),
                  'completed_fits': len(records)*fits_per_job, 'new_jobs': new, 'reused_jobs': reused, 'new_fits': new*fits_per_job,
                  'fit_seconds': sum(r['seconds'] for r in records), 'target': target, 'validation_test_scoring': False, 'reference_replaced': False}
        atomic_json(out/'completion.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/text_accuracy_target.json')
    parser.add_argument('--output', default='results/text_accuracy_target')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    if any(Path(args.output).resolve().is_relative_to(Path(p).resolve()) for p in ['data', 'dataset', 'checkpoints', settings['corpus_directory']]):
        parser.error('Output must be outside original data/models and the corpus')
    matrices, training, provenance = load_inputs(settings)
    def check():
        x, frame, source = load_inputs(settings)
        if source != provenance or not frame.equals(training) or any(not x[k].equals(matrices[k]) for k in matrices):
            raise ValueError('Inputs changed during experiment')
    print(json.dumps(run_experiment(matrices, training, provenance, settings, args.output, args.resume, args.max_new_jobs, check), indent=2))


if __name__ == '__main__':
    main()
