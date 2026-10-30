"""New low-dimensional age/language blocks appended after fold-local semantic/AV PCA."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .crossvalidated_baselines import fit
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import decide, score, summarize

OUTPUT = Path('results/age_language_accuracy')
PROTOCOL = 'fold-local-age-language-block-logistic-v1'
METADATA = Path('data/metadata/official_edaic/detailed_lables.csv')
CORPUS = Path('results/transcript_corpus/participants.json')
# Fixed descriptive word groups, not clinical scores or a validated lexicon.
GROUPS = {
    'first_singular': "i me my mine myself i'm i've i'd i'll".split(),
    'first_plural': "we us our ours ourselves we're we've we'd we'll".split(),
    'second_person': "you your yours yourself yourselves you're you've you'd you'll".split(),
    'third_person': "he him his she her hers they them their theirs".split(),
    'negation': "no not never neither nor nothing nobody nowhere don't doesn't didn't can't couldn't won't wouldn't isn't wasn't aren't weren't haven't hasn't hadn't".split(),
    'absolute_terms': "always never everything nothing everyone nobody completely totally entirely absolutely".split(),
    'hedges': "maybe perhaps probably possibly sometimes somewhat usually guess suppose".split(),
    'negative_affect': "sad unhappy depressed anxious worried scared afraid lonely angry upset hopeless worthless guilty".split(),
    'positive_affect': "happy glad enjoy enjoyed enjoying love loved fun good great excited hopeful".split(),
    'social_words': "friend friends family mother father sister brother wife husband partner children kids people".split(),
    'sleep_energy': "sleep sleeping asleep insomnia tired fatigue exhausted energy awake bed".split(),
    'thinking_words': "think thinking thought know believe remember understand feel feeling".split()
}
LANGUAGE_NAMES = ['log_words', 'mean_word_length', 'moving_type_token_ratio', 'long_word_fraction']+list(GROUPS)


def language_features(text):
    words = str(text).split()
    if not words:
        raise ValueError('Nonempty cleaned transcript required')
    counts = Counter(words); n = len(words)
    windows = [words[i:i+100] for i in range(0, max(n-99, 1), 50)]
    moving_ttr = np.mean([len(set(w))/len(w) for w in windows])
    values = [np.log1p(n), np.mean([len(w) for w in words]), moving_ttr, np.mean([len(w) >= 7 for w in words])]
    values += [sum(counts[w] for w in terms)/n for terms in GROUPS.values()]
    return np.asarray(values, float)


def extra_features(frame):
    metadata = pd.read_csv(METADATA, usecols=['Participant','age'], dtype={'Participant': str}).set_index('Participant')
    if metadata.index.duplicated().any():
        raise ValueError('Duplicate age ownership')
    ages = pd.to_numeric(metadata.loc[frame.participant_id, 'age'], errors='raise').to_numpy(float)
    if np.isinf(ages).any() or ((ages[np.isfinite(ages)] < 18) | (ages[np.isfinite(ages)] > 100)).any():
        raise ValueError('Invalid recorded adult age')
    docs = json.loads(CORPUS.read_text()); by_id = {r['participant_id']: r for r in docs}
    if len(by_id) != len(docs) or any(by_id[r.participant_id]['split'] != r.split for r in frame.itertuples()):
        raise ValueError('Transcript/participant alignment differs')
    age = np.column_stack([ages, (ages-40)**2, np.isnan(ages).astype(float)])
    language = np.stack([language_features(by_id[p]['text']) for p in frame.participant_id])
    return {'age': age, 'language': language, 'age_language': np.column_stack([age, language])}


class BlockProjection:
    def fit(self, x, components):
        self.base_scaler = StandardScaler().fit(x['base'])
        self.pca = PCA(n_components=components, svd_solver='full').fit(self.base_scaler.transform(x['base']))
        self.imputers = {}; self.scalers = {}; self.real_count = len(x['base'])
        for name in ['age','language','age_language']:
            self.imputers[name] = SimpleImputer(strategy='median', keep_empty_features=True).fit(x[name])
            self.scalers[name] = StandardScaler().fit(self.imputers[name].transform(x[name]))
        return self

    def transform(self, x, candidate):
        base = self.pca.transform(self.base_scaler.transform(x['base']))
        name = candidate['feature_set']
        extra = self.scalers[name].transform(self.imputers[name].transform(x[name]))
        return np.column_stack([base, candidate['block_scale']*extra])


class BlockClassifier:
    def __init__(self, candidate, settings):
        self.candidate, self.settings = dict(candidate), dict(settings)

    def fit(self, x, y, projection=None):
        y = np.asarray(y)
        if y.shape != (len(x['base']),) or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('Real fitting rows and five integer classes required')
        if not np.isfinite(x['base']).all() or not np.isfinite(x['language']).all():
            raise ValueError('Nonfinite semantic or language features')
        y = y.astype(int); self.real_count = len(y); self.class_counts = np.bincount(y, minlength=5)
        self.projection = projection if projection is not None else BlockProjection().fit(x, self.settings['components'])
        if self.projection.real_count != len(y):
            raise ValueError('Projection population differs')
        self.model = LogisticRegression(C=self.candidate['C'], solver='lbfgs', max_iter=3000, random_state=self.settings['fit_seed'])
        weights = (len(y)/(5*self.class_counts[y]))**self.candidate['weight_power']
        weights *= len(y)/weights.sum(); self.weights = weights
        # Keep convergence failures visible instead of silently accepting them.
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.model.fit(self.projection.transform(x, self.candidate), y, sample_weight=weights)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(self.projection.transform(x, self.candidate))


def subset(x, indices):
    return {k: np.asarray(v)[indices] for k, v in x.items()}


def candidates(settings):
    return [{'family': 'baseline'}]+[{'family': 'block_logistic', 'feature_set': f, 'block_scale': b, 'C': c, 'weight_power': w}
        for f in settings['feature_sets'] for b in settings['block_scales'] for c in settings['C'] for w in settings['weight_powers']]


def options(inner, old, y, settings):
    index = old['policies']['accuracy']['selection']['index']
    predictions = [np.asarray(old['candidates'][index]['predictions'])]+[decide(p, None) for p in inner]
    return [{'candidate': c, **score(y, p)} for c, p in zip(candidates(settings), predictions)]


def choose(rows, policy):
    if policy not in {'accuracy','new_only','without_age'}:
        raise ValueError('Unknown policy')
    permitted = [i for i, r in enumerate(rows) if policy == 'accuracy' or (i > 0 and
        (policy == 'new_only' or r['candidate']['feature_set'] == 'language'))]
    return max(permitted, key=lambda i: (rows[i]['correct'], rows[i]['macro_f1'], -i))


def predict_state(state, x):
    if state['candidate']['family'] == 'baseline':
        return state['estimator'].predict_proba(x[state['feature']])
    return state['estimator'].predict_proba(x)


def job(x, frame, fold, old, settings, out):
    started = time.monotonic(); y = frame.severity_class.to_numpy(int); search = candidates(settings)
    inner = np.full((len(search)-1, len(y), 5), np.nan); receipts = []
    for part in fold['inner']:
        training = subset(x, part['fit']); projection = BlockProjection().fit(training, settings['components'])
        receipts.append({'fit': part['fit'], 'heldout': part['heldout'], 'real_count': projection.real_count})
        for i, c in enumerate(search[1:]):
            model = BlockClassifier(c, settings).fit(training, y[part['fit']], projection)
            inner[i, part['heldout']] = model.predict_proba(subset(x, part['heldout']))
    inner = inner[:, fold['fit']]; rows = options(inner, old, y[fold['fit']], settings)
    training = subset(x, fold['fit']); projection = BlockProjection().fit(training, settings['components'])
    selected = {}; fitted = {}; out.mkdir(parents=True, exist_ok=True)
    for name in settings['policies']:
        i = choose(rows, name); c = search[i]
        if i not in fitted:
            if i:
                model = BlockClassifier(c, settings).fit(training, y[fold['fit']], projection)
                state = {'estimator': model, 'candidate': c}
            else:
                source = ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
                original = pickle.loads(source.read_bytes())
                state = {'estimator': original['estimator'], 'candidate': c, 'feature': old['policies']['accuracy']['candidate']['feature']}
            fitted[i] = {**state, 'fit': fold['fit'], 'heldout': fold['heldout'], 'protocol': PROTOCOL}
        state = fitted[i]; p = predict_state(state, subset(x, fold['heldout']))
        if not i:
            np.testing.assert_allclose(p, old['policies']['accuracy']['probabilities'], rtol=0, atol=1e-14)
        path = out/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
        np.testing.assert_allclose(p, predict_state(pickle.loads(path.read_bytes()), subset(x, fold['heldout'])), rtol=0, atol=1e-14)
        selected[name] = {'selection': {'index': i}, 'candidate': c, 'threshold': None,
            'probabilities': p.tolist(), 'predictions': p.argmax(1).tolist(), 'checkpoint_sha256': sha256(path),
            'real_training_accuracy': float((predict_state(state, training).argmax(1) == y[fold['fit']]).mean())}
    return {**{k: fold[k] for k in ['seed','fold','fit','heldout']}, 'inner_probabilities': inner.tolist(),
            'options': rows, 'selected': selected, 'inner_receipts': receipts,
            'fits': (len(search)-1)*len(fold['inner'])+sum(i > 0 for i in fitted), 'seconds': time.monotonic()-started}


def validate(r, fold, old, y, settings, directory):
    if any(r[k] != fold[k] for k in ['seed','fold','fit','heldout']) or set(r['selected']) != set(settings['policies']):
        raise ValueError('Fold/policy mismatch')
    if np.asarray(r['inner_probabilities']).shape != (len(candidates(settings))-1, len(fold['fit']), 5):
        raise ValueError('Incomplete inner coverage')
    rows = options(r['inner_probabilities'], old, y[fold['fit']], settings)
    if rows != r['options'] or r['inner_receipts'] != [{'fit': p['fit'], 'heldout': p['heldout'], 'real_count': len(p['fit'])} for p in fold['inner']]:
        raise ValueError('Training ownership or inner results changed')
    for name, result in r['selected'].items():
        i = choose(rows, name)
        if result['selection'] != {'index': i} or result['candidate'] != rows[i]['candidate']:
            raise ValueError('Selection changed')
        if np.asarray(result['probabilities']).shape != (len(fold['heldout']), 5) or decide(result['probabilities'], None).tolist() != result['predictions']:
            raise ValueError('Invalid outer probabilities')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')


def load_inputs():
    x, frame, old_settings, old, provenance = load_baseline()
    if 'split' not in frame:
        frame = frame.assign(split='train')
    extra = extra_features(frame)
    x.update(base=x['semantic_av'], **extra)
    bindings = {**provenance['bindings'], str(METADATA): sha256(METADATA), str(CORPUS): sha256(CORPUS)}
    return x, frame, old_settings, old, {**provenance, 'bindings': bindings}


def run(inputs, settings, output=OUTPUT, resume=False, max_new_jobs=None):
    x, frame, old_settings, old, provenance = inputs; out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing study')
    folds = nested_folds(frame, old_settings); y = frame.severity_class.to_numpy(int)
    if len(folds) != settings['expected_jobs'] or len(folds) != len(old) or frame.participant_id.duplicated().any():
        raise ValueError('Population/coverage changed')
    for fold, baseline in zip(folds, old):
        if any(fold[k] != baseline[k] for k in ['seed','fold','fit','heldout']) or set(fold['fit']) & set(fold['heldout']):
            raise ValueError('Outer participant leakage/ownership')
        for part in fold['inner']:
            if set(part['fit']) & set(part['heldout']) or set(part['fit']+part['heldout']) != set(fold['fit']):
                raise ValueError('Inner participant leakage/ownership')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
        'provenance': provenance, 'source_sha256': sha256(__file__), 'language_names': LANGUAGE_NAMES,
        'feature_hashes': {k: digest(v.tolist()) for k,v in x.items()}, 'old_records_hash': digest(old)}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}; out.mkdir(parents=True, exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        path = out/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Frozen inputs/code changed')
        elif resume:
            raise ValueError('Missing frozen protocol')
        else:
            atomic_json(path, envelope)
        records = []; new = fits = 0
        for fold, baseline in zip(folds, old):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; path = directory/'result.json'
            if path.exists():
                r = load_record(path, ch)
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                r = job(x, frame, fold, baseline, settings, directory); save_record(path, r, ch); new += 1; fits += r['fits']
            validate(r, fold, baseline, y, settings, directory); records.append(r)
            print(f"Age/language: {len(records)}/{len(folds)} jobs; {r['seconds']:.1f}s/job", flush=True)
        for p,h in provenance['bindings'].items():
            if sha256(p) != h:
                raise ValueError('Bound previous result/input changed')
        complete = len(records) == len(folds)
        targets = summarize(frame, records, old, settings, out) if complete else None
        result = {'complete': complete, 'context_hash': ch, 'completed_jobs': len(records), 'new_jobs': new,
                  'new_fits': fits, 'total_fits': sum(r['fits'] for r in records), 'targets': targets,
                  'old_models_refitted': 0, 'synthetic_rows': 0, 'validation_test_scored': False}
        atomic_json(out/'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-jobs', type=int); args = parser.parse_args()
    settings = json.loads(Path('configs/age_language_accuracy.json').read_text())
    print(json.dumps(run(load_inputs(), settings, resume=args.resume, max_new_jobs=args.max_new_jobs), indent=2))


if __name__ == '__main__':
    from .age_language_accuracy import main as module_main
    module_main()
