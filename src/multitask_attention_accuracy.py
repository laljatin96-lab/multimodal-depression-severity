"""Fold-local learned chunk attention with training-only symptom supervision.

Questionnaire answers are auxiliary targets, never inference features. Existing
baseline predictions/checkpoints are read-only controls and are never refitted.
"""
import argparse
import copy
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .chunk_semantic_features import load_verified
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import decide, score, summarize

PROTOCOL = 'training-only-symptom-multitask-chunk-attention-v1'
OUTPUT = Path('results/multitask_attention_accuracy')
METADATA = Path('data/metadata/official_edaic/detailed_lables.csv')


def subset(data, indices):
    return {k: [v[i] for i in indices] if k in {'vectors', 'lengths'} else np.asarray(v)[indices]
            for k, v in data.items()}


class Projection:
    """Participant-weighted PCA; every fit participant contributes unit mass."""
    @staticmethod
    def fit_matrix(x, weights, dimensions):
        x = np.asarray(x, float); weights = np.asarray(weights, float)
        if x.ndim != 2 or not np.isfinite(x).all() or (weights <= 0).any():
            raise ValueError('Invalid fitting features')
        weights = weights / weights.sum()
        mean = weights @ x
        scale = np.sqrt(weights @ ((x-mean)**2)).clip(1e-6)
        z = (x-mean)/scale
        values, vectors = np.linalg.eigh((z*weights[:, None]).T @ z)
        indices = np.argsort(values)[::-1][:min(dimensions, len(x)-1, x.shape[1])]
        # Whitening plus a variance floor limits unstable low-support directions.
        axes = vectors[:, indices]/np.sqrt(values[indices].clip(1e-3))
        return mean, scale, axes

    def fit(self, data, settings):
        n = len(data['vectors'])
        if n < 2 or len(data['lengths']) != n or len(data['av']) != n:
            raise ValueError('Invalid fitting participant count')
        vectors = []; weights = []
        for v, length in zip(data['vectors'], data['lengths']):
            v, length = np.asarray(v), np.asarray(length, float)
            if v.ndim != 2 or len(v) != len(length) or not len(v) or (length <= 0).any() or not np.isfinite(length).all():
                raise ValueError('Invalid chunk lengths')
            vectors.append(v); weights.extend(length/length.sum())
        self.text = self.fit_matrix(np.concatenate(vectors), weights, settings['text_components'])
        self.av = self.fit_matrix(data['av'], np.ones(n), settings['av_components'])
        self.real_count = n
        return self

    @staticmethod
    def transform_matrix(x, params):
        mean, scale, axes = params
        return (((np.asarray(x, float)-mean)/scale) @ axes).astype('float32')

    def transform(self, data):
        n = len(data['vectors']); max_chunks = max(map(len, data['vectors']))
        x = np.zeros((n, max_chunks, self.text[2].shape[1]), 'float32')
        prior = np.full((n, max_chunks), -np.inf, 'float32')
        for i, (v, length) in enumerate(zip(data['vectors'], data['lengths'])):
            length = np.asarray(length, float)
            if len(v) != len(length) or not len(v) or (length <= 0).any() or not np.isfinite(length).all():
                raise ValueError('Invalid inference chunk lengths')
            x[i, :len(v)] = self.transform_matrix(v, self.text)
            prior[i, :len(v)] = np.log(length/length.sum())
        av = self.transform_matrix(data['av'], self.av)
        if not np.isfinite(x).all() or not np.isfinite(av).all():
            raise ValueError('Invalid inference features')
        return tuple(torch.from_numpy(a) for a in (x, prior, av))


class AttentionNetwork(nn.Module):
    def __init__(self, text_dim, av_dim, candidate, settings):
        super().__init__()
        h = settings['hidden']; self.use_av = candidate['use_av']
        self.encoder = nn.Linear(text_dim, h)
        self.dropout = nn.Dropout(settings['dropout'])
        self.attention = nn.Linear(h, 8)
        self.phq_weight = nn.Parameter(torch.empty(8, h, 4))
        nn.init.xavier_uniform_(self.phq_weight)
        self.phq_bias = nn.Parameter(torch.zeros(8, 4))
        extra = av_dim if self.use_av else 0
        self.classifier = nn.Linear(8*h+8+extra, 5)
        self.ptsd = nn.Linear(h+extra, 17)

    def forward(self, x, prior, av):
        h = self.dropout(torch.tanh(self.encoder(x)))
        attention = torch.softmax(self.attention(h)+prior[:, :, None], dim=1)
        pooled = torch.einsum('bca,bch->bah', attention, h)
        phq = torch.einsum('bah,ahk->bak', pooled, self.phq_weight)+self.phq_bias
        expected = (torch.softmax(phq, dim=-1)*torch.arange(4, dtype=x.dtype)).sum(-1)/3
        features = [pooled.flatten(1), expected]
        global_features = [pooled.mean(1)]
        if self.use_av:
            features.append(av); global_features.append(av)
        return self.classifier(torch.cat(features, 1)), phq, self.ptsd(torch.cat(global_features, 1))


class AttentionClassifier:
    def __init__(self, candidate, settings):
        self.candidate = copy.deepcopy(candidate); self.settings = copy.deepcopy(settings)

    def _network(self):
        return AttentionNetwork(self.projection.text[2].shape[1], self.projection.av[2].shape[1],
                                self.candidate, self.settings)

    def fit(self, data, y, clinical, projection=None):
        y = np.asarray(y)
        if y.shape != (len(data['vectors']),) or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('Five classes and real participant targets required')
        mode = self.candidate['mode']; clinical = np.asarray(clinical)
        if mode not in {'class_only', 'phq', 'phq_ptsd'}:
            raise ValueError('Unknown supervision')
        if mode != 'class_only':
            if clinical.shape != (len(y), 25) or not np.isfinite(clinical).all() or not np.array_equal(clinical, clinical.astype(int)):
                raise ValueError('Missing integer fitting symptom targets')
            if (clinical[:, :8] < 0).any() or (clinical[:, :8] > 3).any() or (clinical[:, 8:] < 1).any() or (clinical[:, 8:] > 5).any():
                raise ValueError('Invalid item range')
            if not np.array_equal(np.minimum(clinical[:, :8].sum(1)//5, 4), y):
                raise ValueError('PHQ targets disagree with severity')
        self.projection = projection if projection is not None else Projection().fit(data, self.settings)
        if self.projection.real_count != len(y):
            raise ValueError('Projection fitting population differs')
        inputs = self.projection.transform(data); labels = torch.from_numpy(y.astype('int64'))
        counts = np.bincount(y.astype(int), minlength=5)
        weights = (len(y)/(5*counts[y.astype(int)]))**self.candidate['weight_power']
        weights = torch.tensor(weights/weights.mean(), dtype=torch.float32)
        self.real_count = len(y); self.class_counts = counts.tolist()
        with torch.random.fork_rng():
            torch.manual_seed(self.settings['fit_seed']); self.model = self._network()
            with torch.no_grad():
                self.model.classifier.bias.copy_(torch.tensor(np.log(counts/counts.sum()), dtype=torch.float32))
                if mode != 'class_only':
                    priors = np.stack([np.bincount(clinical[:, j].astype(int), minlength=4)+1 for j in range(8)])
                    self.model.phq_bias.copy_(torch.tensor(np.log(priors/priors.sum(1, keepdims=True)), dtype=torch.float32))
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.settings['learning_rate'],
                                          weight_decay=self.settings['weight_decay'])
            self.loss_endpoints = []
            for epoch in range(self.candidate['epochs']):
                self.model.train(); optimizer.zero_grad(set_to_none=True)
                logits, phq, ptsd = self.model(*inputs)
                loss = (F.cross_entropy(logits, labels, reduction='none', label_smoothing=self.settings['label_smoothing'])*weights).mean()
                if mode != 'class_only':
                    loss = loss+self.settings['phq_loss_weight']*F.cross_entropy(phq.flatten(0, 1), torch.tensor(clinical[:, :8].reshape(-1), dtype=torch.long))
                if mode == 'phq_ptsd':
                    target = torch.tensor((clinical[:, 8:]-1)/4, dtype=torch.float32)
                    loss = loss+self.settings['ptsd_loss_weight']*F.mse_loss(torch.sigmoid(ptsd), target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                if epoch in {0, self.candidate['epochs']-1}:
                    self.loss_endpoints.append(float(loss.detach()))
                loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, data):
        self.model.eval()
        with torch.inference_mode():
            return torch.softmax(self.model(*self.projection.transform(data))[0], dim=1).numpy().astype(float)

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop('model')
        state['weights_numpy'] = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
        return state

    def __setstate__(self, state):
        weights = state.pop('weights_numpy'); self.__dict__.update(state)
        with torch.random.fork_rng():
            self.model = self._network()
        self.model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}); self.model.eval()


def candidates(settings):
    return [{'family': 'baseline'}]+[
        {'family': 'attention', 'mode': mode, 'use_av': av, 'weight_power': w, 'epochs': e}
        for mode in settings['modes'] for av in settings['use_av']
        for w in settings['weight_powers'] for e in settings['epochs']]


def options(inner, old, y, settings):
    baseline = old['candidates'][old['policies']['accuracy']['selection']['index']]['predictions']
    predictions = [baseline]+[decide(p, None) for p in inner]
    return [{'candidate': c, **score(y, np.asarray(pred))} for c, pred in zip(candidates(settings), predictions)]


def choose(rows, policy='accuracy'):
    permitted = [i for i, r in enumerate(rows) if policy == 'accuracy' or
                 (i and ((r['candidate']['mode'] == 'class_only') == (policy == 'class_only')))]
    if policy not in {'accuracy', 'class_only', 'multitask'} or not permitted:
        raise ValueError('Invalid selection policy')
    return max(permitted, key=lambda i: (rows[i]['correct'], rows[i]['macro_f1'], -i))


def predict_state(state, data, baseline_x):
    if state['candidate']['family'] == 'baseline':
        return state['estimator'].predict_proba(baseline_x[state['feature']])
    return state['estimator'].predict_proba(data)


def job(data, clinical, baseline_x, frame, fold, old, settings, out):
    start = time.monotonic(); y = frame.severity_class.to_numpy(int); search = candidates(settings)
    inner = np.full((len(search)-1, len(y), 5), np.nan)
    receipts = []
    for part in fold['inner']:
        training = subset(data, part['fit']); testing = subset(data, part['heldout'])
        projection = Projection().fit(training, settings)
        receipts.append({'fit': part['fit'], 'heldout': part['heldout'], 'projection_real_count': projection.real_count,
                         'clinical_target_participants': part['fit']})
        for i, c in enumerate(search[1:]):
            model = AttentionClassifier(c, settings).fit(training, y[part['fit']], clinical[part['fit']], projection)
            inner[i, part['heldout']] = model.predict_proba(testing)
    inner = inner[:, fold['fit']]
    rows = options(inner, old, y[fold['fit']], settings)
    selected = {}; fitted = {}; out.mkdir(parents=True, exist_ok=True)
    training = subset(data, fold['fit']); testing = subset(data, fold['heldout'])
    projection = Projection().fit(training, settings)
    for name in settings['policies']:
        i = choose(rows, name); c = search[i]
        if i not in fitted:
            if i:
                model = AttentionClassifier(c, settings).fit(training, y[fold['fit']], clinical[fold['fit']], projection)
                state = {'estimator': model, 'candidate': c}
            else:
                path = ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
                with path.open('rb') as f:
                    saved = pickle.load(f)
                state = {'estimator': saved['estimator'], 'candidate': c, 'feature': old['policies']['accuracy']['candidate']['feature']}
            fitted[i] = {**state, 'fit': fold['fit'], 'heldout': fold['heldout'], 'protocol': PROTOCOL}
        state = fitted[i]
        p = predict_state(state, testing, subset(baseline_x, fold['heldout']))
        if not i:
            np.testing.assert_allclose(p, old['policies']['accuracy']['probabilities'], rtol=0, atol=1e-14)
        path = out/f'{name}.pkl'; path.write_bytes(pickle.dumps(state))
        restored = pickle.loads(path.read_bytes())
        np.testing.assert_allclose(p, predict_state(restored, testing, subset(baseline_x, fold['heldout'])), rtol=0, atol=1e-7)
        selected[name] = {'selection': {'index': i}, 'candidate': c, 'threshold': None,
                          'probabilities': p.tolist(), 'predictions': decide(p, None).tolist(),
                          'checkpoint_sha256': sha256(path),
                          'real_training_accuracy': float((predict_state(state, training, subset(baseline_x, fold['fit'])).argmax(1) == y[fold['fit']]).mean())}
    return {'seed': fold['seed'], 'fold': fold['fold'], 'fit': fold['fit'], 'heldout': fold['heldout'],
            'inner_probabilities': inner.tolist(), 'options': rows, 'selected': selected, 'inner_receipts': receipts,
            'fits': (len(search)-1)*len(fold['inner'])+sum(i > 0 for i in fitted), 'seconds': time.monotonic()-start}


def validate(r, fold, old, y, settings, directory):
    if any(r[k] != fold[k] for k in ['seed', 'fold', 'fit', 'heldout']):
        raise ValueError('Fold ownership changed')
    if np.asarray(r['inner_probabilities']).shape != (len(candidates(settings))-1, len(fold['fit']), 5):
        raise ValueError('Incomplete inner coverage')
    rows = options(r['inner_probabilities'], old, y[fold['fit']], settings)
    if rows != r['options'] or set(r['selected']) != set(settings['policies']):
        raise ValueError('Selection changed')
    if len(r['inner_receipts']) != len(fold['inner']):
        raise ValueError('Missing auxiliary fitting receipts')
    for part, receipt in zip(fold['inner'], r['inner_receipts']):
        if receipt != {'fit': part['fit'], 'heldout': part['heldout'], 'projection_real_count': len(part['fit']), 'clinical_target_participants': part['fit']}:
            raise ValueError('Auxiliary fitting ownership changed')
    for name, result in r['selected'].items():
        i = choose(rows, name)
        if result['selection'] != {'index': i} or result['candidate'] != rows[i]['candidate']:
            raise ValueError('Candidate changed')
        if np.asarray(result['probabilities']).shape != (len(fold['heldout']), 5) or decide(result['probabilities'], None).tolist() != result['predictions']:
            raise ValueError('Outer probabilities changed')
        if sha256(directory/f'{name}.pkl') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')


def load_inputs():
    x, frame, old_settings, old_records, provenance = load_baseline()
    chunks, bindings = load_verified()
    by_id = {r['participant_id']: r for r in chunks}
    if len(by_id) != len(chunks) or any(by_id[p]['split'] != 'train' for p in frame.participant_id):
        raise ValueError('Training chunk identity mismatch')
    selected = [by_id[p] for p in frame.participant_id]
    data = {'vectors': [np.asarray(r['vectors'], float) for r in selected],
            'lengths': [r['lengths'] for r in selected], 'av': x['semantic_av'][:, 384:]}
    metadata = pd.read_csv(METADATA, dtype={'Participant': str}).set_index('Participant')
    if metadata.index.duplicated().any():
        raise ValueError('Duplicate symptom ownership')
    columns = [c for c in metadata if c.startswith('PHQ8_')]+[c for c in metadata if c.startswith('PCL-C_')]
    clinical = metadata.loc[frame.participant_id, columns].to_numpy(float)
    provenance = {**provenance, 'bindings': {**provenance['bindings'], **bindings, str(METADATA): sha256(METADATA)},
                  'auxiliary_target_columns': columns, 'questionnaires_are_inputs': False}
    return data, clinical, x, frame, old_settings, old_records, provenance


def run(inputs, settings, output=OUTPUT, resume=False, max_new_jobs=None):
    data, clinical, x, frame, old_settings, old_records, provenance = inputs
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing run')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires frozen protocol')
    folds = nested_folds(frame, old_settings); y = frame.severity_class.to_numpy(int)
    if len(folds) != settings['expected_jobs'] or len(folds) != len(old_records) or frame.participant_id.duplicated().any():
        raise ValueError('Invalid training population/folds')
    for fold, old in zip(folds, old_records):
        if any(fold[k] != old[k] for k in ['seed', 'fold', 'fit', 'heldout']):
            raise ValueError('Baseline folds differ')
        if set(fold['fit']) & set(fold['heldout']):
            raise ValueError('Outer leakage')
        for part in fold['inner']:
            if set(part['fit']) & set(part['heldout']) or set(part['fit']+part['heldout']) != set(fold['fit']):
                raise ValueError('Inner leakage')
    context = {'protocol': PROTOCOL, 'settings': settings, 'folds': folds, 'training': frame.to_dict('records'),
               'source_sha256': sha256(__file__), 'provenance': provenance,
               'features_hash': digest({k: [np.asarray(a).tolist() for a in v] for k, v in data.items()}),
               'training_targets_hash': digest(clinical.tolist()), 'baseline_records_hash': digest(old_records),
               'numpy': np.__version__, 'torch': torch.__version__, 'synthetic_training_rows': 0,
               'note': 'Fixed participant folds; no questionnaire values at inference; reused evaluation data.'}
    ch = digest(context); envelope = {'context': context, 'context_hash': ch}; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings['cpu_threads']); torch.use_deterministic_algorithms(True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != envelope:
                raise ValueError('Frozen protocol changed')
        else:
            atomic_json(out/'protocol.json', envelope)
        records = []; new = 0; fits = 0
        for fold, old in zip(folds, old_records):
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; path = directory/'result.json'
            if path.exists():
                r = load_record(path, ch)
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                r = job(data, clinical, x, frame, fold, old, settings, directory)
                save_record(path, r, ch); new += 1; fits += r['fits']
            validate(r, fold, old, y, settings, directory); records.append(r)
            print(f"Attention study: {len(records)}/{len(folds)} jobs; {r['seconds']:.1f}s/job", flush=True)
        for p, h in provenance['bindings'].items():
            if sha256(p) != h:
                raise ValueError(f'Bound input changed: {p}')
        complete = len(records) == len(folds)
        targets = summarize(frame, records, old_records, settings, out) if complete else None
        receipt = {'complete': complete, 'context_hash': ch, 'completed_jobs': len(records), 'new_jobs': new,
                   'new_fits': fits, 'total_fits': sum(r['fits'] for r in records), 'targets': targets,
                   'old_models_refitted': 0, 'synthetic_records': 0, 'validation_test_scored': False}
        atomic_json(out/'completion.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    settings = json.loads(Path('configs/multitask_attention_accuracy.json').read_text())
    print(json.dumps(run(load_inputs(), settings, resume=args.resume, max_new_jobs=args.max_new_jobs), indent=2))


if __name__ == '__main__':
    from .multitask_attention_accuracy import main as module_main
    module_main()
