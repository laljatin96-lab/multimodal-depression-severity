"""Audit-gated, fold-local imbalance repair of the frozen semantic classifier search."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import semantic_pipeline_audit as audit
from .accuracy_model_comparison import label_metrics
from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .evaluate import write_report
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .raw_frame_uncertainty import METRICS, participant_counts, weighted_metrics

PROTOCOL = 'audited-semantic-severe-repair-v2'


def interpolate_severe(x, y, seed, neighbors=2, max_multiplier=4):
    """Create bounded feature-space training examples; parents are local fitting indices."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or neighbors < 1 or max_multiplier < 1:
        raise ValueError('Invalid interpolation inputs')
    parents = np.flatnonzero(y == 4)
    if len(parents) < 2:
        raise ValueError('Need at least two real severe fitting participants')
    target = max(len(parents), min(int((y == 3).sum()), max_multiplier*len(parents)))
    distances = ((x[parents, None, :] - x[None, parents, :])**2).sum(2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind='stable')[:, :min(neighbors, len(parents)-1)]
    rng = np.random.default_rng(seed)
    generated, lineage = [], []
    for _ in range(target-len(parents)):
        a = int(rng.integers(len(parents)))
        b = int(rng.choice(nearest[a])); fraction = float(rng.random())
        i, j = int(parents[a]), int(parents[b])
        generated.append(x[i]+fraction*(x[j]-x[i]))
        lineage.append({'parent_a': i, 'parent_b': j, 'fraction': fraction})
    if not generated:
        return x.copy(), y.copy(), lineage
    return np.vstack([x, generated]), np.concatenate([y, np.full(len(generated), 4)]), lineage


class ImbalanceLogistic(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, components=None, intervention='none', seed=42, max_iter=2000,
                 synthetic_neighbors=2, synthetic_max_multiplier=4):
        self.C, self.components, self.intervention = C, components, intervention
        self.seed, self.max_iter = seed, max_iter
        self.synthetic_neighbors, self.synthetic_max_multiplier = synthetic_neighbors, synthetic_max_multiplier

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.array_equal(y, y.astype(int)) or set(y) != set(range(5)):
            raise ValueError('Finite real fitting vectors and all five integer classes required')
        y = y.astype(int)
        if self.intervention not in ['none', 'sqrt_weight', 'balanced_weight', 'severe_interpolation']:
            raise ValueError('Unknown imbalance intervention')
        self.n_features_in_ = x.shape[1]
        self.real_count_ = len(y)
        self.real_class_counts_ = np.bincount(y, minlength=5)
        self.scaler_ = StandardScaler()
        transformed = self.scaler_.fit_transform(x)
        self.pca_ = None
        if self.components is not None:
            self.pca_ = PCA(n_components=self.components, svd_solver='full', whiten=False)
            transformed = self.pca_.fit_transform(transformed)
        self.lineage_ = []
        augmented_y = y.copy()
        if self.intervention == 'severe_interpolation':
            transformed, augmented_y, self.lineage_ = interpolate_severe(
                transformed, y, self.seed, self.synthetic_neighbors, self.synthetic_max_multiplier)
        alpha = {'none': 0., 'sqrt_weight': .5, 'balanced_weight': 1., 'severe_interpolation': 0.}[self.intervention]
        class_weight = (len(y)/(5*self.real_class_counts_))**alpha
        weights = class_weight[augmented_y]
        weights *= len(y)/weights.sum()  # Keep the regularization scale comparable across interventions.
        self.fit_class_counts_ = np.bincount(augmented_y, minlength=5)
        self.effective_class_weights_ = np.bincount(augmented_y, weights=weights, minlength=5)
        self.total_weight_ = float(weights.sum())
        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver='lbfgs', random_state=self.seed)
        self.model_.fit(transformed, augmented_y, sample_weight=None if self.intervention == 'none' else weights)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x):
        transformed = self.scaler_.transform(x)
        if self.pca_ is not None:
            transformed = self.pca_.transform(transformed)
        return self.model_.predict_proba(transformed)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(1)]


def candidates(settings):
    # Preserve the original 17-candidate ordering as an exact matched control subset.
    return [{'family': 'majority', 'feature': settings['feature_inputs'][0]}] + [
        {'family': 'logistic', 'feature': feature, 'components': components, 'C': c, 'intervention': intervention}
        for intervention in settings['interventions'] for feature in settings['feature_inputs']
        for components in settings['components'] for c in settings['C_candidates']]


def estimator_for(candidate, settings):
    if candidate['family'] == 'majority':
        return DummyClassifier(strategy='most_frequent')
    return ImbalanceLogistic(C=candidate['C'], components=candidate['components'], intervention=candidate['intervention'],
        seed=settings['fit_seed'], max_iter=settings['max_iter'], synthetic_neighbors=settings['synthetic_neighbors'],
        synthetic_max_multiplier=settings['synthetic_max_multiplier'])


def selection_metrics(y, prediction):
    scores = label_metrics(y, prediction)
    true_positive = int(((y == 4) & (prediction == 4)).sum())
    severe_count, predicted_count = int((y == 4).sum()), int((prediction == 4).sum())
    return {**{k:scores[k] for k in METRICS}, 'correct': int((prediction == y).sum()), 'severe_true_positive': true_positive,
            'severe_support': severe_count, 'severe_predicted': predicted_count,
            'severe_recall': true_positive/severe_count if severe_count else 0.,
            'severe_precision': true_positive/predicted_count if predicted_count else 0.}


def choose(scores, policy, settings):
    if policy not in ['accuracy', 'severe_aware']:
        raise ValueError('Unknown selection policy')
    eligible = [i for i,r in enumerate(scores) if r['severe_recall'] >= settings['minimum_severe_recall']
                and r['severe_precision'] >= settings['minimum_severe_precision']]
    pool = eligible if policy == 'severe_aware' and eligible else list(range(len(scores)))
    chosen = max(pool, key=lambda i: (scores[i]['correct'], scores[i]['macro_f1'], -i))
    return {'index': chosen, 'gate_satisfied': chosen in eligible, 'eligible_candidates': len(eligible),
            'fallback_used': policy == 'severe_aware' and not eligible}


def fit_receipt(estimator, candidate):
    if candidate['family'] == 'majority':
        return {'synthetic_count': 0, 'lineage': []}
    return {'real_count': estimator.real_count_, 'real_class_counts': estimator.real_class_counts_.tolist(),
            'fit_class_counts': estimator.fit_class_counts_.tolist(), 'total_weight': estimator.total_weight_,
            'effective_class_weights': estimator.effective_class_weights_.tolist(),
            'synthetic_count': len(estimator.lineage_), 'lineage': estimator.lineage_}


def run_job(matrices, y, fold, settings, directory):
    start = time.monotonic(); search = candidates(settings); scores = []
    for candidate in search:
        x = matrices[candidate['feature']]
        predicted = np.full(len(y), -1, int); receipts = []
        for part in fold['inner']:
            estimator = estimator_for(candidate, settings)
            fit(estimator, x[part['fit']], y[part['fit']])
            predicted[part['heldout']] = estimator.predict(x[part['heldout']])
            receipts.append(fit_receipt(estimator, candidate))
        scores.append({'candidate': candidate, 'predictions': predicted[fold['fit']].tolist(),
                       **selection_metrics(y[fold['fit']], predicted[fold['fit']]), 'inner_fit_receipts': receipts})
    policies = {}
    directory.mkdir(parents=True, exist_ok=True)
    for policy in settings['policies']:
        selection = choose(scores, policy, settings); candidate = search[selection['index']]
        x = matrices[candidate['feature']]; estimator = estimator_for(candidate, settings)
        fit(estimator, x[fold['fit']], y[fold['fit']])
        p = five_probabilities(estimator, x[fold['heldout']])
        state = {'estimator': estimator, 'candidate': candidate, 'policy': policy, 'protocol': PROTOCOL,
                 'fit': fold['fit'], 'heldout': fold['heldout'], 'feature_count': x.shape[1]}
        path = directory/f'{policy}.pkl'; tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as handle:
            pickle.dump(state, handle)
        tmp.replace(path)
        with path.open('rb') as handle:
            restored = pickle.load(handle)
        np.testing.assert_allclose(p, five_probabilities(restored['estimator'], x[fold['heldout']]), rtol=0, atol=1e-14)
        policies[policy] = {'selection': selection, 'candidate': candidate, 'probabilities': p.tolist(),
                            'checkpoint_parity': True, 'model_sha256': sha256(path),
                            'fit_receipt': fit_receipt(estimator, candidate)}
    return {**{k:fold[k] for k in ['seed','fold','fit','heldout']}, 'candidates': scores, 'policies': policies,
            'fits': len(search)*len(fold['inner'])+len(settings['policies']), 'seconds': time.monotonic()-start}


def validate_receipt(receipt, candidate, fitting_y, settings):
    count = np.bincount(fitting_y, minlength=5)
    if candidate['family'] == 'majority':
        if receipt != {'synthetic_count': 0, 'lineage': []}:
            raise ValueError('Majority receipt changed')
        return
    expected_synthetic = max(0, min(int(count[3]), settings['synthetic_max_multiplier']*int(count[4]))-int(count[4])) if candidate['intervention'] == 'severe_interpolation' else 0
    if receipt['real_count'] != len(fitting_y) or receipt['real_class_counts'] != count.tolist() or receipt['synthetic_count'] != expected_synthetic or len(receipt['lineage']) != expected_synthetic:
        raise ValueError('Synthetic/real fitting support changed')
    for item in receipt['lineage']:
        a, b, fraction = item['parent_a'], item['parent_b'], item['fraction']
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < len(fitting_y) and 0 <= b < len(fitting_y) and a != b and fitting_y[a] == fitting_y[b] == 4 and 0 <= fraction < 1):
            raise ValueError('Synthetic parent outside real severe fitting participants')
    augmented_count = count.copy(); augmented_count[4] += expected_synthetic
    if receipt['fit_class_counts'] != augmented_count.tolist() or not np.isclose(receipt['total_weight'], len(fitting_y)):
        raise ValueError('Effective training population changed')
    alpha = {'none': 0., 'sqrt_weight': .5, 'balanced_weight': 1., 'severe_interpolation': 0.}[candidate['intervention']]
    totals = (len(fitting_y)/(5*count))**alpha*augmented_count
    totals *= len(fitting_y)/totals.sum()
    np.testing.assert_allclose(receipt['effective_class_weights'], totals, rtol=0, atol=1e-10)


def validate_job(record, fold, settings, directory, y):
    if any(record[k] != fold[k] for k in ['seed','fold','fit','heldout']):
        raise ValueError('Fold identity changed')
    search = candidates(settings)
    if [r['candidate'] for r in record['candidates']] != search or record['fits'] != len(search)*len(fold['inner'])+len(settings['policies']):
        raise ValueError('Candidate identity/fit count changed')
    for result in record['candidates']:
        prediction = np.asarray(result['predictions'])
        if prediction.shape != (len(fold['fit']),) or not np.isin(prediction, range(5)).all():
            raise ValueError('Invalid inner predictions')
        expected = selection_metrics(y[fold['fit']], prediction)
        if any(not np.isfinite(result[k]) or abs(result[k]-v) > 1e-12 for k,v in expected.items()):
            raise ValueError('Inner selection metric changed')
        if len(result['inner_fit_receipts']) != len(fold['inner']):
            raise ValueError('Missing fitting receipt')
        for receipt, part in zip(result['inner_fit_receipts'], fold['inner']):
            validate_receipt(receipt, result['candidate'], y[part['fit']], settings)
    if set(record['policies']) != set(settings['policies']):
        raise ValueError('Selection policy set changed')
    for policy, result in record['policies'].items():
        selection = choose(record['candidates'], policy, settings)
        if result['selection'] != selection or result['candidate'] != search[selection['index']] or not result['checkpoint_parity'] or result['model_sha256'] != sha256(directory/f'{policy}.pkl'):
            raise ValueError('Selection/checkpoint changed')
        p = np.asarray(result['probabilities'])
        if p.shape != (len(fold['heldout']), 5) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12):
            raise ValueError('Invalid held-out probabilities')
        validate_receipt(result['fit_receipt'], result['candidate'], y[fold['fit']], settings)


def validate_controls(record, reference):
    """Fail closed if the unchanged control search diverges from the audited run."""
    controls = [r for r in record['candidates'] if r['candidate']['family'] == 'majority' or r['candidate'].get('intervention') == 'none']
    if len(controls) != len(reference['candidates']):
        raise ValueError('Matched control search size differs')
    for new, old in zip(controls, reference['candidates']):
        candidate = {k:v for k,v in new['candidate'].items() if k != 'intervention'}
        if candidate != old['candidate'] or new['predictions'] != old['predictions']:
            raise ValueError('Matched control predictions differ from audited original')


def load_inputs(settings):
    x, frame, original, records, provenance = audit.load_reference(settings['source_config'], settings['source_directory'])
    root = Path(settings['audit_directory']); report = json.loads((root/'audit.json').read_text())
    if not report['passed'] or report['code_sha256'] != sha256(audit.__file__) or report['provenance'] != provenance:
        raise ValueError('Complete compatible semantic audit required')
    for name, expected in report['artifacts'].items():
        if sha256(root/name) != expected:
            raise ValueError('Audit artifact changed')
    if nested_folds(frame, settings) != nested_folds(frame, original):
        raise ValueError('Original participants and folds must be retained')
    return x, frame, {**provenance, 'audit_sha256': sha256(root/'audit.json'),
                     'control_records': [{'seed':r['seed'],'fold':r['fold'],
                         'candidates':[{k:c[k] for k in ['candidate','predictions']} for c in r['candidates']]} for r in records],
                     'reference_records': [{k:r[k] for k in ['seed','fold','fit','heldout','probabilities']} for r in records]}


def summarize(frame, records, settings, out, provenance):
    y = frame.severity_class.to_numpy(int); rows = []; all_predictions = {}
    names = settings['policies']+['majority']+(['previous_semantic'] if provenance.get('reference_records') else [])
    for name in names:
        repeated = []
        for seed in settings['fold_seeds']:
            p = np.full((len(y),5), np.nan); owners = np.zeros(len(y),int)
            source_records = provenance['reference_records'] if name == 'previous_semantic' else records
            for record in source_records:
                if record['seed'] != seed:
                    continue
                if name == 'majority':
                    value = np.eye(5)[np.bincount(y[record['fit']],minlength=5).argmax()]
                elif name == 'previous_semantic':
                    value = record['probabilities']
                else:
                    value = record['policies'][name]['probabilities']
                p[record['heldout']] = value; owners[record['heldout']] += 1
            if not (owners == 1).all() or not np.isfinite(p).all():
                raise ValueError('Incomplete real-participant OOF coverage')
            report = write_report(frame, p, out/'oof'/f'{name}_seed{seed}', name, seed, 'train_nested_oof')
            selected = selection_metrics(y, p.argmax(1)); repeated.append(p.argmax(1))
            rows.append({'model':name,'seed':seed, **{k:report[k] for k in METRICS},
                         **{k:selected[k] for k in ['severe_true_positive','severe_support','severe_predicted','severe_recall','severe_precision']}})
        all_predictions[name] = np.stack(repeated)
    scores = pd.DataFrame(rows); scores.to_csv(out/'repeat_scores.csv',index=False)
    summary = scores.groupby('model')[METRICS+['severe_recall','severe_precision']].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]; summary.to_csv(out/'summary.csv')
    counts = participant_counts(len(y),settings['bootstrap_repetitions'],settings['bootstrap_seed'])
    boot = {n:weighted_metrics(y,p,counts) for n,p in all_predictions.items()}; comparisons = []
    for policy in settings['policies']:
        for reference in [n for n in ['majority','previous_semantic'] if n in boot]:
            interval = np.percentile(boot[policy]-boot[reference],[2.5,97.5],axis=0)
            for i,key in enumerate(METRICS):
                comparisons.append({'model':policy,'reference':reference,'metric':key,
                    'difference':float(summary.loc[policy,key+'_mean']-summary.loc[reference,key+'_mean']),
                    'ci95_low':float(interval[0,i]),'ci95_high':float(interval[1,i])})
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame([{'seed':r['seed'],'fold':r['fold'],'policy':n,**p['candidate'],**p['selection']}
                  for r in records for n,p in r['policies'].items()]).to_csv(out/'outer_choices.csv',index=False)
    targets = {name:{'accuracy':float(summary.loc[name,'accuracy_mean']),
        'internal_accuracy_target_reached':bool(summary.loc[name,'accuracy_mean'] >= settings['target_accuracy']),
        'severe_detected_every_repeat':bool(scores.loc[scores.model.eq(name),'severe_true_positive'].gt(0).all()),
        'independent_confirmation':False} for name in settings['policies']}
    atomic_json(out/'target_status.json',targets)
    return targets


def run_experiment(matrices, frame, provenance, settings, output, resume=False, max_new_jobs=None, verify_inputs=None):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing repair experiment; use --resume')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires a protocol')
    if not 0 < settings['target_accuracy'] <= 1 or not 1 <= settings['cpu_threads'] <= 2 or settings['max_iter'] < 1:
        raise ValueError('Invalid target/thread/iteration settings')
    if settings['interventions'][0] != 'none' or len(set(settings['interventions'])) != len(settings['interventions']) or settings['policies'] != ['accuracy','severe_aware']:
        raise ValueError('Retain matched control and both fixed selection policies')
    if any(not 0 < settings[k] <= 1 for k in ['minimum_severe_recall','minimum_severe_precision']) or settings['synthetic_neighbors'] < 1 or settings['synthetic_max_multiplier'] < 1:
        raise ValueError('Invalid severe constraint or synthetic limit')
    frame = frame.reset_index(drop=True); y = frame.severity_class.to_numpy(int)
    if frame.participant_id.duplicated().any() or set(y) != set(range(5)) or not np.array_equal(y, frame.severity_class):
        raise ValueError('Unique real participants and five integer labels required')
    for feature in settings['feature_inputs']:
        if matrices[feature].ndim != 2 or len(matrices[feature]) != len(y) or not np.isfinite(matrices[feature]).all():
            raise ValueError('Invalid feature matrix')
    folds = nested_folds(frame, settings); support = audit.check_folds(folds,y,settings['fold_seeds'])
    if support['inner_fit_class_min'][4] < 2:
        raise ValueError('Insufficient independent severe fitting parents')
    fits_per_job = len(candidates(settings))*settings['inner_folds']+len(settings['policies'])
    if len(folds) != settings['expected_jobs'] or len(folds)*fits_per_job != settings['expected_fits'] or (max_new_jobs is not None and max_new_jobs < 1):
        raise ValueError('Declared budget mismatch')
    context = {'protocol':PROTOCOL,'settings':settings,'training':frame.to_dict('records'),'folds':folds,'provenance':provenance,
        'feature_hashes':{k:digest(np.asarray(matrices[k]).tolist()) for k in settings['feature_inputs']},'code_sha256':sha256(__file__),
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'support_code_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ['semantic_pipeline_audit.py','semantic_accuracy_target.py','encoder_kernel_accuracy.py','crossvalidated_baselines.py','modality_diagnostics.py','evaluate.py','raw_frame_uncertainty.py']}}
    ch = digest(context); envelope = {'context':context,'context_hash':ch}; out.mkdir(parents=True,exist_ok=True)
    with output_lock(out), threadpool_limits(limits=settings['cpu_threads']), offline_network():
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != envelope:
                raise ValueError('Inputs/code/protocol changed')
        else:
            atomic_json(out/'protocol.json',envelope)
        records = []; new = reused = 0
        for fold in folds:
            directory = out/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"; path = directory/'result.json'
            if path.exists():
                record = load_record(path,ch); reused += 1
            elif max_new_jobs is not None and new >= max_new_jobs:
                continue
            else:
                record = run_job(matrices,y,fold,settings,directory); save_record(path,record,ch); new += 1
            validate_job(record,fold,settings,directory,y)
            if provenance.get('control_records'):
                controls = [r for r in provenance['control_records'] if r['seed'] == fold['seed'] and r['fold'] == fold['fold']]
                if len(controls) != 1:
                    raise ValueError('Missing matched control reference')
                validate_controls(record,controls[0])
            records.append(record)
            print(f'Severe repair: {len(records)}/{len(folds)} jobs; {new} new, {reused} reused',flush=True)
        if verify_inputs:
            verify_inputs()
        complete = len(records) == len(folds)
        targets = summarize(frame,records,settings,out,provenance) if complete else None
        result = {'complete':complete,'context_hash':ch,'completed_jobs':len(records),'expected_jobs':len(folds),
            'completed_fits':len(records)*fits_per_job,'new_jobs':new,'reused_jobs':reused,'new_fits':new*fits_per_job,
            'fit_seconds':sum(r['seconds'] for r in records),'targets':targets,'validation_test_scoring':False,'reference_replaced':False}
        atomic_json(out/'completion.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/semantic_severe_repair.json')
    p.add_argument('--output',default='results/semantic_severe_repair_v2')
    p.add_argument('--resume',action='store_true'); p.add_argument('--max-new-jobs',type=int)
    args = p.parse_args(); settings = json.loads(Path(args.config).read_text())
    if any(Path(args.output).resolve().is_relative_to(Path(root).resolve()) for root in
           ['data','dataset','checkpoints',settings['source_directory'],settings['audit_directory'],'results/transcript_corpus','results/semantic_features']):
        p.error('Output must be outside original inputs/audit')
    x, frame, provenance = load_inputs(settings)
    def check():
        other, current, after = load_inputs(settings)
        if after != provenance or not frame.equals(current) or any(not np.array_equal(x[k],other[k]) for k in x):
            raise ValueError('Inputs changed during repair')
    print(json.dumps(run_experiment(x,frame,provenance,settings,args.output,args.resume,args.max_new_jobs,check),indent=2))


if __name__ == '__main__':
    # Keep custom estimator pickle identities importable after a `python -m` run.
    from .semantic_severe_repair import main as module_main
    module_main()
