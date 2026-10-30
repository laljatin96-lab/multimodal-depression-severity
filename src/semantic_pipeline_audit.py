"""Read-only replay audit of the completed MiniLM 48.47% selection procedure."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from . import semantic_accuracy_target as source
from .batch_inference import atomic_json, load_record
from .common import digest
from .crossvalidated_baselines import fit, five_probabilities
from .encoder_kernel_accuracy import offline_network
from .evaluate import metrics
from .final_evaluation import sha256, verify_plan
from .import_official_metadata import reconcile
from .modality_diagnostics import nested_folds

PROTOCOL = 'semantic-pipeline-readonly-replay-audit-v1'
DEFAULT_CONFIG = 'configs/semantic_accuracy_target.json'
DEFAULT_SOURCE = 'results/semantic_accuracy_target'


def check_folds(folds, y, seeds):
    all_indices = set(range(len(y)))
    counts = []
    for seed in seeds:
        owners = np.zeros(len(y), int)
        for fold in [f for f in folds if f['seed'] == seed]:
            a, b = fold['fit'], fold['heldout']
            if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b) or set(a) | set(b) != all_indices:
                raise ValueError('Invalid outer fold ownership')
            owners[b] += 1
            inner_owners = np.zeros(len(y), int)
            for part in fold['inner']:
                c, d = part['fit'], part['heldout']
                if len(set(c)) != len(c) or len(set(d)) != len(d) or set(c) & set(d) or set(c) | set(d) != set(a):
                    raise ValueError('Invalid inner fold ownership')
                inner_owners[d] += 1
                count = np.bincount(y[c], minlength=5)
                if (count == 0).any():
                    raise ValueError('A fitting class is absent')
                counts.append(count.tolist())
            if not (inner_owners[a] == 1).all() or inner_owners[b].any():
                raise ValueError('Incomplete inner coverage')
        if not (owners == 1).all():
            raise ValueError('Incomplete outer coverage')
    return {'inner_fit_class_min': np.min(counts, axis=0).tolist(),
            'inner_fit_class_max': np.max(counts, axis=0).tolist()}


def load_reference(config=DEFAULT_CONFIG, directory=DEFAULT_SOURCE):
    """Validate live inputs against the sealed original context, without rewriting it."""
    settings = json.loads(Path(config).read_text())
    matrices, training, provenance = source.load_inputs(settings)
    folds = nested_folds(training, settings)
    root = Path(directory)
    envelope = json.loads((root/'protocol.json').read_text())
    context = {'protocol': source.PROTOCOL, 'settings': settings, 'training': training.to_dict('records'),
               'folds': folds, 'provenance': provenance,
               'feature_hashes': {k: digest(np.asarray(matrices[k]).tolist()) for k in settings['feature_inputs']},
               'code_sha256': sha256(source.__file__),
               'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    completion = json.loads((root/'completion.json').read_text())
    if envelope != {'context': context, 'context_hash': digest(context)} or not completion['complete'] or completion['context_hash'] != envelope['context_hash']:
        raise ValueError('Original semantic context/completion changed')
    y = training.severity_class.to_numpy(int)
    check_folds(folds, y, settings['fold_seeds'])
    records, bindings = [], {}
    for fold in folds:
        job = root/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
        record = load_record(job/'result.json', envelope['context_hash'])
        source.validate_job(record, fold, settings, job, y)
        records.append(record)
    for path in sorted(root.rglob('*')):
        if path.is_file():
            bindings[str(path)] = sha256(path)
    return matrices, training, settings, records, {'source_context_hash': envelope['context_hash'],
            'source_bindings': bindings, 'source_provenance': provenance,
            'config_sha256': sha256(config), 'audit_loader_sha256': sha256(__file__)}


def check_saved_report(frame, p, root):
    saved = pd.read_csv(root/'predictions.csv', dtype={'participant_id': str})
    for column in ['participant_id', 'gender', 'severity_class']:
        np.testing.assert_array_equal(saved[column], frame[column])
    if not saved.split.eq('train_nested_oof').all():
        raise ValueError('Unexpected score split')
    np.testing.assert_allclose(saved[[f'p{i}' for i in range(5)]], p, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(saved.predicted_label, p.argmax(1))
    computed = metrics(frame.severity_class, p)
    report = json.loads((root/'metrics.json').read_text())
    if any(report[k] != v for k, v in computed.items()):
        raise ValueError('Saved aggregate/per-class metrics do not match predictions')
    groups = {str(g): metrics(part.severity_class, p[frame.index.get_indexer(part.index)])
              for g, part in frame.groupby('gender', dropna=False)}
    if report['group_metrics'] != groups:
        raise ValueError('Saved group metrics changed')
    np.testing.assert_allclose(pd.read_csv(root/'confusion_matrix.csv'), computed['confusion_matrix'])
    return computed


def run(output, config=DEFAULT_CONFIG, directory=DEFAULT_SOURCE):
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserving existing audit; choose a new output')
    matrices, frame, settings, records, provenance = load_reference(config, directory)
    y = frame.severity_class.to_numpy(int)
    plan = verify_plan(settings['frozen_plan'])
    official, issues = reconcile('data/metadata/official_edaic')
    metadata = pd.read_csv('data/metadata/participants.csv', dtype={'participant_id': str})
    pd.testing.assert_frame_equal(metadata.reset_index(drop=True), official.reset_index(drop=True))
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), official.loc[official.split.eq('train'), frame.columns].reset_index(drop=True))
    corpus = json.loads((Path(settings['corpus_directory'])/'participants.json').read_text())
    embedded = json.loads((Path(settings['feature_directory'])/'embeddings.json').read_text())
    if len({r['text_sha256'] for r in corpus}) != len(corpus):
        raise ValueError('Duplicate transcript content between unique participants')
    if len({digest(r['embedding']) for r in embedded}) != len(embedded):
        raise ValueError('Duplicate whole-interview embeddings')
    np.testing.assert_array_equal(matrices['semantic_av'][:, :384], matrices['semantic'])
    corpus_receipt = json.loads((Path(settings['corpus_directory'])/'completion.json').read_text())
    fold_support = check_folds(nested_folds(frame, settings), y, settings['fold_seeds'])
    rows, severe_rows, fits = [], [], 0
    with offline_network(), threadpool_limits(limits=2):
        for fold, record in zip(nested_folds(frame, settings), records):
            job = Path(directory)/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
            # Only load a locally produced checkpoint after its sealed hash validates.
            with (job/'model.pkl').open('rb') as handle:
                state = pickle.load(handle)
            if any(state[k] != v for k, v in {'protocol': source.PROTOCOL, 'candidate': record['selected_candidate'],
                                              'fit': fold['fit'], 'heldout': fold['heldout']}.items()):
                raise ValueError('Checkpoint payload identity differs from record')
            estimator = state['estimator']
            np.testing.assert_array_equal(estimator.classes_, np.arange(5))
            x = matrices[record['selected_candidate']['feature']]
            if state['feature_count'] != x.shape[1]:
                raise ValueError('Checkpoint input dimension changed')
            if record['selected_candidate']['family'] != 'majority':
                scaler = estimator.named_steps['standardscaler']
                np.testing.assert_allclose(scaler.mean_, x[fold['fit']].mean(0), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(scaler.var_, x[fold['fit']].var(0), rtol=1e-12, atol=1e-12)
            expected = np.asarray(record['probabilities'])
            np.testing.assert_allclose(five_probabilities(estimator, x[fold['heldout']]), expected, rtol=0, atol=1e-14)
            for saved in record['candidates']:
                x_inner = matrices[saved['candidate']['feature']]
                predicted = np.full(len(y), -1)
                for inner in fold['inner']:
                    model = source.estimator_for(saved['candidate'], settings)
                    fit(model, x_inner[inner['fit']], y[inner['fit']]); fits += 1
                    predicted[inner['heldout']] = model.predict(x_inner[inner['heldout']])
                np.testing.assert_array_equal(predicted[fold['fit']], saved['predictions'])
            replay = source.estimator_for(record['selected_candidate'], settings)
            fit(replay, x[fold['fit']], y[fold['fit']]); fits += 1
            np.testing.assert_allclose(five_probabilities(replay, x[fold['heldout']]), expected, rtol=0, atol=1e-12)
            print(f"Audit replay: {len(rows)+1}/{len(records)} jobs", flush=True)
            rows.append({'seed': fold['seed'], 'fold': fold['fold'], 'inner_fits': len(record['candidates'])*len(fold['inner']),
                         'saved_checkpoint_parity': True, 'all_inner_predictions_reproduced': True, 'selected_refit_parity': True})
        repeated = []
        for seed in settings['fold_seeds']:
            p = np.full((len(y), 5), np.nan)
            for record in records:
                if record['seed'] == seed:
                    p[record['heldout']] = record['probabilities']
            metric = check_saved_report(frame, p, Path(directory)/'oof'/f'selected_strategy_seed{seed}')
            repeated.append({'seed': seed, **{k: metric[k] for k in ['accuracy','macro_f1','balanced_accuracy']},
                'severe_correct': int(((p.argmax(1) == 4) & (y == 4)).sum()),
                'severe_predictions': int((p.argmax(1) == 4).sum()),
                'severe_average_precision': float(average_precision_score(y == 4, p[:, 4])),
                'severe_roc_auc': float(roc_auc_score(y == 4, p[:, 4]))})
            for i in np.flatnonzero(y == 4):
                severe_rows.append({'participant_id': frame.participant_id.iloc[i], 'seed': seed,
                                    'predicted_class': int(p[i].argmax()), 'severe_probability': float(p[i, 4]),
                                    'severe_rank_within_classes': int(1+(p[i] > p[i, 4]).sum())})
    expected_summary = pd.read_csv(Path(directory)/'summary.csv', index_col='model')
    for metric in ['accuracy','macro_f1','balanced_accuracy']:
        values = [r[metric] for r in repeated]
        np.testing.assert_allclose([np.mean(values), np.std(values, ddof=1)],
            expected_summary.loc['selected_strategy', [metric+'_mean',metric+'_std']], rtol=0, atol=1e-14)
    # Revalidate live bindings before writing a success receipt.
    _, frame_after, _, _, after = load_reference(config, directory)
    if after != provenance or not frame_after.equals(frame):
        raise ValueError('Inputs changed during audit')
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out/'replay_checks.csv', index=False)
    pd.DataFrame(repeated).to_csv(out/'scores_and_severe_detection.csv', index=False)
    pd.DataFrame(severe_rows).to_csv(out/'private_severe_errors.csv', index=False)
    result = {'protocol': PROTOCOL, 'passed': True, 'code_sha256': sha256(__file__), 'provenance': provenance,
        'participants': len(official), 'physical_sources': corpus_receipt['physical_sources'],
        'archives': corpus_receipt['archives'], 'folders': corpus_receipt['folders'], 'training_participants': len(y),
        'training_class_counts': np.bincount(y, minlength=5).tolist(), **fold_support,
        'reproduced_jobs': len(rows), 'audit_replay_fits': fits,
        'observed_mean_accuracy': float(np.mean([r['accuracy'] for r in repeated])),
        'label_item_sums_and_official_splits_verified': True, 'original_files_written': False,
        'validation_test_scored': False, 'frozen_plan_bindings_verified': len(plan['bound_files']),
        'raw_archive_hash_scope': 'Existing complete-read hashes and current source stamps/bindings verified; archive bytes not reread in this replay audit',
        'findings': ['No replay, score mapping, fold ownership, checkpoint or saved-metric discrepancy found',
                     'Accuracy-only selection permits zero severe recall; original candidates have no imbalance intervention',
                     'Only seven independent severe training participants; synthetic features cannot add independent evidence',
                     'Whole-interview speaker identity and public encoder pretraining overlap remain unverified',
                     '48.47% describes repeated nested training selection, not an independently tested deployment model'],
        'artifacts': {p.name: sha256(p) for p in out.glob('*.csv')}}
    atomic_json(out/'audit.json', result)
    return {k:v for k,v in result.items() if k not in ['provenance','artifacts']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results/semantic_pipeline_audit')
    args = p.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == '__main__':
    main()
