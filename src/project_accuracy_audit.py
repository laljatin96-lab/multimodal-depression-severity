"""Read-only evidence audit of the current severity pipeline; never retrain old experiments."""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json
from .evaluate import metrics
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .semantic_severe_repair import fit_receipt, validate_receipt

OUTPUT = Path('results/project_accuracy_audit_20261027')
STUDIES = [
    'modality_diagnostics', 'raw_frame_comparison', 'temporal_summary_comparison',
    'accuracy_model_comparison', 'ensemble_accuracy_target', 'text_accuracy_target',
    'semantic_accuracy_target', 'foundation_accuracy_target', 'encoder_kernel_accuracy',
    'semantic_severe_repair_v2', 'severe_detection', 'chunk_severe_detection',
    'deep_fusion_accuracy', 'score_regression_accuracy', 'catboost_late_fusion',
    'baseline_refinement', 'targeted512_refinement', 'imbalance_accuracy',
    'supervised_accuracy', 'baseline_averaging_accuracy', 'multitask_attention_accuracy',
    'decision_offset_accuracy']


def assert_preserved(snapshot):
    before = json.loads(Path(snapshot).read_text())
    changed = [p for p, v in before.items() if not Path(p).is_file() or
               Path(p).stat().st_mtime_ns != v['mtime_ns'] or sha256(p) != v['sha256']]
    if changed:
        raise ValueError(f'Previously saved files changed: {changed}')
    return len(before)


def label_checks(metadata, detailed):
    if metadata.participant_id.duplicated().any() or detailed.index.duplicated().any():
        raise ValueError('Duplicate participant identity')
    columns = [c for c in detailed if c.startswith('PHQ8_')]
    items = detailed.loc[metadata.participant_id, columns].to_numpy(float)
    if items.shape != (len(metadata), 8) or not np.isfinite(items).all() or not np.array_equal(items, items.astype(int)) or (items < 0).any() or (items > 3).any():
        raise ValueError('Invalid questionnaire items')
    if not np.array_equal(items.sum(1), metadata.phq8_score.to_numpy(float)):
        raise ValueError('Total/item mismatch')
    if not np.array_equal(np.minimum(items.sum(1)//5, 4), metadata.severity_class.to_numpy(int)):
        raise ValueError('Severity mapping mismatch')
    if set(metadata.split) != {'train', 'validation', 'test'}:
        raise ValueError('Unexpected split')
    return {'participants': len(metadata), 'duplicate_ids': 0, 'total_item_mismatches': 0, 'class_mismatches': 0,
            'split_counts': metadata.groupby(['split', 'severity_class']).size().unstack(fill_value=0).to_dict('index')}


def errors(frame, probabilities):
    y = frame.severity_class.to_numpy(int); p = np.asarray(probabilities, float)
    if p.ndim != 3 or p.shape[1:] != (len(y), 5) or not np.isfinite(p).all() or not np.allclose(p.sum(2), 1):
        raise ValueError('Invalid repeated held-out probabilities')
    pred = p.argmax(2); cm = np.zeros((5, 5), int)
    for row in pred:
        np.add.at(cm, (y, row), 1)
    repeated_errors = (cm.sum(1)-cm.diagonal()).tolist()
    result = {'accuracy': float((pred == y).mean()), 'majority_accuracy': float((y == 0).mean()),
              'recall': (cm.diagonal()/cm.sum(1)).tolist(), 'confusion_matrix': cm.tolist(),
              'class_errors': repeated_errors, 'always_wrong': int((pred != y).all(0).sum()),
              'ever_wrong': int((pred != y).any(0).sum()), 'majority_prediction_fraction': float((pred == 0).mean()),
              'mild_moderate_error_fraction': sum(repeated_errors[1:3])/sum(repeated_errors),
              'repeats': len(p), 'unique_participants': len(y)}
    private = frame[['participant_id', 'severity_class']].copy()
    private['wrong_repeats'] = (pred != y).sum(0)
    private['mean_true_class_score'] = p[:, np.arange(len(y)), y].mean(0)
    private['mean_winning_score'] = p.max(2).mean(0)
    private['review_flag_not_verified_mislabel'] = ((private.wrong_repeats == len(p)) &
        (private.mean_true_class_score < .1) & (private.mean_winning_score > .6))
    for i in range(len(p)):
        private[f'repeat_{i+1}_prediction'] = pred[i]
    return result, private


def ledger():
    rows = []; skipped = []
    for study in STUDIES:
        path = Path('results')/study/'summary.csv'
        if not path.exists():
            skipped.append(str(path)); continue
        table = pd.read_csv(path)
        if 'accuracy_mean' not in table:
            skipped.append(str(path)); continue
        for _, r in table.iterrows():
            identity = {str(k): str(v) for k, v in r.items() if not str(k).endswith(('_mean', '_std'))}
            rows.append({'study': study, 'variant': json.dumps(identity, sort_keys=True),
                         'nested_accuracy': float(r['accuracy_mean']),
                         'macro_f1': float(r['macro_f1_mean']) if 'macro_f1_mean' in r and pd.notna(r['macro_f1_mean']) else None,
                         'summary_path': str(path), 'summary_sha256': sha256(path),
                         'evidence_scope': 'Saved training-participant CV; historical selection rules differ; exploratory',
                         'repeat_action': 'Reuse saved evidence; no old fits required'})
    return pd.DataFrame(rows), skipped


def audit():
    if (OUTPUT/'audit.json').exists():
        raise FileExistsError('Preserving completed audit')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, frame, settings, records, provenance = load_baseline()
    metadata_path = Path('data/metadata/participants.csv')
    detailed_path = Path('data/metadata/official_edaic/detailed_lables.csv')
    metadata = pd.read_csv(metadata_path, dtype={'participant_id': str})
    detailed = pd.read_csv(detailed_path, dtype={'Participant': str}).set_index('Participant')
    labels = label_checks(metadata, detailed)
    y = frame.severity_class.to_numpy(int); folds = nested_folds(frame, settings)
    probabilities = {seed: np.full((len(y), 5), np.nan) for seed in settings['fold_seeds']}
    owners = {seed: np.zeros(len(y), int) for seed in settings['fold_seeds']}
    fit_accuracy = []; inner_counts = []; outer_counts = []; replayed = 0; lineage_checked = 0
    with threadpool_limits(limits=2):
        for fold, record in zip(folds, records):
            if set(fold['fit']) & set(fold['heldout']) or set(fold['fit']+fold['heldout']) != set(range(len(y))):
                raise ValueError('Outer leakage/coverage')
            for part in fold['inner']:
                if set(part['fit']) & set(part['heldout']) or set(part['fit']+part['heldout']) != set(fold['fit']):
                    raise ValueError('Inner leakage/coverage')
                inner_counts.append(int((y[part['fit']] == 4).sum()))
            outer_counts.append(int((y[fold['fit']] == 4).sum()))
            saved = record['policies']['accuracy']; candidate = saved['candidate']
            path = ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"/'accuracy.pkl'
            state = pickle.loads(path.read_bytes()); model = state['estimator']; fitting = x[candidate['feature']][fold['fit']]
            p = model.predict_proba(x[candidate['feature']][fold['heldout']])
            np.testing.assert_allclose(p, saved['probabilities'], rtol=0, atol=1e-14)
            receipt = fit_receipt(model, candidate); validate_receipt(receipt, candidate, y[fold['fit']], settings)
            if candidate['family'] != 'majority':
                np.testing.assert_allclose(model.scaler_.mean_, fitting.mean(0), rtol=0, atol=1e-12)
                np.testing.assert_allclose(model.scaler_.var_, fitting.var(0), rtol=1e-12, atol=1e-12)
                lineage_checked += len(model.lineage_)
            fit_accuracy.append(float((model.predict(fitting) == y[fold['fit']]).mean()))
            probabilities[fold['seed']][fold['heldout']] = p; owners[fold['seed']][fold['heldout']] += 1; replayed += 1
    if any(not (v == 1).all() for v in owners.values()):
        raise ValueError('OOF coverage differs')
    prediction_errors, private = errors(frame, np.stack(list(probabilities.values())))
    private.to_csv(OUTPUT/'private_error_review.csv', index=False)
    quality = pd.read_csv('results/feature_diagnostics/participant_quality.csv', dtype={'participant_id': str}).set_index('participant_id').loc[frame.participant_id]
    private_quality = private.join(quality[['bins', 'audio_observed_fraction', 'visual_observed_fraction']].reset_index(drop=True))
    private_quality.to_csv(OUTPUT/'private_errors_and_quality.csv', index=False)
    quality_groups = private_quality.groupby(private_quality.wrong_repeats.eq(3))[
        ['bins', 'visual_observed_fraction']].agg(['count', 'mean', 'median'])
    quality_groups.to_csv(OUTPUT/'quality_by_error_consistency.csv')
    docs = json.loads(Path('results/transcript_corpus/participants.json').read_text())
    features = {k: {'shape': list(v.shape), 'nonfinite_values': int((~np.isfinite(v)).sum()),
                    'constant_columns': int((v.std(0) == 0).sum()), 'duplicate_rows': len(v)-len(np.unique(v, axis=0))} for k, v in x.items()}
    corpus = {'participants': len(docs), 'empty_documents': sum(not d['text'].strip() for d in docs),
              'duplicate_text_hashes': len(docs)-len({d['text_sha256'] for d in docs}),
              'words_range': [min(d['words'] for d in docs), max(d['words'] for d in docs)],
              'explicit_score_rows_removed': sum(d['explicit_score_rows_removed'] for d in docs), 'speaker_identity_available': False}
    table, skipped = ledger(); table.to_csv(OUTPUT/'experiment_ledger.csv', index=False)
    bindings = {str(p): sha256(p) for p in [metadata_path, detailed_path, Path(__file__),
        Path('results/transcript_corpus/participants.json'), Path('results/feature_diagnostics/participant_quality.csv'),
        Path('results/timing_audit/timing_assessment.json'), Path('results/multitask_attention_accuracy/verification.json')]}
    result = {'date': '2026-10-27', 'labels': labels, 'features': features, 'corpus': corpus,
              'prediction_errors': prediction_errors, 'mean_fold_training_accuracy': float(np.mean(fit_accuracy)),
              'severe_support': {'outer_fit': sorted(set(outer_counts)), 'inner_fit': sorted(set(inner_counts))},
              'baseline_checkpoints_replayed': replayed, 'synthetic_parent_receipts_checked': lineage_checked,
              'inner_outer_split_checks_passed': True, 'fitting_scalers_verified': True, 'models_fitted': 0,
              'test_predictions_or_scores_computed': False, 'validation_predictions_or_scores_computed': False,
              'labels_changed': 0, 'participants_excluded': 0, 'ledger_rows': len(table), 'ledger_skipped_schemas': skipped,
              'data_quality': quality[['bins','audio_observed_fraction','visual_observed_fraction']].agg(['min','median','max']).to_dict(),
              'bindings': bindings, 'baseline_bindings_verified': len(provenance['bindings']),
              'independent_confirmation': False,
              'scope': 'All participant metadata and cached text; training feature matrices, all baseline outer checkpoints, fold/preprocessing receipts and saved study results. Source archive integrity uses existing audited hashes/current stamps; no complete raw archive reread or independent clinical/speaker validation.'}
    result['previous_files_and_mtimes_preserved'] = assert_preserved(OUTPUT/'preservation_before.json')
    atomic_json(OUTPUT/'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in {'bindings', 'data_quality'}}, indent=2))
    return result


if __name__ == '__main__':
    audit()
