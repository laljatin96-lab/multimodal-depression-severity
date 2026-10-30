"""Read-only label, feature and generalization audit of the preserved best baseline."""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .batch_inference import atomic_json
from .crossvalidated_baselines import five_probabilities
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds
from .severe_case_diagnosis import load_baseline, ROOT
from .severe_detection_study import score


def diagnose(output='results/baseline_refinement_diagnosis'):
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserving existing diagnosis')
    x, frame, settings, records, provenance = load_baseline()
    y = frame.severity_class.to_numpy(int)
    paths = ['data/metadata/participants.csv', 'data/metadata/official_edaic/detailed_lables.csv']
    metadata = pd.read_csv(paths[0], dtype={'participant_id': str}).set_index('participant_id').loc[frame.participant_id]
    detail = pd.read_csv(paths[1], dtype={'Participant': str}).set_index('Participant').loc[frame.participant_id]
    items = detail.filter(regex='^PHQ8_[1-8]_').apply(pd.to_numeric, errors='raise').to_numpy(float)
    totals = metadata.phq8_score.to_numpy(float)
    if items.shape != (len(y), 8) or not np.isfinite(items).all() or (items < 0).any() or (items > 3).any() or not np.array_equal(items, items.astype(int)):
        raise ValueError('Invalid item responses')
    if not np.array_equal(items.sum(1), totals) or not np.array_equal(np.minimum(totals//5, 4), y) or not (metadata.split == 'train').all():
        raise ValueError('Label or training split inconsistency')
    rows, probability_rows, counts = [], [], []
    folds = {(f['seed'], f['fold']): f for f in nested_folds(frame, settings)}
    with threadpool_limits(limits=2):
        for r in records:
            result = r['policies']['accuracy']
            path = ROOT/'jobs'/f"seed{r['seed']}_fold{r['fold']}"/'accuracy.pkl'
            with path.open('rb') as f:
                state = pickle.load(f)
            p = five_probabilities(state['estimator'], x[result['candidate']['feature']][r['fit']])
            inner = r['candidates'][result['selection']['index']]['predictions']
            outer = np.asarray(result['probabilities'])
            for name, indices, prediction in [('fitting', r['fit'], p.argmax(1)), ('inner_selected', r['fit'], np.asarray(inner)), ('outer', r['heldout'], outer.argmax(1))]:
                rows.append({'seed': r['seed'], 'fold': r['fold'], 'evaluation': name, **score(y[indices], prediction)})
            counts.append({'seed': r['seed'], 'fold': r['fold'], 'outer_fitting': np.bincount(y[r['fit']], minlength=5).tolist(),
                           'inner_fitting': [np.bincount(y[f['fit']], minlength=5).tolist()
                                             for f in folds[(r['seed'], r['fold'])]['inner']]})
            for index, probabilities in zip(r['heldout'], outer):
                probability_rows.append({'participant_id': frame.participant_id.iloc[index], 'seed': r['seed'], 'true_label': int(y[index]),
                                         'predicted_label': int(probabilities.argmax()), 'true_probability': float(probabilities[y[index]]),
                                         'winning_probability': float(probabilities.max()), 'severe_probability': float(probabilities[4])})
    feature_quality = {name: {'shape': list(a.shape), 'finite': bool(np.isfinite(a).all()),
                              'constant_columns': int((np.ptp(a, axis=0) == 0).sum()),
                              'duplicate_rows': int(len(a)-len(np.unique(a, axis=0))),
                              'norm_range': [float(np.linalg.norm(a, axis=1).min()), float(np.linalg.norm(a, axis=1).max())]}
                       for name, a in x.items()}
    probs = pd.DataFrame(probability_rows)
    review = []
    for pid, group in probs.groupby('participant_id', sort=True):
        if len(group) != len(settings['fold_seeds']):
            raise ValueError('Incomplete participant probability coverage')
        if (group.predicted_label != group.true_label).all() and group.true_probability.mean() < .1 and group.winning_probability.mean() > .6:
            review.append({'participant_id': pid, 'true_label': int(group.true_label.iloc[0]),
                           'predicted_labels': group.predicted_label.tolist(), 'mean_true_probability': float(group.true_probability.mean()),
                           'mean_winning_probability': float(group.winning_probability.mean()),
                           'note': 'Review flag only; not evidence of mislabeling; original label retained.'})
    out.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(out/'generalization.csv', index=False)
    probs.to_csv(out/'private_probability_audit.csv', index=False)
    atomic_json(out/'private_label_review_flags.json', {'criteria': 'Wrong in all repeats, mean true-class probability <0.1, mean winning probability >0.6; no relabeling.', 'participants': review})
    atomic_json(out/'fitting_class_counts.json', counts)
    means = table.groupby('evaluation')[['accuracy', 'macro_f1', 'balanced_accuracy']].mean().to_dict('index')
    report = {'participants': len(y), 'training_counts': np.bincount(y, minlength=5).tolist(), 'all_totals_match_items': True,
              'all_class_mappings_match': True, 'feature_quality': feature_quality, 'mean_fold_metrics': means,
              'review_flags': len(review), 'label_changes': 0, 'refits': 0, 'validation_test_scored': False,
              'note': 'Fitting and selected inner estimates are optimistic; outer scores are repeated exploratory training CV, not independent test evidence.',
              'limitations': ['Item consistency does not independently authenticate participant identity or clinical status.',
                             'Finite cached features do not verify speaker identity, speech quality or semantic relevance.',
                             'No new raw archive reads; reuse verified full-dataset inventory and caches.'],
              'code_sha256': sha256(__file__), 'metadata_hashes': {p: sha256(p) for p in paths}, 'provenance': provenance}
    atomic_json(out/'diagnosis.json', report)
    print(json.dumps({k: report[k] for k in ['participants', 'training_counts', 'all_totals_match_items', 'feature_quality', 'mean_fold_metrics', 'review_flags', 'refits']}, indent=2))
    return report


if __name__ == '__main__':
    diagnose()
