"""Replay selected local models on train/validation to diagnose generalization."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from .common import digest, encode_gender, joined_manifest, write_json
from .evaluate import metrics, write_report
from .exploratory_baselines import load_audited_sequences, pool_modalities
from .independent_datasets import independent_loader
from .independent_train import PROTOCOL, build_model, predictions
from .train import choose_device


def sensitivity(frame, probabilities):
    """Leave-one-participant-out sensitivity, not a confidence interval."""
    y = frame.severity_class.to_numpy()
    full = metrics(y, probabilities)['macro_f1']
    return pd.DataFrame([
        {'participant_id': frame.iloc[i].participant_id,
         'severity_class': int(y[i]),
         'macro_f1_without_participant': metrics(np.delete(y, i), np.delete(probabilities, i, axis=0))['macro_f1'],
         'full_macro_f1': full}
        for i in range(len(frame))
    ])


def diagnose(output, baseline_root='results/exploratory_baselines', matrix_root='checkpoints/independent_lstm'):
    out, baselines, neural = Path(output), Path(baseline_root), Path(matrix_root)
    if out.exists():
        raise FileExistsError(f'Preserving existing diagnostics: {out}')
    protocol = json.loads((neural/'protocol.json').read_text())
    config = protocol['config']
    manifest = joined_manifest(config)
    sequences, sources = load_audited_sequences(config, manifest)
    provenance = {'protocol': PROTOCOL, 'manifest_hash': digest(manifest.to_dict('records')), 'features': sources}
    if protocol['provenance'] != provenance:
        raise ValueError('Neural source/manifest provenance changed')
    baseline_protocol = json.loads((baselines/'protocol.json').read_text())
    if baseline_protocol['manifest_hash'] != provenance['manifest_hash'] or json.loads((baselines/'feature_provenance.json').read_text()) != sources:
        raise ValueError('Summary baseline source/manifest provenance changed')
    # Test records remain in the coverage manifest, but are never predicted.
    active = manifest[manifest.split.isin(['train', 'validation'])].reset_index(drop=True)
    matrix = np.stack([pool_modalities(sequences[pid]) for pid in active.participant_id])
    genders = np.stack([encode_gender(value, config['gender_categories']) for value in active.gender])
    train_ids = active.loc[active.split.eq('train'), 'participant_id'].tolist()
    out.mkdir(parents=True)
    manifest[['participant_id', 'split']].to_csv(out/'participant_coverage.csv', index=False)
    write_json(out/'protocol.json', {
        'purpose': 'Post-selection training/validation diagnostics; no model fitting or test predictions',
        'provenance': provenance,
        'limitations': ['In-sample training scores are optimistic', 'Validation already used for prior selection',
                        'Leave-one-out sensitivity is not an uncertainty interval',
                        'Inherited unverified AV timing and unsegmented audio limitations remain'],
    })
    rows, checks, class_rows, sensitivity_rows = [], [], [], []

    def save(name, seed, split, frame, probabilities, source):
        destination = out/f'{name}_seed{seed}'/split
        report = write_report(frame, probabilities, destination, name, seed, split)
        if split == 'validation':
            old = pd.read_csv(source/'validation/predictions.csv', dtype={'participant_id': str})
            if old.participant_id.tolist() != frame.participant_id.tolist():
                raise ValueError('Stored validation participant order changed')
            np.testing.assert_allclose(probabilities, old[[f'p{i}' for i in range(5)]], rtol=1e-6, atol=1e-7)
            checks.append({'model': name, 'seed': seed, 'saved_validation_reproduced': True})
            if name == 'LR-AV':
                sensitivity_rows.extend(sensitivity(frame, probabilities).to_dict('records'))
        pred_counts = np.bincount(probabilities.argmax(1), minlength=5)
        rows.append({'model': name, 'seed': seed, 'split': split,
                     **{key: report[key] for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']},
                     **{f'predicted_class_{i}': int(n) for i, n in enumerate(pred_counts)}})
        class_rows.extend({'model': name, 'seed': seed, 'split': split, **item} for item in report['per_class'])

    for row in pd.read_csv(baselines/'validation_summary.csv').itertuples():
        source = Path(row.run)
        with (source/'model.pkl').open('rb') as handle:
            state = pickle.load(handle)  # Trusted, locally produced artifact only.
        if state['fit_participant_ids'] != train_ids:
            raise ValueError('Summary training participant IDs changed')
        features = genders if state['gender_only'] else np.column_stack([matrix, genders]) if state['gender_input'] else matrix
        estimator = state['estimator']
        for split in ['train', 'validation']:
            mask = active.split.eq(split).to_numpy()
            probs = np.zeros((mask.sum(), 5))
            probs[:, estimator.classes_.astype(int)] = estimator.predict_proba(features[mask])
            save(row.model, baseline_protocol['seed'], split, active[mask].reset_index(drop=True), probs, source)

    torch.set_num_threads(2)
    for row in pd.read_csv(neural/'validation_runs.csv').itertuples():
        source = Path(row.run)
        state = torch.load(source/'best.pt', map_location='cpu', weights_only=False)
        if state['provenance'] != provenance or state['protocol'] != PROTOCOL:
            raise ValueError('Checkpoint data/protocol mismatch')
        device = choose_device(state['config']['device'])
        model = build_model(sequences, state['config'], state['gender']).to(device)
        model.load_state_dict(state['model'])
        for split in ['train', 'validation']:
            frame = active[active.split.eq(split)].reset_index(drop=True)
            ids, probs = predictions(model, independent_loader(frame, sequences, state['scalers'], state['config']), device)
            if ids != frame.participant_id.tolist():
                raise ValueError('Neural prediction order changed')
            save(row.model, row.seed, split, frame, probs, source)

    scores = pd.DataFrame(rows)
    scores.to_csv(out/'scores.csv', index=False)
    pd.DataFrame(class_rows).to_csv(out/'per_class.csv', index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out/'logistic_validation_sensitivity.csv', index=False)
    mean = scores.groupby(['model', 'split'])[['accuracy', 'macro_f1', 'balanced_accuracy']].mean().unstack('split')
    mean.to_csv(out/'summary.csv')
    lines = ['# Generalization diagnostics', '',
             f'All {len(manifest)} participants remain accounted for. Saved models were replayed on training and validation only. No fitting, new tuning, or test prediction occurred.', '',
             'Neural values are means of all three saved seeds. Training scores use the validation-selected checkpoint; they are in-sample diagnostics.', '',
             '| Model | Train macro-F1 | Validation macro-F1 | Train minus validation |',
             '|---|---:|---:|---:|']
    for name, row in mean.iterrows():
        train, val = row[('macro_f1', 'train')], row[('macro_f1', 'validation')]
        lines.append(f'| {name} | {train:.4f} | {val:.4f} | {train-val:+.4f} |')
    influence = pd.DataFrame(sensitivity_rows)
    severe = influence[influence.severity_class.eq(4)]
    lines += ['', f"For the AV logistic baseline, leaving out one validation participant gives fixed-five-class macro-F1 between {influence.macro_f1_without_participant.min():.4f} and {influence.macro_f1_without_participant.max():.4f}. This is a sensitivity diagnostic, not a confidence interval."]
    if len(severe) == 1:
        lines += ['', f"Without the sole severe-class validation participant, AV logistic macro-F1 is {severe.iloc[0].macro_f1_without_participant:.4f}; the five-class denominator is retained. The original full validation score remains the reported result."]
    lines += ['', 'Class prediction counts, per-class train/validation metrics, and local participant predictions accompany this report. Generalization gaps do not identify a causal explanation. Original timing, speech isolation, sparse class, and validation selection limitations remain.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    write_json(out/'verification.json', {'saved_models': checks, 'test_evaluated': False, 'complete': True})
    return {'report': str(out/'REPORT.md'), 'models_verified': len(checks), 'test_evaluated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/generalization_diagnostics')
    parser.add_argument('--baselines', default='results/exploratory_baselines')
    parser.add_argument('--matrix', default='checkpoints/independent_lstm')
    args = parser.parse_args()
    print(json.dumps(diagnose(args.output, args.baselines, args.matrix), indent=2))
