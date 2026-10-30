"""Participant summary baselines with explicit unresolved timing/speech limitations.

This separate experiment pools each modality independently. It never promotes
unverified caches to temporal research caches or assumes simultaneous AV frames.
It retains the previously cached overlap crop and 0.5-second binning, so these
are exploratory results, pending verification of preprocessing provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import VERSION, digest, encode_gender, joined_manifest, load_config, write_json
from .evaluate import write_report
from .full_dataset import discover_sources, source_stamp
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS


def pool_modalities(sequence):
    """No cross-modal pairing, correlation, windows or sequence modeling."""
    summaries = []
    for modality in ['audio', 'visual']:
        values = np.asarray(sequence[modality], dtype=np.float64)
        mask = np.asarray(sequence[modality + '_mask'])
        if values.ndim != 2 or mask.shape != (len(values),) or mask.dtype != bool or not mask.any():
            raise ValueError('Invalid modality shape or empty observation mask')
        observed = values[mask]
        if not np.isfinite(observed).all():
            raise ValueError('Nonfinite observed feature')
        summaries.extend([observed.mean(0), observed.std(0)])
    return np.concatenate(summaries)


def load_audited_sequences(config, manifest):
    """Read audited arrays for explicitly exploratory, independent-modality use."""
    root = Path(config.get('full_dataset_report', 'results/full_dataset'))
    summary = json.loads((root / 'summary.json').read_text())
    run = json.loads((root / 'run.json').read_text())
    if not summary['complete'] or Path(run['config']['dataset']).resolve() != Path(config['dataset']).resolve():
        raise ValueError('A complete audit of this dataset is required')
    discovered, unknown = discover_sources(config['dataset'])
    sources = pd.read_csv(root / 'sources.csv', dtype=str, keep_default_na=False)
    if unknown or set(sources.source) != {str(s['path']) for s in discovered}:
        raise ValueError('Source inventory changed; rerun the full dataset audit')
    reports = {}
    for key in sources.source_key:
        info = json.loads((root / 'sources' / f'{key}.json').read_text())
        if not info['integrity_checked'] or source_stamp(Path(info['source'])) != info['source_stamp']:
            raise ValueError('Source changed or integrity check missing; rerun full dataset audit')
        reports[info['source']] = info
    participants = pd.read_csv(root / 'participants.csv', dtype={'participant_id': str}, keep_default_na=False).set_index('participant_id')
    if set(participants.index) != set(manifest.participant_id) or participants.index.duplicated().any():
        raise ValueError('All audited participants must be retained exactly once')
    sequences, provenance, hashes = {}, {}, set()
    for pid in manifest.participant_id:
        row = participants.loc[pid]
        if str(row.feature_ready).lower() != 'true' or row.duplicate_status == 'conflicting_features':
            raise ValueError(f'Unresolved or missing primary features for {pid}')
        selected = reports[row.selected_source]
        info = selected['modalities']['mfcc']
        if info['version'] != VERSION or info['audio_columns'] != AUDIO_COLUMNS or info['visual_columns'] != VISUAL_COLUMNS:
            raise ValueError('Unsupported or inconsistent cached feature schema')
        if any(info['settings'][k] != config[k] for k in ['audio_kind', 'bin_seconds', 'confidence_threshold', 'audio_time_offset', 'visual_time_offset', 'minimum_window_steps']):
            raise ValueError('Cached feature settings differ from experiment config')
        path = Path(selected['artifacts']['mfcc_npz'])
        if path != Path(row.audio_feature_path):
            raise ValueError('Source and participant cache references disagree')
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in ['audio', 'visual', 'audio_mask', 'visual_mask', 'timestamps']}
            if any(len(value) != len(values['timestamps']) for value in values.values()) or not np.isfinite(values['timestamps']).all() or not np.allclose(np.diff(values['timestamps']), config['bin_seconds']):
                raise ValueError('Invalid cached timestamp grid')
            # Validate observed values/masks independently for both modalities.
            pooled = pool_modalities(values)
            content_hash = hashlib.sha256(b''.join(values[key].tobytes() for key in values)).hexdigest()
            if content_hash in hashes:
                raise ValueError('Duplicate participant feature content')
            hashes.add(content_hash)
            sequences[pid] = values
        provenance[pid] = {'cache': str(path), 'npz_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_fingerprint': info['fingerprint'], 'summary_sha256': hashlib.sha256(pooled.tobytes()).hexdigest()}
    return sequences, provenance


def load_summary_matrix(config, manifest):
    sequences, provenance = load_audited_sequences(config, manifest)
    return np.stack([pool_modalities(sequences[pid]) for pid in manifest.participant_id]), provenance


def fit_candidates(matrix, manifest, config, output, grid=(0.01, 0.1, 1.0, 10.0)):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f'Preserving existing experiment: {out}')
    if len(matrix) != len(manifest) or not np.isfinite(matrix).all() or manifest.participant_id.duplicated().any():
        raise ValueError('Invalid participant summaries')
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', {'status': 'EXPLORATORY', 'seed': config['seed'], 'regularization_C_candidates': list(grid), 'selection_metric': 'validation fixed-five-class macro-F1; ties prefer smaller C', 'test_evaluation': False, 'fit_participants': 'train only', 'feature_definition': 'Independent per-modality means and population standard deviations of observed cached 0.5-second bins; 128 AV features', 'limitations': ['Timestamp origins not verified; existing overlap crop retained', 'Unsegmented audio may contain interviewer speech', 'Validation used for C selection; selected validation metrics are optimistic', 'No claim of clinical validity or faithful paper reproduction'], 'config': config, 'manifest_hash': digest(manifest.to_dict('records'))})
    train_mask = manifest.split.eq('train').to_numpy()
    validation_mask = manifest.split.eq('validation').to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError('Training and validation participants required')
    train_ids = manifest.loc[train_mask, 'participant_id'].tolist()
    y = manifest.loc[train_mask, 'severity_class'].to_numpy()
    validation = manifest.loc[validation_mask].reset_index(drop=True)
    gender = np.stack([encode_gender(value, config['gender_categories']) for value in manifest.gender])
    rows, selected_rows = [], []
    definitions = [('majority', False, False), ('LR-AV', False, False), ('LR-AV-balanced', False, True), ('LR-AV-gender', True, False), ('LR-AV-gender-balanced', True, True), ('LR-gender-only', True, False)]
    for name, with_gender, balanced in definitions:
        features = gender if name == 'LR-gender-only' else np.column_stack([matrix, gender]) if with_gender else matrix
        candidates = (1.0,) if name in ['majority', 'LR-gender-only'] else grid
        best = None
        for c in candidates:
            estimator = DummyClassifier(strategy='most_frequent') if name == 'majority' else make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced' if balanced else None, max_iter=2000, solver='lbfgs', random_state=config['seed']))
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                estimator.fit(features[train_mask], y)
            probabilities = np.zeros((len(validation), 5))
            probabilities[:, estimator.classes_.astype(int)] = estimator.predict_proba(features[validation_mask])
            candidate = out / name / f'C_{c:g}'
            report = write_report(validation, probabilities, candidate / 'validation', name, config['seed'], 'validation')
            write_json(candidate / 'training_provenance.json', {'fit_participant_ids': train_ids, 'C': c, 'gender_input': with_gender, 'balanced': balanced, 'feature_count': features.shape[1], 'train_class_counts': np.bincount(y, minlength=5).tolist()})
            with (candidate / 'model.pkl').open('wb') as handle:
                pickle.dump({'estimator': estimator, 'gender_input': with_gender, 'gender_only': name == 'LR-gender-only', 'gender_categories': config['gender_categories'], 'protocol': 'EXPLORATORY participant summaries', 'fit_participant_ids': train_ids}, handle)
            result = {'model': name, 'C': c, 'run': str(candidate), **{key: report[key] for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']}}
            rows.append(result)
            if best is None or result['macro_f1'] > best['macro_f1']:
                best = result
        selected_rows.append(best)
        print(f"{name}: selected C={best['C']:g}, validation macro-F1={best['macro_f1']:.4f}, accuracy={best['accuracy']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out / 'all_candidates.csv', index=False)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(out / 'validation_summary.csv', index=False)
    write_json(out / 'completion.json', {'complete': True, 'fitted_candidates': len(rows), 'train_participants': int(train_mask.sum()), 'validation_participants': int(validation_mask.sum()), 'reserved_test_participants': int(manifest.split.eq('test').sum()), 'test_predictions_created': False})
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--output', default='results/exploratory_baselines')
    args = parser.parse_args()
    config = load_config(args.config)
    if Path(args.output).exists():
        raise FileExistsError('Choose a new output path to preserve previous results')
    manifest = joined_manifest(config)
    matrix, provenance = load_summary_matrix(config, manifest)
    selected = fit_candidates(matrix, manifest, config, args.output)
    from .train import record_result
    for row in selected.itertuples():
        report = json.loads((Path(row.run) / 'validation/metrics.json').read_text())
        record_result(row.run, {**report, 'model': 'EXPLORATORY-' + report['model']}, config)
    write_json(Path(args.output) / 'feature_provenance.json', provenance)
    manifest[['participant_id', 'split']].to_csv(Path(args.output) / 'participant_coverage.csv', index=False)
    np.savez_compressed(Path(args.output) / 'participant_summaries.npz', participant_ids=manifest.participant_id.to_numpy(dtype=str), features=matrix)
    lines = ['# Exploratory participant-summary baselines', '', 'Real supplied dataset, original questionnaire targets. All 275 participants are accounted for in the official split. Only training participants fit the models; test participants remain reserved.', '', 'These models independently pool each modality. They use no temporal AV pairing. The existing overlap crop, 0.5-second binning and unsegmented audio remain preprocessing limitations. Temporal LSTM training remains blocked pending verified timing/speech protocol.', '', '| Model | C | Validation accuracy | Macro-F1 (5 classes) | Balanced accuracy |', '|---|---:|---:|---:|---:|']
    for row in selected.itertuples():
        lines.append(f'| {row.model} | {row.C:g} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} |')
    lines += ['', 'Regularization was selected using the same four C values for each AV variant. The majority and gender-only diagnostics use one fixed setting. These selected validation scores are exploratory and optimistic; they are not final test performance. Logistic fitting is deterministic here, so repeating identical optimization under different seeds would not provide independent evidence.', '', 'See all_candidates.csv, validation_summary.csv, protocol.json, feature_provenance.json and each candidate directory for saved models, predictions, group metrics, confusion matrices and training participant IDs.', '']
    (Path(args.output) / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
