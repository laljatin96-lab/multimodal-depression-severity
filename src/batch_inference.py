"""Resumable local inference accounting for every supplied interview source."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, write_json
from .exploratory_baselines import pool_modalities
from .full_dataset import discover_sources, source_stamp
from .reference_inference import DEFAULT_BUNDLE, load_bundle, probabilities, read_cache, read_features, sha256

FORMAT = 'all-source-batch-inference-v1'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def save_record(path, record, context_hash):
    atomic_json(path, {'context_hash': context_hash, 'record': record, 'record_sha256': digest(record)})


def load_record(path, context_hash):
    envelope = json.loads(Path(path).read_text())
    if envelope['context_hash'] != context_hash or envelope['record_sha256'] != digest(envelope['record']):
        raise ValueError(f'Resume record integrity/context mismatch: {path}')
    return envelope['record']


@contextmanager
def output_lock(root):
    with (root/'.batch.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another batch process is using this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stamp_or_error(path):
    try:
        return source_stamp(path)
    except (OSError, ValueError) as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


def inventory(dataset):
    sources, unknown = discover_sources(dataset)
    rows = [{'source_key': s['source_key'], 'source': str(s['path'].resolve()),
             'participant_id': s['participant_id'], 'kind': s['kind'],
             'source_stamp': stamp_or_error(s['path'])} for s in sources]
    return sources, rows, sorted(unknown)


def primary_hashes(report, pid):
    names = [f'{pid}_OpenSMILE2.3.0_mfcc.csv', f'{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv']
    return {name: report['feature_sha256'][name] for name in names}


def audited_cache_index(dataset, sources, audit, frozen_plan, contract):
    """Reuse only caches already bound to a verified frozen plan.

    All physical source stamps and selected raw feature hashes still have to
    match the audit. Identical copies may share a bound canonical cache.
    """
    from .final_evaluation import verify_plan
    root = Path(audit)
    plan = verify_plan(frozen_plan)
    if contract.get('frozen_plan_sha256') != digest(plan):
        raise ValueError('Bundle and frozen cache plan differ')
    summary = json.loads((root/'summary.json').read_text())
    run = json.loads((root/'run.json').read_text())
    if not summary['complete'] or Path(run['config']['dataset']).resolve() != Path(dataset).resolve():
        raise ValueError('A complete audit of this dataset is required for cache reuse')
    table = pd.read_csv(root/'sources.csv', dtype=str, keep_default_na=False)
    if table.source_key.duplicated().any() or set(table.source_key) != {s['source_key'] for s in sources}:
        raise ValueError('Source inventory differs from the complete audit')
    reports, producers, bindings = {}, {}, {}
    for item in table.itertuples():
        report_path = root/'sources'/f'{item.source_key}.json'
        report = json.loads(report_path.read_text())
        if not report['integrity_checked'] or source_stamp(Path(report['source'])) != report['source_stamp']:
            raise ValueError('Audited source changed or lacks integrity verification')
        if report['source_key'] != item.source_key or Path(report['source']).resolve() != Path(item.source).resolve():
            raise ValueError('Audit source references disagree')
        if 'mfcc' not in report['modalities']:
            raise ValueError('Audited source lacks usable primary features; use a new raw batch to report failures')
        reports[item.source_key] = report
        producers[str(Path(report['artifacts']['mfcc_npz']).resolve())] = report
        bindings[str(report_path.resolve())] = sha256(report_path)
    ids = {s['participant_id'] for s in sources}
    features = plan['source_provenance']
    if ids != set(features):
        raise ValueError('Frozen feature bindings must cover every participant exactly')
    canonical = {}
    for pid, feature in features.items():
        path = Path(feature['cache'])
        producer = producers[str(path.resolve())]
        if producer['participant_id'] != pid or producer['modalities']['mfcc']['fingerprint'] != feature['source_fingerprint']:
            raise ValueError('Frozen cache producer/fingerprint mismatch')
        if sha256(path) != feature['npz_sha256']:
            raise ValueError('Frozen cache checksum changed')
        sidecar = path.with_suffix('.json')
        metadata = json.loads(sidecar.read_text())
        if metadata['fingerprint'] != feature['source_fingerprint']:
            raise ValueError('Frozen cache sidecar fingerprint changed')
        canonical[pid] = {'cache': str(path), 'npz_sha256': feature['npz_sha256'],
                          'sidecar_sha256': sha256(sidecar), 'summary_sha256': feature['summary_sha256'],
                          'primary_feature_sha256': primary_hashes(producer, pid)}
    index = {}
    for source in sources:
        report = reports[source['source_key']]
        pid = source['participant_id']
        if Path(report['source']).resolve() != source['path'].resolve() or primary_hashes(report, pid) != canonical[pid]['primary_feature_sha256']:
            raise ValueError('A physical source conflicts with its frozen primary feature cache')
        index[source['source_key']] = {**canonical[pid], 'source_sha256': report['source_sha256']}
    for path in [root/'summary.json', root/'run.json', root/'sources.csv', Path(frozen_plan)]:
        bindings[str(path.resolve())] = sha256(path)
    return index, bindings


def sequence_hash(sequence):
    hasher = hashlib.sha256()
    for name in sorted(sequence):
        value = sequence[name]
        hasher.update(digest({'array': name, 'shape': value.shape, 'dtype': str(value.dtype)}).encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def process_source(source, expected_stamp, reuse, contract):
    if source['participant_id'] is None:
        raise ValueError('Archive filename does not identify a numeric participant')
    if stamp_or_error(source['path']) != expected_stamp or isinstance(expected_stamp, dict):
        raise ValueError('Source is unavailable, unsafe, or changed before processing')
    if reuse is not None:
        sequence, provenance = read_cache(reuse['cache'], contract)
        if provenance['files'][reuse['cache']] != reuse['npz_sha256'] or provenance['record_id'] != source['participant_id']:
            raise ValueError('Frozen cache content/participant changed')
        if sha256(Path(reuse['cache']).with_suffix('.json')) != reuse['sidecar_sha256']:
            raise ValueError('Frozen cache sidecar changed')
        raw_hashes, content_hash = reuse['primary_feature_sha256'], reuse['source_sha256']
        mode = 'verified_frozen_cache'
    else:
        sequence, provenance = read_features(contract, source=source['path'])
        raw_hashes, content_hash = provenance['selected_feature_sha256'], provenance['source_sha256']
        mode = 'full_source_read'
    values = pool_modalities(sequence)
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError('Invalid participant feature summary')
    summary_hash = hashlib.sha256(values.tobytes()).hexdigest()
    if reuse is not None and summary_hash != reuse['summary_sha256']:
        raise ValueError('Frozen feature summary checksum changed')
    if stamp_or_error(source['path']) != expected_stamp:
        raise ValueError('Source changed while processing')
    return {'status': 'ready', 'read_mode': mode, 'source_sha256': content_hash,
            'primary_feature_sha256': raw_hashes, 'sequence_sha256': sequence_hash(sequence),
            'summary': values.tolist(), 'summary_sha256': summary_hash,
            'observations': {'bins': len(sequence['timestamps']),
                             'audio_observed_fraction': float(sequence['audio_mask'].mean()),
                             'visual_observed_fraction': float(sequence['visual_mask'].mean())}}


def participant_prediction(pid, copies, contract, parameters, split):
    if any(row['status'] != 'ready' for row in copies):
        return {'participant_id': pid, 'status': 'source_failure', 'source_count': len(copies),
                'split': split, 'reason': 'At least one physical source could not be verified'}
    signatures = {(digest(row['primary_feature_sha256']), row['sequence_sha256']) for row in copies}
    if len(signatures) != 1:
        return {'participant_id': pid, 'status': 'conflicting_copies', 'source_count': len(copies),
                'split': split, 'reason': 'Same-ID sources have different primary feature content'}
    chosen = next((row for row in copies if Path(row['source']).name == f'{pid}_P.tar.gz'), copies[0])
    values = np.asarray(chosen['summary'], dtype=np.float64)
    if hashlib.sha256(values.tobytes()).hexdigest() != chosen['summary_sha256']:
        raise ValueError('Saved source summary checksum mismatch')
    probs = probabilities(parameters, values[None, :])[0]
    label = int(probs.argmax())
    return {'participant_id': pid, 'status': 'predicted', 'source_count': len(copies), 'split': split,
            'selected_source': chosen['source'], 'all_source_keys': [row['source_key'] for row in copies],
            'model': contract['model'], 'model_weights_sha256': contract['weights_sha256'],
            'predicted_class': label, 'predicted_questionnaire_category': contract['class_labels'][label],
            'probabilities': probs.tolist(), 'class_order': list(range(5)),
            'sequence_sha256': chosen['sequence_sha256'], 'summary_sha256': chosen['summary_sha256'],
            'observations': chosen['observations'], 'limitations': contract['limitations']}


def save_tables(out, source_records, participants, unknown, reused_sources, reused_participants):
    source_rows = [{key: row.get(key, '') for key in ['source_key', 'source', 'participant_id', 'kind', 'status', 'read_mode', 'error', 'source_sha256']} for row in source_records]
    pd.DataFrame(source_rows).to_csv(out/'sources.csv', index=False)
    pd.DataFrame([row for row in source_rows if row['status'] != 'ready'], columns=source_rows[0].keys()).to_csv(out/'source_errors.csv', index=False)
    columns = ['participant_id', 'status', 'split', 'source_count', 'reason', 'selected_source']
    pd.DataFrame([{key: row.get(key, '') for key in columns} for row in participants], columns=columns).to_csv(out/'participants.csv', index=False)
    predicted = [row for row in participants if row['status'] == 'predicted']
    probability_columns = [f'p{i}' for i in range(5)]
    columns = ['participant_id', 'split', 'source_count', 'predicted_class', 'predicted_questionnaire_category', *probability_columns]
    pd.DataFrame([{**{key: row[key] for key in columns if key not in probability_columns},
                   **{f'p{i}': value for i, value in enumerate(row['probabilities'])}} for row in predicted], columns=columns).to_csv(out/'predictions.csv', index=False)
    hash_groups = {}
    for row in predicted:
        hash_groups.setdefault(row['sequence_sha256'], []).append(row['participant_id'])
    cross_id_duplicates = [ids for ids in hash_groups.values() if len(ids) > 1]
    complete = not unknown and all(row['status'] == 'ready' for row in source_records) and len(predicted) == len(participants)
    result = {'complete': complete, 'sources_accounted_for': len(source_records),
              'archives_accounted_for': sum(row['kind'] == 'archive' for row in source_records),
              'folders_accounted_for': sum(row['kind'] == 'directory' for row in source_records),
              'source_failures': sum(row['status'] != 'ready' for row in source_records),
              'unrecognized_items': unknown, 'unique_participants': len(participants),
              'predicted_participants': len(predicted), 'blocked_participants': len(participants)-len(predicted),
              'same_id_duplicate_groups': sum(row['source_count'] > 1 for row in participants),
              'cross_id_identical_feature_groups': cross_id_duplicates,
              'cache_reused_sources': sum(row.get('read_mode') == 'verified_frozen_cache' for row in source_records),
              'raw_read_sources': sum(row.get('read_mode') == 'full_source_read' for row in source_records),
              'source_records_reused_this_invocation': reused_sources,
              'participant_records_reused_this_invocation': reused_participants,
              'prediction_counts_by_declared_split': pd.Series([row['split'] for row in predicted], dtype=str).value_counts().to_dict(),
              'models_fitted': 0, 'questionnaire_labels_used': False, 'gender_used': False,
              'performance_metrics_computed': False,
              'interpretation': 'Operational predictions from one frozen model, including any previously seen train/validation/test records. No new generalization estimate.'}
    atomic_json(out/'completion.json', result)
    (out/'REPORT.md').write_text(
        '# Dataset batch inference\n\n'
        f"Accounted for {result['archives_accounted_for']} archives and {result['folders_accounted_for']} extracted folders, representing {len(participants)} participants. "
        f"Produced {len(predicted)} participant predictions; {result['blocked_participants']} participants are blocked and {len(unknown)} items are unrecognized. "
        f"Complete: {complete}.\n\n"
        f"Same-ID duplicate groups: {result['same_id_duplicate_groups']}. Identical copies are represented once. "
        'Conflicting copies or any failed physical copy block that participant. Every source has a row in sources.csv; failures are explicit. '
        'Cross-ID identical feature groups, when present, are flagged in completion.json and remain separate records; no performance metrics weight these records.\n\n'
        f"Verified cache reuse covered {result['cache_reused_sources']} sources; {result['raw_read_sources']} sources were fully read in raw mode. "
        'Cache reuse validates the original complete inventory, source stamps, frozen NPZ hashes, selected feature hashes, preprocessing contract and summary hashes. '
        'It does not claim that archives were decompressed again.\n\n'
        f"Prediction counts by declared split: {result['prediction_counts_by_declared_split']}. "
        'These are operational outputs, not an additional evaluation. Train/validation/test records may have been used or evaluated previously. '
        'No questionnaire labels, gender inputs, model fitting, threshold tuning or new performance metrics are used. '
        'The previously reported weak model performance and unverified timing/speech limitations remain.\n\n'
        'Each source and participant record is saved with context/content checksums. Resume requires unchanged model, code, inventory and inputs. '
        'Existing output requires explicit --resume; changed input requires a new output directory. '
        'All results stay local. See predictions.csv, participants.csv, source_errors.csv, completion.json and the source_records/participant_records/ folders.\n')
    return result


def run_batch(dataset, bundle, output, audit=None, frozen_plan=None, splits=None, resume=False):
    out = Path(output)
    if out.resolve().is_relative_to(Path(dataset).resolve()):
        raise ValueError('Batch output must be outside the input dataset folder')
    if out.exists() and not resume:
        raise FileExistsError(f'Preserving existing batch; use --resume for unchanged inputs: {out}')
    if resume and not (out/'run.json').exists():
        raise ValueError('Resume requires an existing batch run.json')
    if (audit is None) != (frozen_plan is None):
        raise ValueError('Provide both --audit and --frozen-plan for verified cache reuse')
    contract, parameters = load_bundle(bundle)
    sources, rows, unknown = inventory(dataset)
    reuse, bindings = audited_cache_index(dataset, sources, audit, frozen_plan, contract) if audit else ({}, {})
    split_lookup = {}
    if splits:
        split_frame = pd.read_csv(splits, usecols=['participant_id', 'split'], dtype=str, keep_default_na=False)
        if split_frame.participant_id.duplicated().any() or set(split_frame.participant_id) != {s['participant_id'] for s in sources if s['participant_id'] is not None} or not set(split_frame.split) <= {'train', 'validation', 'test'}:
            raise ValueError('Optional split manifest must cover each discovered participant exactly once')
        split_lookup = dict(zip(split_frame.participant_id, split_frame.split))
        bindings[str(Path(splits).resolve())] = sha256(splits)
    implementation = {name: sha256(Path(__file__).with_name(name)) for name in
                      ['batch_inference.py', 'reference_inference.py', 'prepare_features.py', 'exploratory_baselines.py', 'full_dataset.py', 'common.py']}
    context = {'format': FORMAT, 'dataset': str(Path(dataset).resolve()), 'model_contract': contract,
               'implementation_sha256': implementation, 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
               'sources': rows, 'unrecognized_items': unknown,
               'cache_index': reuse, 'input_bindings': bindings, 'declared_splits': split_lookup}
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        if (out/'run.json').exists():
            if not resume:
                raise FileExistsError('Preserving existing batch; use --resume explicitly')
            saved = json.loads((out/'run.json').read_text())
            if saved['context_hash'] != context_hash or saved['context'] != context:
                raise ValueError('Resume inputs/model/code changed; preserve this run and choose a new output directory')
        else:
            atomic_json(out/'run.json', {'context': context, 'context_hash': context_hash})
        pd.DataFrame(rows).drop(columns='source_stamp').to_csv(out/'source_inventory.csv', index=False)
        records, reused_sources = [], 0
        for index, (source, row) in enumerate(zip(sources, rows), 1):
            path = out/'source_records'/f"{source['source_key']}.json"
            previous = load_record(path, context_hash) if path.exists() else None
            if previous is not None and any(previous.get(key) != value for key, value in row.items()):
                raise ValueError('Resume record identity does not match the expected physical source')
            if previous is not None and previous['status'] == 'ready':
                record = previous
                reused_sources += 1
            else:
                try:
                    record = {**row, **process_source(source, row['source_stamp'], reuse.get(source['source_key']), contract)}
                except Exception as exc:
                    record = {**row, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                save_record(path, record, context_hash)
            records.append(record)
            if index % 25 == 0 or index == len(sources):
                print(f"Sources accounted for: {index}/{len(sources)}; failures: {sum(r['status'] != 'ready' for r in records)}", flush=True)
        groups = {}
        for record in records:
            if record['participant_id'] is not None:
                groups.setdefault(record['participant_id'], []).append(record)
        participants, reused_participants = [], 0
        for pid, copies in sorted(groups.items()):
            path = out/'participant_records'/f'{pid}.json'
            existing = load_record(path, context_hash) if path.exists() else None
            if existing is not None and (existing.get('participant_id') != pid or existing.get('source_count') != len(copies)
                                         or existing.get('split') != split_lookup.get(pid, 'unspecified')):
                raise ValueError('Resume record identity does not match the expected participant')
            if existing is not None and existing['status'] == 'predicted':
                if existing.get('all_source_keys') != [row['source_key'] for row in copies] or existing.get('model_weights_sha256') != contract['weights_sha256']:
                    raise ValueError('Resume participant sources/model do not match')
                participant = existing
                reused_participants += 1
            else:
                try:
                    participant = participant_prediction(pid, copies, contract, parameters, split_lookup.get(pid, 'unspecified'))
                except Exception as exc:
                    participant = {'participant_id': pid, 'status': 'inference_failure', 'source_count': len(copies),
                                   'split': split_lookup.get(pid, 'unspecified'), 'reason': f'{type(exc).__name__}: {exc}'}
                save_record(path, participant, context_hash)
            participants.append(participant)
        _, final_rows, final_unknown = inventory(dataset)
        if rows != final_rows or unknown != final_unknown or any(sha256(path) != value for path, value in bindings.items()):
            raise ValueError('Inputs changed during processing; results are not a completed batch')
        result = save_tables(out, records, participants, unknown, reused_sources, reused_participants)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--bundle', default=DEFAULT_BUNDLE)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit', help='Complete existing source audit; requires --frozen-plan')
    parser.add_argument('--frozen-plan', help='Existing frozen cache hashes; requires --audit')
    parser.add_argument('--splits', help='Optional ID/split manifest for annotation only; no labels used')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run_batch(args.dataset, args.bundle, args.output, args.audit, args.frozen_plan, args.splits, args.resume)
    print(json.dumps(result, indent=2))
    if not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
