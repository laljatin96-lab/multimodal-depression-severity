"""Audit-bound, order-invariant eGeMAPS summaries for original training people only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import zipfile

import numpy as np
import pandas as pd

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .final_evaluation import sha256
from .full_dataset import MAX_SELECTED_BYTES, safe_member, source_stamp

OUTPUT = Path('results/prosody_training_features')
PROTOCOL = 'audit-bound-recording-level-egemaps-v1'
COLUMNS = ['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3', 'slope0-500_sma3',
    'slope500-1500_sma3', 'spectralFlux_sma3', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3',
    'F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz', 'HNRdBACF_sma3nz',
    'logRelF0-H1-H2_sma3nz', 'logRelF0-H1-A3_sma3nz', 'F1frequency_sma3nz', 'F1bandwidth_sma3nz',
    'F1amplitudeLogRelF0_sma3nz', 'F2frequency_sma3nz', 'F2amplitudeLogRelF0_sma3nz',
    'F3frequency_sma3nz', 'F3amplitudeLogRelF0_sma3nz']
STATISTICS = ['mean', 'std', 'p10', 'median', 'p90']
NAMES = [f'{column}:{stat}' for column in COLUMNS for stat in STATISTICS]+['voiced_frame_fraction']


def summarize(blob):
    frame = pd.read_csv(io.BytesIO(blob), sep=';')
    if frame.columns.tolist() != ['name', 'frameTime']+COLUMNS or len(frame) < 2:
        raise ValueError('Unexpected eGeMAPS schema or insufficient frames')
    values = frame[COLUMNS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite acoustic measurements; no participant may be silently discarded')
    voiced = values[:, 10] > 0
    features = []
    for j in range(len(COLUMNS)):
        a = values[voiced, j] if j >= 10 else values[:, j]
        if len(a):
            q = np.quantile(a, [.1, .5, .9])
            features.extend([float(a.mean()), float(a.std()), *q.tolist()])
        else:
            # Explicit missing summaries; model imputation must use fitting people only.
            features.extend([None]*5)
    features.append(float(voiced.mean()))
    t = pd.to_numeric(frame.frameTime, errors='coerce').to_numpy(float)
    return {'features': features, 'frames': len(frame), 'voiced_frames': int(voiced.sum()),
        'nonincreasing_time_steps': int((np.diff(t) <= 0).sum()), 'nonfinite_times': int((~np.isfinite(t)).sum()),
        'negative_times': int((t < 0).sum()), 'timestamps_used_as_features': False,
        'alignment_used': False, 'speaker_identity': 'unverified whole supplied recording'}


def selected_member(report):
    name = f"{report['participant_id']}_OpenSMILE2.3.0_egemaps.csv"
    matches = [m for m in report['members'] if safe_member(m['member']) == name]
    if len(matches) != 1 or not 0 < matches[0]['bytes'] <= MAX_SELECTED_BYTES:
        raise ValueError('Expected one bounded audited eGeMAPS member')
    member = matches[0]
    if member['sha256'] != report['feature_sha256'][name]:
        raise ValueError('Audited eGeMAPS member hashes disagree')
    return member


def read_egemaps(report):
    """Read one audited member, without extracting or executing archive files."""
    path = Path(report['source'])
    member = selected_member(report)
    if not report['integrity_checked'] or source_stamp(path) != report['source_stamp']:
        raise ValueError('Source changed or lacks a complete integrity audit')
    name, expected_size = member['member'], member['bytes']
    if path.is_dir():
        target = path / name
        if any(p.is_symlink() for p in [target, *target.parents]) or not target.resolve().is_relative_to(path.resolve()):
            raise ValueError('Unsafe extracted member')
        blob = target.read_bytes()
    elif path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            matches = [m for m in archive.infolist() if m.filename == name]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate selected ZIP member')
            info = matches[0]
            if info.file_size != expected_size or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe or changed ZIP member')
            blob = archive.read(info)  # Also verifies this member's CRC.
    else:
        blob = None
        with tarfile.open(path, mode='r|*', bufsize=1024**2) as archive:
            for info in archive:
                safe_member(info.name)
                if info.name != name:
                    continue
                if not info.isfile() or info.size != expected_size:
                    raise ValueError('Unsafe or changed tar member')
                blob = archive.extractfile(info).read(expected_size+1)
                break  # Deliberately not a new archive-wide integrity scan.
        if blob is None:
            raise ValueError('Missing selected tar member')
    if len(blob) != expected_size or hashlib.sha256(blob).hexdigest() != member['sha256']:
        raise ValueError('Raw eGeMAPS size/checksum differs from the complete audit')
    if source_stamp(path) != report['source_stamp']:
        raise ValueError('Source changed during targeted read')
    return blob


def context():
    metadata = Path('data/metadata/participants.csv')
    frame = pd.read_csv(metadata, usecols=['participant_id', 'split'], dtype=str)
    ids = frame.loc[frame.split == 'train', 'participant_id'].tolist()
    if len(ids) != 163 or len(set(ids)) != len(ids):
        raise ValueError('Original training coverage changed')
    reports = []; bindings = {}
    for path in sorted(Path('results/full_dataset/sources').glob('*.json')):
        report = json.loads(path.read_text())
        if report['participant_id'] in ids:
            selected_member(report)
            if not report['integrity_checked'] or source_stamp(Path(report['source'])) != report['source_stamp']:
                raise ValueError('Original source changed')
            reports.append(report); bindings[str(path)] = sha256(path)
    if set(r['participant_id'] for r in reports) != set(ids):
        raise ValueError('Missing original participant source')
    ctx = {'protocol': PROTOCOL, 'training_ids': ids, 'source_records': bindings,
        'source_stamps': {r['source_key']: r['source_stamp'] for r in reports}, 'feature_names': NAMES,
        'metadata_sha256': sha256(metadata), 'code_sha256': sha256(__file__),
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
        'definition': 'Recording-level mean/std/p10/median/p90. First10 columns use all frames; final13 use F0>0 frames. Include voiced fraction. No temporal alignment or sequence reconstruction.',
        'targets_used': False, 'validation_test_extracted': False}
    return ctx, reports


def process(report):
    blob = read_egemaps(report)
    return {**{k: report[k] for k in ['participant_id', 'source_key', 'source', 'source_stamp']},
        'member_sha256': hashlib.sha256(blob).hexdigest(), **summarize(blob)}


def validate(record, report):
    if any(record[k] != report[k] for k in ['participant_id', 'source_key', 'source', 'source_stamp']):
        raise ValueError('Source ownership differs')
    if record['member_sha256'] != selected_member(report)['sha256']:
        raise ValueError('Source member hash differs')
    a = np.asarray(record['features'], float)
    if a.shape != (116,) or np.isinf(a).any() or record['frames'] < 2 or not 0 <= record['voiced_frames'] <= record['frames']:
        raise ValueError('Invalid participant acoustic summary')
    if not np.isfinite(a[:50]).all() or not np.isfinite(a[-1]) or a[-1] != record['voiced_frames']/record['frames']:
        raise ValueError('Invalid coverage features')
    if bool(np.isnan(a[50:-1]).any()) != (record['voiced_frames'] == 0):
        raise ValueError('Missing voiced-summary ownership differs')


def combine(records, ids):
    rows = []
    for pid in ids:
        copies = sorted((r for r in records if r['participant_id'] == pid), key=lambda r: r['source_key'])
        if not copies or any(r['member_sha256'] != copies[0]['member_sha256'] or r['features'] != copies[0]['features'] for r in copies):
            raise ValueError('Missing or conflicting participant copies')
        rows.append({'participant_id': pid, 'features': copies[0]['features'],
            'source_keys': [r['source_key'] for r in copies], 'frames': copies[0]['frames'],
            'voiced_frames': copies[0]['voiced_frames'], 'nonincreasing_time_steps': copies[0]['nonincreasing_time_steps']})
    return rows


def prepare(resume=False):
    if OUTPUT.exists() and not resume:
        raise FileExistsError('Preserving existing acoustic cache')
    ctx, reports = context(); ch = digest(ctx); envelope = {'context': ctx, 'context_hash': ch}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with output_lock(OUTPUT):
        path = OUTPUT/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Feature context changed')
        else:
            if resume:
                raise ValueError('Missing protocol')
            atomic_json(path, envelope)
        records = {}; pending = []
        for r in reports:
            p = OUTPUT/'sources'/f"{r['source_key']}.json"
            if p.exists():
                value = load_record(p, ch); validate(value, r); records[r['source_key']] = value
            else:
                pending.append(r)
        if not pending and (OUTPUT/'completion.json').exists():
            return json.loads((OUTPUT/'completion.json').read_text())
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(process, r): r for r in pending}
            for job in as_completed(jobs):
                r = jobs[job]; value = job.result(); validate(value, r)
                save_record(OUTPUT/'sources'/f"{r['source_key']}.json", value, ch)
                records[r['source_key']] = value
                print(f"Acoustic sources: {len(records)}/{len(reports)}", flush=True)
        if context()[0] != ctx:
            raise ValueError('Sources changed during extraction')
        rows = combine(list(records.values()), ctx['training_ids'])
        atomic_json(OUTPUT/'participants.json', rows)
        receipt = {'complete': True, 'context_hash': ch, 'participants': len(rows), 'sources': len(reports),
            'features': len(NAMES), 'participants_with_nonincreasing_times': sum(r['nonincreasing_time_steps'] > 0 for r in rows),
            'participants_without_voiced_frames': sum(r['voiced_frames'] == 0 for r in rows),
            'participants_sha256': sha256(OUTPUT/'participants.json'),
            'records': {k: sha256(OUTPUT/'sources'/f'{k}.json') for k in records},
            'validation_test_extracted': False, 'labels_used': False, 'full_archive_reaudit': False}
        atomic_json(OUTPUT/'completion.json', receipt)
    return receipt


def load_verified():
    ctx, reports = context(); ch = digest(ctx)
    if json.loads((OUTPUT/'protocol.json').read_text()) != {'context': ctx, 'context_hash': ch}:
        raise ValueError('Feature context changed')
    done = json.loads((OUTPUT/'completion.json').read_text())
    if not done['complete'] or done['context_hash'] != ch or done['participants_sha256'] != sha256(OUTPUT/'participants.json'):
        raise ValueError('Acoustic cache incomplete or changed')
    records = []
    for r in reports:
        p = OUTPUT/'sources'/f"{r['source_key']}.json"
        if sha256(p) != done['records'][r['source_key']]:
            raise ValueError('Acoustic source record changed')
        value = load_record(p, ch); validate(value, r); records.append(value)
    rows = json.loads((OUTPUT/'participants.json').read_text())
    if combine(records, ctx['training_ids']) != rows:
        raise ValueError('Acoustic participant alignment changed')
    return rows, {str(OUTPUT/n): sha256(OUTPUT/n) for n in ['protocol.json', 'completion.json', 'participants.json']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.resume), indent=2))

