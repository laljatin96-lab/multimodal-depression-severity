"""Versioned, label-free raw-frame summaries bound to the complete source audit.

Only selected MFCC members are read. Archive-wide integrity comes from the
previous full audit plus unchanged source stamps, not this targeted read.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import stat
import tarfile
import time
import zipfile

import numpy as np
import pandas as pd

from .batch_inference import atomic_json, audited_cache_index, load_record, output_lock, save_record
from .common import digest
from .exploratory_baselines import load_audited_sequences, pool_modalities
from .final_evaluation import sha256, verify_plan
from .full_dataset import MAX_SELECTED_BYTES, discover_sources, safe_member, source_stamp
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS, bin_frames, temporal_frame

FORMAT = 'raw-frame-mfcc-summary-v1'
FAMILIES = ['MFCC', 'delta', 'delta_delta']
FEATURE_NAMES = ([f'raw_audio_mean:{c}' for c in AUDIO_COLUMNS]
                 + [f'raw_audio_std:{c}' for c in AUDIO_COLUMNS]
                 + [f'cached_visual_mean:{c}' for c in VISUAL_COLUMNS]
                 + [f'cached_visual_std:{c}' for c in VISUAL_COLUMNS])


def selected_member(report):
    name = f"{report['participant_id']}_OpenSMILE2.3.0_mfcc.csv"
    matches = [m for m in report['members'] if safe_member(m['member']) == name]
    if len(matches) != 1 or not 0 < matches[0]['bytes'] <= MAX_SELECTED_BYTES:
        raise ValueError('Expected one bounded audited MFCC member')
    member = matches[0]
    if member['sha256'] != report['feature_sha256'][name]:
        raise ValueError('Audited MFCC member hashes disagree')
    return member


def read_mfcc(report):
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
        raise ValueError('Raw MFCC size/checksum differs from the complete audit')
    if source_stamp(path) != report['source_stamp']:
        raise ValueError('Source changed during targeted read')
    return blob


def summarize_raw(blob, sequence, config):
    frame, times = temporal_frame(blob, 'frameTime', ';')
    times += config['audio_time_offset']
    values = frame[AUDIO_COLUMNS].to_numpy(np.float64)
    edges = np.r_[sequence['timestamps'], sequence['timestamps'][-1]+config['bin_seconds']]
    cached, observed = bin_frames(times, values, np.ones(len(times), bool), edges)
    if not np.array_equal(cached, sequence['audio']) or not np.array_equal(observed, sequence['audio_mask']):
        raise ValueError('Raw MFCC does not exactly reproduce the frozen cache')
    indices = np.searchsorted(edges, times, side='right')-1
    keep = (indices >= 0) & (indices < len(cached))
    x, indices = values[keep], indices[keep]
    if len(x) < 2:
        raise ValueError('Fewer than two frames in the exact cached interval')
    counts = np.bincount(indices, minlength=len(cached))
    means = np.zeros_like(cached, dtype=np.float64)
    np.add.at(means, indices, x)
    means[observed] /= counts[observed, None]
    mean, variance = x.mean(0), x.var(0)
    between = ((means[observed]-mean)**2*counts[observed, None]).sum(0)/len(x)
    within = ((x-means[indices])**2).mean(0)
    np.testing.assert_allclose(variance, between+within, rtol=1e-10, atol=1e-10)
    ratio = np.divide(between, variance, out=np.zeros(39), where=variance > 0)
    summary = np.r_[mean, np.sqrt(variance), pool_modalities(sequence)[78:]]
    if summary.shape != (128,) or not np.isfinite(summary).all():
        raise ValueError('Invalid raw-frame summary')
    return {'summary': summary.tolist(),
            'summary_sha256': hashlib.sha256(summary.tobytes()).hexdigest(),
            'raw_frames_total': len(values), 'raw_frames_in_cached_interval': len(x),
            'interval_start': float(edges[0]), 'interval_end_exclusive': float(edges[-1]),
            'observed_audio_bins': int(observed.sum()), 'cache_reproduced_exactly': True,
            'variance_decomposition_verified': True,
            'variance': [{'feature': name, 'family': FAMILIES[i//13],
                          'raw_variance': float(variance[i]), 'within_bin_variance': float(within[i]),
                          'between_bin_variance': float(between[i]),
                          'retained_variance_fraction': float(np.clip(ratio[i], 0, 1)) if variance[i] > 0 else None}
                         for i, name in enumerate(AUDIO_COLUMNS)]}


def validate_record(record, pid, report):
    if (record.get('participant_id') != pid or record.get('source_key') != report['source_key']
            or record.get('raw_mfcc_sha256') != selected_member(report)['sha256']):
        raise ValueError('Saved raw-frame record identity/source mismatch')
    values = np.asarray(record['summary'], dtype=np.float64)
    if (values.shape != (128,) or not np.isfinite(values).all()
            or hashlib.sha256(values.tobytes()).hexdigest() != record['summary_sha256']):
        raise ValueError('Saved raw-frame summary checksum/shape mismatch')
    if not record['cache_reproduced_exactly'] or not record['variance_decomposition_verified']:
        raise ValueError('Missing raw-frame verification')


def input_state(plan_path):
    plan = verify_plan(plan_path)
    config = plan['feature_config']
    coverage = pd.read_csv(config['splits'], usecols=['participant_id', 'split'], dtype=str, keep_default_na=False)
    if coverage.participant_id.duplicated().any() or set(coverage.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected unique participants and complete split annotations')
    sources, unknown = discover_sources(config['dataset'])
    if unknown:
        raise ValueError('Unrecognized dataset sources')
    audit = Path(config.get('full_dataset_report', 'results/full_dataset'))
    _, bindings = audited_cache_index(config['dataset'], sources, audit, plan_path, {'frozen_plan_sha256': digest(plan)})
    sequences, provenance = load_audited_sequences(config, coverage)
    if provenance != plan['source_provenance']:
        raise ValueError('Feature cache provenance differs from frozen study')
    reports = [json.loads((audit/'sources'/f"{s['source_key']}.json").read_text()) for s in sources]
    selected = {}
    for pid in coverage.participant_id:
        copies = [r for r in reports if r['participant_id'] == pid]
        def cost(r):
            member = selected_member(r)
            index = r['members'].index(member)
            return (r['kind'] != 'directory', sum(m['bytes'] for m in r['members'][:index]), r['source'])
        selected[pid] = min(copies, key=cost)
    context = {'format': FORMAT, 'frozen_plan_sha256': digest(plan), 'feature_config': config,
               'feature_names': FEATURE_NAMES, 'coverage': coverage.to_dict('records'),
               'audit_bindings': bindings, 'source_provenance': provenance,
               'sources': [{k: r[k] for k in ['source', 'source_key', 'participant_id', 'kind', 'source_stamp', 'source_sha256']}
                           for r in reports], 'selected_sources': {pid: r['source_key'] for pid, r in selected.items()},
               'implementation_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                        ['raw_frame_features.py', 'prepare_features.py', 'full_dataset.py',
                                         'exploratory_baselines.py', 'batch_inference.py', 'final_evaluation.py', 'common.py']},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
               'definition': '39 raw-frame means + 39 population stds in the exact frozen crop; unchanged 50 cached visual summaries',
               'read_mode': 'Targeted MFCC read checked against original full audit; all physical source stamps and duplicate hashes checked',
               'limitations': ['Post-final exploratory representation; no independent evaluation claim',
                               'Raw frames receive equal weight; old summaries weighted observed bins equally',
                               'Common-time crop and unverified timing/speaker isolation retained',
                               'Participant summaries preserve variance, not temporal ordering',
                               'No labels or gender used and no model fitting by feature preparation']}
    return context, coverage, sequences, selected


def process_participant(pid, report, sequence, config):
    started = time.monotonic()
    result = summarize_raw(read_mfcc(report), sequence, config)
    return {'participant_id': pid, 'source_key': report['source_key'], 'source': report['source'],
            'raw_mfcc_sha256': selected_member(report)['sha256'], 'seconds': time.monotonic()-started, **result}


def prepare(plan_path, output, resume=False, workers=2):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving raw-frame outputs; use --resume for unchanged inputs')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires an existing protocol')
    if not 1 <= workers <= 2:
        raise ValueError('Use one or two workers for bounded memory/CPU use')
    context, coverage, sequences, selected = input_state(plan_path)
    if out.resolve().is_relative_to(Path(context['feature_config']['dataset']).resolve()):
        raise ValueError('Output must be outside the original dataset')
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        protocol = {'context': context, 'context_hash': context_hash}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != protocol:
                raise ValueError('Raw-frame inputs/code/protocol changed; choose a new output')
        else:
            atomic_json(out/'protocol.json', protocol)
        (out/'participants').mkdir(exist_ok=True)
        records, pending, reused = {}, [], 0
        for pid in coverage.participant_id:
            path = out/'participants'/f'{pid}.json'
            if path.exists():
                record = load_record(path, context_hash)
                validate_record(record, pid, selected[pid])
                records[pid] = record
                reused += 1
            else:
                pending.append(pid)
        failures = []
        print(f"Raw-frame preparation: {len(selected)} participants, {len(context['sources'])} sources; {reused} reusable", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_participant, pid, selected[pid], sequences[pid], context['feature_config']): pid for pid in pending}
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    record = future.result()
                    validate_record(record, pid, selected[pid])
                    save_record(out/'participants'/f'{pid}.json', record, context_hash)
                    records[pid] = record
                except Exception as exc:
                    failures.append({'participant_id': pid, 'error': f'{type(exc).__name__}: {exc}'})
                if len(records) % 10 == 0 or failures or len(records) == len(selected):
                    print(f"Prepared {len(records)}/{len(selected)}; failures {len(failures)}", flush=True)
        current, _, _, _ = input_state(plan_path)
        if current != context:
            raise ValueError('Inputs changed during preparation; completion cannot be confirmed')
        coverage.to_csv(out/'participant_coverage.csv', index=False)
        pd.DataFrame([{**s, 'used_summary_participant_id': s['participant_id'],
                       'raw_mfcc_read_here': s['source_key'] == selected[s['participant_id']]['source_key'],
                       'primary_features_match_selected': True} for s in context['sources']]).to_csv(out/'source_coverage.csv', index=False)
        artifacts = {}
        if not failures:
            ordered = [records[pid] for pid in coverage.participant_id]
            np.savez_compressed(out/'participant_summaries.npz', participant_ids=coverage.participant_id.to_numpy(dtype=str),
                                features=np.asarray([r['summary'] for r in ordered]), feature_names=np.asarray(FEATURE_NAMES))
            pd.DataFrame([{'participant_id': r['participant_id'], **v} for r in ordered for v in r['variance']]).to_csv(out/'variance_retention.csv', index=False)
            pd.DataFrame([{k: v for k, v in r.items() if k not in ['summary', 'variance']} for r in ordered]).to_csv(out/'quality.csv', index=False)
            artifacts = {p.name: sha256(p) for p in [out/'participant_summaries.npz', out/'variance_retention.csv', out/'quality.csv',
                                                    out/'participant_coverage.csv', out/'source_coverage.csv']}
        result = {'complete': not failures, 'format': FORMAT, 'context_hash': context_hash,
                  'participants': len(selected), 'prepared_participants': len(records), 'physical_sources': len(context['sources']),
                  'archives': sum(s['kind'] == 'archive' for s in context['sources']),
                  'folders': sum(s['kind'] == 'directory' for s in context['sources']),
                  'new_raw_reads_this_invocation': len(records)-reused, 'reused_participants_this_invocation': reused,
                  'raw_frames_in_cached_interval': sum(r['raw_frames_in_cached_interval'] for r in records.values()),
                  'failures': failures, 'model_fits': 0, 'predictions_created': False, 'artifacts': artifacts,
                  'record_sha256': {pid: sha256(out/'participants'/f'{pid}.json') for pid in records}}
        atomic_json(out/'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-plan', default='results/final_evaluation_plan.json')
    parser.add_argument('--output', default='results/raw_frame_features')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.frozen_plan, args.output, args.resume, args.workers)
    print(json.dumps({k: v for k, v in result.items() if k not in ['record_sha256', 'artifacts']}, indent=2))
    raise SystemExit(0 if result['complete'] else 2)


if __name__ == '__main__':
    main()
