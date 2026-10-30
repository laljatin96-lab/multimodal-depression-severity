"""Local-only, label-free transcript cache with complete physical-source coverage."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import tarfile
import unicodedata
import zipfile

import numpy as np
import pandas as pd

from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest, write_json
from .final_evaluation import sha256
from .full_dataset import HashingReader, safe_member, source_stamp
from .raw_frame_features import input_state

PROTOCOL = 'local-interview-text-v1'
SCORE_PATTERN = re.compile(r'\b(?:phq(?:[\s_-]*(?:8|9))?|patient health questionnaire|questionnaire score|depression score|severity label|severity class)\b', re.I)
MAX_TEXT_BYTES = 10*1024**2


def clean_transcript(blob):
    text = blob.decode('utf-8-sig', errors='strict')
    header = text.splitlines()[0]
    delimiter = csv.Sniffer().sniff(header, delimiters=',;\t').delimiter
    frame = pd.read_csv(io.StringIO(text), sep=delimiter, keep_default_na=False)
    if set(frame.columns) != {'Start_Time', 'End_Time', 'Text', 'Confidence'}:
        raise ValueError('Unexpected transcript schema; speaker identity must not be invented')
    cleaned, removed = [], 0
    for value in frame.Text:
        row = unicodedata.normalize('NFKC', str(value)).lower()
        if SCORE_PATTERN.search(row):
            removed += 1
            continue
        row = re.sub(r'<[^>]*>', ' ', row)
        row = re.sub(r'\d+', ' ', row)  # No participant IDs, dates or numeric scores as tokens.
        row = re.sub(r'[^a-z\s\x27]', ' ', row)
        cleaned.append(' '.join(row.split()))
    document = ' '.join(t for t in cleaned if t)
    start = pd.to_numeric(frame.Start_Time, errors='coerce').to_numpy(float)
    end = pd.to_numeric(frame.End_Time, errors='coerce').to_numpy(float)
    return {'text': document, 'text_sha256': hashlib.sha256(document.encode()).hexdigest(),
            'rows': len(frame), 'words': len(document.split()), 'explicit_score_rows_removed': removed,
            'invalid_time_rows': int((~np.isfinite(start) | ~np.isfinite(end) | (start < 0) | (end < start)).sum()),
            'speaker_field_available': False, 'timestamps_used_as_features': False}


def transcript_member(report):
    matches = [m for m in report['members'] if safe_member(m['member']) == f"{report['participant_id']}_Transcript.csv"]
    if len(matches) != 1 or not 0 < matches[0]['bytes'] <= MAX_TEXT_BYTES:
        raise ValueError('Expected exactly one bounded transcript per supplied source')
    return matches[0]


def read_transcript(report):
    path, member = Path(report['source']), transcript_member(report)
    if not report['integrity_checked'] or source_stamp(path) != report['source_stamp']:
        raise ValueError('Source changed or not audited')
    name, blob = member['member'], None
    if path.is_dir():
        hashes = {}
        for p in sorted(path.rglob('*')):
            if p.is_symlink():
                raise ValueError('Unsafe symlink in source')
            if p.is_file() and p.name != '.DS_Store':
                with p.open('rb') as handle:
                    hashes[str(p.relative_to(path))] = hashlib.file_digest(handle, 'sha256').hexdigest()
        actual_hash = digest(hashes)
        target = path/name
        if not target.resolve().is_relative_to(path.resolve()):
            raise ValueError('Unsafe transcript path')
        blob = target.read_bytes()
    elif path.suffix.lower() == '.zip':
        with path.open('rb') as handle:
            actual_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        with zipfile.ZipFile(path) as archive:
            matches = [m for m in archive.infolist() if m.filename == name]
            if len(matches) != 1 or matches[0].file_size != member['bytes'] or stat.S_ISLNK(matches[0].external_attr >> 16):
                raise ValueError('Transcript ZIP member changed')
            blob = archive.read(matches[0])
    else:
        with path.open('rb') as raw:
            reader = HashingReader(raw)
            compressed = path.name.endswith(('.gz', '.tgz'))
            stream = gzip.GzipFile(fileobj=reader, mode='rb') if compressed else reader
            actual_members = []
            try:
                with tarfile.open(fileobj=stream, mode='r|') as archive:
                    for info in archive:
                        safe_member(info.name)
                        if info.isdir():
                            continue
                        if not info.isfile():
                            raise ValueError('Nonregular archive member')
                        actual_members.append((info.name, info.size))
                        if info.name == name:
                            if blob is not None or info.size != member['bytes']:
                                raise ValueError('Duplicate or changed transcript member')
                            blob = archive.extractfile(info).read(MAX_TEXT_BYTES+1)
                while stream.read(1024**2):
                    pass
            finally:
                if compressed:
                    stream.close()
            while reader.read(1024**2):
                pass
            if reader.bytes_read != path.stat().st_size or actual_members != [(m['member'], m['bytes']) for m in report['members']]:
                raise ValueError('Archive coverage/inventory differs from complete audit')
            actual_hash = reader.hasher.hexdigest()
    if actual_hash != report['source_sha256'] or source_stamp(path) != report['source_stamp']:
        raise ValueError('Full source hash/stamp differs from original audit')
    if blob is None or len(blob) != member['bytes']:
        raise ValueError('Missing or truncated transcript')
    return blob


def build_context(plan_path):
    source, coverage, _, _ = input_state(plan_path)
    audit = Path(source['feature_config'].get('full_dataset_report', 'results/full_dataset'))
    reports = [json.loads((audit/'sources'/f"{r['source_key']}.json").read_text()) for r in source['sources']]
    for report in reports:
        transcript_member(report)
    return {'protocol': PROTOCOL, 'source_context': source, 'coverage': coverage.to_dict('records'),
            'implementation_sha256': sha256(__file__), 'explicit_score_pattern': SCORE_PATTERN.pattern,
            'numeric_tokens_removed': True, 'labels_or_gender_used': False, 'uploads': False}, reports


def validate_record(record, report):
    if any(record[k] != report[k] for k in ['source_key', 'source', 'participant_id', 'source_stamp', 'source_sha256']):
        raise ValueError('Transcript record source identity changed')
    if record['text_sha256'] != hashlib.sha256(record['text'].encode()).hexdigest() or not record['full_source_hash_verified']:
        raise ValueError('Transcript record checksum/verification changed')


def process(report):
    blob = read_transcript(report)
    return {**{k: report[k] for k in ['source_key', 'source', 'participant_id', 'source_stamp', 'source_sha256']},
            'raw_transcript_sha256': hashlib.sha256(blob).hexdigest(), 'full_source_hash_verified': True,
            **clean_transcript(blob)}


def prepare(plan_path, output, resume=False, workers=2):
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Preserving existing corpus; use --resume')
    if resume and not (out/'protocol.json').exists():
        raise ValueError('Resume requires an existing corpus protocol')
    if not 1 <= workers <= 2:
        raise ValueError('Use one or two workers')
    context, reports = build_context(plan_path)
    dataset = context.get('source_context', {}).get('feature_config', {}).get('dataset', 'dataset')
    if any(out.resolve().is_relative_to(Path(p).resolve()) for p in [dataset, 'data', 'checkpoints']):
        raise ValueError('Corpus output must be outside original data/model directories')
    context_hash = digest(context)
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        envelope = {'context': context, 'context_hash': context_hash}
        if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != envelope:
            raise ValueError('Corpus source/code/protocol changed')
        atomic_json(out/'protocol.json', envelope)
        records, pending = {}, []
        for report in reports:
            path = out/'sources'/f"{report['source_key']}.json"
            if path.exists():
                record = load_record(path, context_hash)
                validate_record(record, report)
                records[report['source_key']] = record
            else:
                pending.append(report)
        reused = len(records)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = {executor.submit(process, r): r for r in pending}
            for future in as_completed(tasks):
                report = tasks[future]
                record = future.result()
                validate_record(record, report)
                save_record(out/'sources'/f"{report['source_key']}.json", record, context_hash)
                records[report['source_key']] = record
                if len(records) % 20 == 0 or len(records) == len(reports):
                    print(f'Transcripts: {len(records)}/{len(reports)} physical sources', flush=True)
        participants = []
        for item in context['coverage']:
            copies = [r for r in records.values() if r['participant_id'] == item['participant_id']]
            if not copies or len({r['raw_transcript_sha256'] for r in copies}) != 1:
                raise ValueError('Missing/conflicting transcript copies for participant')
            chosen = sorted(copies, key=lambda r: r['source_key'])[0]
            participants.append({**item, 'text': chosen['text'], 'text_sha256': chosen['text_sha256'], 'copies': len(copies),
                                 'words': chosen['words'], 'explicit_score_rows_removed': chosen['explicit_score_rows_removed']})
        nonempty = [p['text_sha256'] for p in participants if p['text']]
        if len(set(nonempty)) != len(nonempty):
            raise ValueError('Identical nonempty transcript content across different participant IDs')
        current, _ = build_context(plan_path)
        if current != context:
            raise ValueError('Sources changed during corpus preparation')
        atomic_json(out/'participants.json', participants)
        pd.DataFrame([{k:v for k,v in r.items() if k not in ['text']} for r in participants]).to_csv(out/'coverage.csv', index=False)
        result = {'complete': True, 'context_hash': context_hash, 'participants': len(participants), 'physical_sources': len(reports),
                  'archives': sum(r['kind'] == 'archive' for r in reports), 'folders': sum(r['kind'] == 'directory' for r in reports),
                  'new_sources': len(pending), 'reused_sources': reused, 'empty_documents': sum(not r['text'] for r in participants),
                  'explicit_score_rows_removed': sum(r['explicit_score_rows_removed'] for r in participants),
                  'artifacts': {n: sha256(out/n) for n in ['participants.json', 'coverage.csv']},
                  'records': {key: sha256(out/'sources'/f'{key}.json') for key in records}}
        atomic_json(out/'completion.json', result)
    return result


def load_verified(plan_path, output):
    out = Path(output)
    context, reports = build_context(plan_path)
    envelope = json.loads((out/'protocol.json').read_text())
    receipt = json.loads((out/'completion.json').read_text())
    if envelope != {'context': context, 'context_hash': digest(context)} or not receipt['complete'] or receipt['context_hash'] != envelope['context_hash']:
        raise ValueError('Corpus incomplete or source bindings changed')
    if set(receipt['records']) != {r['source_key'] for r in reports}:
        raise ValueError('Corpus source coverage changed')
    for report in reports:
        path = out/'sources'/f"{report['source_key']}.json"
        if sha256(path) != receipt['records'][report['source_key']]:
            raise ValueError('Transcript source record changed')
        validate_record(load_record(path, envelope['context_hash']), report)
    for name, expected in receipt['artifacts'].items():
        if sha256(out/name) != expected:
            raise ValueError('Corpus participant artifact changed')
    participants = json.loads((out/'participants.json').read_text())
    if [{k:p[k] for k in ['participant_id', 'split']} for p in participants] != context['coverage']:
        raise ValueError('Corpus participant order/splits changed')
    return participants, {'protocol': sha256(out/'protocol.json'), 'completion': sha256(out/'completion.json'), 'participants': sha256(out/'participants.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-plan', default='results/final_evaluation_plan.json')
    p.add_argument('--output', default='results/transcript_corpus')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    result = prepare(args.frozen_plan, args.output, args.resume)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','artifacts']}, indent=2))

if __name__ == '__main__':
    main()
