"""Audit-bound, training-only transcript-confidence filtering; no speaker inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from .batch_inference import atomic_json, load_record
from .common import digest
from .final_evaluation import sha256
from .full_dataset import safe_member, source_stamp
from .transcript_corpus import clean_transcript, transcript_member as audited_member

ROOT = Path('results/confidence_features')
THRESHOLD = .8  # Fixed before any fitting or outer predictions.


def transcript_member(report):
    envelope = json.loads(Path('results/transcript_corpus/protocol.json').read_text())
    context = envelope.get('context', envelope.get('protocol'))
    if context is None or digest(context) != envelope['context_hash']:
        raise ValueError('Original transcript audit context changed')
    row = load_record(Path('results/transcript_corpus/sources')/f"{report['source_key']}.json", envelope['context_hash'])
    if (not row['full_source_hash_verified'] or any(row[k] != report[k] for k in
            ['participant_id','source_key','source','source_stamp','source_sha256'])):
        raise ValueError('Original transcript integrity receipt differs')
    return {**audited_member(report), 'sha256': row['raw_transcript_sha256']}


def filtered_document(blob):
    original = clean_transcript(blob)
    delimiter = csv.Sniffer().sniff(blob.decode('utf-8-sig').splitlines()[0], delimiters=',;\t').delimiter
    frame = pd.read_csv(io.BytesIO(blob), sep=delimiter, keep_default_na=False)
    confidence = pd.to_numeric(frame.Confidence, errors='raise').to_numpy(float)
    if not len(confidence) or not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError('Invalid transcript confidence annotations')
    keep = confidence >= THRESHOLD
    selected = clean_transcript(frame.loc[keep].to_csv(index=False, sep=';').encode())
    fallback = not bool(selected['words'])
    if fallback:
        selected = original  # Never discard an original participant.
    return {'text': selected['text'], 'original_text_sha256': original['text_sha256'],
        'text_sha256': selected['text_sha256'], 'rows': len(frame), 'retained_rows': int(keep.sum()),
        'original_words': original['words'], 'retained_words': selected['words'],
        'confidence_mean': float(confidence.mean()), 'confidence_min': float(confidence.min()),
        'invalid_time_rows': original['invalid_time_rows'], 'empty_filter_fallback': fallback,
        'explicit_score_rows_removed': original['explicit_score_rows_removed'],
        'speaker_field_available': False, 'timestamps_used_as_features': False}


def read_member(report):
    """Reuse full integrity audit; verify the requested member and source stamps."""
    path = Path(report['source']); member = transcript_member(report)
    if not report['integrity_checked'] or source_stamp(path) != report['source_stamp']:
        raise ValueError('Source changed or lacks full integrity audit')
    if path.is_dir():
        target = path/member['member']
        if any(p.is_symlink() for p in [target, *target.parents]) or not target.resolve().is_relative_to(path.resolve()):
            raise ValueError('Unsafe transcript member')
        blob = target.read_bytes()
    else:
        blob = None
        with tarfile.open(path, mode='r|*', bufsize=1024**2) as archive:
            for info in archive:
                safe_member(info.name)
                if info.name != member['member']:
                    continue
                if not info.isfile() or info.size != member['bytes']:
                    raise ValueError('Changed or unsafe transcript member')
                blob = archive.extractfile(info).read(member['bytes']+1)
                break
    if blob is None or len(blob) != member['bytes'] or hashlib.sha256(blob).hexdigest() != member['sha256']:
        raise ValueError('Transcript differs from full integrity audit')
    if source_stamp(path) != report['source_stamp']:
        raise ValueError('Source changed during targeted read')
    return blob


def prepare(resume=False):
    if ROOT.exists() and not resume:
        raise FileExistsError('Preserving existing transcript features')
    metadata = pd.read_csv('data/metadata/participants.csv', dtype={'participant_id': str}, usecols=['participant_id', 'split'])
    ids = metadata.loc[metadata.split == 'train', 'participant_id'].tolist()
    if len(ids) != 163 or len(set(ids)) != 163:
        raise ValueError('Original training population changed')
    corpus = {r['participant_id']: r for r in json.loads(Path('results/transcript_corpus/participants.json').read_text())}
    reports = {pid: [] for pid in ids}; bindings = {}
    for path in sorted(Path('results/full_dataset/sources').glob('*.json')):
        r = json.loads(path.read_text())
        if r['participant_id'] in reports:
            reports[r['participant_id']].append(r); bindings[str(path)] = sha256(path)
    selected = {}
    for pid, copies in reports.items():
        if not copies or len({transcript_member(r)['sha256'] for r in copies}) != 1:
            raise ValueError('Missing or conflicting participant copies')
        for r in copies:
            if not r['integrity_checked'] or source_stamp(Path(r['source'])) != r['source_stamp']:
                raise ValueError('Source changed')
        selected[pid] = sorted(copies, key=lambda r: (not Path(r['source']).is_dir(), r['source']))[0]
    ctx = {'training_ids': ids, 'threshold': THRESHOLD, 'source_reports': bindings,
        'source_sha256': sha256(__file__), 'cleaner_sha256': sha256('src/transcript_corpus.py'),
        'corpus_sha256': sha256('results/transcript_corpus/participants.json'),
        'targets_used': False, 'speaker_attribution': 'unverified; no speech attributed or removed by identity',
        'confidence_interpretation': 'Supplied recognition-confidence field; calibration and accuracy relation unverified',
        'validation_test_processed': False}
    ROOT.mkdir(parents=True, exist_ok=True); protocol = ROOT/'protocol.json'
    envelope = {'context': ctx, 'context_hash': digest(ctx)}
    if protocol.exists():
        if json.loads(protocol.read_text()) != envelope:
            raise ValueError('Feature context changed')
    else:
        atomic_json(protocol, envelope)
    (ROOT/'documents').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = {}
        for pid, r in selected.items():
            path = ROOT/'documents'/f'{pid}.json'
            if not path.exists():
                pending[executor.submit(read_member, r)] = (pid, path)
        total = len(pending)
        for index, future in enumerate(as_completed(pending), 1):
            pid, path = pending[future]; row = filtered_document(future.result())
            if row['original_text_sha256'] != corpus[pid]['text_sha256'] or corpus[pid]['split'] != 'train':
                raise ValueError('Transcript-to-participant alignment differs from original corpus')
            atomic_json(path, {'participant_id': pid, 'source_member_sha256': transcript_member(selected[pid])['sha256'],
                'source_keys': [r['source_key'] for r in reports[pid]], **row})
            print(f'Transcript annotations {index}/{total}', flush=True)
    rows = [json.loads((ROOT/'documents'/f'{pid}.json').read_text()) for pid in ids]
    for pid, row in zip(ids, rows):
        if (row['participant_id'] != pid or row['source_member_sha256'] != transcript_member(selected[pid])['sha256']
                or row['original_text_sha256'] != corpus[pid]['text_sha256']
                or row['text_sha256'] != hashlib.sha256(row['text'].encode()).hexdigest()):
            raise ValueError('Cached document ownership or checksum changed')
    output = ROOT/'annotation_coverage.json'
    if not output.exists():
        atomic_json(output, {'participants': len(rows), 'physical_sources': sum(map(len, reports.values())),
            'speaker_annotated_participants': 0, 'confidence_annotated_participants': len(rows),
            'changed_documents': sum(r['text_sha256'] != r['original_text_sha256'] for r in rows),
            'empty_filter_fallbacks': sum(r['empty_filter_fallback'] for r in rows),
            'invalid_time_rows': sum(r['invalid_time_rows'] for r in rows),
            'word_retention_min_median_max': np.quantile([r['retained_words']/r['original_words'] for r in rows], [0,.5,1]).tolist(),
            'source_speaker_identity_verified': False, 'temporal_alignment_used': False})
    return rows


def encode():
    import torch
    from threadpoolctl import threadpool_limits
    from .encoder_kernel_accuracy import offline_network
    from .semantic_features import local_encoder, encode_document
    from .age_language_accuracy import language_features
    rows = prepare(resume=True)
    if (ROOT/'features.npz').exists():
        raise FileExistsError('Preserving encoded features')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    (ROOT/'embeddings').mkdir(exist_ok=True); embeddings = []
    with threadpool_limits(limits=2), offline_network():
        tokenizer, model = local_encoder(Path('checkpoints/pretrained_minilm'))
        for i, r in enumerate(rows):
            path = ROOT/'embeddings'/f"{r['participant_id']}.json"
            if not path.exists():
                atomic_json(path, {'participant_id': r['participant_id'], 'text_sha256': r['text_sha256'],
                    **encode_document(r['text'], tokenizer, model)})
            record = json.loads(path.read_text())
            if record['participant_id'] != r['participant_id'] or record['text_sha256'] != r['text_sha256']:
                raise ValueError('Embedding ownership changed')
            embeddings.append(record['embedding'])
            if (i+1)%10 == 0:
                print(f'Confidence-filtered embeddings {i+1}/{len(rows)}', flush=True)
    np.savez_compressed(ROOT/'features.npz', participant_id=np.asarray([r['participant_id'] for r in rows]),
        semantic=np.asarray(embeddings), language=np.stack([language_features(r['text']) for r in rows]))
    atomic_json(ROOT/'completion.json', {'features_sha256': sha256(ROOT/'features.npz'), 'participants': len(rows),
        'protocol_sha256': sha256(ROOT/'protocol.json'), 'documents': {str(ROOT/'documents'/f"{r['participant_id']}.json"):
        sha256(ROOT/'documents'/f"{r['participant_id']}.json") for r in rows},
        'encoder_files': {str(p): sha256(p) for p in Path('checkpoints/pretrained_minilm').glob('*') if p.is_file()},
        'encoding_code_sha256': sha256('src/semantic_features.py'), 'language_code_sha256': sha256('src/age_language_accuracy.py')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['prepare','encode'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.resume)
    else:
        encode()
