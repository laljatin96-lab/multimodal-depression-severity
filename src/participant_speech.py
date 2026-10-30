"""Source-bound human/provider speech annotations; no inferred speaker identity.

This module checks annotation provenance and coordinates, not the truth of a
human attestation. Unverified inputs retain the original participant pipeline.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd

from .batch_inference import atomic_json, load_record, output_lock
from .common import digest
from .final_evaluation import sha256
from .full_dataset import source_stamp
from .transcript_corpus import clean_transcript

ROOT = Path('results/participant_speech')
ANNOTATIONS = Path('data/annotations/participant_speech')
VERSION = 'participant-speech-v1'
ROLES = {'unknown', 'participant', 'interviewer', 'overlap'}


def training_sources():
    """Return label-free training source contracts from completed integrity audits."""
    from . import wavlm_features_complete as previous
    from .confidence_features import transcript_member
    metadata = pd.read_csv('data/metadata/participants.csv', usecols=['participant_id', 'split'], dtype=str)
    ids = metadata.loc[metadata.split == 'train', 'participant_id'].tolist()
    if len(ids) != 163 or len(set(ids)) != len(ids):
        raise ValueError('Expected the original 163 distinct training participants')
    _, old_provenance = previous.load_training(ids)
    protocol = json.loads((previous.ROOT/'protocol.json').read_text())
    transcript_protocol = json.loads(Path('results/transcript_corpus/protocol.json').read_text())
    transcript_receipt = json.loads(Path('results/transcript_corpus/completion.json').read_text())
    if (digest(transcript_protocol['context']) != transcript_protocol['context_hash']
            or not transcript_receipt['complete']
            or transcript_receipt['context_hash'] != transcript_protocol['context_hash']):
        raise ValueError('Transcript audit changed')
    bindings = {str(previous.ROOT/'protocol.json'): sha256(previous.ROOT/'protocol.json'),
        str(previous.ROOT/'completion.json'): sha256(previous.ROOT/'completion.json'),
        'results/transcript_corpus/protocol.json': sha256('results/transcript_corpus/protocol.json'),
        'results/transcript_corpus/completion.json': sha256('results/transcript_corpus/completion.json')}
    reports = {}
    for path in protocol['context']['source_reports']:
        report = json.loads(Path(path).read_text())
        reports[report['source_key']] = (path, report)
    sources = []
    for pid in ids:
        audio_path = previous.ROOT/'participants'/f'{pid}.json'
        audio = load_record(audio_path, protocol['context_hash'])
        report_path, report = reports[audio['source_keys'][0]]
        transcript_path = Path('results/transcript_corpus/sources')/f"{report['source_key']}.json"
        if sha256(transcript_path) != transcript_receipt['records'][report['source_key']]:
            raise ValueError('Transcript audit record changed')
        transcript = load_record(transcript_path, transcript_protocol['context_hash'])
        member = transcript_member(report)
        if (transcript['participant_id'] != pid or audio['participant_id'] != pid
                or transcript['speaker_field_available'] or not transcript['full_source_hash_verified']
                or transcript['raw_transcript_sha256'] != member['sha256']):
            raise ValueError('Source identity or known transcript schema differs')
        quality = audio['quality']
        sources.append({'participant_id': pid, 'source_key': report['source_key'],
            'source_report_path': report_path, 'source_report_sha256': sha256(report_path),
            'raw_transcript_sha256': member['sha256'], 'raw_wav_sha256': audio['raw_wav_sha256'],
            'transcript_rows': transcript['rows'], 'wav_duration_seconds': quality['samples']/quality['sample_rate'],
            'wav_sample_rate': quality['sample_rate'], 'wav_samples': quality['samples'],
            'csv_time_origin_verified': False, 'speaker_identity_verified': False})
        for path in [audio_path, transcript_path, Path(report_path)]:
            bindings[str(path)] = sha256(path)
    return ids, sources, {'bindings': bindings, 'previous_audio_features': old_provenance,
        'labels_loaded_by_source_contract': False, 'validation_test_processed': False}


def read_source(source):
    from .confidence_features import read_member
    from .wavlm_features import read_wav
    report_path = Path(source['source_report_path'])
    if sha256(report_path) != source['source_report_sha256']:
        raise ValueError('Source report changed')
    report = json.loads(report_path.read_text())
    if report['participant_id'] != source['participant_id'] or source_stamp(Path(report['source'])) != report['source_stamp']:
        raise ValueError('Participant source changed')
    transcript, wav = read_member(report), read_wav(report)
    if (hashlib.sha256(transcript).hexdigest() != source['raw_transcript_sha256']
            or hashlib.sha256(wav).hexdigest() != source['raw_wav_sha256']):
        raise ValueError('Original source bytes differ from bound participant')
    if clean_transcript(transcript)['rows'] != source['transcript_rows']:
        raise ValueError('Transcript row ownership differs')
    return transcript, wav


def annotation_template(source):
    common = {'verified': False, 'method': '', 'reviewer_id': '', 'verified_at_utc': '',
        'evidence_reference': '', 'speaker_identity_confirmed': False, 'complete_review': False}
    return {'annotation_version': VERSION, 'status': 'draft',
        **{key: source[key] for key in ['participant_id', 'raw_wav_sha256', 'raw_transcript_sha256']},
        'audio_review': {**common, 'time_unit': 'seconds', 'time_origin': 'raw_wav_sample_0',
            'timestamp_unit_confirmed': False, 'reviewed_ranges': [], 'intervals': []},
        'transcript_review': {**common, 'alignment_verified': False, 'event_checks': [],
            'rows': [{'row_index': i, 'role': 'unknown'} for i in range(source['transcript_rows'])]}}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError('Finite numeric WAV-relative seconds required')
    return float(value)


def _spans(rows, duration, roles=False):
    if not isinstance(rows, list):
        raise ValueError('Interval list required')
    last = 0.
    for row in rows:
        if set(row) != ({'start', 'end', 'role'} if roles else {'start', 'end'}):
            raise ValueError('Unexpected interval fields')
        start, end = _number(row['start']), _number(row['end'])
        if start < 0 or end <= start or end > duration or start < last:
            raise ValueError('Intervals must be ordered, nonoverlapping and within the bound WAV')
        if roles and row['role'] not in ROLES:
            raise ValueError('Unknown speaker role')
        last = end
    return rows


def _attested(section):
    if not isinstance(section['verified'], bool) or not isinstance(section['complete_review'], bool):
        raise ValueError('Explicit boolean verification/coverage required')
    if not section['verified']:
        return False
    if section['method'] not in {'human_review', 'provider_annotation'}:
        raise ValueError('Automatic diarization does not verify speaker identity')
    for name in ['reviewer_id', 'evidence_reference', 'verified_at_utc']:
        if not isinstance(section[name], str) or not section[name].strip():
            raise ValueError('Reviewer, evidence reference and UTC timestamp required')
    try:
        timestamp = datetime.fromisoformat(section['verified_at_utc'].replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid verification timestamp') from error
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError('Verification timestamp must explicitly use UTC')
    if section['speaker_identity_confirmed'] is not True:
        raise ValueError('Speaker identity requires explicit human/provider evidence')
    return section['complete_review']


def validate_annotation(annotation, source):
    fallback = {'text_rows': [], 'audio_intervals': [], 'text_verified': False, 'audio_verified': False,
        'text_reason': 'missing_verified_annotation', 'audio_reason': 'missing_verified_annotation'}
    if annotation is None:
        return fallback
    if not isinstance(annotation, dict):
        raise ValueError('Annotation object required')
    if set(annotation) != set(annotation_template(source)) or annotation['annotation_version'] != VERSION:
        raise ValueError('Unexpected annotation schema')
    if annotation['status'] not in {'draft', 'reviewed'}:
        raise ValueError('Unexpected annotation status')
    for key in ['participant_id', 'raw_wav_sha256', 'raw_transcript_sha256']:
        if annotation[key] != source[key]:
            raise ValueError('Annotation source hash or participant differs')
    for key in ['raw_wav_sha256', 'raw_transcript_sha256']:
        if re.fullmatch('[0-9a-f]{64}', annotation[key]) is None:
            raise ValueError('Expected SHA256 source binding')
    duration = _number(source['wav_duration_seconds']); audio = annotation['audio_review']; text = annotation['transcript_review']
    template = annotation_template(source)
    for key in ['audio_review', 'transcript_review']:
        if set(annotation[key]) != set(template[key]):
            raise ValueError('Unexpected review fields; targets and questionnaire data are forbidden')
    intervals = _spans(audio['intervals'], duration, True); ranges = _spans(audio['reviewed_ranges'], duration)
    if audio['time_unit'] != 'seconds' or audio['time_origin'] != 'raw_wav_sample_0':
        raise ValueError('Explicit seconds from the bound raw WAV sample zero required')
    rows = text['rows']; indices = []
    if not isinstance(rows, list):
        raise ValueError('Transcript row list required')
    for row in rows:
        if set(row) != {'row_index', 'role'} or type(row['row_index']) is not int or row['role'] not in ROLES:
            raise ValueError('Invalid transcript row role')
        indices.append(row['row_index'])
    if len(set(indices)) != len(indices) or any(i < 0 or i >= source['transcript_rows'] for i in indices):
        raise ValueError('Duplicate or out-of-range transcript row')
    events = text['event_checks']
    if not isinstance(events, list):
        raise ValueError('Explicit audible alignment checks required')
    for event in events:
        if (set(event) != {'row_index', 'wav_time_seconds', 'evidence'}
                or type(event['row_index']) is not int or not 0 <= event['row_index'] < source['transcript_rows']
                or not isinstance(event['evidence'], str) or not event['evidence'].strip()
                or not 0 <= _number(event['wav_time_seconds']) < duration):
            raise ValueError('Invalid transcript/WAV event check')
    audio_complete, text_complete = _attested(audio), _attested(text)
    if (audio['verified'] or text['verified']) and annotation['status'] != 'reviewed':
        raise ValueError('A draft cannot be used as a verified annotation')
    result = {**fallback, 'text_reason': 'unverified_or_incomplete_review', 'audio_reason': 'unverified_or_incomplete_review'}
    if audio_complete:
        if audio['timestamp_unit_confirmed'] is not True:
            raise ValueError('Audio timebase requires explicit verification')
        if not ranges or ranges[0]['start'] != 0 or ranges[-1]['end'] != duration or any(a['end'] != b['start'] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('Complete audio review must cover the full bound WAV without gaps')
        if any(r['role'] == 'unknown' for r in intervals):
            raise ValueError('Unknown audio identity cannot be promoted as complete review')
        usable = [r for r in intervals if r['role'] == 'participant' and r['end']-r['start'] >= .1]
        result.update(audio_intervals=usable, audio_verified=bool(usable),
            audio_reason='verified_participant_segments' if usable else 'no_usable_verified_participant_segments')
    if text_complete:
        if (text['alignment_verified'] is not True or len({e['row_index'] for e in events}) < 2
                or len({e['wav_time_seconds'] for e in events}) < 2):
            raise ValueError('Text identity requires separate row-to-audio verification and two audible event checks')
        if sorted(indices) != list(range(source['transcript_rows'])) or any(r['role'] == 'unknown' for r in rows):
            raise ValueError('Complete transcript review must assign every original row')
        selected = sorted(r['row_index'] for r in rows if r['role'] == 'participant')
        result.update(text_rows=selected, text_verified=bool(selected),
            text_reason='verified_participant_rows' if selected else 'no_usable_verified_participant_rows')
    return result


def transcript_frame(blob):
    import csv
    delimiter = csv.Sniffer().sniff(blob.decode('utf-8-sig').splitlines()[0], delimiters=',;\t').delimiter
    frame = pd.read_csv(io.BytesIO(blob), sep=delimiter, keep_default_na=False)
    clean_transcript(blob)  # Reuse the frozen strict schema and score exclusion.
    return frame


def selected_document(blob, annotation, source):
    if hashlib.sha256(blob).hexdigest() != source['raw_transcript_sha256']:
        raise ValueError('Transcript source hash differs')
    original = clean_transcript(blob); decision = validate_annotation(annotation, source)
    if original['rows'] != source['transcript_rows']:
        raise ValueError('Transcript row count differs')
    selected = original
    if decision['text_verified']:
        frame = transcript_frame(blob)
        selected = clean_transcript(frame.iloc[decision['text_rows']].to_csv(index=False).encode())
    fallback = not decision['text_verified'] or not bool(selected['words'])
    return {**(original if fallback else selected), 'fallback': fallback,
        'fallback_reason': ('empty_cleaned_verified_text' if decision['text_verified'] else decision['text_reason']) if fallback else None,
        'retained_row_indices': list(range(original['rows'])) if fallback else decision['text_rows'],
        'speaker_identity_verified': not fallback, 'targets_used': False}


def select_audio_segments(waveform, annotation, source, sample_rate=16000):
    if type(sample_rate) is not int or sample_rate <= 0 or np.asarray(waveform).ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError('Finite mono waveform and explicit sample rate required')
    if abs(len(waveform)/sample_rate-source['wav_duration_seconds']) > 1/sample_rate:
        raise ValueError('Decoded waveform duration differs from the bound source')
    decision = validate_annotation(annotation, source)
    if not decision['audio_verified']:
        return [waveform], decision['audio_reason']
    segments = [waveform[int(np.ceil(r['start']*sample_rate)):int(np.floor(r['end']*sample_rate))]
                for r in decision['audio_intervals']]
    segments = [s for s in segments if len(s) >= int(np.ceil(.1*sample_rate))]
    return (segments, 'verified_participant_segments') if segments else ([waveform], 'no_usable_verified_participant_segments')


def cohort_plan(sources, annotations):
    ids = [s['participant_id'] for s in sources]
    if len(set(ids)) != len(ids) or set(annotations)-set(ids):
        raise ValueError('Duplicate or nontraining participant annotations')
    return [{'participant_id': s['participant_id'], **validate_annotation(annotations.get(s['participant_id']), s)} for s in sources]


def prepare(annotations_root=ANNOTATIONS, resume=False):
    for protected in ['data', 'dataset', 'checkpoints', annotations_root]:
        if ROOT.resolve().is_relative_to(Path(protected).resolve()):
            raise ValueError('Input cache must be outside source, annotation and model directories')
    if ROOT.exists() and not resume:
        raise FileExistsError('Preserving existing input cache; use --resume')
    ids, sources, provenance = training_sources(); annotations_root = Path(annotations_root)
    annotations, annotation_hashes = {}, {}
    for path in sorted(annotations_root.glob('*.json')):
        annotation = json.loads(path.read_text()); pid = annotation['participant_id']
        if path.stem != pid or pid in annotations:
            raise ValueError('Annotation filename/participant ownership differs')
        annotations[pid] = annotation; annotation_hashes[str(path)] = sha256(path)
    plan = cohort_plan(sources, annotations)
    context = {'protocol': VERSION, 'sources': sources, 'training_ids': ids,
        'provenance': provenance, 'annotation_hashes': annotation_hashes, 'source_code_sha256': sha256(__file__),
        'fallback': 'Retain exact original features independently for each unverified/unusable modality; never omit a participant',
        'targets_used': False, 'validation_test_processed': False,
        'support_hashes': {p: sha256(p) for p in ['src/transcript_corpus.py', 'src/confidence_features.py',
            'src/wavlm_features.py', 'src/wavlm_features_complete.py', 'src/semantic_features.py', 'src/age_language_accuracy.py']}}
    envelope = {'context': context, 'context_hash': digest(context)}
    if resume:
        receipt = json.loads((ROOT/'completion.json').read_text())
        if json.loads((ROOT/'contract.json').read_text()) != envelope:
            raise ValueError('Frozen source/annotation contract changed; use a new cache directory')
        if any(sha256(path) != expected for path, expected in receipt['artifacts'].items()):
            raise ValueError('Completed annotation cache changed')
        return {**receipt, 'new_encoder_runs': 0, 'new_fits': 0, 'rewritten_files': 0}
    # Reuse verified cached features; only approved participant inputs can cause encoding.
    from .wavlm_accuracy import load_inputs
    x, frame, _, _, _ = load_inputs()
    if frame.participant_id.tolist() != ids:
        raise ValueError('Cached feature participant alignment differs')
    semantic_av, language, speech = x['semantic_av'].copy(), x['language'].copy(), x['speech'].copy()
    text_encoder = audio_encoder = None; new_text = new_audio = 0
    documents = {r['participant_id']: r for r in json.loads(Path('results/transcript_corpus/participants.json').read_text())}
    ROOT.mkdir(parents=True); rows = []
    from .encoder_kernel_accuracy import offline_network
    from threadpoolctl import threadpool_limits
    with output_lock(ROOT), offline_network(), threadpool_limits(limits=2):
        atomic_json(ROOT/'contract.json', envelope)
        for i, (source, decision) in enumerate(zip(sources, plan)):
            pid = source['participant_id']; document = None; segments = None
            if decision['text_verified'] or decision['audio_verified']:
                blob, wav = read_source(source)
                document = selected_document(blob, annotations[pid], source)
                if decision['audio_verified']:
                    from .wavlm_features_complete import decode_compatible_pcm
                    waveform, _ = decode_compatible_pcm(wav)
                    segments, reason = select_audio_segments(waveform, annotations[pid], source)
                    if reason != 'verified_participant_segments':
                        decision = {**decision, 'audio_verified': False, 'audio_reason': reason}
            if document is None:
                document = {'text': documents[pid]['text'], 'text_sha256': documents[pid]['text_sha256'],
                    'fallback': True, 'fallback_reason': decision['text_reason'], 'targets_used': False,
                    'speaker_identity_verified': False}
            if not document['fallback']:
                from .semantic_features import local_encoder, encode_document
                from .age_language_accuracy import language_features
                import torch
                torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
                if text_encoder is None:
                    text_encoder = local_encoder(Path('checkpoints/pretrained_minilm'))
                semantic_av[i, :384] = encode_document(document['text'], *text_encoder)['embedding']
                language[i] = language_features(document['text']); new_text += 1
            else:
                decision = {**decision, 'text_verified': False, 'text_reason': document['fallback_reason']}
            if decision['audio_verified']:
                import torch
                from . import wavlm_features as core
                if audio_encoder is None:
                    audio_encoder = core.load_encoder()
                processor, model = audio_encoder; clips = []
                for segment in segments:
                    starts, width = core.clip_starts(len(segment))
                    clips.extend(segment[s:s+width] for s in starts)
                indices = np.linspace(0, len(clips)-1, min(64, len(clips)), dtype=int)
                with torch.inference_mode():
                    frames = [model(**processor(clips[j], sampling_rate=16000, return_tensors='pt')).last_hidden_state[0].cpu().numpy() for j in indices]
                speech[i], _ = core.pooled_moments(frames); new_audio += 1
            rows.append({'participant_id': pid, 'decision': decision, 'document': document})
        np.testing.assert_array_equal(semantic_av[:, 384:], x['semantic_av'][:, 384:])
        for i, row in enumerate(rows):
            if not row['decision']['text_verified']:
                np.testing.assert_array_equal(semantic_av[i], x['semantic_av'][i]); np.testing.assert_array_equal(language[i], x['language'][i])
            if not row['decision']['audio_verified']:
                np.testing.assert_array_equal(speech[i], x['speech'][i])
        atomic_json(ROOT/'participant_inputs.json', rows)
        np.savez_compressed(ROOT/'features.npz', participant_id=np.asarray(ids), semantic_av=semantic_av, language=language, speech=speech)
        receipt = {'complete': True, 'participants': len(ids), 'context_hash': envelope['context_hash'],
            'verified_text_participants': new_text, 'verified_audio_participants': new_audio,
            'text_fallback_participants': len(ids)-new_text, 'audio_fallback_participants': len(ids)-new_audio,
            'changed_text_feature_participants': int(((semantic_av[:, :384] != x['semantic_av'][:, :384]).any(1) | (language != x['language']).any(1)).sum()),
            'changed_audio_feature_participants': int((speech != x['speech']).any(1).sum()),
            'new_encoder_runs': new_text+new_audio, 'new_fits': 0, 'dropped_participants': 0,
            'targets_used_for_input_selection': False, 'validation_test_processed': False,
            'av_features_unchanged': True, 'original_fallback_features_bit_identical': True,
            'status': 'blocked_input_annotations' if new_text == new_audio == 0 else 'verified_inputs_ready_for_locked_comparison',
            'artifacts': {str(ROOT/name): sha256(ROOT/name) for name in ['contract.json', 'participant_inputs.json', 'features.npz']}}
        atomic_json(ROOT/'completion.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', default=str(ANNOTATIONS)); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', default=str(ROOT))
    args = parser.parse_args(); ROOT = Path(args.output)
    print(json.dumps(prepare(args.annotations, args.resume), indent=2))
