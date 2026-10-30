"""Source and identity guards, using synthetic review annotations only."""
import copy
import hashlib

import numpy as np
import pytest

from src import participant_speech as s


@pytest.fixture
def source():
    blob = (b'Start_Time,End_Time,Text,Confidence\n'
            b'0,0.5,A quiet day,0.9\n0.6,1.2,A question,0.8\n1.3,2,PHQ8 score 17,0.9\n')
    return {'participant_id': '487', 'raw_transcript_sha256': hashlib.sha256(blob).hexdigest(),
        'raw_wav_sha256': 'b'*64, 'transcript_rows': 3, 'wav_duration_seconds': 3.,
        'wav_sample_rate': 48000, 'wav_samples': 144000, 'blob': blob}


def reviewed(source, audio=False, text=False):
    annotation = s.annotation_template(source); annotation['status'] = 'reviewed'
    for key, enabled in [('audio_review', audio), ('transcript_review', text)]:
        if enabled:
            annotation[key].update(verified=True, complete_review=True,
                method='human_review', reviewer_id='synthetic-test-fixture',
                verified_at_utc='2026-10-29T00:00:00Z', evidence_reference='toy-only-not-a-real-review',
                speaker_identity_confirmed=True)
    if audio:
        annotation['audio_review'].update(timestamp_unit_confirmed=True,
            reviewed_ranges=[{'start': 0., 'end': 3.}],
            intervals=[{'start': .25, 'end': .75, 'role': 'participant'},
                       {'start': 1., 'end': 2., 'role': 'overlap'}])
    if text:
        annotation['transcript_review'].update(alignment_verified=True,
            rows=[{'row_index': 0, 'role': 'participant'}, {'row_index': 1, 'role': 'interviewer'},
                  {'row_index': 2, 'role': 'participant'}],
            event_checks=[{'row_index': 0, 'wav_time_seconds': .2, 'evidence': 'toy-event-a'},
                          {'row_index': 1, 'wav_time_seconds': 1.2, 'evidence': 'toy-event-b'}])
    return annotation


def test_templates_and_missing_drafts_keep_unknown_identity(source):
    draft = s.annotation_template(source)
    assert draft['status'] == 'draft'
    assert all(row['role'] == 'unknown' for row in draft['transcript_review']['rows'])
    for annotation in [None, draft]:
        decision = s.validate_annotation(annotation, source)
        assert not decision['audio_verified'] and not decision['text_verified']
        row = s.selected_document(source['blob'], annotation, source)
        assert row['fallback'] and row['text'] == 'a quiet day a question'


def test_text_and_audio_identity_are_independent_and_scores_stay_excluded(source):
    audio = s.validate_annotation(reviewed(source, audio=True), source)
    assert audio['audio_verified'] and not audio['text_verified']
    text = reviewed(source, text=True)
    decision = s.validate_annotation(text, source)
    assert decision['text_verified'] and not decision['audio_verified']
    row = s.selected_document(source['blob'], text, source)
    assert row['text'] == 'a quiet day' and not row['fallback']
    assert row['explicit_score_rows_removed'] == 1 and not row['targets_used']


@pytest.mark.parametrize('key', ['participant_id', 'raw_wav_sha256', 'raw_transcript_sha256'])
def test_swapped_sources_rejected_even_for_drafts(source, key):
    a = s.annotation_template(source); a[key] = 'wrong'
    with pytest.raises(ValueError, match='source hash or participant'):
        s.validate_annotation(a, source)


@pytest.mark.parametrize('change', [
    lambda a: a.update(severity_class=4),
    lambda a: a['transcript_review'].update(PHQ8_score=20),
    lambda a: a['audio_review'].update(time_unit='milliseconds'),
    lambda a: a['audio_review'].update(time_origin='transcript_start'),
    lambda a: a['transcript_review']['rows'].append({'row_index': 0, 'role': 'unknown'}),
    lambda a: a['transcript_review']['rows'][0].update(row_index=99),
    lambda a: a['transcript_review']['rows'][0].update(role='speaker_0'),
])
def test_malformed_or_target_derived_annotations_fail_closed(source, change):
    a = s.annotation_template(source); change(a)
    with pytest.raises(ValueError):
        s.validate_annotation(a, source)


@pytest.mark.parametrize('start,end', [(-1, .5), (0, 0), (1, .5), (0, 3.1), (0, float('nan'))])
def test_invalid_audio_bounds_rejected(source, start, end):
    a = s.annotation_template(source)
    a['audio_review']['intervals'] = [{'start': start, 'end': end, 'role': 'participant'}]
    with pytest.raises(ValueError):
        s.validate_annotation(a, source)


def test_overlapping_intervals_and_incomplete_audio_coverage_rejected(source):
    a = reviewed(source, audio=True)
    a['audio_review']['intervals'][1]['start'] = .5
    with pytest.raises(ValueError, match='nonoverlapping'):
        s.validate_annotation(a, source)
    a = reviewed(source, audio=True)
    a['audio_review']['reviewed_ranges'] = [{'start': 0., 'end': 2.}]
    with pytest.raises(ValueError, match='full bound WAV'):
        s.validate_annotation(a, source)


def test_automatic_diarization_or_draft_cannot_establish_identity(source):
    a = reviewed(source, audio=True); a['audio_review']['method'] = 'automatic_diarization'
    with pytest.raises(ValueError, match='does not verify'):
        s.validate_annotation(a, source)
    a = reviewed(source, text=True); a['status'] = 'draft'
    with pytest.raises(ValueError, match='draft'):
        s.validate_annotation(a, source)


@pytest.mark.parametrize('field,value', [('alignment_verified', False), ('event_checks', []),
    ('rows', [{'row_index': 0, 'role': 'participant'}])])
def test_complete_text_requires_full_rows_and_separate_alignment_evidence(source, field, value):
    a = reviewed(source, text=True); a['transcript_review'][field] = value
    with pytest.raises(ValueError):
        s.validate_annotation(a, source)


def test_partial_or_no_participant_review_has_explicit_fallback(source):
    a = reviewed(source, text=True); a['transcript_review']['complete_review'] = False
    assert s.selected_document(source['blob'], a, source)['fallback']
    a = reviewed(source, text=True)
    for row in a['transcript_review']['rows']:
        row['role'] = 'interviewer'
    assert s.selected_document(source['blob'], a, source)['fallback']
    a = reviewed(source, text=True)
    for row in a['transcript_review']['rows']:
        row['role'] = 'participant' if row['row_index'] == 2 else 'interviewer'
    row = s.selected_document(source['blob'], a, source)
    assert row['fallback_reason'] == 'empty_cleaned_verified_text'


def test_seconds_use_decoded_rate_and_round_inward(source):
    a = reviewed(source, audio=True); waveform = np.arange(48000, dtype=float)
    a['audio_review']['intervals'][0].update(start=.25001, end=.75001)
    clips, reason = s.select_audio_segments(waveform, a, source, sample_rate=16000)
    np.testing.assert_array_equal(clips[0], waveform[4001:12000])
    assert len(clips) == 1 and reason == 'verified_participant_segments'
    with pytest.raises(ValueError, match='duration'):
        s.select_audio_segments(waveform, a, source, sample_rate=48000)
    clips, reason = s.select_audio_segments(waveform, None, source)
    assert clips[0] is waveform and reason == 'missing_verified_annotation'


def test_fallback_cohort_keeps_all163_in_original_order(source):
    sources = [{**source, 'participant_id': str(1000+i)} for i in range(163)]
    plan = s.cohort_plan(sources, {})
    assert [row['participant_id'] for row in plan] == [row['participant_id'] for row in sources]
    assert all(not row['text_verified'] and not row['audio_verified'] for row in plan)
    with pytest.raises(ValueError, match='nontraining'):
        s.cohort_plan(sources, {'validation-person': s.annotation_template(source)})


def test_actual_transcript_hash_is_checked_before_extraction(source):
    with pytest.raises(ValueError, match='source hash'):
        s.selected_document(source['blob']+b'\n', None, source)
