"""Local review packet tests; no real speaker identities or model fits."""
import hashlib
import io
import json
from types import SimpleNamespace
import wave

import pytest

from src import participant_speech as core
from src import participant_speech_review as s


@pytest.fixture
def packet_api(monkeypatch):
    transcript = (b'Start_Time,End_Time,Text,Confidence\n'
        b'0,0.1,</script><script>alert(1)</script>,0.9\n'
        b'0.2,0.3,PHQ8 score 17,0.8\n0.4,0.5,=UNTRUSTED,0.9\n')
    output = io.BytesIO()
    with wave.open(output, 'wb') as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
        writer.writeframes(b'\0\0'*16000)
    wav = output.getvalue()
    source = {'participant_id': '487', 'raw_transcript_sha256': hashlib.sha256(transcript).hexdigest(),
        'raw_wav_sha256': hashlib.sha256(wav).hexdigest(), 'transcript_rows': 3,
        'wav_duration_seconds': 1., 'wav_sample_rate': 16000,
        'source_key': 'toy', 'source_report_path': 'toy_report.json', 'severity_class': 4}
    ids = ['487']+[str(1000+i) for i in range(162)]
    api = SimpleNamespace(training_sources=lambda: (ids, [source], {'bindings': {}}),
        read_source=lambda _: (transcript, wav), annotation_template=core.annotation_template)
    monkeypatch.setattr(s, '_api', lambda: api)
    return source, transcript, wav, api


def test_packet_hides_targets_redacts_score_rows_and_exports_only_drafts(packet_api, tmp_path):
    source, transcript, wav, api = packet_api; out = tmp_path/'packet'
    result = s.packet('487', out)
    assert result['complete'] and not result['targets_loaded']
    assert (out/'recording.wav').read_bytes() == wav
    rows = json.loads((out/'transcript_rows.json').read_text())
    assert rows[1]['score_redacted'] and '17' not in rows[1]['text']
    assert "'=UNTRUSTED" in (out/'transcript_rows.csv').read_text()
    html = (out/'review.html').read_text()
    assert '</script><script>alert(1)</script>' not in html
    assert '\\u003c/script' in html and 'connect-src \'none\'' in html
    draft = json.loads((out/'annotation_draft.json').read_text())
    assert draft['status'] == 'draft' and all(row['role'] == 'unknown' for row in draft['transcript_review']['rows'])
    assert not core.validate_annotation(draft, source)['text_verified']
    manifest = json.loads((out/'manifest.json').read_text())
    assert 'severity_class' not in json.dumps(manifest) and 'severity_class' not in html
    assert not manifest['manifest']['context']['speaker_identities_inferred']


def test_resume_is_immutable_and_detects_tampering(packet_api, tmp_path):
    out = tmp_path/'packet'; s.packet('487', out)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    assert s.packet('487', out, resume=True)['new_files'] == 0
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        s.packet('487', out)
    (out/'review.html').write_text('changed')
    with pytest.raises(ValueError, match='file changed'):
        s.packet('487', out, resume=True)


def test_nontraining_id_and_source_hash_swap_rejected(packet_api, tmp_path):
    _, _, _, api = packet_api
    with pytest.raises(ValueError, match='training cohort'):
        s.packet('306', tmp_path/'excluded')
    api.read_source = lambda _: (b'swapped', b'swapped')
    with pytest.raises(ValueError, match='source hash'):
        s.packet('487', tmp_path/'swapped')


def test_safe_draft_resets_every_attestation(packet_api):
    source, _, _, _ = packet_api; template = core.annotation_template(source)
    template['status'] = 'reviewed'
    for name in ['audio_review', 'transcript_review']:
        template[name].update(verified=True, complete_review=True, speaker_identity_confirmed=True,
            reviewer_id='untrusted', method='human_review', evidence_reference='untrusted', verified_at_utc='untrusted')
    draft = s.safe_draft(template, source, 3)
    for section in [draft['audio_review'], draft['transcript_review']]:
        assert section['verified'] is section['complete_review'] is section['speaker_identity_confirmed'] is False
        assert not section['reviewer_id'] and not section['method']


def test_outside_audio_timestamp_is_flagged_not_guessed_or_shifted():
    blob = b'Start_Time,End_Time,Text,Confidence\n4,5,words,0.8\n'
    row = s.parse_review_rows(blob, 1, 1.)[0]
    assert row['start_seconds'] == 4 and not row['audio_jump_available'] and row['suggested_end_outside_audio']


def test_invalid_recognition_confidence_rejected():
    with pytest.raises(ValueError, match='confidence'):
        s.parse_review_rows(b'Start_Time,End_Time,Text,Confidence\n0,1,words,nan\n', 1, 2.)
