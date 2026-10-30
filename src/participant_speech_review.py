"""Create one local, target-blind speaker-review packet; never verify its draft."""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unicodedata

from .common import digest
from .final_evaluation import sha256
from .transcript_corpus import SCORE_PATTERN


FORMAT = 'local-participant-speech-review-v1'
ROOT = Path('results/participant_speech_review')
SOURCE_FIELDS = ['participant_id', 'raw_transcript_sha256', 'raw_wav_sha256',
                 'transcript_rows', 'wav_duration_seconds', 'wav_sample_rate',
                 'source_key', 'source_report_path']
REDACTION = '[REDACTED: questionnaire or score content]'
ROLES = ['unknown', 'participant', 'interviewer', 'overlap']


def _api():
    from . import participant_speech
    return participant_speech


def parse_review_rows(blob, expected_rows, duration):
    """Retain raw row indices; redact the existing cleaner's score-pattern rows."""
    text = blob.decode('utf-8-sig', errors='strict')
    if not text.splitlines():
        raise ValueError('Empty transcript')
    delimiter = csv.Sniffer().sniff(text.splitlines()[0], delimiters=',;\t').delimiter
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None or set(reader.fieldnames) != {'Start_Time', 'End_Time', 'Text', 'Confidence'}:
        raise ValueError('Unexpected transcript schema')
    rows = []
    for index, row in enumerate(reader):
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed transcript row')
        start, end, confidence = [float(row[key]) for key in ['Start_Time', 'End_Time', 'Confidence']]
        if not all(math.isfinite(value) for value in [start, end]) or start < 0 or end < start:
            raise ValueError('Invalid transcript time suggestion')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid recognition confidence')
        raw = row['Text']
        redacted = bool(SCORE_PATTERN.search(unicodedata.normalize('NFKC', raw).lower()))
        rows.append({'row_index': index, 'start_seconds': start, 'end_seconds': end,
                     'text': REDACTION if redacted else raw, 'confidence': confidence,
                     'score_redacted': redacted, 'audio_jump_available': start < duration,
                     'suggested_end_outside_audio': end > duration})
    if len(rows) != expected_rows or not rows:
        raise ValueError('Transcript row count differs from source contract')
    return rows


def safe_draft(template, source, row_count):
    draft = copy.deepcopy(template)
    for key in ['participant_id', 'raw_transcript_sha256', 'raw_wav_sha256']:
        if draft.get(key) != source[key]:
            raise ValueError('Annotation template/source identity differs')
    if draft.get('annotation_version') != 'participant-speech-v1':
        raise ValueError('Unsupported annotation draft schema')
    draft['status'] = 'draft'
    for name in ['audio_review', 'transcript_review']:
        section = draft[name]
        for key in ['verified', 'speaker_identity_confirmed', 'complete_review']:
            section[key] = False
        for key in ['method', 'reviewer_id', 'verified_at_utc', 'evidence_reference']:
            section[key] = ''
    audio = draft['audio_review']
    audio.update(time_unit='seconds', time_origin='raw_wav_sample_0', timestamp_unit_confirmed=False,
                 reviewed_ranges=[], intervals=[])
    draft['transcript_review'].update(alignment_verified=False, event_checks=[],
                                      rows=[{'row_index': i, 'role': 'unknown'} for i in range(row_count)])
    return draft


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')


def _review_csv(rows):
    buffer = io.StringIO(newline='')
    fields = ['row_index', 'start_seconds', 'end_seconds', 'text', 'confidence',
              'score_redacted', 'audio_jump_available', 'suggested_end_outside_audio']
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        safe = dict(row)
        # This is a display export; prevent spreadsheet formula execution without
        # changing the original transcript, JSON rows, row IDs or source hashes.
        if safe['text'].lstrip().startswith(('=', '+', '-', '@')):
            safe['text'] = "'"+safe['text']
        writer.writerow(safe)
    return buffer.getvalue().encode('utf-8')


def render_html(source, rows, draft):
    payload = json.dumps({'participant': source['participant_id'], 'duration': source['wav_duration_seconds'],
                          'rows': rows, 'draft': draft, 'roles': ROLES}, ensure_ascii=False, allow_nan=False)
    for character, escaped in [('&', '\\u0026'), ('<', '\\u003c'), ('>', '\\u003e'),
                               ('\u2028', '\\u2028'), ('\u2029', '\\u2029')]:
        payload = payload.replace(character, escaped)
    return HTML.replace('__PACKET_DATA__', payload).encode('utf-8')


def packet(participant_id='487', output=None, resume=False):
    participant_id = str(participant_id)
    if not participant_id.isdigit():
        raise ValueError('A numeric original training participant ID is required')
    out = ROOT/participant_id if output is None else Path(output)
    if out.is_symlink():
        raise ValueError('Review output cannot be a symlink')
    for protected in ['data', 'dataset', 'checkpoints']:
        if out.resolve().is_relative_to(Path(protected).resolve()):
            raise ValueError('Review packet must be outside source and model directories')
    if out.exists() and not resume:
        raise FileExistsError('Preserving the existing review packet; use --resume to verify')
    if resume and not out.is_dir():
        raise FileNotFoundError('Resume requires an existing review packet')
    api = _api()
    ids, sources, provenance = api.training_sources()
    if len(ids) != 163 or len(set(ids)) != 163 or participant_id not in ids:
        raise ValueError('Review participant must belong to the unchanged163-person training cohort')
    matching = [source for source in sources if source['participant_id'] == participant_id]
    if len(matching) != 1:
        raise ValueError('Expected one canonical training-source contract')
    source = {key: matching[0][key] for key in SOURCE_FIELDS}
    transcript, wav = api.read_source(matching[0])
    for name, blob in [('transcript', transcript), ('wav', wav)]:
        if hashlib.sha256(blob).hexdigest() != source[f'raw_{name}_sha256']:
            raise ValueError('Raw review source hash differs from its contract')
    rows = parse_review_rows(transcript, source['transcript_rows'], source['wav_duration_seconds'])
    draft = safe_draft(api.annotation_template(matching[0]), source, len(rows))
    context = {'format': FORMAT, 'participant_id': participant_id, 'source': source,
               'training_ids_sha256': digest(ids), 'source_provenance_sha256': digest(provenance),
               'implementation_sha256': sha256(__file__), 'targets_loaded': False,
               'questionnaire_rows_redacted': sum(row['score_redacted'] for row in rows),
               'speaker_identities_inferred': False, 'annotation_status': 'draft',
               'source_files_modified': False, 'uploads': False}
    if resume:
        envelope = json.loads((out/'manifest.json').read_text())
        manifest = envelope['manifest']
        if digest(manifest) != envelope['manifest_sha256'] or manifest['context'] != context:
            raise ValueError('Review packet/source context changed')
        for filename, expected in manifest['files'].items():
            path = out/filename
            if Path(filename).name != filename or path.is_symlink() or not path.is_file() or sha256(path) != expected:
                raise ValueError('Review packet file changed')
        expected_names = {'recording.wav', 'transcript_rows.csv', 'transcript_rows.json',
                          'annotation_draft.json', 'review.html', 'README.md'}
        if set(manifest['files']) != expected_names:
            raise ValueError('Review packet coverage differs')
        return {'complete': True, 'verified_existing_packet': True, 'new_files': 0,
                'participant_id': participant_id, 'output': str(out.resolve()),
                'annotation_status': 'draft', 'targets_loaded': False}
    readme = f'''# Local speaker review for participant {participant_id}

Open review.html in a local browser. The original WAV copy and all page assets
remain local. The page loads no remote scripts and uploads no data.

Roles begin as unknown. Assign a role only from direct source review. Transcript
timestamps are unverified suggestions; their relationship to this WAV has not
been established. Audio intervals and transcript-row roles remain separate.
Recognition confidence is not speaker confidence. Questionnaire/score-pattern
rows are redacted using the existing transcript cleaner's exclusion rule; raw
source hashes and row numbering remain unchanged.

Export creates a DRAFT. It never verifies identity, time units, alignment or full
coverage. Follow the project annotation instructions for separate human/provider
attestation before ingestion. The review CSV is a display copy (formula-like
strings are prefixed with an apostrophe); the JSON rows retain the displayed text.

The packet contains no questionnaire targets, demographics, folds or predictions.
Source hash bindings are in manifest.json. Keep exported drafts under a new file
name; --resume verifies original packet files without rewriting them. The raw WAV
copy is byte-identical to the supplied source, including its original headers.
'''.encode('utf-8')
    contents = {'recording.wav': wav, 'transcript_rows.csv': _review_csv(rows),
                'transcript_rows.json': _json_bytes(rows), 'annotation_draft.json': _json_bytes(draft),
                'review.html': render_html(source, rows, draft), 'README.md': readme}
    manifest = {'context': context, 'files': {name: hashlib.sha256(blob).hexdigest() for name, blob in contents.items()}}
    contents['manifest.json'] = _json_bytes({'manifest': manifest, 'manifest_sha256': digest(manifest)})
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{out.name}.review-', dir=out.parent))
    try:
        for filename, blob in contents.items():
            (staging/filename).write_bytes(blob)
        if out.exists():
            raise FileExistsError('Preserving a review packet created concurrently')
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'complete': True, 'verified_existing_packet': False, 'new_files': len(contents),
            'participant_id': participant_id, 'output': str(out.resolve()),
            'review_page': str((out/'review.html').resolve()), 'annotation_status': 'draft',
            'questionnaire_rows_redacted': context['questionnaire_rows_redacted'], 'targets_loaded': False}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local participant speech review</title>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file:; connect-src 'none'; img-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<style>
body{font:16px system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 20px;color:#182431;background:#f5f7fa}
h1,h2{line-height:1.2}section{background:white;border:1px solid #d6dfe8;border-radius:10px;padding:18px;margin:18px 0}
audio{width:100%;margin:12px 0}button,select,input{font:inherit;padding:7px;margin:3px}button{cursor:pointer}button:disabled{cursor:default}
label{display:inline-block;margin:6px}input[type=number]{width:120px}table{width:100%;border-collapse:collapse}th,td{padding:9px;text-align:left;border-bottom:1px solid #d6dfe8;vertical-align:top}
.notice{border-left:5px solid #b57010;padding:10px 15px;background:#fff6e5}.muted{color:#516173;font-size:.9rem}.scroll{overflow:auto;max-height:65vh}
#message{min-height:1.5em}ul{padding-left:22px}#export{font-weight:700;background:#163f6b;color:white;border:0;border-radius:5px}
</style></head><body>
<h1>Speaker review: participant <span id="participant"></span></h1>
<p class="notice">This is an unverified draft. Choose roles only after direct listening. Transcript timestamps are suggestions and may not align with this recording. Audio and transcript reviews are separate. Export never verifies speaker identity, alignment or complete coverage.</p>
<section><h2>Original recording</h2><p>No speaker or channel has been automatically identified.</p>
<audio id="audio" controls preload="metadata" src="recording.wav"></audio>
<p class="muted">Source duration: <span id="duration"></span> seconds. All interval times are seconds from the start of this source WAV. Audio headers are preserved.</p></section>
<section><h2>Draft audio intervals</h2><p>Add only intervals you directly reviewed. Leave uncertain roles unknown; these intervals do not assign transcript-row roles.</p>
<label>Start <input id="start" type="number" min="0" step="0.001" value="0"></label><button id="start-here">Use playhead for start</button>
<label>End <input id="end" type="number" min="0" step="0.001" value="0"></label><button id="end-here">Use playhead for end</button>
<label>Role <select id="audio-role"></select></label><button id="add-interval">Add interval</button><ul id="intervals"></ul></section>
<section><h2>Transcript rows</h2><p>Jump controls use unverified CSV time suggestions. Listening to a jump does not verify global alignment. Questionnaire/score rows are redacted; recognition confidence is shown only for source review.</p>
<div class="scroll"><table><thead><tr><th>Row / suggested time</th><th>Transcript</th><th>Recognition confidence</th><th>Draft role</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section><h2>Optional alignment evidence notes</h2><p>Record audible event correspondences after checking them directly. These notes remain unverified until separate attestation.</p>
<label>Row index <input id="event-row" type="number" min="0" step="1" value="0"></label>
<label>WAV time <input id="event-time" type="number" min="0" step="0.001" value="0"></label>
<label>Audible event <input id="event-evidence" type="text" maxlength="500"></label><button id="add-event">Add note</button><ul id="events"></ul></section>
<section><button id="export">Export unverified draft JSON</button><p id="message" role="status"></p><p class="muted">The export keeps all verification and completeness attestations false. Follow the separate annotation instructions before ingestion. Nothing is uploaded.</p></section>
<script id="packet-data" type="application/json">__PACKET_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('packet-data').textContent);
const draft=JSON.parse(JSON.stringify(data.draft));
const byId=id=>document.getElementById(id), audio=byId('audio');
byId('participant').textContent=data.participant;byId('duration').textContent=String(data.duration);
function message(text){byId('message').textContent=text;}
function roles(select){for(const role of data.roles){const option=document.createElement('option');option.value=role;option.textContent=role;select.append(option);}select.value='unknown';}
roles(byId('audio-role'));
for(const row of data.rows){
 const tr=document.createElement('tr'),timing=document.createElement('td'),text=document.createElement('td'),confidence=document.createElement('td'),roleCell=document.createElement('td');
 const label=document.createElement('div');label.textContent=`${row.row_index}: ${row.start_seconds}–${row.end_seconds}s`;timing.append(label);
 const jump=document.createElement('button');jump.textContent=row.audio_jump_available?'Jump and listen':'Outside audio range';jump.disabled=!row.audio_jump_available;
 jump.addEventListener('click',()=>{audio.currentTime=row.start_seconds;const play=audio.play();if(play)play.catch(()=>message('Use the audio controls to start playback.'));});timing.append(jump);
 if(row.suggested_end_outside_audio){const warning=document.createElement('div');warning.textContent='Suggested end exceeds source audio.';warning.className='muted';timing.append(warning);}
 text.textContent=row.text;confidence.textContent=String(row.confidence);const select=document.createElement('select');roles(select);select.setAttribute('aria-label',`Role for row ${row.row_index}`);
 select.addEventListener('change',()=>{draft.transcript_review.rows[row.row_index].role=select.value;});roleCell.append(select);tr.append(timing,text,confidence,roleCell);byId('rows').append(tr);
}
byId('start-here').addEventListener('click',()=>byId('start').value=audio.currentTime.toFixed(3));
byId('end-here').addEventListener('click',()=>byId('end').value=audio.currentTime.toFixed(3));
function paintIntervals(){byId('intervals').replaceChildren();draft.audio_review.intervals.forEach((interval,index)=>{const li=document.createElement('li');li.textContent=`${interval.start}–${interval.end}s: ${interval.role} `;const remove=document.createElement('button');remove.textContent='Remove';remove.addEventListener('click',()=>{draft.audio_review.intervals.splice(index,1);paintIntervals();});li.append(remove);byId('intervals').append(li);});}
byId('add-interval').addEventListener('click',()=>{const start=Number(byId('start').value),end=Number(byId('end').value),role=byId('audio-role').value;
 if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>data.duration){message('Choose a finite interval within the source duration, with end after start.');return;}
 if(draft.audio_review.intervals.some(interval=>start<interval.end&&end>interval.start)){message('Intervals must not overlap; use the overlap role for a single shared interval.');return;}
 draft.audio_review.intervals.push({start,end,role});draft.audio_review.intervals.sort((a,b)=>a.start-b.start);paintIntervals();message('Added a draft audio interval.');});
function paintEvents(){byId('events').replaceChildren();draft.transcript_review.event_checks.forEach((event,index)=>{const li=document.createElement('li');li.textContent=`Row ${event.row_index}, WAV ${event.wav_time_seconds}s: ${event.evidence} `;const remove=document.createElement('button');remove.textContent='Remove';remove.addEventListener('click',()=>{draft.transcript_review.event_checks.splice(index,1);paintEvents();});li.append(remove);byId('events').append(li);});}
byId('add-event').addEventListener('click',()=>{const row_index=Number(byId('event-row').value),wav_time_seconds=Number(byId('event-time').value),evidence=byId('event-evidence').value.trim();
 if(!Number.isInteger(row_index)||row_index<0||row_index>=data.rows.length||!Number.isFinite(wav_time_seconds)||wav_time_seconds<0||wav_time_seconds>=data.duration||!evidence){message('Choose a source row, an in-range WAV time and an audible event description.');return;}
 draft.transcript_review.event_checks.push({row_index,wav_time_seconds,evidence});paintEvents();message('Added an unverified alignment note.');});
byId('export').addEventListener('click',()=>{const result=JSON.parse(JSON.stringify(draft));result.status='draft';
 for(const section of [result.audio_review,result.transcript_review]){for(const key of ['verified','speaker_identity_confirmed','complete_review'])section[key]=false;for(const key of ['method','reviewer_id','verified_at_utc','evidence_reference'])section[key]='';}
 result.audio_review.timestamp_unit_confirmed=false;result.transcript_review.alignment_verified=false;
 const blob=new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${data.participant}_speech_draft_export.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);message('Exported an unverified draft. No verification or completeness was asserted.');});
</script></body></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant-id', '--participant', dest='participant_id', default='487')
    parser.add_argument('--output'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(packet(args.participant_id, args.output, args.resume), indent=2))


if __name__ == '__main__':
    main()
