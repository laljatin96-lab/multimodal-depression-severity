"""Frozen WavLM features from audit-bound original training recordings only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import time

import numpy as np
import pandas as pd

from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .final_evaluation import sha256
from .full_dataset import safe_member, source_stamp

ROOT = Path('results/wavlm_features')
ENCODER = Path('checkpoints/pretrained_wavlm_base_plus')
PROTOCOL = 'frozen-wavlm-whole-interview-64x4s-mean-std-v1'
SAMPLE_RATE = 16000
CLIP_SECONDS = 4
MAX_CLIPS = 64


def decode_pcm(blob):
    """Read complete PCM data; record an incorrect outer RIFF length only.

    No source header is rewritten and no missing data is synthesized. All inner
    chunks, format fields and byte counts must be internally consistent.
    """
    if len(blob) < 44 or blob[:4] != b'RIFF' or blob[8:12] != b'WAVE':
        raise ValueError('Expected a bounded RIFF/WAVE input')
    declared = struct.unpack_from('<I', blob, 4)[0]+8
    if declared < len(blob):
        raise ValueError('RIFF length excludes supplied trailing bytes')
    chunks = {}; position = 12
    while position < len(blob):
        if position+8 > len(blob):
            raise ValueError('Truncated WAV chunk header')
        name, size = struct.unpack_from('<4sI', blob, position); position += 8
        end = position+size
        if end > len(blob):
            raise ValueError('Truncated WAV chunk payload')
        if name in (b'fmt ', b'data'):
            if name in chunks:
                raise ValueError('Duplicate WAV format or data chunk')
            chunks[name] = memoryview(blob)[position:end]
        position = end+(size % 2)
        if position > len(blob):
            raise ValueError('Missing odd-length WAV chunk padding')
    if set(chunks) != {b'fmt ', b'data'} or len(chunks[b'fmt ']) < 16:
        raise ValueError('Missing WAV format or data')
    codec, channels, rate, byte_rate, block, bits = struct.unpack_from('<HHIIHH', chunks[b'fmt '], 0)
    if codec != 1 or bits != 16 or not 1 <= channels <= 2 or rate != SAMPLE_RATE:
        raise ValueError('Expected original 16kHz PCM16 mono/stereo recording')
    data = chunks[b'data']
    if block != channels*2 or byte_rate != rate*block or not len(data) or len(data) % block:
        raise ValueError('Inconsistent PCM byte/sample counts')
    samples = np.frombuffer(data, dtype='<i2').reshape(-1, channels).astype(np.float32)/32768
    # Channel averaging does not assign a channel or any speech to a person.
    waveform = samples.mean(axis=1)
    if len(waveform) < rate//10 or not np.isfinite(waveform).all():
        raise ValueError('Insufficient or invalid waveform')
    return waveform, {'sample_rate': rate, 'channels': channels, 'samples': len(waveform),
        'seconds': len(waveform)/rate, 'pcm_bytes': len(data), 'physical_bytes': len(blob),
        'declared_riff_bytes': declared, 'riff_size_overstated': declared > len(blob),
        'pcm_data_complete': True, 'source_modified': False}


def clip_starts(samples):
    if not isinstance(samples, (int, np.integer)) or samples < SAMPLE_RATE//10:
        raise ValueError('Insufficient waveform samples')
    width = min(samples, SAMPLE_RATE*CLIP_SECONDS)
    count = min(MAX_CLIPS, max(1, samples//width))
    starts = np.linspace(0, samples-width, count, dtype=np.int64)
    if len(set(starts.tolist())) != count or (count > 1 and (np.diff(starts) < width).any()):
        raise ValueError('Overlapping or duplicate clip ownership')
    return starts, width


def pooled_moments(frame_arrays):
    count = 0; total = np.zeros(768, np.float64); squares = total.copy()
    for array in frame_arrays:
        array = np.asarray(array, np.float64)
        if array.ndim != 2 or array.shape[1] != 768 or not len(array) or not np.isfinite(array).all():
            raise ValueError('Invalid encoder hidden frames')
        count += len(array); total += array.sum(0); squares += np.square(array).sum(0)
    if not count:
        raise ValueError('No hidden frames')
    mean = total/count
    return np.r_[mean, np.sqrt(np.maximum(squares/count-mean*mean, 0))], count


def encoder_contract():
    receipt = json.loads((ENCODER/'download_manifest.json').read_text())
    if receipt['revision'] != '4c66d4806a428f2e922ccfa1a962776e232d487b':
        raise ValueError('Unexpected public encoder revision')
    for name, record in receipt['files'].items():
        if sha256(ENCODER/name) != record['sha256']:
            raise ValueError('Pinned encoder file changed')
    return {'root': str(ENCODER), 'manifest_sha256': sha256(ENCODER/'download_manifest.json'),
        'revision': receipt['revision'], 'files': {name:r['sha256'] for name,r in receipt['files'].items()}}


def load_encoder():
    import torch
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    from .encoder_kernel_accuracy import offline_network
    encoder_contract()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    with offline_network():
        model, info = WavLMModel.from_pretrained(ENCODER, local_files_only=True, weights_only=True,
                                                output_loading_info=True)
        if any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise ValueError('Incomplete or unexpected pretrained parameters')
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(ENCODER, local_files_only=True)
    if processor.sampling_rate != SAMPLE_RATE or processor.do_normalize or not processor.return_attention_mask:
        raise ValueError('Frozen publisher preprocessing differs')
    return processor, model


def encode_blob(blob, processor, model):
    import torch
    waveform, quality = decode_pcm(blob); starts, width = clip_starts(len(waveform))
    frame_arrays = []; rms = []; clipped = []
    with torch.inference_mode():
        for start in starts:
            clip = waveform[start:start+width]
            rms.append(float(np.sqrt(np.mean(np.square(clip, dtype=np.float64)))))
            clipped.append(float((np.abs(clip) >= .999).mean()))
            inputs = processor(clip, sampling_rate=SAMPLE_RATE, return_tensors='pt')
            # No padding, batches, augmentation, score-driven selection or fine-tuning.
            frame_arrays.append(model(**inputs).last_hidden_state[0].cpu().numpy())
    features, frames = pooled_moments(frame_arrays)
    return {'features': features.tolist(), 'raw_wav_sha256': hashlib.sha256(blob).hexdigest(),
        'quality': quality, 'clip_start_samples': starts.tolist(), 'clip_samples': width,
        'unique_covered_seconds': len(starts)*width/SAMPLE_RATE,
        'coverage_fraction': len(starts)*width/len(waveform), 'encoder_frames': frames,
        'clip_rms': rms, 'clip_clipped_fraction': clipped,
        'low_energy_clips': sum(value < 1e-4 for value in rms),
        'speaker_identity': 'Unverified supplied whole-interview recording',
        'alignment_used': False, 'targets_used': False}


def wav_member(report):
    expected = f"{report['participant_id']}_AUDIO.wav"
    members = [m for m in report['members'] if safe_member(m['member']) == expected]
    if len(members) != 1 or not 44 <= members[0]['bytes'] <= 1024**3:
        raise ValueError('Expected one bounded audited WAV member')
    return members[0]


def read_wav(report):
    path = Path(report['source']); member = wav_member(report)
    if not report['integrity_checked'] or source_stamp(path) != report['source_stamp']:
        raise ValueError('Source differs from complete archive audit')
    if path.is_dir():
        target = path/member['member']
        if any(p.is_symlink() for p in [target,*target.parents]) or not target.resolve().is_relative_to(path.resolve()):
            raise ValueError('Unsafe WAV member')
        blob = target.read_bytes()
    else:
        blob = None
        with tarfile.open(path, mode='r|*', bufsize=1024**2) as archive:
            for info in archive:
                safe_member(info.name)
                if info.name == member['member']:
                    if not info.isfile() or info.size != member['bytes']:
                        raise ValueError('Changed or unsafe WAV member')
                    blob = archive.extractfile(info).read(member['bytes']+1)
                    break
    if blob is None or len(blob) != member['bytes'] or source_stamp(path) != report['source_stamp']:
        raise ValueError('Changed or incomplete WAV')
    return blob


def prepare(resume=False):
    import torch, transformers
    from threadpoolctl import threadpool_limits
    from .encoder_kernel_accuracy import offline_network
    if ROOT.exists() and not resume:
        raise FileExistsError('Preserving original WavLM feature evidence')
    metadata = pd.read_csv('data/metadata/participants.csv', usecols=['participant_id','split'], dtype=str)
    ids = metadata.loc[metadata.split == 'train','participant_id'].tolist()
    if len(ids) != 163 or len(set(ids)) != 163:
        raise ValueError('Original training population changed')
    reports = {pid:[] for pid in ids}; bindings = {}
    for path in sorted(Path('results/full_dataset/sources').glob('*.json')):
        report = json.loads(path.read_text())
        if report['participant_id'] in reports:
            wav_member(report)
            if not report['integrity_checked'] or source_stamp(Path(report['source'])) != report['source_stamp']:
                raise ValueError('Original source changed')
            reports[report['participant_id']].append(report); bindings[str(path)] = sha256(path)
    if any(not copies for copies in reports.values()):
        raise ValueError('Missing original recording')
    context = {'protocol': PROTOCOL, 'ids': ids, 'metadata_sha256': sha256('data/metadata/participants.csv'),
        'source_reports': bindings, 'source_stamps': {r['source_key']:r['source_stamp'] for copies in reports.values() for r in copies},
        'encoder': encoder_contract(), 'source_code_sha256': sha256(__file__),
        'sampler': {'max_clips':MAX_CLIPS,'clip_seconds':CLIP_SECONDS,'sample_rate':SAMPLE_RATE,
                    'starts':'Equally spaced nonoverlapping integer sample starts over entire recording'},
        'pooling':'Final hidden frame mean and population std over all selected frames, including between-clip variability',
        'speaker_attribution_verified':False,'validation_test_processed':False,'targets_used':False,
        'versions': {'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__}}
    ch = digest(context); envelope = {'context':context,'context_hash':ch}
    ROOT.mkdir(parents=True,exist_ok=True)
    with output_lock(ROOT), threadpool_limits(limits=2), offline_network():
        protocol = ROOT/'protocol.json'
        if protocol.exists():
            if json.loads(protocol.read_text()) != envelope:
                raise ValueError('WavLM feature context changed')
        else:
            if resume:
                raise ValueError('Resume requires frozen context')
            atomic_json(protocol,envelope)
        processor, model = load_encoder(); rows = []
        (ROOT/'participants').mkdir(exist_ok=True)
        for index,pid in enumerate(ids,1):
            copies = sorted(reports[pid],key=lambda r:(not Path(r['source']).is_dir(),r['source']))
            path = ROOT/'participants'/f'{pid}.json'
            if path.exists():
                row = load_record(path,ch)
            else:
                started = time.monotonic(); blob = read_wav(copies[0]); hash_value = hashlib.sha256(blob).hexdigest()
                for report in copies[1:]:
                    if hashlib.sha256(read_wav(report)).hexdigest() != hash_value:
                        raise ValueError('Duplicate-source WAV content differs')
                row = {'participant_id':pid,'source_keys':[r['source_key'] for r in copies],
                    'audited_source_sha256':{r['source_key']:r['source_sha256'] for r in copies},
                    **encode_blob(blob,processor,model),'seconds':time.monotonic()-started}
                save_record(path,row,ch)
            if row['participant_id'] != pid or row['source_keys'] != [r['source_key'] for r in copies]:
                raise ValueError('Cached recording ownership differs')
            features = np.asarray(row['features'])
            if features.shape != (1536,) or not np.isfinite(features).all() or row['targets_used']:
                raise ValueError('Invalid frozen speech feature cache')
            rows.append(row)
            print(f"WavLM features {index}/{len(ids)}; {row['seconds']:.1f}s/person; coverage {100*row['coverage_fraction']:.1f}%",flush=True)
        if (ROOT/'completion.json').exists():
            return json.loads((ROOT/'completion.json').read_text())
        np.savez_compressed(ROOT/'features.npz',participant_id=np.asarray(ids),speech=np.asarray([r['features'] for r in rows]))
        report = {'complete':True,'participants':len(ids),'physical_sources':sum(map(len,reports.values())),
            'context_hash':ch,'features_sha256':sha256(ROOT/'features.npz'),
            'participant_records':{str(ROOT/'participants'/f'{pid}.json'):sha256(ROOT/'participants'/f'{pid}.json') for pid in ids},
            'riff_overstated_people':sum(r['quality']['riff_size_overstated'] for r in rows),
            'all_pcm_data_complete':all(r['quality']['pcm_data_complete'] for r in rows),
            'coverage_min_median_max':np.quantile([r['coverage_fraction'] for r in rows],[0,.5,1]).tolist(),
            'low_energy_clips':sum(r['low_energy_clips'] for r in rows),'validation_test_processed':False,
            'training_only':True,'raw_source_hash_reverification':'Prior complete archive hashes and unchanged source stamps; targeted WAV bytes now hashed; duplicate WAV copies checked exactly'}
        atomic_json(ROOT/'completion.json',report)
        return report


def load_training(ids):
    protocol = json.loads((ROOT/'protocol.json').read_text()); receipt = json.loads((ROOT/'completion.json').read_text())
    context = protocol['context']; ch = protocol['context_hash']
    if (digest(context) != ch or receipt['context_hash'] != ch or not receipt['complete']
            or sha256(__file__) != context['source_code_sha256'] or ids != context['ids']
            or encoder_contract() != context['encoder'] or sha256(ROOT/'features.npz') != receipt['features_sha256']):
        raise ValueError('Frozen speech context/features changed')
    for path,expected in {**context['source_reports'],**receipt['participant_records']}.items():
        if sha256(path) != expected:
            raise ValueError('Speech feature provenance changed')
    for path in context['source_reports']:
        report = json.loads(Path(path).read_text())
        if source_stamp(Path(report['source'])) != context['source_stamps'][report['source_key']]:
            raise ValueError('Audited source changed')
    with np.load(ROOT/'features.npz',allow_pickle=False) as cache:
        if cache['participant_id'].tolist() != ids:
            raise ValueError('Speech participant alignment changed')
        speech = cache['speech'].copy()
    if speech.shape != (len(ids),1536) or not np.isfinite(speech).all():
        raise ValueError('Invalid speech vectors')
    return speech, {'protocol_sha256':sha256(ROOT/'protocol.json'),'completion_sha256':sha256(ROOT/'completion.json'),
        'features_sha256':receipt['features_sha256'],'source_code_sha256':sha256(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--resume',action='store_true')
    print(json.dumps(prepare(parser.parse_args().resume),indent=2))
