"""Complete mixed-rate WAV extraction while preserving valid frozen computations."""
import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
import struct
import time

import numpy as np
from scipy.signal import resample_poly

from . import wavlm_features as original
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .final_evaluation import sha256
from .full_dataset import source_stamp

ROOT = Path('results/wavlm_features_complete')
ENCODER = original.ENCODER
encoder_contract = original.encoder_contract
load_encoder = original.load_encoder


def runtime_contract():
    import scipy, torch, transformers
    return {'numpy':np.__version__,'scipy':scipy.__version__,
            'torch':torch.__version__,'transformers':transformers.__version__}


def decode_compatible_pcm(blob):
    """Validate actual PCM16 format and resample supported rates deterministically.

    The legacy strict parser is retained unchanged. For other valid rates, a
    temporary header permits that same structural/sample validation. Only the
    decoded waveform is resampled; source bytes and files are never changed.
    """
    if len(blob) < 44 or blob[:4] != b'RIFF' or blob[8:12] != b'WAVE':
        raise ValueError('Expected a bounded RIFF/WAVE input')
    position = 12; fmt = None
    while position < len(blob):
        if position+8 > len(blob):
            raise ValueError('Truncated WAV chunk header')
        name, size = struct.unpack_from('<4sI', blob, position); position += 8
        if position+size > len(blob):
            raise ValueError('Truncated WAV chunk payload')
        if name == b'fmt ':
            if size < 16:
                raise ValueError('Incomplete WAV format')
            fmt = position
            break
        position += size+(size % 2)
    if fmt is None:
        raise ValueError('Missing WAV format')
    codec, channels, rate, byte_rate, block, bits = struct.unpack_from('<HHIIHH', blob, fmt)
    if (codec != 1 or bits != 16 or channels not in (1,2)
            or rate not in (8000,16000,22050,24000,32000,44100,48000,96000)):
        raise ValueError('Unsupported original PCM format')
    if block != 2*channels or byte_rate != rate*block:
        raise ValueError('Inconsistent original PCM byte/sample counts')
    if rate == original.SAMPLE_RATE:
        waveform, quality = original.decode_pcm(blob)
    else:
        temporary = bytearray(blob)
        struct.pack_into('<I', temporary, fmt+4, original.SAMPLE_RATE)
        struct.pack_into('<I', temporary, fmt+8, original.SAMPLE_RATE*block)
        waveform, quality = original.decode_pcm(bytes(temporary))
        quality['sample_rate'] = rate
        quality['seconds'] = quality['samples']/rate
    divisor = gcd(rate, original.SAMPLE_RATE)
    up, down = original.SAMPLE_RATE//divisor, rate//divisor
    if up != down:
        waveform = resample_poly(waveform, up, down, window=('kaiser',5.0), padtype='constant')
    if not np.isfinite(waveform).all() or len(waveform) < original.SAMPLE_RATE//10:
        raise ValueError('Insufficient or invalid resampled waveform')
    quality.update({'encoder_sample_rate':original.SAMPLE_RATE,'encoder_samples':len(waveform),
        'resampled':rate != original.SAMPLE_RATE,
        'resampling':{'method':'scipy.signal.resample_poly','up':up,'down':down,
                      'window':['kaiser',5.0],'padtype':'constant'},
        'decoder_header_adjusted_in_memory':rate != original.SAMPLE_RATE,'source_modified':False})
    return waveform, quality


def encode_blob(blob, processor, model):
    import torch
    waveform, quality = decode_compatible_pcm(blob)
    starts, width = original.clip_starts(len(waveform)); arrays = []; rms = []; clipped = []
    with torch.inference_mode():
        for start in starts:
            clip = waveform[start:start+width]
            rms.append(float(np.sqrt(np.mean(np.square(clip,dtype=np.float64)))))
            clipped.append(float((np.abs(clip)>=.999).mean()))
            inputs = processor(clip,sampling_rate=original.SAMPLE_RATE,return_tensors='pt')
            arrays.append(model(**inputs).last_hidden_state[0].cpu().numpy())
    features,frames = original.pooled_moments(arrays)
    return {'features':features.tolist(),'raw_wav_sha256':hashlib.sha256(blob).hexdigest(),'quality':quality,
        'clip_start_samples':starts.tolist(),'clip_start_sample_rate':original.SAMPLE_RATE,'clip_samples':width,
        'unique_covered_seconds':len(starts)*width/original.SAMPLE_RATE,
        'coverage_fraction':len(starts)*width/len(waveform),'encoder_frames':frames,'clip_rms':rms,
        'clip_clipped_fraction':clipped,'low_energy_clips':sum(v<1e-4 for v in rms),
        'speaker_identity':'Unverified supplied whole-interview recording','alignment_used':False,'targets_used':False}


def prepare(resume=False):
    from threadpoolctl import threadpool_limits
    from .encoder_kernel_accuracy import offline_network
    if ROOT.exists() and not resume:
        raise FileExistsError('Preserving completed mixed-rate feature evidence')
    previous = json.loads((original.ROOT/'protocol.json').read_text()); old = previous['context']; old_hash = previous['context_hash']
    if (digest(old) != old_hash or sha256(original.__file__) != old['source_code_sha256']
            or encoder_contract() != old['encoder']
            or sha256('data/metadata/participants.csv') != old['metadata_sha256']):
        raise ValueError('Original extraction inputs changed')
    ids = old['ids']; reports = {pid:[] for pid in ids}
    for path, expected in old['source_reports'].items():
        if sha256(path) != expected:
            raise ValueError('Source integrity receipt changed')
        r = json.loads(Path(path).read_text())
        if source_stamp(Path(r['source'])) != old['source_stamps'][r['source_key']]:
            raise ValueError('Original audio source changed')
        reports[r['participant_id']].append(r)
    reused = {pid:sha256(original.ROOT/'participants'/f'{pid}.json') for pid in ids
              if (original.ROOT/'participants'/f'{pid}.json').exists()}
    context = {**old,'protocol':'frozen-wavlm-mixed-rate-completion-v1','versions':runtime_contract(),
        'source_code_sha256':sha256(__file__),'legacy_source_code_sha256':sha256(original.__file__),
        'legacy_protocol_sha256':sha256(original.ROOT/'protocol.json'),'legacy_context_hash':old_hash,
        'reused_participant_records':reused,
        'format_correction':'Strict PCM validation; deterministic polyphase resampling to16kHz for valid original non16kHz recordings; source bytes preserved',
        'resampling':{'method':'scipy.signal.resample_poly','window':['kaiser',5.0],'padtype':'constant'}}
    ch = digest(context); envelope = {'context':context,'context_hash':ch}
    ROOT.mkdir(parents=True,exist_ok=True)
    with output_lock(ROOT),threadpool_limits(limits=2),offline_network():
        protocol = ROOT/'protocol.json'
        if protocol.exists():
            if json.loads(protocol.read_text()) != envelope:
                raise ValueError('Mixed-rate completion context changed')
        else:
            if resume:
                raise ValueError('Missing mixed-rate completion context')
            atomic_json(protocol,envelope)
        processor,model = load_encoder(); (ROOT/'participants').mkdir(exist_ok=True);rows=[]
        for index,pid in enumerate(ids,1):
            copies = sorted(reports[pid],key=lambda r:(not Path(r['source']).is_dir(),r['source']))
            path = ROOT/'participants'/f'{pid}.json'
            if path.exists():
                row = load_record(path,ch)
            elif pid in reused:
                old_path = original.ROOT/'participants'/f'{pid}.json'
                if sha256(old_path) != reused[pid]:
                    raise ValueError('Valid completed computation changed')
                row = {**load_record(old_path,old_hash),'reused_feature_record_sha256':reused[pid],
                    'clip_start_sample_rate':original.SAMPLE_RATE,'new_encoder_runs':0}
                save_record(path,row,ch)
            else:
                started=time.monotonic();blob=original.read_wav(copies[0]);h=hashlib.sha256(blob).hexdigest()
                for report in copies[1:]:
                    if hashlib.sha256(original.read_wav(report)).hexdigest()!=h:
                        raise ValueError('Duplicate recording content differs')
                row={'participant_id':pid,'source_keys':[r['source_key'] for r in copies],
                    'audited_source_sha256':{r['source_key']:r['source_sha256'] for r in copies},
                    **encode_blob(blob,processor,model),'seconds':time.monotonic()-started,'new_encoder_runs':1}
                save_record(path,row,ch)
            if (row['participant_id']!=pid or row['source_keys']!=[r['source_key'] for r in copies]
                    or np.asarray(row['features']).shape!=(1536,) or not np.isfinite(row['features']).all()
                    or row['targets_used'] or not row['quality']['pcm_data_complete']):
                raise ValueError('Invalid recovered feature ownership/data')
            rows.append(row)
            if pid not in reused or index%25==0:
                print(f"Completed WavLM {index}/{len(ids)}; reused={pid in reused}; {row['quality']['sample_rate']}Hz",flush=True)
        if (ROOT/'completion.json').exists():
            return json.loads((ROOT/'completion.json').read_text())
        np.savez_compressed(ROOT/'features.npz',participant_id=np.asarray(ids),speech=np.asarray([r['features'] for r in rows]))
        result={'complete':True,'participants':len(ids),'physical_sources':sum(map(len,reports.values())),
            'context_hash':ch,'features_sha256':sha256(ROOT/'features.npz'),
            'participant_records':{str(ROOT/'participants'/f'{pid}.json'):sha256(ROOT/'participants'/f'{pid}.json') for pid in ids},
            'riff_overstated_people':sum(r['quality']['riff_size_overstated'] for r in rows),
            'all_pcm_data_complete':all(r['quality']['pcm_data_complete'] for r in rows),
            'coverage_min_median_max':np.quantile([r['coverage_fraction'] for r in rows],[0,.5,1]).tolist(),
            'low_energy_clips':sum(r['low_energy_clips'] for r in rows),'validation_test_processed':False,
            'training_only':True,'reused_encoder_computations':len(reused),'new_encoder_computations':len(ids)-len(reused),
            'resampled_people':sum(r['quality'].get('resampled',False) for r in rows),
            'source_sample_rates':{str(rate):sum(r['quality']['sample_rate']==rate for r in rows)
                for rate in sorted({r['quality']['sample_rate'] for r in rows})},
            'raw_source_hash_reverification':'Prior complete archive hashes and unchanged source stamps; targeted WAV bytes hashed and duplicate copies checked'}
        atomic_json(ROOT/'completion.json',result)
        return result


def load_training(ids):
    protocol=json.loads((ROOT/'protocol.json').read_text());receipt=json.loads((ROOT/'completion.json').read_text())
    context=protocol['context'];ch=protocol['context_hash']
    if (digest(context)!=ch or receipt['context_hash']!=ch or not receipt['complete'] or ids!=context['ids']
            or sha256(__file__)!=context['source_code_sha256']
            or sha256(original.__file__)!=context['legacy_source_code_sha256']
            or sha256(original.ROOT/'protocol.json')!=context['legacy_protocol_sha256']
            or sha256('data/metadata/participants.csv')!=context['metadata_sha256']
            or encoder_contract()!=context['encoder'] or runtime_contract()!=context['versions']
            or sha256(ROOT/'features.npz')!=receipt['features_sha256']):
        raise ValueError('Completed speech feature context changed')
    checks={**context['source_reports'],**receipt['participant_records']}
    checks.update({str(original.ROOT/'participants'/f'{pid}.json'):h for pid,h in context['reused_participant_records'].items()})
    for path,expected in checks.items():
        if sha256(path)!=expected:
            raise ValueError('Speech feature provenance changed')
    for path in context['source_reports']:
        r=json.loads(Path(path).read_text())
        if source_stamp(Path(r['source']))!=context['source_stamps'][r['source_key']]:
            raise ValueError('Original recording source changed')
    with np.load(ROOT/'features.npz',allow_pickle=False) as cache:
        if cache['participant_id'].tolist()!=ids:
            raise ValueError('Speech participant alignment differs')
        speech=cache['speech'].copy()
    if speech.shape!=(len(ids),1536) or not np.isfinite(speech).all():
        raise ValueError('Invalid speech vectors')
    return speech,{'protocol_sha256':sha256(ROOT/'protocol.json'),'completion_sha256':sha256(ROOT/'completion.json'),
        'features_sha256':receipt['features_sha256'],'source_code_sha256':sha256(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true')
    print(json.dumps(prepare(parser.parse_args().resume),indent=2))
