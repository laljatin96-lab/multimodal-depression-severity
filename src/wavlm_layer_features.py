"""Two fixed frozen audio representations; omit exactly-zero sampled clips only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import wavlm_features as core
from . import wavlm_features_complete as previous
from .batch_inference import atomic_json, load_record, save_record, output_lock
from .common import digest
from .final_evaluation import sha256

ROOT = Path('results/wavlm_layer_features')
REPRESENTATIONS = ('final_nonzero', 'layer6_nonzero')
encoder_contract = previous.encoder_contract
runtime_contract = previous.runtime_contract


def nonzero_clips(waveform):
    """Retain original sampling positions, including arbitrarily quiet nonzero audio."""
    starts, width = core.clip_starts(len(waveform))
    keep = np.array([np.any(waveform[s:s+width] != 0) for s in starts], dtype=bool)
    if not keep.any():
        raise ValueError('All sampled clips are exactly zero; supply usable audio')
    return starts, width, keep


def load_encoder(representation):
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown frozen representation')
    processor, model = core.load_encoder()
    if representation == 'layer6_nonzero':
        model.encoder.layers = model.encoder.layers[:6]
    return processor, model


def encode_waveform(waveform, processor, model, both=False):
    import torch
    starts, width, keep = nonzero_clips(waveform)
    arrays = []; middle = []
    with torch.inference_mode():
        for start in starts[keep]:
            inputs = processor(waveform[start:start+width], sampling_rate=core.SAMPLE_RATE, return_tensors='pt')
            output = model(**inputs, output_hidden_states=both)
            arrays.append(output.last_hidden_state[0].cpu().numpy())
            if both:
                if len(output.hidden_states) != 13:
                    raise ValueError('Expected full twelve-layer encoder')
                middle.append(output.hidden_states[6][0].cpu().numpy())
    features, frames = core.pooled_moments(arrays)
    row = {'features': features.tolist(), 'encoder_frames': frames,
        'original_clip_start_samples': starts.tolist(), 'clip_start_samples': starts[keep].tolist(),
        'removed_clip_start_samples': starts[~keep].tolist(), 'clip_samples': width,
        'clip_start_sample_rate': core.SAMPLE_RATE, 'exact_zero_clips_removed': int((~keep).sum()),
        'unique_covered_seconds': int(keep.sum())*width/core.SAMPLE_RATE,
        'coverage_fraction': int(keep.sum())*width/len(waveform),
        'targets_used': False, 'alignment_used': False,
        'speaker_identity': 'Unverified supplied whole-interview recording'}
    if both:
        vector, middle_frames = core.pooled_moments(middle)
        if middle_frames != frames:
            raise ValueError('Hidden layer frame ownership differs')
        row['layer6_features'] = vector.tolist()
    return row


def encode_blob(blob, representation, processor, model):
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown frozen representation')
    expected = 6 if representation == 'layer6_nonzero' else 12
    if len(model.encoder.layers) != expected:
        raise ValueError('Encoder layer count differs from inference representation')
    waveform, quality = previous.decode_compatible_pcm(blob)
    return {**encode_waveform(waveform, processor, model), 'quality': quality,
        'raw_wav_sha256': hashlib.sha256(blob).hexdigest(), 'representation': representation}


def layer_probe(waveform, processor, model):
    """Check the shortened encoder against the corresponding full hidden state."""
    import torch
    starts, width, keep = nonzero_clips(waveform)
    inputs = processor(waveform[starts[keep][0]:starts[keep][0]+width],
        sampling_rate=core.SAMPLE_RATE, return_tensors='pt')
    layers = model.encoder.layers
    with torch.inference_mode():
        full = model(**inputs, output_hidden_states=True)
        model.encoder.layers = layers[:6]
        try:
            prefix = model(**inputs, output_hidden_states=True)
        finally:
            model.encoder.layers = layers
    a = full.hidden_states[6].cpu().numpy(); b = prefix.last_hidden_state.cpu().numpy()
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    if len(full.hidden_states) != 13 or len(prefix.hidden_states) != 7:
        raise ValueError('Unexpected hidden state indexing')
    return {'full_hidden_state_count': 13, 'prefix_hidden_state_count': 7,
        'full_hidden_state_index': 6, 'maximum_absolute_difference': float(np.abs(a-b).max()),
        'absolute_tolerance': 1e-6, 'targets_used': False}


def dependencies():
    old_protocol = json.loads((previous.ROOT/'protocol.json').read_text())
    old_context = old_protocol['context']; ids = old_context['ids']
    vectors, provenance = previous.load_training(ids)
    old_rows = {pid: load_record(previous.ROOT/'participants'/f'{pid}.json', old_protocol['context_hash']) for pid in ids}
    reports = {}
    for path in old_context['source_reports']:
        report = json.loads(Path(path).read_text()); reports[report['source_key']] = report
    context = {'protocol': 'fixed-nonzero-final-and-layer6-frozen-wavlm-v1', 'ids': ids,
        'source_code_sha256': sha256(__file__), 'core_source_sha256': sha256(core.__file__),
        'previous_provenance': provenance, 'encoder': encoder_contract(), 'versions': runtime_contract(),
        'old_participant_records': {str(previous.ROOT/'participants'/f'{pid}.json'): sha256(previous.ROOT/'participants'/f'{pid}.json') for pid in ids},
        'representations': list(REPRESENTATIONS), 'middle_hidden_state_index': 6,
        'clip_policy': 'Original fixed up-to64 four-second positions; omit iff every decoded resampled mono sample equals exactly zero; never replace positions',
        'pooling': 'Global mean and population standard deviation of all retained frames',
        'targets_used': False, 'training_participants_only': True, 'speaker_identity_verified': False,
        'nonzero_final_reuse': 'Bit-identical previous final-layer vector when all original clips are nonzero'}
    return ids, vectors, old_rows, reports, context


def prepare(resume=False):
    from threadpoolctl import threadpool_limits
    from .encoder_kernel_accuracy import offline_network
    if ROOT.exists() and not resume:
        raise FileExistsError('Preserving feature evidence; use --resume')
    ids, vectors, old_rows, reports, context = dependencies(); ch = digest(context)
    envelope = {'context': context, 'context_hash': ch}
    ROOT.mkdir(parents=True, exist_ok=True)
    with output_lock(ROOT), threadpool_limits(limits=2), offline_network():
        path = ROOT/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError('Feature context changed')
        else:
            if resume:
                raise ValueError('Missing feature context')
            atomic_json(path, envelope)
        (ROOT/'participants').mkdir(exist_ok=True)
        if (ROOT/'completion.json').exists():
            load_training(ids)
            return json.loads((ROOT/'completion.json').read_text())
        processor, model = core.load_encoder(); full_layers = model.encoder.layers; rows = []
        for i, pid in enumerate(ids):
            row_path = ROOT/'participants'/f'{pid}.json'; old = old_rows[pid]
            if row_path.exists():
                row = load_record(row_path, ch)
            else:
                started = time.monotonic()
                blob = core.read_wav(reports[old['source_keys'][0]])
                if hashlib.sha256(blob).hexdigest() != old['raw_wav_sha256']:
                    raise ValueError('Original raw recording hash differs')
                waveform, quality = previous.decode_compatible_pcm(blob)
                starts, width, keep = nonzero_clips(waveform)
                if (starts.tolist() != old['clip_start_samples'] or width != old['clip_samples']
                        or (~keep).tolist() != (np.asarray(old['clip_rms']) == 0).tolist()):
                    raise ValueError('Verified original sampler/zero-mask changed')
                if not (ROOT/'layer_probe.json').exists():
                    model.encoder.layers = full_layers
                    atomic_json(ROOT/'layer_probe.json', {**layer_probe(waveform, processor, model),
                        'context_hash': ch, 'raw_wav_sha256': old['raw_wav_sha256']})
                reuse = bool(keep.all()); model.encoder.layers = full_layers[:6] if reuse else full_layers
                encoded = encode_waveform(waveform, processor, model, both=not reuse)
                final = vectors[i].tolist() if reuse else encoded['features']
                middle = encoded['features'] if reuse else encoded.pop('layer6_features')
                row = {**encoded, 'participant_id': pid, 'source_keys': old['source_keys'],
                    'raw_wav_sha256': old['raw_wav_sha256'], 'quality': quality,
                    'old_participant_record_sha256': context['old_participant_records'][str(previous.ROOT/'participants'/f'{pid}.json')],
                    'final_nonzero': final, 'layer6_nonzero': middle,
                    'final_features_reused': reuse, 'seconds': time.monotonic()-started}
                row.pop('features')
                save_record(row_path, row, ch)
            if row['participant_id'] != pid or row['targets_used'] or row['raw_wav_sha256'] != old['raw_wav_sha256']:
                raise ValueError('Feature ownership changed')
            for key in REPRESENTATIONS:
                if np.asarray(row[key]).shape != (1536,) or not np.isfinite(row[key]).all():
                    raise ValueError('Invalid frozen features')
            rows.append(row)
            print(f"Audio layer features {i+1}/{len(ids)}; zero_clips={row['exact_zero_clips_removed']}; final_reused={row['final_features_reused']}", flush=True)
        np.savez_compressed(ROOT/'features.npz', participant_id=np.asarray(ids),
            **{key: np.asarray([r[key] for r in rows]) for key in REPRESENTATIONS})
        result = {'complete': True, 'context_hash': ch, 'participants': len(ids),
            'physical_sources': sum(len(r['source_keys']) for r in rows),
            'features_sha256': sha256(ROOT/'features.npz'),
            'participant_records': {str(ROOT/'participants'/f'{pid}.json'): sha256(ROOT/'participants'/f'{pid}.json') for pid in ids},
            'layer_probe_sha256': sha256(ROOT/'layer_probe.json'),
            'original_sampled_clips': sum(len(r['original_clip_start_samples']) for r in rows),
            'retained_sampled_clips': sum(len(r['clip_start_samples']) for r in rows),
            'exact_zero_clips_removed': sum(r['exact_zero_clips_removed'] for r in rows),
            'participants_with_zero_clips': sum(r['exact_zero_clips_removed'] > 0 for r in rows),
            'final_vectors_reused': sum(r['final_features_reused'] for r in rows),
            'final_vectors_recomputed': sum(not r['final_features_reused'] for r in rows),
            'new_layer6_vectors': len(ids), 'dropped_participants': 0,
            'training_only': True, 'validation_test_processed': False, 'targets_used': False}
        atomic_json(ROOT/'completion.json', result)
        return result


def load_training(ids):
    protocol = json.loads((ROOT/'protocol.json').read_text()); receipt = json.loads((ROOT/'completion.json').read_text())
    context = protocol['context']; ch = protocol['context_hash']
    if (digest(context) != ch or receipt['context_hash'] != ch or not receipt['complete']
            or ids != context['ids'] or sha256(__file__) != context['source_code_sha256']
            or sha256(core.__file__) != context['core_source_sha256']
            or encoder_contract() != context['encoder'] or runtime_contract() != context['versions']
            or sha256(ROOT/'features.npz') != receipt['features_sha256']
            or sha256(ROOT/'layer_probe.json') != receipt['layer_probe_sha256']):
        raise ValueError('Layer feature context changed')
    _, provenance = previous.load_training(ids)
    if provenance != context['previous_provenance']:
        raise ValueError('Previous frozen feature dependency changed')
    for path, expected in {**receipt['participant_records'], **context['old_participant_records']}.items():
        if sha256(path) != expected:
            raise ValueError('Participant feature dependency changed')
    with np.load(ROOT/'features.npz', allow_pickle=False) as cache:
        if cache['participant_id'].tolist() != ids:
            raise ValueError('Participant alignment changed')
        features = {key: cache[key].copy() for key in REPRESENTATIONS}
    if any(a.shape != (len(ids), 1536) or not np.isfinite(a).all() for a in features.values()):
        raise ValueError('Invalid frozen feature array')
    return features, {'protocol_sha256': sha256(ROOT/'protocol.json'),
        'completion_sha256': sha256(ROOT/'completion.json'), 'features_sha256': receipt['features_sha256'],
        'source_code_sha256': sha256(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--resume', action='store_true')
    print(json.dumps(prepare(parser.parse_args().resume), indent=2))
