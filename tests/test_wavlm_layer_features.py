"""Deterministic clip ownership and layer indexing using a tiny fake encoder."""
import hashlib
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import wavlm_layer_features as s


class Processor:
    def __init__(self):
        self.clips = []

    def __call__(self, waveform, *, sampling_rate, return_tensors):
        assert sampling_rate == 16000 and return_tensors == 'pt'
        self.clips.append(waveform.copy())
        return {'input_values': torch.as_tensor(waveform)[None, :]}


class Encoder:
    """An explicit residual stack with one hidden state before each layer."""
    def __init__(self, layers=12, corrupt_prefix=False):
        self.encoder = SimpleNamespace(layers=list(range(layers)))
        self.corrupt_prefix = corrupt_prefix
        self.calls = 0

    def __call__(self, input_values, *, output_hidden_states=False):
        self.calls += 1
        n = input_values.shape[1]
        hidden = input_values[:, [0, n//2, n-1], None].repeat(1, 1, 768)
        states = [hidden]
        for layer in self.encoder.layers:
            hidden = hidden+(layer+1)/4
            states.append(hidden)
        if self.corrupt_prefix and len(self.encoder.layers) == 6:
            hidden = hidden+1
        return SimpleNamespace(last_hidden_state=hidden,
                               hidden_states=tuple(states) if output_hidden_states else None)


def waveform():
    width = 16000*4
    return np.concatenate([np.ones(width, np.float32), np.zeros(width, np.float32),
                           np.full(width, 3, np.float32)])


def test_exact_zero_selection_keeps_all_quiet_nonzero_and_original_positions():
    wave = waveform()
    wave[:64000] = np.float32(1e-9)  # Below the previous descriptive RMS threshold.
    wave[-64000:] = np.nextafter(np.float32(0), np.float32(1))
    starts, width, keep = s.nonzero_clips(wave)
    np.testing.assert_array_equal(starts, [0, 64000, 128000])
    np.testing.assert_array_equal(keep, [True, False, True])
    assert width == 64000


def test_all_zero_audio_fails_explicitly_before_any_encoder_call():
    processor, encoder = Processor(), Encoder()
    with pytest.raises(ValueError, match='All sampled clips are exactly zero'):
        s.encode_waveform(np.zeros(16000, np.float32), processor, encoder)
    assert not processor.clips and encoder.calls == 0


def test_short_quiet_nonzero_audio_is_retained_without_padding_or_replacement():
    wave = np.full(3200, 1e-12, np.float32)
    starts, width, keep = s.nonzero_clips(wave)
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(keep, [True])
    assert width == len(wave)
    processor, encoder = Processor(), Encoder()
    row = s.encode_waveform(wave, processor, encoder)
    assert len(processor.clips) == 1 and len(processor.clips[0]) == len(wave)
    assert row['clip_start_samples'] == [0] and not row['removed_clip_start_samples']
    assert row['coverage_fraction'] == 1


def test_both_layer_poolings_share_retained_frames_and_include_between_clip_variance():
    processor, encoder = Processor(), Encoder()
    row = s.encode_waveform(waveform(), processor, encoder, both=True)
    assert encoder.calls == len(processor.clips) == 2
    assert row['original_clip_start_samples'] == [0, 64000, 128000]
    assert row['clip_start_samples'] == [0, 128000]
    assert row['removed_clip_start_samples'] == [64000]
    assert row['exact_zero_clips_removed'] == 1 and row['encoder_frames'] == 6
    assert row['unique_covered_seconds'] == 8 and row['coverage_fraction'] == 2/3
    final, middle = np.asarray(row['features']), np.asarray(row['layer6_features'])
    np.testing.assert_array_equal(final[:768], 2+sum(range(1, 13))/4)
    np.testing.assert_array_equal(middle[:768], 2+sum(range(1, 7))/4)
    np.testing.assert_array_equal(final[768:], 1)
    np.testing.assert_array_equal(middle[768:], 1)
    assert not row['targets_used'] and not row['alignment_used']


def test_prefix_encoder_pooled_vector_matches_full_hidden_state_six():
    full = s.encode_waveform(waveform(), Processor(), Encoder(), both=True)
    prefix = s.encode_waveform(waveform(), Processor(), Encoder(layers=6))
    np.testing.assert_array_equal(prefix['features'], full['layer6_features'])
    assert prefix['clip_start_samples'] == full['clip_start_samples']
    assert prefix['encoder_frames'] == full['encoder_frames']


def test_layer_probe_verifies_hidden_index_and_restores_original_layers():
    processor, encoder = Processor(), Encoder()
    layers = encoder.encoder.layers
    row = s.layer_probe(waveform(), processor, encoder)
    assert encoder.encoder.layers is layers
    assert len(layers) == 12 and encoder.calls == 2
    assert row['full_hidden_state_count'] == 13 and row['prefix_hidden_state_count'] == 7
    assert row['full_hidden_state_index'] == 6 and row['maximum_absolute_difference'] == 0
    assert row['absolute_tolerance'] == 1e-6 and not row['targets_used']


def test_probe_failure_also_restores_model_layer_ownership():
    encoder = Encoder(corrupt_prefix=True)
    layers = encoder.encoder.layers
    with pytest.raises(AssertionError):
        s.layer_probe(waveform(), Processor(), encoder)
    assert encoder.encoder.layers is layers and len(layers) == 12


def test_both_mode_rejects_unexpected_hidden_state_count():
    with pytest.raises(ValueError, match='Expected full twelve-layer encoder'):
        s.encode_waveform(waveform(), Processor(), Encoder(layers=6), both=True)


@pytest.mark.parametrize('representation,layer_count', [('final_nonzero', 12), ('layer6_nonzero', 6)])
def test_loading_only_shortens_the_requested_representation(monkeypatch, representation, layer_count):
    processor, encoder = Processor(), Encoder()
    monkeypatch.setattr(s.core, 'load_encoder', lambda: (processor, encoder))
    returned_processor, returned_encoder = s.load_encoder(representation)
    assert returned_processor is processor and returned_encoder is encoder
    assert len(encoder.encoder.layers) == layer_count


def test_unknown_representation_and_wrong_layer_count_fail_before_decoding(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Wrong representation must fail before raw decoding')
    monkeypatch.setattr(s.previous, 'decode_compatible_pcm', forbidden)
    with pytest.raises(ValueError, match='Unknown frozen representation'):
        s.load_encoder('unknown')
    with pytest.raises(ValueError, match='Unknown frozen representation'):
        s.encode_blob(b'not audio', 'unknown', Processor(), Encoder())
    with pytest.raises(ValueError, match='Encoder layer count differs'):
        s.encode_blob(b'not audio', 'layer6_nonzero', Processor(), Encoder(layers=12))


def test_blob_path_records_source_hash_without_assigning_speaker_identity():
    samples = np.concatenate([np.full(64000, 100, '<i2'), np.zeros(64000, '<i2')])
    fmt = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, 16000, 32000, 2, 16)
    data = b'data'+struct.pack('<I', samples.nbytes)+samples.tobytes()
    blob = b'RIFF'+struct.pack('<I', 4+len(fmt)+len(data))+b'WAVE'+fmt+data
    before = hashlib.sha256(blob).hexdigest()
    row = s.encode_blob(blob, 'layer6_nonzero', Processor(), Encoder(layers=6))
    assert row['raw_wav_sha256'] == hashlib.sha256(blob).hexdigest() == before
    assert row['representation'] == 'layer6_nonzero' and row['exact_zero_clips_removed'] == 1
    assert row['quality']['pcm_data_complete'] and not row['quality']['source_modified']
    assert 'Unverified' in row['speaker_identity'] and not row['targets_used']
