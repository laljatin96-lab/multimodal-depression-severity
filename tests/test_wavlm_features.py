"""WAV decoding and frozen feature mathematics without loading an encoder."""
import struct

import numpy as np
import pytest

from src import wavlm_features as s


def chunk(name, payload):
    return name + struct.pack('<I', len(payload)) + payload + b'\0'*(len(payload) % 2)


def wav(samples=None, *, channels=1, byte_rate=None, before_data=b'', after_data=b'',
        data=None, riff_extra=0):
    if samples is None:
        samples = np.arange(1600, dtype=np.int16)
    if data is None:
        data = np.asarray(samples, dtype='<i2').tobytes()
    block = channels*2
    fmt = struct.pack('<HHIIHH', 1, channels, s.SAMPLE_RATE,
                      s.SAMPLE_RATE*block if byte_rate is None else byte_rate, block, 16)
    body = b'WAVE' + chunk(b'fmt ', fmt) + before_data + chunk(b'data', data) + after_data
    return b'RIFF' + struct.pack('<I', len(body)+riff_extra) + body


def test_pcm16_values_and_channel_averaging_are_preserved():
    mono = np.resize(np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16), 1600)
    decoded, quality = s.decode_pcm(wav(mono))
    np.testing.assert_array_equal(decoded, mono.astype(np.float32)/32768)
    assert quality['samples'] == 1600 and quality['seconds'] == .1
    assert quality['channels'] == 1 and quality['pcm_data_complete']
    assert not quality['riff_size_overstated'] and not quality['source_modified']
    stereo = np.column_stack([mono, -mono.astype(np.int32)//2]).astype(np.int16)
    decoded, quality = s.decode_pcm(wav(stereo, channels=2))
    np.testing.assert_array_equal(decoded, (stereo.astype(np.float32)/32768).mean(axis=1))
    assert quality['channels'] == 2 and quality['samples'] == len(mono)


def test_complete_data_with_overstated_outer_header_is_recorded_without_repair():
    original = wav(riff_extra=200000)
    decoded, quality = s.decode_pcm(original)
    assert quality['riff_size_overstated'] and quality['pcm_data_complete']
    assert quality['declared_riff_bytes'] == len(original)+200000
    assert quality['physical_bytes'] == len(original) and not quality['source_modified']
    np.testing.assert_array_equal(decoded, np.arange(1600, dtype=np.float32)/32768)
    assert original == wav(riff_extra=200000)


@pytest.mark.parametrize('removed', [1, 9, 100])
def test_truncated_data_is_rejected_even_when_outer_header_is_overstated(removed):
    with pytest.raises(ValueError, match='Truncated WAV chunk payload'):
        s.decode_pcm(wav(riff_extra=200000)[:-removed])


def test_inconsistent_byte_rate_is_rejected():
    with pytest.raises(ValueError, match='Inconsistent PCM'):
        s.decode_pcm(wav(byte_rate=12345))


@pytest.mark.parametrize('name,payload', [
    (b'fmt ', struct.pack('<HHIIHH', 1, 1, 16000, 32000, 2, 16)),
    (b'data', np.zeros(1600, dtype='<i2').tobytes()),
])
def test_duplicate_format_and_data_chunks_are_rejected(name, payload):
    with pytest.raises(ValueError, match='Duplicate WAV'):
        s.decode_pcm(wav(after_data=chunk(name, payload)))


def test_odd_pcm_data_is_not_completed_by_riff_padding():
    # A valid RIFF pad byte is outside the PCM chunk and cannot complete a sample.
    with pytest.raises(ValueError, match='Inconsistent PCM'):
        s.decode_pcm(wav(data=b'\0'*3201))


def test_unknown_odd_chunk_is_skipped_using_its_riff_padding():
    original = wav(before_data=chunk(b'JUNK', b'abc'))
    decoded, quality = s.decode_pcm(original)
    assert len(decoded) == 1600 and quality['pcm_data_complete']


def test_outer_header_cannot_hide_supplied_bytes():
    value = bytearray(wav())
    struct.pack_into('<I', value, 4, len(value)-10)
    with pytest.raises(ValueError, match='RIFF length excludes'):
        s.decode_pcm(bytes(value))


@pytest.mark.parametrize('seconds,count,width_seconds', [
    (.1, 1, .1), (3.5, 1, 3.5), (4, 1, 4), (7.9, 1, 4),
    (8, 2, 4), (13, 3, 4), (256, 64, 4), (1200, 64, 4),
])
def test_sampling_is_deterministic_bounded_and_nonoverlapping(seconds, count, width_seconds):
    samples = int(seconds*s.SAMPLE_RATE)
    starts, width = s.clip_starts(samples)
    repeat, repeat_width = s.clip_starts(np.int64(samples))
    np.testing.assert_array_equal(starts, repeat)
    assert width == repeat_width == int(width_seconds*s.SAMPLE_RATE)
    assert len(starts) == len(set(starts.tolist())) == count
    assert starts[0] == 0 and starts[-1]+width <= samples
    assert count*width <= samples
    if count > 1:
        assert starts[-1]+width == samples
        assert (np.diff(starts) >= width).all()
        assert np.diff(starts).max()-np.diff(starts).min() <= 1


@pytest.mark.parametrize('samples', [0, 1599, 1600.0, -1])
def test_invalid_or_insufficient_sampling_inputs_are_rejected(samples):
    with pytest.raises(ValueError, match='Insufficient waveform'):
        s.clip_starts(samples)


def test_pooled_std_includes_between_clip_variation_and_frame_count_weights():
    a = np.ones((2, 768))
    b = np.full((6, 768), 3.)
    features, count = s.pooled_moments([a, b])
    expected = np.concatenate([a, b])
    assert count == 8 and features.shape == (1536,)
    np.testing.assert_allclose(features[:768], expected.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(features[768:], expected.std(axis=0), atol=1e-15)
    assert (features[768:] > 0).all()  # Both individual clip standard deviations are zero.


def test_pooling_is_independent_of_frame_partition_and_clip_order():
    frames = np.random.default_rng(71).normal(size=(17, 768)).astype(np.float32)
    together, count = s.pooled_moments([frames])
    partitioned, other_count = s.pooled_moments([frames[8:], frames[:3], frames[3:8]])
    assert count == other_count == 17
    np.testing.assert_allclose(together, partitioned, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(together[:768], frames.astype(float).mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(together[768:], frames.astype(float).std(axis=0), atol=1e-14)


@pytest.mark.parametrize('arrays', [[], [np.zeros((0, 768))], [np.zeros((2, 767))],
                                    [np.full((2, 768), np.nan)], [np.ones(768)]])
def test_invalid_hidden_frames_are_rejected(arrays):
    with pytest.raises(ValueError, match='hidden frames'):
        s.pooled_moments(arrays)
