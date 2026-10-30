"""Mixed-rate PCM recovery without loading the encoder or fitting models."""
import hashlib
import struct

import numpy as np
import pytest
from scipy.signal import resample_poly

from src import wavlm_features as original
from src import wavlm_features_complete as s


def chunk(name, payload):
    return name+struct.pack('<I', len(payload))+payload+b'\0'*(len(payload) % 2)


def wav(samples, *, rate=48000, channels=2, codec=1, bits=16,
        byte_rate=None, before_fmt=b'', before_data=b'', after_data=b'', data=None):
    block = channels*bits//8
    fmt = struct.pack('<HHIIHH', codec, channels, rate,
                      rate*block if byte_rate is None else byte_rate, block, bits)
    if data is None:
        data = np.asarray(samples, dtype='<i2').tobytes()
    body = b'WAVE'+before_fmt+chunk(b'fmt ', fmt)+before_data+chunk(b'data', data)+after_data
    return b'RIFF'+struct.pack('<I', len(body))+body


def stereo_frames(count=4800):
    t = np.arange(count)/48000
    left = np.round(10000*np.sin(2*np.pi*431*t)).astype(np.int16)
    right = np.round(6000*np.cos(2*np.pi*137*t)).astype(np.int16)
    return np.column_stack([left, right])


@pytest.mark.parametrize('channels', [1, 2])
def test_16khz_passthrough_is_exactly_legacy_waveform(channels, monkeypatch):
    values = np.random.default_rng(6).integers(-20000, 20000, (3200, channels), dtype=np.int16)
    blob = wav(values, rate=16000, channels=channels)
    expected, legacy = original.decode_pcm(blob)
    def forbidden(*args, **kwargs):
        raise AssertionError('16kHz data must not be resampled')
    monkeypatch.setattr(s, 'resample_poly', forbidden)
    actual, quality = s.decode_compatible_pcm(blob)
    np.testing.assert_array_equal(actual, expected)
    for key, value in legacy.items():
        assert quality[key] == value
    assert not quality['resampled'] and not quality['decoder_header_adjusted_in_memory']
    assert quality['sample_rate'] == quality['encoder_sample_rate'] == 16000
    assert quality['samples'] == quality['encoder_samples'] == 3200
    assert quality['resampling']['up'] == quality['resampling']['down'] == 1


def test_48khz_stereo_averaging_and_polyphase_resampling_are_deterministic():
    frames = stereo_frames(24000)
    blob = wav(frames)
    digest = hashlib.sha256(blob).hexdigest()
    actual, quality = s.decode_compatible_pcm(blob)
    repeated, repeated_quality = s.decode_compatible_pcm(blob)
    mono = (frames.astype(np.float32)/32768).mean(axis=1)
    expected = resample_poly(mono, 1, 3, padtype='constant', window=('kaiser', 5.0))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, repeated)
    assert quality == repeated_quality
    assert hashlib.sha256(blob).hexdigest() == digest
    assert quality['channels'] == 2 and quality['sample_rate'] == 48000
    assert quality['samples'] == len(frames) and quality['seconds'] == .5
    assert quality['encoder_sample_rate'] == 16000 and quality['encoder_samples'] == 8000
    assert quality['pcm_bytes'] == frames.nbytes and quality['physical_bytes'] == len(blob)
    assert quality['resampled'] and quality['decoder_header_adjusted_in_memory']
    assert not quality['source_modified'] and quality['pcm_data_complete']
    assert quality['resampling']['up'] == 1 and quality['resampling']['down'] == 3


def test_48khz_noninteger_resampled_duration_retains_original_duration():
    frames = stereo_frames(4801)
    actual, quality = s.decode_compatible_pcm(wav(frames))
    assert len(actual) == (len(frames)+2)//3
    assert quality['samples'] == 4801 and quality['encoder_samples'] == 1601
    assert quality['seconds'] == 4801/48000
    assert abs(quality['seconds']-quality['encoder_samples']/16000) < 1/16000


def test_polyphase_downsampling_suppresses_above_nyquist_tone():
    # Naive every-third-sample decimation aliases this 12kHz tone to 4kHz.
    frames = np.round(16000*np.sin(2*np.pi*12000*np.arange(4800)/48000)).astype(np.int16)
    actual, _ = s.decode_compatible_pcm(wav(frames, channels=1))
    assert np.sqrt(np.mean(actual[100:-100]**2)) < .005
    assert np.sqrt(np.mean((frames[::3].astype(float)/32768)**2)) > .2


def test_bad_original_byte_rate_rejected_before_temporary_header_adaptation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Original malformed format must fail before legacy decoding')
    monkeypatch.setattr(original, 'decode_pcm', forbidden)
    with pytest.raises(ValueError, match='Inconsistent original PCM'):
        s.decode_compatible_pcm(wav(stereo_frames(), byte_rate=192001))


def test_48khz_accepts_metadata_before_format_and_before_or_after_data():
    frames = stereo_frames()
    baseline, _ = s.decode_compatible_pcm(wav(frames))
    blob = wav(frames, before_fmt=chunk(b'JUNK', b'abc'),
               before_data=chunk(b'LIST', b'INFOtest'), after_data=chunk(b'LIST', b'INFOtail'))
    actual, quality = s.decode_compatible_pcm(blob)
    np.testing.assert_array_equal(actual, baseline)
    assert quality['physical_bytes'] == len(blob) and quality['pcm_data_complete']


@pytest.mark.parametrize('removed', [1, 8, 100])
def test_truncated_48khz_pcm_is_rejected(removed):
    with pytest.raises(ValueError, match='Truncated WAV chunk payload'):
        s.decode_compatible_pcm(wav(stereo_frames())[:-removed])


@pytest.mark.parametrize('name,payload', [
    (b'fmt ', struct.pack('<HHIIHH', 1, 2, 48000, 192000, 4, 16)),
    (b'data', np.zeros((4800, 2), dtype='<i2').tobytes()),
])
def test_duplicate_format_and_audio_chunks_remain_rejected(name, payload):
    with pytest.raises(ValueError, match='Duplicate WAV'):
        s.decode_compatible_pcm(wav(stereo_frames(), after_data=chunk(name, payload)))


@pytest.mark.parametrize('options', [dict(bits=8), dict(bits=24), dict(codec=3),
                                    dict(codec=65534), dict(rate=11025), dict(channels=3)])
def test_unsupported_original_formats_are_not_reinterpreted(options):
    with pytest.raises(ValueError, match='Unsupported original PCM'):
        s.decode_compatible_pcm(wav(stereo_frames(), **options))


def test_partial_stereo_pcm_frame_is_rejected_even_with_riff_padding():
    with pytest.raises(ValueError, match='Inconsistent PCM'):
        s.decode_compatible_pcm(wav(stereo_frames(), data=b'\0'*19201))


def test_insufficient_actual_duration_is_not_made_valid_by_rate_metadata():
    with pytest.raises(ValueError, match='Insufficient or invalid resampled waveform'):
        s.decode_compatible_pcm(wav(stereo_frames(2400)))
