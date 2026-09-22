import os
import numpy as np
import soundfile as sf

from modules.audio.spectrogram import generate_spectrogram


def create_test_audio(path, duration=2, sample_rate=16000):
    """Create a simple test WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sample_rate)


def test_valid_audio_generates_spectrogram(tmp_path):
    audio_path = tmp_path / "test.wav"
    output_path = tmp_path / "spectrogram.png"

    create_test_audio(audio_path)

    spectrogram, sample_rate = generate_spectrogram(
        str(audio_path),
        str(output_path)
    )

    assert spectrogram is not None
    assert sample_rate == 16000


def test_invalid_file_raises_error(tmp_path):
    invalid_path = tmp_path / "missing.wav"
    output_path = tmp_path / "spectrogram.png"

    try:
        generate_spectrogram(
            str(invalid_path),
            str(output_path)
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True


def test_spectrogram_output_file_created(tmp_path):
    audio_path = tmp_path / "test.wav"
    output_path = tmp_path / "spectrogram.png"

    create_test_audio(audio_path)

    generate_spectrogram(
        str(audio_path),
        str(output_path)
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_sample_rate_is_16khz(tmp_path):
    audio_path = tmp_path / "test.wav"
    output_path = tmp_path / "spectrogram.png"

    create_test_audio(audio_path)

    _, sample_rate = generate_spectrogram(
        str(audio_path),
        str(output_path)
    )

    assert sample_rate == 16000


def test_spectrogram_dimensions(tmp_path):
    audio_path = tmp_path / "test.wav"
    output_path = tmp_path / "spectrogram.png"

    create_test_audio(audio_path)

    spectrogram, _ = generate_spectrogram(
        str(audio_path),
        str(output_path)
    )

    assert spectrogram.shape[0] == 128
    assert spectrogram.ndim == 2