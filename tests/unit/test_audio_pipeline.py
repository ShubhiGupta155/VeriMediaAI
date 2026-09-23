import soundfile as sf
import numpy as np

from modules.audio.pipeline import run_audio_pipeline


def create_test_audio(path, duration=2, sample_rate=16000):
    """Create a simple test WAV file."""
    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    sf.write(path, audio, sample_rate)


def test_audio_pipeline_success(tmp_path):
    audio_path = tmp_path / "test.wav"

    create_test_audio(audio_path)

    result = run_audio_pipeline(
        str(audio_path),
        str(tmp_path / "results")
    )

    assert result["status"] == "success"
    assert result["module"] == "audio"


def test_audio_pipeline_sample_rate(tmp_path):
    audio_path = tmp_path / "test.wav"

    create_test_audio(audio_path)

    result = run_audio_pipeline(
        str(audio_path),
        str(tmp_path / "results")
    )

    assert result["sample_rate"] == 16000


def test_audio_pipeline_spectrogram(tmp_path):
    audio_path = tmp_path / "test.wav"
    output_dir = tmp_path / "results"

    create_test_audio(audio_path)

    result = run_audio_pipeline(
        str(audio_path),
        str(output_dir)
    )

    assert result["spectrogram_shape"][0] == 128
    assert result["spectrogram_shape"][1] > 0
    assert result["spectrogram_path"]


def test_audio_pipeline_score(tmp_path):
    audio_path = tmp_path / "test.wav"

    create_test_audio(audio_path)

    result = run_audio_pipeline(
        str(audio_path),
        str(tmp_path / "results")
    )

    assert 0.0 <= result["audio_score"] <= 1.0


def test_audio_pipeline_missing_file(tmp_path):
    audio_path = tmp_path / "missing.wav"

    try:
        run_audio_pipeline(
            str(audio_path),
            str(tmp_path / "results")
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True