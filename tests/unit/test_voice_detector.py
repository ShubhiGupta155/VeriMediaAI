import numpy as np
import soundfile as sf

from modules.audio.voice_detector import detect_voice_manipulation


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


def test_valid_audio_detection(tmp_path):
    audio_path = tmp_path / "test.wav"

    create_test_audio(audio_path)

    result = detect_voice_manipulation(
        str(audio_path)
    )

    assert result is not None
    assert "audio_score" in result
    assert "status" in result


def test_missing_file_raises_error(tmp_path):
    audio_path = tmp_path / "missing.wav"

    try:
        detect_voice_manipulation(
            str(audio_path)
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True


def test_sample_rate_is_16khz(tmp_path):
    audio_path = tmp_path / "test.wav"

    create_test_audio(audio_path)

    result = detect_voice_manipulation(
        str(audio_path)
    )

    assert result["sample_rate"] == 16000


def test_audio_score_range(tmp_path):
    audio_path = tmp_path / "test.wav"

    create_test_audio(audio_path)

    result = detect_voice_manipulation(
        str(audio_path)
    )

    assert 0.0 <= result["audio_score"] <= 1.0


def test_model_version_present(tmp_path):
    audio_path = tmp_path / "test.wav"

    create_test_audio(audio_path)

    result = detect_voice_manipulation(
        str(audio_path)
    )

    assert result["model_version"] == "voice-baseline-v1"
