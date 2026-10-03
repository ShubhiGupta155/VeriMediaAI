import numpy as np
import soundfile as sf
import pytest

from modules.lipsync.audio_alignment import (
    detect_speech_activity,
    calculate_audio_mouth_alignment,
)


def test_missing_file_raises_error(tmp_path):
    audio_path = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        detect_speech_activity(str(audio_path))


def test_empty_audio_raises_error(tmp_path):
    audio_path = tmp_path / "empty.wav"

    sf.write(
        str(audio_path),
        np.array([], dtype=np.float32),
        16000
    )

    with pytest.raises(ValueError):
        detect_speech_activity(str(audio_path))


def test_successful_speech_detection():
    result = detect_speech_activity("sample.wav")

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert isinstance(result["speech_intervals"], list)
    assert isinstance(result["sample_rate"], int)
    assert isinstance(result["duration"], float)
    assert isinstance(result["speech_segments"], int)

    assert result["sample_rate"] == 16000
    assert result["duration"] > 0
    assert result["speech_segments"] >= 0


def test_audio_mouth_alignment_full_match():
    speech_intervals = [
        (1.0, 2.0),
        (3.0, 4.0)
    ]

    mouth_intervals = [
        (1.0, 2.0),
        (3.0, 4.0)
    ]

    result = calculate_audio_mouth_alignment(
        speech_intervals,
        mouth_intervals
    )

    assert result["alignment_score"] == 1.0
    assert result["matched_intervals"] == 2
    assert result["mismatch_intervals"] == []


def test_audio_mouth_alignment_partial_match():
    speech_intervals = [
        (1.0, 2.0),
        (3.0, 4.0)
    ]

    mouth_intervals = [
        (1.1, 2.1),
        (5.0, 6.0)
    ]

    result = calculate_audio_mouth_alignment(
        speech_intervals,
        mouth_intervals
    )

    assert result["alignment_score"] == 0.5
    assert result["matched_intervals"] == 1
    assert result["mismatch_intervals"] == [(3.0, 4.0)]


def test_audio_mouth_alignment_no_speech():
    result = calculate_audio_mouth_alignment(
        [],
        [(1.0, 2.0)]
    )

    assert result["alignment_score"] == 0.0
    assert result["matched_intervals"] == 0
    assert result["mismatch_intervals"] == []