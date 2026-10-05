import numpy as np
import soundfile as sf
import pytest

from modules.lipsync.audio_alignment import (
    detect_audio_activity,
    calculate_audio_mouth_alignment,
    extract_mouth_intervals,
)


def test_missing_file_raises_error(tmp_path):
    audio_path = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        detect_audio_activity(str(audio_path))


def test_empty_audio_raises_error(tmp_path):
    audio_path = tmp_path / "empty.wav"

    sf.write(
        str(audio_path),
        np.array([], dtype=np.float32),
        16000
    )

    with pytest.raises(ValueError):
        detect_audio_activity(str(audio_path))


def test_successful_audio_activity_detection(tmp_path):
    audio_path = tmp_path / "test_audio.wav"

    sample_rate = 16000

    # Generate 1 second of simple audio activity.
    audio = np.ones(
        sample_rate,
        dtype=np.float32
    ) * 0.1

    sf.write(
        str(audio_path),
        audio,
        sample_rate
    )

    result = detect_audio_activity(
        str(audio_path)
    )

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert isinstance(
        result["audio_activity_intervals"],
        list
    )
    assert isinstance(result["sample_rate"], int)
    assert isinstance(result["duration"], float)
    assert isinstance(result["activity_segments"], int)

    assert result["sample_rate"] == 16000
    assert result["duration"] > 0
    assert result["activity_segments"] >= 0


def test_alignment_overlapping_intervals():
    audio_intervals = [
        (1.0, 2.0)
    ]

    mouth_intervals = [
        (1.5, 2.5)
    ]

    result = calculate_audio_mouth_alignment(
        audio_intervals,
        mouth_intervals
    )

    assert result["alignment_score"] == 1.0
    assert result["matched_intervals"] == 1
    assert result["mismatch_intervals"] == []


def test_alignment_exactly_at_tolerance_boundary():
    audio_intervals = [
        (1.0, 2.0)
    ]

    mouth_intervals = [
        (2.2, 3.0)
    ]

    result = calculate_audio_mouth_alignment(
        audio_intervals,
        mouth_intervals,
        tolerance=0.2
    )

    assert result["alignment_score"] == 1.0
    assert result["matched_intervals"] == 1


def test_alignment_within_tolerance():
    audio_intervals = [
        (1.0, 2.0)
    ]

    mouth_intervals = [
        (2.1, 3.0)
    ]

    result = calculate_audio_mouth_alignment(
        audio_intervals,
        mouth_intervals,
        tolerance=0.2
    )

    assert result["alignment_score"] == 1.0
    assert result["matched_intervals"] == 1


def test_alignment_outside_tolerance():
    audio_intervals = [
        (1.0, 2.0)
    ]

    mouth_intervals = [
        (2.3, 3.0)
    ]

    result = calculate_audio_mouth_alignment(
        audio_intervals,
        mouth_intervals,
        tolerance=0.2
    )

    assert result["alignment_score"] == 0.0
    assert result["matched_intervals"] == 0
    assert result["mismatch_intervals"] == [
        (1.0, 2.0)
    ]


def test_empty_mouth_intervals():
    audio_intervals = [
        (1.0, 2.0)
    ]

    result = calculate_audio_mouth_alignment(
        audio_intervals,
        []
    )

    assert result["alignment_score"] == 0.0
    assert result["matched_intervals"] == 0
    assert result["mismatch_intervals"] == [
        (1.0, 2.0)
    ]


def test_both_interval_lists_empty():
    result = calculate_audio_mouth_alignment(
        [],
        []
    )

    assert result["alignment_score"] == 0.0
    assert result["matched_intervals"] == 0
    assert result["mismatch_intervals"] == []


def test_invalid_negative_interval():
    audio_intervals = [
        (-1.0, 2.0)
    ]

    with pytest.raises(ValueError):
        calculate_audio_mouth_alignment(
            audio_intervals,
            [(1.0, 2.0)]
        )


def test_invalid_reversed_interval():
    audio_intervals = [
        (2.0, 1.0)
    ]

    with pytest.raises(ValueError):
        calculate_audio_mouth_alignment(
            audio_intervals,
            [(1.0, 2.0)]
        )


def test_invalid_interval_format():
    audio_intervals = [
        (1.0,)
    ]

    with pytest.raises(ValueError):
        calculate_audio_mouth_alignment(
            audio_intervals,
            [(1.0, 2.0)]
        )


def test_completely_non_overlapping_intervals():
    audio_intervals = [
        (1.0, 2.0),
        (5.0, 6.0)
    ]

    mouth_intervals = [
        (10.0, 11.0)
    ]

    result = calculate_audio_mouth_alignment(
        audio_intervals,
        mouth_intervals
    )

    assert result["alignment_score"] == 0.0
    assert result["matched_intervals"] == 0
    assert result["mismatch_intervals"] == [
        (1.0, 2.0),
        (5.0, 6.0)
    ]


def test_extract_mouth_intervals():
    movement_intervals = [
        {
            "start": 0.0,
            "end": 0.1,
            "movement": 2.5
        },
        {
            "start": 0.1,
            "end": 0.2,
            "movement": 3.0
        }
    ]

    result = extract_mouth_intervals(
        movement_intervals
    )

    assert result == [
        (0.0, 0.1),
        (0.1, 0.2)
    ]


def test_extract_mouth_intervals_invalid_input():
    with pytest.raises(ValueError):
        extract_mouth_intervals("invalid")


def test_extract_mouth_intervals_missing_keys():
    movement_intervals = [
        {
            "start": 0.0,
            "movement": 2.5
        }
    ]

    with pytest.raises(ValueError):
        extract_mouth_intervals(
            movement_intervals
        )