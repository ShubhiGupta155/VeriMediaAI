from modules.lipsync.mouth_tracker import (
    calculate_mouth_movement,
)


def test_no_landmarks_returns_zero():
    result = calculate_mouth_movement([], [])

    assert result == 0.0


def test_mismatched_landmarks_returns_zero():
    previous = [
        (10, 10),
        (20, 10),
    ]

    current = [
        (12, 12),
    ]

    result = calculate_mouth_movement(
        previous,
        current
    )

    assert result == 0.0


def test_mouth_movement_calculation():
    previous = [
        (0, 0),
        (10, 0),
        (0, 10),
        (10, 10),
    ]

    current = [
        (3, 4),
        (13, 4),
        (3, 14),
        (13, 14),
    ]

    result = calculate_mouth_movement(
        previous,
        current
    )

    assert result == 5.0


def test_invalid_video_raises_error(tmp_path):
    video_path = tmp_path / "missing.mp4"

    try:
        from modules.lipsync.mouth_tracker import track_mouth_movement

        track_mouth_movement(str(video_path))

        assert False, "Expected ValueError"
    except ValueError:
        assert True