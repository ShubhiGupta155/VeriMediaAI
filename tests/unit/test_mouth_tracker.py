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


def test_successful_video_tracking():
    from modules.lipsync.mouth_tracker import track_mouth_movement

    video_path = "tests/fixtures/test_mouth_tracker.mp4"

    result = track_mouth_movement(video_path)

    assert isinstance(result, dict)

    assert "status" in result
    assert "frame_count" in result
    assert "movement_frames" in result
    assert "average_mouth_movement" in result
    
    assert "movement_intervals" in result
    assert isinstance(result["movement_intervals"], list)

    assert result["status"] == "success"
    assert isinstance(result["frame_count"], int)
    assert isinstance(result["movement_frames"], int)
    assert isinstance(result["average_mouth_movement"], float)

    assert result["frame_count"] > 0
    assert result["movement_frames"] >= 0
    assert result["average_mouth_movement"] >= 0.0

    if result["movement_intervals"]:
     interval = result["movement_intervals"][0]

    assert isinstance(interval, dict)
    assert "start" in interval
    assert "end" in interval
    assert "movement" in interval

    assert isinstance(interval["start"], float)
    assert isinstance(interval["end"], float)
    assert isinstance(interval["movement"], float)

    assert interval["start"] >= 0.0
    assert interval["end"] >= interval["start"]
    assert interval["movement"] >= 0.0


def test_mouth_movement_below_threshold_is_ignored():
    previous = [
        (0, 0),
        (10, 0),
        (0, 10),
        (10, 10)
    ]

    current = [
        (1, 0),
        (11, 0),
        (1, 10),
        (11, 10)
    ]

    movement = calculate_mouth_movement(
        previous,
        current
    )

    assert movement == 1.0
    assert movement < 2.0


def test_mouth_movement_above_threshold_is_detected():
    previous = [
        (0, 0),
        (10, 0),
        (0, 10),
        (10, 10)
    ]

    current = [
        (3, 4),
        (13, 4),
        (3, 14),
        (13, 14)
    ]

    movement = calculate_mouth_movement(
        previous,
        current
    )

    assert movement == 5.0
    assert movement >= 2.0