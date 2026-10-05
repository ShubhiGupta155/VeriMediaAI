import os

import librosa


def detect_audio_activity(
    file_path,
    top_db=30
):
    """
    Detect non-silent audio activity intervals.

    Note:
        librosa.effects.split() detects energy-based non-silent
        regions, not true speech. These intervals are used as
        a speech/activity approximation for lip-sync analysis.

    Returns:
        dict containing audio activity intervals and timing information.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    if audio.size == 0:
        raise ValueError(
            "Audio file contains no samples"
        )

    intervals = librosa.effects.split(
        audio,
        top_db=top_db
    )

    audio_activity_intervals = [
        (
            float(start / sample_rate),
            float(end / sample_rate)
        )
        for start, end in intervals
    ]

    return {
        "status": "success",
        "audio_activity_intervals": audio_activity_intervals,
        "sample_rate": sample_rate,
        "duration": float(len(audio) / sample_rate),
        "activity_segments": len(audio_activity_intervals)
    }


def _validate_intervals(intervals, name):
    """
    Validate interval format.

    Each interval must contain:
        (start_time, end_time)

    Both values must be non-negative and end_time must
    be greater than or equal to start_time.
    """

    if not isinstance(intervals, list):
        raise ValueError(
            f"{name} must be a list of intervals"
        )

    for interval in intervals:
        if (
            not isinstance(interval, (tuple, list))
            or len(interval) != 2
        ):
            raise ValueError(
                f"Invalid interval in {name}: {interval}"
            )

        start, end = interval

        if start < 0 or end < 0:
            raise ValueError(
                f"Negative interval values are not allowed: {interval}"
            )

        if end < start:
            raise ValueError(
                f"Reversed interval is not allowed: {interval}"
            )


def _interval_gap(
    first_start,
    first_end,
    second_start,
    second_end
):
    """
    Calculate temporal gap between two intervals.

    Returns:
        0.0 when intervals overlap.
        Positive value when intervals are separated.
    """

    if first_end >= second_start and second_end >= first_start:
        return 0.0

    if first_end < second_start:
        return float(second_start - first_end)

    return float(first_start - second_end)


def calculate_audio_mouth_alignment(
    audio_activity_intervals,
    mouth_intervals,
    tolerance=0.2
):
    """
    Compare audio activity intervals with mouth movement intervals.

    An interval is considered aligned when:
    1. The intervals overlap, or
    2. The temporal gap between them is within tolerance.

    The alignment score is the proportion of audio activity
    intervals that have corresponding mouth activity.

    Note:
        This is an interval-count-based baseline metric.
        It is not a complete duration-based lip-sync score.
    """

    _validate_intervals(
        audio_activity_intervals,
        "audio_activity_intervals"
    )

    _validate_intervals(
        mouth_intervals,
        "mouth_intervals"
    )

    if tolerance < 0:
        raise ValueError(
            "Tolerance must be non-negative"
        )

    if not audio_activity_intervals:
        return {
            "alignment_score": 0.0,
            "matched_intervals": 0,
            "mismatch_intervals": []
        }

    matched_intervals = 0
    mismatch_intervals = []

    for audio_start, audio_end in audio_activity_intervals:

        matched = False

        for mouth_start, mouth_end in mouth_intervals:

            gap = _interval_gap(
                audio_start,
                audio_end,
                mouth_start,
                mouth_end
            )

            if gap <= tolerance + 1e-9:
                matched = True
                break

        if matched:
            matched_intervals += 1
        else:
            mismatch_intervals.append(
                (audio_start, audio_end)
            )

    alignment_score = (
        matched_intervals /
        len(audio_activity_intervals)
    )

    return {
        "alignment_score": float(alignment_score),
        "matched_intervals": matched_intervals,
        "mismatch_intervals": mismatch_intervals
    }
def extract_mouth_intervals(movement_intervals):
    """
    Convert mouth movement interval dictionaries into
    (start, end) intervals for audio-mouth alignment.
    """

    if not isinstance(movement_intervals, list):
        raise ValueError(
            "movement_intervals must be a list"
        )

    mouth_intervals = []

    for interval in movement_intervals:
        if not isinstance(interval, dict):
            raise ValueError(
                f"Invalid movement interval: {interval}"
            )

        if "start" not in interval or "end" not in interval:
            raise ValueError(
                f"Missing start/end in movement interval: {interval}"
            )

        mouth_intervals.append(
            (
                float(interval["start"]),
                float(interval["end"])
            )
        )

    return mouth_intervals