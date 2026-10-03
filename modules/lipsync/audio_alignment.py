import os

import librosa


def detect_speech_activity(
    file_path,
    top_db=30
):
    """
    Detect speech-active time intervals in an audio file.

    Returns:
        dict containing speech intervals and audio timing information.
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

    speech_intervals = [
        (
            float(start / sample_rate),
            float(end / sample_rate)
        )
        for start, end in intervals
    ]

    return {
        "status": "success",
        "speech_intervals": speech_intervals,
        "sample_rate": sample_rate,
        "duration": float(len(audio) / sample_rate),
        "speech_segments": len(speech_intervals)
    }


def calculate_audio_mouth_alignment(
    speech_intervals,
    mouth_intervals,
    tolerance=0.2
):
    """
    Compare speech intervals with mouth movement intervals.

    Returns:
        dict containing alignment score and mismatch intervals.
    """

    if not speech_intervals:
        return {
            "alignment_score": 0.0,
            "matched_intervals": 0,
            "mismatch_intervals": []
        }

    matched_intervals = 0
    mismatch_intervals = []

    for speech_start, speech_end in speech_intervals:

        matched = False

        for mouth_start, mouth_end in mouth_intervals:

            overlap_start = max(
                speech_start,
                mouth_start
            )

            overlap_end = min(
                speech_end,
                mouth_end
            )

            if overlap_end >= overlap_start - tolerance:
                matched = True
                break

        if matched:
            matched_intervals += 1
        else:
            mismatch_intervals.append(
                (speech_start, speech_end)
            )

    alignment_score = (
        matched_intervals / len(speech_intervals)
    )

    return {
        "alignment_score": float(alignment_score),
        "matched_intervals": matched_intervals,
        "mismatch_intervals": mismatch_intervals
    }