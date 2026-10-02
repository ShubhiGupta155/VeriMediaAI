import os

import librosa
import numpy as np


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