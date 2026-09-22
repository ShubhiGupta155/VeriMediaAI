import os
import numpy as np
import librosa


def detect_voice_manipulation(file_path, threshold=0.5):
    """
    Analyze an audio file for basic synthetic/manipulated voice indicators.

    Returns:
        dict containing manipulation score, status, and model version.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load audio at 16 kHz
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("Audio file contains no samples")

    # Extract basic spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    # Calculate normalized feature statistics
    centroid_mean = float(np.mean(spectral_centroid))
    bandwidth_mean = float(np.mean(spectral_bandwidth))

    # Basic heuristic score.
    # This is a baseline indicator, NOT a trained deepfake detector.
    score = float(
        np.clip(
            (centroid_mean / sample_rate) +
            (bandwidth_mean / sample_rate),
            0.0,
            1.0
        )
    )

    status = "suspicious" if score >= threshold else "normal"

    return {
        "audio_score": score,
        "status": status,
        "sample_rate": sample_rate,
        "spectral_centroid": centroid_mean,
        "spectral_bandwidth": bandwidth_mean,
        "model_version": "voice-baseline-v1"
    }


if __name__ == "__main__":

    input_file = "sample.wav"

    try:
        result = detect_voice_manipulation(input_file)

        print("Voice analysis successful!")
        print(f"Audio score: {result['audio_score']:.4f}")
        print(f"Status: {result['status']}")
        print(f"Sample rate: {result['sample_rate']} Hz")
        print(f"Model version: {result['model_version']}")

    except Exception as error:
        print(f"Error: {error}")