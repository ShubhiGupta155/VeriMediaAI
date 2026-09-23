import os

from modules.audio.preprocessing import load_and_preprocess_audio
from modules.audio.spectrogram import generate_spectrogram
from modules.audio.voice_detector import detect_voice_manipulation


def run_audio_pipeline(
    file_path,
    output_dir="results"
):
    """
    Run the complete baseline audio analysis pipeline.

    Pipeline stages:
    1. Audio preprocessing
    2. Spectrogram generation
    3. Voice manipulation detection
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Audio preprocessing
    audio, sample_rate = load_and_preprocess_audio(
        file_path
    )

    # Step 2: Spectrogram generation
    spectrogram_path = os.path.join(
        output_dir,
        "spectrogram.png"
    )

    spectrogram, spectrogram_sr = generate_spectrogram(
        file_path,
        spectrogram_path
    )

    # Step 3: Voice manipulation detection
    voice_result = detect_voice_manipulation(
        file_path
    )

    return {
        "module": "audio",
        "status": "success",
        "sample_rate": sample_rate,
        "duration": len(audio) / sample_rate,
        "spectrogram_shape": spectrogram.shape,
        "spectrogram_path": spectrogram_path,
        "audio_score": voice_result["audio_score"],
        "voice_status": voice_result["status"],
        "spectral_centroid": voice_result["spectral_centroid"],
        "spectral_bandwidth": voice_result["spectral_bandwidth"],
        "model_version": voice_result["model_version"],
    }


if __name__ == "__main__":

    input_file = "sample.wav"

    try:
        result = run_audio_pipeline(input_file)

        print("Audio pipeline completed successfully!")
        print(f"Sample rate: {result['sample_rate']} Hz")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Spectrogram shape: {result['spectrogram_shape']}")
        print(f"Audio score: {result['audio_score']:.4f}")
        print(f"Voice status: {result['voice_status']}")
        print(f"Model version: {result['model_version']}")
        print(f"Spectrogram: {result['spectrogram_path']}")

    except Exception as error:
        print(f"Error: {error}")