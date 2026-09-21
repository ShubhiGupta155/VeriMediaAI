import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def generate_spectrogram(
    file_path,
    output_path="results/spectrogram.png",
    n_fft=2048,
    hop_length=512
):
    """
    Generate a Mel-spectrogram from an audio file.

    Parameters:
        file_path: Path to input audio file
        output_path: Path to save spectrogram image
        n_fft: FFT window size
        hop_length: Number of samples between frames

    Returns:
        mel_spectrogram: Mel-spectrogram in dB
        sample_rate: Audio sample rate
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load audio
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Generate Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128
    )

    # Convert to decibel scale
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Create output directory
    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Plot spectrogram
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-Spectrogram")
    plt.tight_layout()

    # Save image
    plt.savefig(output_path)
    plt.close()

    print(f"Spectrogram saved: {output_path}")

    return mel_db, sample_rate


if __name__ == "__main__":

    input_file = "sample.wav"
    output_file = "results/spectrogram.png"

    try:
        spectrogram, sr = generate_spectrogram(
            input_file,
            output_file
        )

        print("Spectrogram generation successful!")
        print(f"Sample rate: {sr} Hz")
        print(f"Spectrogram shape: {spectrogram.shape}")

    except Exception as error:
        print(f"Error: {error}")