import os
import numpy as np
import librosa
import soundfile as sf


def load_and_preprocess_audio(file_path, target_sr=16000):
    """
    Load and preprocess audio for VeriMedia AI.

    Steps:
    1. Load audio
    2. Convert to mono
    3. Resample to target sample rate
    4. Remove DC offset
    5. Normalize amplitude
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load audio and convert to mono
    audio, sample_rate = librosa.load(
        file_path,
        sr=target_sr,
        mono=True
    )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize amplitude
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude > 0:
        audio = audio / max_amplitude

    return audio, sample_rate


def save_processed_audio(audio, sample_rate, output_path):
    """Save processed audio as WAV."""

    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    sf.write(output_path, audio, sample_rate)

    print(f"Processed audio saved: {output_path}")


if __name__ == "__main__":

    input_file = "sample.wav"
    output_file = "results/processed_audio.wav"

    try:
        audio, sr = load_and_preprocess_audio(input_file)

        print("Audio preprocessing successful!")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(audio) / sr:.2f} seconds")
        print(f"Samples: {len(audio)}")

        save_processed_audio(
            audio,
            sr,
            output_file
        )

    except Exception as error:
        print(f"Error: {error}")