from pathlib import Path

import cv2
import numpy as np


def analyze_frequency(image_path):
    """
    Analyze an image in the frequency domain using a 2D Fourier Transform.

    Returns frequency-domain statistics that can be used as
    supporting forensic evidence.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image_float = image.astype(np.float32)

    # 2D Fourier Transform
    fft = np.fft.fft2(image_float)

    # Move low frequencies to the center.
    fft_shifted = np.fft.fftshift(fft)

    # Magnitude spectrum
    magnitude = np.abs(fft_shifted)

    # Log scaling makes the spectrum easier to analyze numerically.
    log_magnitude = np.log1p(magnitude)

    height, width = image.shape

    center_y = height // 2
    center_x = width // 2

    # Define a central low-frequency region.
    radius = max(1, min(height, width) // 20)

    y_start = max(0, center_y - radius)
    y_end = min(height, center_y + radius + 1)
    x_start = max(0, center_x - radius)
    x_end = min(width, center_x + radius + 1)

    low_frequency_region = log_magnitude[
        y_start:y_end,
        x_start:x_end
    ]

    total_energy = float(np.sum(log_magnitude ** 2))
    low_frequency_energy = float(np.sum(low_frequency_region ** 2))

    if total_energy > 0:
        low_frequency_ratio = low_frequency_energy / total_energy
    else:
        low_frequency_ratio = 0.0

    # Exclude the center to estimate high-frequency energy.
    high_frequency_mask = np.ones_like(log_magnitude, dtype=bool)
    high_frequency_mask[y_start:y_end, x_start:x_end] = False

    high_frequency_values = log_magnitude[high_frequency_mask]

    if high_frequency_values.size > 0:
        high_frequency_energy = float(
            np.sum(high_frequency_values ** 2)
        )
    else:
        high_frequency_energy = 0.0

    if total_energy > 0:
        high_frequency_ratio = high_frequency_energy / total_energy
    else:
        high_frequency_ratio = 0.0

    evidence = []

    if high_frequency_ratio > 0.75:
        evidence.append(
        "A high proportion of measured frequency-domain energy lies "
        "outside the defined low-frequency region; this is a supporting "
        "frequency-domain indicator and is not, by itself, evidence of manipulation."
        )
    else:
        evidence.append(
            "Frequency-domain energy is primarily concentrated toward "
            "lower frequencies."
        )

    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "total_frequency_energy": round(total_energy, 6),
        "low_frequency_energy": round(low_frequency_energy, 6),
        "low_frequency_ratio": round(low_frequency_ratio, 6),
        "high_frequency_energy": round(high_frequency_energy, 6),
        "high_frequency_ratio": round(high_frequency_ratio, 6),
        "evidence": evidence,
    }