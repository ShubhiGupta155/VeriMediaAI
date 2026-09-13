from pathlib import Path

import cv2
import numpy as np


def analyze_noise(image_path):
    """
    Analyze local noise/residual characteristics of an image.

    The residual is obtained by subtracting a Gaussian-smoothed
    version of the image from the original image.

    Returns statistical indicators that can later be used as
    supporting forensic evidence.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image_float = image.astype(np.float32)

    # Estimate the low-frequency content.
    blurred = cv2.GaussianBlur(
        image_float,
        (5, 5),
        0
    )

    # High-frequency residual/noise component.
    residual = image_float - blurred

    mean_residual = float(np.mean(residual))
    std_residual = float(np.std(residual))
    mean_absolute_residual = float(np.mean(np.abs(residual)))

    # Divide the image into four regions and calculate
    # the residual standard deviation in each region.
    height, width = image.shape

    half_height = height // 2
    half_width = width // 2

    regions = {
        "top_left": residual[:half_height, :half_width],
        "top_right": residual[:half_height, half_width:],
        "bottom_left": residual[half_height:, :half_width],
        "bottom_right": residual[half_height:, half_width:],
    }

    regional_std = {
        name: round(float(np.std(region)), 6)
        for name, region in regions.items()
    }

    regional_values = list(regional_std.values())

    regional_mean = float(np.mean(regional_values))

    if regional_mean > 0:
        regional_variation = float(
            np.std(regional_values) / regional_mean
        )
    else:
        regional_variation = 0.0

    evidence = []

    if regional_variation > 0.35:
        evidence.append(
        "Regional noise variation was detected; this is a supporting "
        "forensic indicator and is not, by itself, evidence of manipulation."
        )
    else:
        evidence.append(
            "Noise characteristics are relatively consistent across "
            "the analyzed image regions."
        )

    return {
        "filename": path.name,
        "mean_residual": round(mean_residual, 6),
        "noise_std": round(std_residual, 6),
        "mean_absolute_residual": round(mean_absolute_residual, 6),
        "regional_noise_std": regional_std,
        "regional_noise_variation": round(regional_variation, 6),
        "evidence": evidence,
    }