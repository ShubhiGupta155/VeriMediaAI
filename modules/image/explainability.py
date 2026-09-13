from pathlib import Path

import cv2
import numpy as np


def generate_forensic_heatmap(image_path, output_path):
    """
    Generate a forensic heatmap from local high-frequency residuals.

    The heatmap highlights regions with relatively stronger residual
    activity. It is supporting forensic evidence and should not be
    interpreted as proof of manipulation.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32)

    # Estimate smooth image content.
    blurred = cv2.GaussianBlur(gray_float, (5, 5), 0)

    # Calculate high-frequency residual.
    residual = np.abs(gray_float - blurred)

    # Smooth the residual so we get regional rather than pixel-level
    # variation.
    regional_residual = cv2.GaussianBlur(
        residual,
        (31, 31),
        0
    )

    # Normalize to 0-255.
    minimum = float(regional_residual.min())
    maximum = float(regional_residual.max())

    if maximum > minimum:
        normalized = (
            (regional_residual - minimum)
            / (maximum - minimum)
            * 255.0
        )
    else:
        normalized = np.zeros_like(regional_residual)

    heatmap_gray = normalized.astype(np.uint8)

    # Apply OpenCV's standard heatmap representation.
    heatmap = cv2.applyColorMap(
        heatmap_gray,
        cv2.COLORMAP_JET
    )

    # Blend heatmap with original image.
    overlay = cv2.addWeighted(
        image,
        0.65,
        heatmap,
        0.35,
        0
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), overlay):
        raise IOError(f"Unable to save heatmap: {output_path}")

    # Identify the strongest region.
    _, max_value, _, max_location = cv2.minMaxLoc(
        regional_residual
    )

    height, width = gray.shape

    x, y = max_location

    normalized_peak = (
        float(max_value - minimum)
        / float(maximum - minimum)
        if maximum > minimum
        else 0.0
    )

    return {
        "filename": image_path.name,
        "heatmap_path": str(output_path),
        "image_width": width,
        "image_height": height,
        "peak_x": int(x),
        "peak_y": int(y),
        "peak_score": round(normalized_peak, 6),
        "description": (
            "Heatmap highlights regions with stronger local "
            "noise/residual activity. It is supporting forensic "
            "evidence and is not proof of manipulation."
        ),
    }