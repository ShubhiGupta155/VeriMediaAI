from pathlib import Path

from modules.image.metadata import extract_metadata
from modules.image.compression import analyze_compression
from modules.image.noise import analyze_noise
from modules.image.frequency import analyze_frequency
from modules.image.detector import detect_ai_generated
from modules.image.explainability import generate_forensic_heatmap


def analyze_image(image_path, generate_heatmap=False, heatmap_path=None):
    """
    Run the complete Member 1 image-forensics pipeline.

    Parameters
    ----------
    image_path : str or Path
        Path to the image being analyzed.

    generate_heatmap : bool
        Whether to generate the forensic heatmap.

    heatmap_path : str or Path, optional
        Output path for the heatmap.

    Returns
    -------
    dict
        Structured image-forensics result.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = {
        "image": {
            "path": str(image_path),
            "filename": image_path.name,
        },
        "metadata": None,
        "compression": None,
        "noise": None,
        "frequency": None,
        "ai_detection": None,
        "explainability": None,
    }

    # 1. Metadata
    result["metadata"] = extract_metadata(image_path)

    # 2. Compression
    result["compression"] = analyze_compression(image_path)

    # 3. Noise / residual
    result["noise"] = analyze_noise(image_path)

    # 4. Frequency-domain analysis
    result["frequency"] = analyze_frequency(image_path)

    # 5. AI-generated image detector
    result["ai_detection"] = detect_ai_generated(image_path)

    # 6. Optional forensic heatmap
    if generate_heatmap:
        if heatmap_path is None:
            heatmap_path = (
                image_path.parent
                / f"{image_path.stem}_forensic_heatmap.jpg"
            )

        result["explainability"] = generate_forensic_heatmap(
            image_path,
            heatmap_path,
        )

    return result
# Evaluation note:
# The current noise, frequency, compression, and AI-detector outputs
# are baseline forensic indicators. Their thresholds and probabilities
# have not yet been quantitatively calibrated against the VeriMedia AI
# evaluation dataset. Dataset-based evaluation and calibration are
# planned as a subsequent research/evaluation stage.