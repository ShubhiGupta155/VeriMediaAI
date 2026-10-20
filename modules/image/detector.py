from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "delpot/steganograph-ia-detector"

_processor = None
_model = None


def load_detector():
    """
    Load and cache the pretrained AI-generated image detector.
    """

    global _processor, _model

    if _processor is None or _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

        _model.eval()

    return _processor, _model


def detect_ai_generated(image_path):
    """
    Detect whether an image is more likely to be real
    or AI-generated.

    Returns probabilities and the predicted class.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(path).convert("RGB")

    processor, model = load_detector()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    labels = model.config.id2label

    # Normalize model labels and validate the expected classes.
    normalized_labels = {
        str(label).strip().lower(): int(index)
        for index, label in labels.items()
    }

    expected_labels = {"real", "ai_generated"}

    if set(normalized_labels) != expected_labels:
        raise ValueError(
            f"Unsupported model labels: {labels}. "
            f"Expected exactly {sorted(expected_labels)}."
        )

    scores = {
        label: float(probabilities[index])
        for label, index in normalized_labels.items()
    }

    predicted_index = int(torch.argmax(probabilities))
    predicted_label = str(labels[predicted_index]).strip().lower()

    return {
        "filename": path.name,
        "predicted_label": predicted_label,
        "real_probability": round(scores["real"], 6),
        "ai_generated_probability": round(
            scores["ai_generated"], 6
        ),
        "confidence": round(
            float(probabilities[predicted_index]), 6
        ),
    }
