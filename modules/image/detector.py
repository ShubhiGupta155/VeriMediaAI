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

    scores = {
        labels[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    predicted_index = int(torch.argmax(probabilities))
    predicted_label = labels[predicted_index]

    return {
        "filename": path.name,
        "predicted_label": predicted_label,
        "real_probability": round(scores.get("real", 0.0), 6),
        "ai_generated_probability": round(
            scores.get("ai_generated", 0.0), 6
        ),
        "confidence": round(
            float(probabilities[predicted_index]), 6
        ),
    }