import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "delpot/steganograph-ia-detector"

processor = None
model = None


def load_model():
    global processor, model

    if processor is None or model is None:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME
        )

        model.eval()

    return processor, model


def analyze_image(image):

    processor, model = load_model()

    # Make sure image is RGB
    image = image.convert("RGB")

    # Preprocess
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Model inference
    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # Get probabilities
    real_probability = probabilities[0].item()
    ai_probability = probabilities[1].item()

    # Determine prediction
    if ai_probability >= real_probability:
        prediction = "AI-generated"
    else:
        prediction = "Real"

    return {
        "prediction": prediction,
        "ai_probability": ai_probability,
        "real_probability": real_probability
    }