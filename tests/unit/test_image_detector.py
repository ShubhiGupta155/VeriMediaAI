from types import SimpleNamespace

import pytest
import torch

from modules.image import detector


class FakeProcessor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


class FakeModel:
    def __init__(self, labels, logits):
        self.config = SimpleNamespace(id2label=labels)
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, **inputs):
        return SimpleNamespace(logits=self.logits)


def test_detect_maps_probabilities_by_label(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    from PIL import Image
    Image.new("RGB", (32, 32)).save(image_path)

    # Deliberately reverse the class indices.
    model = FakeModel({0: "ai_generated", 1: "real"}, [3.0, 1.0])

    monkeypatch.setattr(
        detector,
        "load_detector",
        lambda: (FakeProcessor(), model),
    )

    result = detector.detect_ai_generated(image_path)

    assert result["predicted_label"] == "ai_generated"
    assert result["ai_generated_probability"] > result["real_probability"]
    assert result["ai_generated_probability"] + result["real_probability"] == pytest.approx(1.0, abs=1e-5)


def test_detect_missing_image_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        detector,
        "load_detector",
        lambda: pytest.fail("Model should not load for a missing image"),
    )

    with pytest.raises(FileNotFoundError):
        detector.detect_ai_generated(tmp_path / "missing.jpg")
def test_detect_rejects_unexpected_labels(monkeypatch, tmp_path):
    from PIL import Image

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32)).save(image_path)

    model = FakeModel(
        {0: "LABEL_0", 1: "LABEL_1"},
        [3.0, 1.0],
    )

    monkeypatch.setattr(
        detector,
        "load_detector",
        lambda: (FakeProcessor(), model),
    )

    with pytest.raises(ValueError, match="Unsupported model labels"):
        detector.detect_ai_generated(image_path)