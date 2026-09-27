from io import BytesIO

from PIL import Image

from backend.schemas.analysis import ModuleStatus
from backend.services import orchestration_service as service


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_score_maps_to_suspicion_score(monkeypatch):
    def fake_detector(image_path):
        assert image_path.endswith("normalized_image.png")
        return {
            "predicted_label": "AI-generated",
            "real_probability": 0.10,
            "ai_generated_probability": 0.90,
            "confidence": 0.90,
        }

    monkeypatch.setattr(service, "detect_ai_generated", fake_detector)

    result = service.process_uploaded_media(
        png_bytes(), "sample.png", "image", "VM-TEST-1"
    )

    module_result = result["module_results"][0]
    assert module_result["scores"]["suspicion_score"] == 0.90
    assert result["fusion"]["fused_score"] == 0.90


def test_audio_score_maps_to_suspicion_score(monkeypatch):
    def fake_audio_pipeline(file_path, output_dir):
        assert file_path.endswith("upload.wav")
        assert "VM-TEST-2" in output_dir
        return {
            "audio_score": 0.80,
            "model_version": "audio-test-1",
            "spectrogram_path": f"{output_dir}/spectrogram.png",
        }

    monkeypatch.setattr(service, "run_audio_pipeline", fake_audio_pipeline)

    result = service.process_uploaded_media(
        b"fake audio bytes", "sample.wav", "audio", "VM-TEST-2"
    )

    module_result = result["module_results"][0]
    assert module_result["scores"]["suspicion_score"] == 0.80
    assert module_result["model_versions"]["detector"] == "audio-test-1"


def test_video_is_explicitly_unavailable_for_now():
    result = service.process_uploaded_media(
        b"video bytes", "sample.mp4", "video", "VM-TEST-3"
    )

    module_result = result["module_results"][0]
    assert module_result["status"] == ModuleStatus.UNAVAILABLE.value
    assert module_result["scores"] == {}
    assert result["fusion"]["fused_score"] is None


def test_detector_exception_becomes_failed_result(monkeypatch):
    def broken_detector(image_path):
        raise RuntimeError("mock detector failure")

    monkeypatch.setattr(service, "detect_ai_generated", broken_detector)

    result = service.process_uploaded_media(
        png_bytes(), "sample.png", "image", "VM-TEST-4"
    )

    module_result = result["module_results"][0]
    assert module_result["status"] == ModuleStatus.FAILED.value
    assert "mock detector failure" in module_result["errors"][0]
    assert result["fusion"]["fused_score"] is None