from io import BytesIO

from PIL import Image

from backend.schemas.analysis import MediaType, ModuleStatus
from backend.schemas.evidence import EvidenceItem
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


def test_video_adapter_preserves_evidence_conversion(monkeypatch, tmp_path):
    representative_video_result = {
        "status": "success",
        "scores": {
            "suspicion_score": 0.88,
            "confidence": 0.92,
            "temporal_inconsistency_score": 0.35,
        },
        "evidence": [
            {
                "type": "temporal_anomaly",
                "description": "Sustained facial manipulation between 1.5s and 3.5s across 25 frames.",
                "severity": 0.85,
                "confidence": 0.94,
                "timestamp_seconds": 1.5,
                "location": {
                    "start_time_seconds": 1.5,
                    "end_time_seconds": 3.5,
                    "start_frame": 15,
                    "end_frame": 40,
                },
            },
            {
                "type": "deepfake_face_swap",
                "description": "High facial feature boundary warping at 5.0s.",
                "severity": "high",
                "confidence": 0.89,
                "start_time_seconds": 5.0,
                "end_time_seconds": 6.2,
            },
            "Identified 2 suspicious timestamp interval(s).",
        ],
        "timestamps": [
            {
                "start_time_seconds": 1.5,
                "end_time_seconds": 3.5,
                "severity": "high",
                "mean_fake_probability": 0.88,
            },
            {
                "start_time_seconds": 5.0,
                "end_time_seconds": 6.2,
                "severity": "medium",
                "mean_fake_probability": 0.72,
            },
        ],
        "warnings": [],
        "errors": [],
        "model_versions": {"deepfake_detector": "faceforensics++_xception"},
    }

    monkeypatch.setattr(
        service, "analyze_video", lambda path: representative_video_result
    )

    dummy_video_path = tmp_path / "test_video.mp4"
    dummy_video_path.write_bytes(b"mock_bytes")

    module_result = service._analyze_video(
        dummy_video_path, "VM-TEST-VIDEO-1", tmp_path
    )

    # 1. Verify schema & status
    assert module_result.status == ModuleStatus.SUCCESS
    assert module_result.media_type == MediaType.VIDEO
    assert module_result.module == "video"
    assert module_result.scores["suspicion_score"] == 0.88

    # 2. Verify evidence items are converted to EvidenceItem objects
    assert len(module_result.evidence) == 3
    for ev in module_result.evidence:
        assert isinstance(ev, EvidenceItem)
        assert ev.source_module == "video"

    # 3. Verify descriptions are preserved exactly
    assert (
        module_result.evidence[0].description
        == "Sustained facial manipulation between 1.5s and 3.5s across 25 frames."
    )
    assert (
        module_result.evidence[1].description
        == "High facial feature boundary warping at 5.0s."
    )
    assert (
        module_result.evidence[2].description
        == "Identified 2 suspicious timestamp interval(s)."
    )

    # 4. Verify severity, confidence, and timestamps are preserved
    ev0 = module_result.evidence[0]
    assert ev0.severity == 0.85
    assert ev0.confidence == 0.94
    assert ev0.timestamp_seconds == 1.5
    assert ev0.location == {
        "start_time_seconds": 1.5,
        "end_time_seconds": 3.5,
        "start_frame": 15,
        "end_frame": 40,
    }

    ev1 = module_result.evidence[1]
    assert ev1.severity == 0.9  # "high" severity mapped to 0.9
    assert ev1.confidence == 0.89
    assert ev1.timestamp_seconds == 5.0
    assert ev1.location == {
        "start_time_seconds": 5.0,
        "end_time_seconds": 6.2,
    }

    ev2 = module_result.evidence[2]
    assert ev2.severity == 0.6
    assert ev2.confidence == 0.92  # fallback to scores.confidence
    assert ev2.timestamp_seconds is None
    assert ev2.location is None

    # 5. Verify timestamps are still copied into the separate timestamps field
    assert len(module_result.timestamps) == 2
    assert module_result.timestamps[0]["start_time_seconds"] == 1.5
    assert module_result.timestamps[1]["start_time_seconds"] == 5.0


def test_video_adapter_handles_empty_evidence_safely(monkeypatch, tmp_path):
    empty_video_result = {
        "status": "success",
        "scores": {
            "suspicion_score": 0.12,
            "confidence": 0.95,
        },
        "evidence": [],
        "timestamps": [],
        "warnings": [],
        "errors": [],
        "model_versions": {"deepfake_detector": "faceforensics++_xception"},
    }

    monkeypatch.setattr(
        service, "analyze_video", lambda path: empty_video_result
    )

    dummy_video_path = tmp_path / "clean_video.mp4"
    dummy_video_path.write_bytes(b"clean_bytes")

    module_result = service._analyze_video(
        dummy_video_path, "VM-TEST-VIDEO-EMPTY", tmp_path
    )

    assert module_result.status == ModuleStatus.SUCCESS
    assert module_result.evidence == []
    assert module_result.timestamps == []
    assert module_result.scores["suspicion_score"] == 0.12
