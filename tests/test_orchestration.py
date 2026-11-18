from io import BytesIO

import numpy as np
from PIL import Image

from backend.schemas.analysis import MediaType, ModuleStatus
from backend.schemas.evidence import EvidenceItem
from backend.services import orchestration_service as service
from modules.video import pipeline as video_pipeline
from modules.video.timestamp_analyzer import (
    extract_suspicious_timestamps,
    format_timestamp_evidence,
)


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


def test_video_adapter_links_actual_pipeline_evidence_and_intervals(monkeypatch, tmp_path):
    """Correlate the actual analyze_video output with its timestamp intervals."""
    fake_probabilities = iter([0.9, 0.9, 0.1, 0.1, 0.7, 0.7, 0.1, 0.1])
    frame_numbers = [0, 30, 60, 90, 300, 330, 360, 390]
    timestamps = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
    frames = [
        {
            "frame": np.zeros((32, 32, 3), dtype=np.uint8),
            "frame_number": frame_number,
            "timestamp_seconds": timestamp,
        }
        for frame_number, timestamp in zip(frame_numbers, timestamps)
    ]

    def fake_detect_deepfake(crop, model):
        fake_probability = next(fake_probabilities)
        return {
            "fake_probability": fake_probability,
            "real_probability": 1.0 - fake_probability,
            "label": "fake" if fake_probability > 0.5 else "real",
            "confidence": max(fake_probability, 1.0 - fake_probability),
        }

    monkeypatch.setattr(
        video_pipeline,
        "extract_video_metadata",
        lambda path: {
            "filename": "sample.mp4",
            "duration_seconds": 14.0,
            "fps": 30.0,
            "frame_count": 420,
        },
    )
    monkeypatch.setattr(
        video_pipeline, "extract_frames", lambda *args, **kwargs: frames
    )
    monkeypatch.setattr(
        video_pipeline,
        "detect_faces",
        lambda frame: [{"bbox": {"x": 2, "y": 2, "width": 20, "height": 20}}],
    )
    monkeypatch.setattr(
        video_pipeline, "load_deepfake_model", lambda **kwargs: object()
    )
    monkeypatch.setattr(video_pipeline, "detect_deepfake", fake_detect_deepfake)

    dummy_video_path = tmp_path / "actual_shape_video.mp4"
    dummy_video_path.write_bytes(b"actual_pipeline_mock")

    module_result = service._analyze_video(
        dummy_video_path, "VM-TEST-VIDEO-ACTUAL", tmp_path
    )

    assert module_result.status == ModuleStatus.SUCCESS
    assert module_result.scores["suspicion_score"] == 0.45
    assert len(module_result.evidence) == 4

    # 1. Verify summary evidence item
    ev_summary = module_result.evidence[0]
    assert ev_summary.type == "temporal_summary"
    assert ev_summary.description == "Identified 2 suspicious timestamp interval(s)."
    assert ev_summary.severity == 0.6
    assert ev_summary.timestamp_seconds is None

    # 2. Verify interval 0 statement is correlated with interval metadata
    ev_int0 = module_result.evidence[1]
    assert ev_int0.type == "sustained_deepfake_signal"
    assert "Interval [0.00s - 2.00s]" in ev_int0.description
    assert ev_int0.severity == 0.9
    assert ev_int0.confidence == 0.7702
    assert ev_int0.timestamp_seconds == 0.0
    assert ev_int0.location == {
        "start_time_seconds": 0.0,
        "end_time_seconds": 2.0,
        "start_frame": 0,
        "end_frame": 60,
    }

    # 3. Verify interval 1 statement is correlated with interval metadata
    ev_int1 = module_result.evidence[2]
    assert ev_int1.type == "sustained_deepfake_signal"
    assert "Interval [10.00s - 11.00s]" in ev_int1.description
    assert ev_int1.severity == 0.6
    assert ev_int1.confidence == 0.7
    assert ev_int1.timestamp_seconds == 10.0
    assert ev_int1.location == {
        "start_time_seconds": 10.0,
        "end_time_seconds": 11.0,
        "start_frame": 300,
        "end_frame": 330,
    }

    # 4. Verify instability warning evidence item
    ev_instab = module_result.evidence[3]
    assert ev_instab.type == "temporal_instability"
    assert "High temporal instability" in ev_instab.description
    assert ev_instab.severity == 0.8
    assert ev_instab.timestamp_seconds is None

    # 5. Verify timestamps field is still copied separately
    assert len(module_result.timestamps) == 2
    assert module_result.timestamps[0]["start_time_seconds"] == 0.0
    assert module_result.timestamps[1]["start_time_seconds"] == 10.0


def test_video_adapter_does_not_invent_unmatched_interval_metadata():
    source_intervals = extract_suspicious_timestamps({
        "suspicious_segments": [
            {
                "start_time_seconds": 1.0,
                "end_time_seconds": 2.0,
                "start_frame": 30,
                "end_frame": 60,
                "mean_fake_probability": 0.8,
                "severity": "high",
            }
        ]
    })
    statement = format_timestamp_evidence(source_intervals)[1]
    mismatched_interval = {
        **source_intervals[0],
        "start_time_seconds": 4.0,
        "end_time_seconds": 5.0,
        "start_frame": 120,
        "end_frame": 150,
    }

    evidence = service._build_video_evidence({
        "evidence": [statement],
        "timestamps": [mismatched_interval],
        "scores": {"confidence": 0.99},
    })

    assert evidence == []


def test_video_adapter_preserves_available_metadata_from_incomplete_interval():
    source_intervals = extract_suspicious_timestamps({
        "suspicious_segments": [
            {
                "start_time_seconds": 2.0,
                "end_time_seconds": 3.0,
                "start_frame": 60,
                "end_frame": 90,
                "mean_fake_probability": 0.75,
                "severity": "medium",
            }
        ]
    })
    statement = format_timestamp_evidence(source_intervals)[1]
    incomplete_interval = {
        "start_frame": 60,
        "end_frame": 90,
        "severity": "medium",
        "mean_fake_probability": 0.75,
        "evidence_type": "sustained_deepfake_signal",
    }

    evidence = service._build_video_evidence({
        "evidence": [statement],
        "timestamps": [incomplete_interval],
        "scores": {"confidence": 0.99},
    })

    assert len(evidence) == 1
    assert evidence[0].severity == 0.6
    assert evidence[0].confidence == 0.75
    assert evidence[0].timestamp_seconds is None
    assert evidence[0].location == {"start_frame": 60, "end_frame": 90}


def test_video_adapter_skips_malformed_interval_metadata(caplog):
    source_intervals = extract_suspicious_timestamps({
        "suspicious_segments": [
            {
                "start_time_seconds": 2.0,
                "end_time_seconds": 3.0,
                "start_frame": 60,
                "end_frame": 90,
                "mean_fake_probability": 0.75,
                "severity": "medium",
            }
        ]
    })
    statement = format_timestamp_evidence(source_intervals)[1]

    evidence = service._build_video_evidence({
        "evidence": [statement],
        "timestamps": [{
            "start_time_seconds": "not-a-time",
            "end_time_seconds": None,
            "start_frame": "not-a-frame",
            "severity": "medium",
        }],
        "scores": {"confidence": 0.99},
    })

    assert evidence == []
    assert "Skipping unmatched video interval evidence" in caplog.text


def test_video_adapter_does_not_fallback_for_missing_interval_confidence(caplog):
    source_intervals = extract_suspicious_timestamps({
        "suspicious_segments": [
            {
                "start_time_seconds": 2.0,
                "end_time_seconds": 3.0,
                "start_frame": 60,
                "end_frame": 90,
                "mean_fake_probability": 0.75,
                "severity": "medium",
            }
        ]
    })
    statement = format_timestamp_evidence(source_intervals)[1]
    incomplete_interval = {
        "start_time_seconds": 2.0,
        "end_time_seconds": 3.0,
        "start_frame": 60,
        "end_frame": 90,
        "severity": "medium",
    }

    evidence = service._build_video_evidence({
        "evidence": [statement],
        "timestamps": [incomplete_interval],
        "scores": {"confidence": 0.99},
    })

    assert evidence == []
    assert "missing or invalid severity/confidence metadata" in caplog.text


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
