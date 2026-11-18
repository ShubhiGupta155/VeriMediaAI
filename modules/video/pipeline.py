"""
Video Forensics Pipeline Orchestrator for VeriMedia AI.

Connects and executes the individual video forensics stages:
Video Input
    ↓
Metadata Extraction
    ↓
Frame Extraction
    ↓
Face Detection
    ↓
Face Cropping
    ↓
Xception Deepfake Detection
    ↓
Frame-Level Predictions
    ↓
Temporal Consistency Analysis
    ↓
Suspicious Timestamp Analysis
    ↓
Standardized Forensic Result

Returns structured results suitable for backend consumption and multimodal fusion.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from modules.video.metadata import extract_video_metadata
from modules.video.frame_extractor import extract_frames
from modules.video.face_detector import detect_faces
from modules.video.deepfake_detector import (
    DeepfakeModel,
    detect_deepfake,
    load_deepfake_model,
)
from modules.video.temporal_analysis import analyze_temporal_consistency
from modules.video.timestamp_analyzer import (
    extract_suspicious_timestamps,
    format_timestamp_evidence,
)

logger = logging.getLogger(__name__)


def analyze_video(
    video_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    sample_interval: float = 1.0,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    enter_threshold: float = 0.65,
    recovery_threshold: float = 0.50,
    tau: float = 1.5,
    max_gap_seconds: float = 3.0,
    min_segment_frames: int = 2,
    min_segment_duration: float = 1.0,
    model: Optional[DeepfakeModel] = None,
) -> Dict[str, Any]:
    """
    Run the complete end-to-end video forensics pipeline on a video file.

    Parameters
    ----------
    video_path : str or Path
        Path to the target video file.
    model_path : str or Path, optional
        Path to local FaceForensics++ Xception PyTorch checkpoint.
        If None, attempts to resolve from VERIMEDIA_XCEPTION_CHECKPOINT env var
        or pre-cached model instance.
    sample_interval : float, default=1.0
        Time interval in seconds between sampled frames (must be > 0).
    max_frames : int, optional
        Maximum number of sampled frames to process (if specified, must be > 0).
    device : str, optional
        Device for deepfake model inference ('cuda' or 'cpu').
    enter_threshold : float, default=0.65
        Temporal hysteresis entry threshold for suspicious segment detection.
    recovery_threshold : float, default=0.50
        Temporal hysteresis recovery/exit threshold.
    tau : float, default=1.5
        Time constant in seconds for time-aware EWMA smoothing.
    max_gap_seconds : float, default=3.0
        Maximum time gap in seconds before EWMA state is reset.
    min_segment_frames : int, default=2
        Minimum frame count required for a suspicious segment.
    min_segment_duration : float, default=1.0
        Minimum duration in seconds required for a suspicious segment.
    model : DeepfakeModel, optional
        Pre-loaded DeepfakeModel instance to reuse across pipeline runs.

    Returns
    -------
    dict
        Standardized video-forensics result containing:
        - module: str ("video")
        - status: str ("success", "insufficient_evidence", "failed")
        - scores: dict (contains "suspicion_score", "temporal_inconsistency_score", etc.)
        - video_metadata: dict
        - frame_predictions: list of dict
        - temporal_analysis: dict
        - timestamps: list of dict (suspicious timestamp intervals)
        - evidence: list of str
        - warnings: list of str
        - errors: list of str
        - model_versions: dict

    Raises
    ------
    FileNotFoundError
        If video_path does not exist.
    ValueError
        If video_path is unreadable or parameters are invalid.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    warnings: List[str] = []
    errors: List[str] = []

    # 1. Video Metadata Extraction
    metadata = extract_video_metadata(str(path))

    # 2. Frame Extraction
    frames = extract_frames(
        str(path),
        sample_interval=sample_interval,
        max_frames=max_frames,
    )

    if not frames:
        warnings.append("No frames could be extracted from video.")
        return {
            "module": "video",
            "status": "insufficient_evidence",
            "scores": {},
            "video_metadata": metadata,
            "frame_predictions": [],
            "temporal_analysis": {
                "total_predictions": 0,
                "status": "empty",
                "evidence": ["No frames available for face detection."],
            },
            "timestamps": [],
            "evidence": ["Insufficient video frames to conduct deepfake analysis."],
            "warnings": warnings,
            "errors": errors,
            "model_versions": {"deepfake_detector": "faceforensics++_xception"},
        }

    # 3. Model Loading (or resolve pre-cached/configured model)
    detector_model = model
    if detector_model is None:
        try:
            detector_model = load_deepfake_model(model_path=model_path, device=device)
        except Exception as e:
            errors.append(f"ModelLoadError: {e}")
            return {
                "module": "video",
                "status": "failed",
                "scores": {},
                "video_metadata": metadata,
                "frame_predictions": [],
                "temporal_analysis": {},
                "timestamps": [],
                "evidence": [],
                "warnings": warnings,
                "errors": errors,
                "model_versions": {"deepfake_detector": "faceforensics++_xception"},
            }

    if detector_model is None:
        warnings.append(
            "No Xception checkpoint supplied or configured. Provide a local checkpoint path "
            "or set VERIMEDIA_XCEPTION_CHECKPOINT."
        )
        return {
            "module": "video",
            "status": "insufficient_evidence",
            "scores": {},
            "video_metadata": metadata,
            "frame_predictions": [],
            "temporal_analysis": {
                "total_predictions": 0,
                "status": "empty",
                "evidence": ["Deepfake detection skipped: checkpoint not provided."],
            },
            "timestamps": [],
            "evidence": ["Deepfake analysis unavailable: model checkpoint was not configured."],
            "warnings": warnings,
            "errors": errors,
            "model_versions": {"deepfake_detector": "faceforensics++_xception"},
        }

    # 4. Face Detection & Deepfake Inference per Frame
    frame_predictions: List[Dict[str, Any]] = []
    frames_with_faces = 0

    for item in frames:
        frame_img = item["frame"]
        frame_num = item["frame_number"]
        t_sec = item["timestamp_seconds"]

        try:
            detections = detect_faces(frame_img)
        except Exception as e:
            logger.warning(f"Face detection failed on frame {frame_num}: {e}")
            continue

        if not detections:
            continue

        frames_with_faces += 1

        # Evaluate detected faces in this frame
        # If multiple faces exist, evaluate the primary / largest face crop
        primary_detection = max(
            detections,
            key=lambda d: d["bbox"]["width"] * d["bbox"]["height"],
        )
        bbox = primary_detection["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # Validate and clamp crop bounds
        img_h, img_w = frame_img.shape[:2]
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(0, min(x + w, img_w))
        y2 = max(0, min(y + h, img_h))

        face_crop = frame_img[y1:y2, x1:x2]
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        try:
            pred = detect_deepfake(face_crop, model=detector_model)
            frame_predictions.append({
                "frame_number": frame_num,
                "timestamp_seconds": t_sec,
                "fake_probability": pred["fake_probability"],
                "real_probability": pred["real_probability"],
                "label": pred["label"],
                "confidence": pred.get("confidence", max(pred["fake_probability"], pred["real_probability"])),
                "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
            })
        except Exception as e:
            logger.warning(f"Inference failed on frame {frame_num}: {e}")
            continue

    # 5. Handle No-Face or Zero-Prediction Edge Case
    if not frame_predictions:
        warnings.append(
            f"No recognizable faces detected across {len(frames)} extracted video frames."
        )
        return {
            "module": "video",
            "status": "insufficient_evidence",
            "scores": {},
            "video_metadata": metadata,
            "frame_predictions": [],
            "temporal_analysis": {
                "total_predictions": 0,
                "status": "empty",
                "evidence": ["No faces detected in sampled video frames."],
            },
            "timestamps": [],
            "evidence": ["No facial evidence detected to perform deepfake verification."],
            "warnings": warnings,
            "errors": errors,
            "model_versions": {"deepfake_detector": "faceforensics++_xception"},
        }

    # 6. Temporal Consistency Analysis
    temporal_summary = analyze_temporal_consistency(
        frame_predictions,
        enter_threshold=enter_threshold,
        recovery_threshold=recovery_threshold,
        tau=tau,
        max_gap_seconds=max_gap_seconds,
        min_frames=min_segment_frames,
        min_duration=min_segment_duration,
    )

    # 7. Suspicious Timestamp Extraction
    suspicious_intervals = extract_suspicious_timestamps(temporal_summary)
    evidence_statements = format_timestamp_evidence(
        suspicious_intervals,
        inconsistency_score=temporal_summary.get("temporal_inconsistency_score"),
    )

    # 8. Standardized Output Formulation
    # suspicion_score is defined as global_mean_fake_prob (range 0.0 - 1.0)
    # Higher value = stronger deepfake indicator
    suspicion_score = float(temporal_summary["global_mean_fake_prob"])
    confidence_val = float(
        sum(p["confidence"] for p in frame_predictions) / len(frame_predictions)
    )

    scores = {
        "suspicion_score": round(suspicion_score, 4),
        "confidence": round(confidence_val, 4),
        "peak_fake_probability": float(temporal_summary["global_peak_fake_prob"]),
        "temporal_inconsistency_score": float(temporal_summary["temporal_inconsistency_score"]),
    }

    return {
        "module": "video",
        "status": "success",
        "scores": scores,
        "video_metadata": metadata,
        "frame_predictions": frame_predictions,
        "temporal_analysis": temporal_summary,
        "timestamps": suspicious_intervals,
        "evidence": evidence_statements,
        "warnings": warnings,
        "errors": errors,
        "model_versions": {"deepfake_detector": "faceforensics++_xception"},
    }
