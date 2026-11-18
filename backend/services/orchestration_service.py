import logging
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from backend.schemas.analysis import MediaType, ModuleResult, ModuleStatus
from backend.schemas.evidence import EvidenceItem
from modules.audio.pipeline import run_audio_pipeline
from modules.fusion.pipeline import run_fusion
from modules.image.detector import detect_ai_generated
from modules.video.pipeline import analyze_video

logger = logging.getLogger(__name__)


def _failed_result(
    analysis_id: str,
    media_type: MediaType,
    module: str,
    error: Exception,
) -> ModuleResult:
    return ModuleResult(
        analysis_id=analysis_id,
        media_type=media_type,
        module=module,
        status=ModuleStatus.FAILED,
        errors=[f"{type(error).__name__}: {error}"],
    )


def _analyze_image(
    file_path: Path,
    analysis_id: str,
    workspace: Path,
) -> ModuleResult:
    # Decode and normalize to RGB PNG before calling the detector.
    # This also makes unsupported image formats fail clearly here.
    normalized_path = workspace / "normalized_image.png"
    with Image.open(file_path) as image:
        image.convert("RGB").save(normalized_path, format="PNG")

    raw: dict[str, Any] = detect_ai_generated(str(normalized_path))
    score = float(raw["ai_generated_probability"])
    confidence = float(raw["confidence"])

    warnings = []
    model_version = raw.get("model_version")
    if not model_version:
        # Ask the image-module owner for the exact version when available.
        model_version = "not-reported-by-module"
        warnings.append("Image detector did not report a model version.")

    return ModuleResult(
        analysis_id=analysis_id,
        media_type=MediaType.IMAGE,
        module="image",
        status=ModuleStatus.SUCCESS,
        scores={
            "suspicion_score": score,
            "confidence": confidence,
        },
        evidence=[],
        artifacts=[],
        timestamps=[],
        warnings=warnings,
        errors=[],
        model_versions={"detector": str(model_version)},
    )


def _analyze_audio(
    file_path: Path,
    analysis_id: str,
    workspace: Path,
) -> ModuleResult:
    output_dir = workspace / "audio_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = run_audio_pipeline(
        str(file_path),
        output_dir=str(output_dir),
    )
    score = float(raw["audio_score"])

    warnings = [
        "Audio spectrogram output is temporary and is removed after analysis."
    ]
    model_version = raw.get("model_version")
    if not model_version:
        model_version = "not-reported-by-module"
        warnings.append("Audio pipeline did not report a model version.")

    return ModuleResult(
        analysis_id=analysis_id,
        media_type=MediaType.AUDIO,
        module="audio",
        status=ModuleStatus.SUCCESS,
        scores={"suspicion_score": score},
        evidence=[],
        artifacts=[],
        timestamps=[],
        warnings=warnings,
        errors=[],
        model_versions={"detector": str(model_version)},
    )


_SEVERITY_TEXT_TO_FLOAT: dict[str, float] = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
    "critical": 1.0,
}


def _map_severity(severity_input: Any, default: float = 0.6) -> float:
    """Map string or numeric severity to a float in [0.0, 1.0]."""
    if isinstance(severity_input, (int, float)):
        return max(0.0, min(1.0, float(severity_input)))
    if isinstance(severity_input, str):
        cleaned = severity_input.strip().lower()
        if cleaned in _SEVERITY_TEXT_TO_FLOAT:
            return _SEVERITY_TEXT_TO_FLOAT[cleaned]
        try:
            return max(0.0, min(1.0, float(cleaned)))
        except ValueError:
            pass
    return default


_INTERVAL_EVIDENCE_RE = re.compile(
    r"Interval\s+\[\s*(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\s*\]"
    r"\s+\(frames\s+(\d+)\.\.(\d+)\):\s*(.*)"
)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return None
    return parsed if math.isfinite(parsed) else None


def _frame_number(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _interval_severity(interval: dict[str, Any]) -> float | None:
    value = interval.get("severity")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = _finite_float(value)
        if parsed is not None and 0.0 <= parsed <= 1.0:
            return parsed
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in _SEVERITY_TEXT_TO_FLOAT:
            return _SEVERITY_TEXT_TO_FLOAT[cleaned]
        parsed = _finite_float(cleaned)
        if parsed is not None and 0.0 <= parsed <= 1.0:
            return parsed
    return None


def _interval_confidence(interval: dict[str, Any]) -> float | None:
    for key in ("mean_fake_probability", "confidence"):
        parsed = _finite_float(interval.get(key))
        if parsed is not None and 0.0 <= parsed <= 1.0:
            return parsed
    return None


def _interval_location(interval: dict[str, Any]) -> dict[str, Any] | None:
    location: dict[str, Any] = {}
    for key in ("start_time_seconds", "end_time_seconds"):
        value = _finite_float(interval.get(key))
        if value is not None:
            location[key] = value
    for key in ("start_frame", "end_frame"):
        value = _frame_number(interval.get(key))
        if value is not None:
            location[key] = value
    return location or None


def _find_matching_interval(
    statement: str,
    intervals: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the unique structured interval matching a formatted pipeline statement."""
    if not isinstance(statement, str):
        return None

    match = _INTERVAL_EVIDENCE_RE.fullmatch(statement.strip())
    if match is None:
        return None

    statement_start = float(match.group(1))
    statement_end = float(match.group(2))
    statement_start_frame = int(match.group(3))
    statement_end_frame = int(match.group(4))
    matching_intervals = []

    for interval in intervals:
        if not isinstance(interval, dict):
            continue

        start_t = _finite_float(interval.get("start_time_seconds"))
        end_t = _finite_float(interval.get("end_time_seconds"))
        start_frame = _frame_number(interval.get("start_frame"))
        end_frame = _frame_number(interval.get("end_frame"))
        has_time_bounds = start_t is not None and end_t is not None
        has_frame_bounds = start_frame is not None and end_frame is not None
        if not has_time_bounds and not has_frame_bounds:
            continue

        if start_t is not None and f"{start_t:.2f}" != f"{statement_start:.2f}":
            continue
        if end_t is not None and f"{end_t:.2f}" != f"{statement_end:.2f}":
            continue
        if start_frame is not None and start_frame != statement_start_frame:
            continue
        if end_frame is not None and end_frame != statement_end_frame:
            continue
        matching_intervals.append(interval)

    return matching_intervals[0] if len(matching_intervals) == 1 else None


def _build_video_evidence(
    raw_video_result: dict[str, Any],
) -> list[EvidenceItem]:
    """Convert raw video evidence into a list of EvidenceItem objects.

    Handles evidence entries provided as:
    - Formatted evidence statement strings (e.g. from format_timestamp_evidence())
      which are correlated with structured intervals in ``raw_video_result["timestamps"]``
      to preserve severity, confidence, timestamps, and bounding frames.
    - Structured dicts (with description/reason, severity, confidence, timestamp).
    - Existing EvidenceItem instances.

    Preserves exact descriptions, maps severity, preserves confidence and
    timestamps, and safely handles missing optional fields.
    """
    raw_evidence = raw_video_result.get("evidence")
    timestamps = raw_video_result.get("timestamps") or []

    if not raw_evidence and not timestamps:
        return []

    evidence_items: list[EvidenceItem] = []
    scores = raw_video_result.get("scores") or {}
    fallback_confidence = scores.get("confidence")
    if fallback_confidence is not None:
        try:
            fallback_confidence = max(0.0, min(1.0, float(fallback_confidence)))
        except (ValueError, TypeError):
            fallback_confidence = 0.5
    else:
        fallback_confidence = 0.5

    for entry in (raw_evidence or []):
        if isinstance(entry, EvidenceItem):
            evidence_items.append(entry)
            continue

        if isinstance(entry, str):
            desc = entry.strip()
            if not desc:
                continue

            # Check if this statement corresponds to a known structured interval
            matched_interval = _find_matching_interval(
                desc, timestamps if isinstance(timestamps, list) else []
            )
            if matched_interval is not None:
                severity_val = _interval_severity(matched_interval)
                conf = _interval_confidence(matched_interval)

                if severity_val is None or conf is None:
                    logger.warning(
                        "Skipping video interval evidence with missing or invalid "
                        "severity/confidence metadata: %s",
                        desc,
                    )
                    continue

                ts = _finite_float(matched_interval.get("start_time_seconds"))
                loc = _interval_location(matched_interval)

                ev_type = str(
                    matched_interval.get("evidence_type")
                    or "sustained_deepfake_signal"
                )

                evidence_items.append(
                    EvidenceItem(
                        type=ev_type,
                        description=desc,
                        severity=severity_val,
                        confidence=conf,
                        source_module="video",
                        location=loc,
                        timestamp_seconds=ts,
                    )
                )
                continue

            if desc.startswith("Interval ["):
                logger.warning(
                    "Skipping unmatched video interval evidence without reliable "
                    "timestamp metadata: %s",
                    desc,
                )
                continue

            # Non-interval statements (summary line, instability warning, edge-case notice)
            lower_desc = desc.lower()
            if "instability" in lower_desc or "temporal variation" in lower_desc:
                ev_type = "temporal_instability"
                sev = 0.8 if "high" in lower_desc else 0.5
            elif "suspicious timestamp interval" in lower_desc:
                ev_type = "temporal_summary"
                sev = 0.6
            else:
                ev_type = "video_forensic_evidence"
                sev = 0.5

            evidence_items.append(
                EvidenceItem(
                    type=ev_type,
                    description=desc,
                    severity=sev,
                    confidence=fallback_confidence,
                    source_module="video",
                    location=None,
                    timestamp_seconds=None,
                )
            )
            continue

        if isinstance(entry, dict):
            description = (
                entry.get("description")
                or entry.get("reason")
                or entry.get("statement")
            )
            if not description or not str(description).strip():
                continue
            description = str(description).strip()

            ev_type = str(
                entry.get("type")
                or entry.get("evidence_type")
                or "video_forensic_evidence"
            )

            severity_val = _map_severity(entry.get("severity"), default=0.6)

            conf = entry.get("confidence")
            if conf is None:
                conf = entry.get("mean_fake_probability")
            if conf is not None:
                try:
                    conf = max(0.0, min(1.0, float(conf)))
                except (ValueError, TypeError):
                    conf = fallback_confidence
            else:
                conf = fallback_confidence

            ts = entry.get("timestamp_seconds")
            if ts is None:
                ts = entry.get("timestamp")
            if ts is None:
                ts = entry.get("start_time_seconds")
            if ts is not None:
                try:
                    ts = float(ts)
                except (ValueError, TypeError):
                    ts = None

            loc = entry.get("location")
            if loc is None and (
                entry.get("start_time_seconds") is not None
                or entry.get("end_time_seconds") is not None
            ):
                loc_data = {}
                if entry.get("start_time_seconds") is not None:
                    loc_data["start_time_seconds"] = entry["start_time_seconds"]
                if entry.get("end_time_seconds") is not None:
                    loc_data["end_time_seconds"] = entry["end_time_seconds"]
                if entry.get("start_frame") is not None:
                    loc_data["start_frame"] = entry["start_frame"]
                if entry.get("end_frame") is not None:
                    loc_data["end_frame"] = entry["end_frame"]
                loc = loc_data if loc_data else None

            source_mod = str(entry.get("source_module") or "video")

            evidence_items.append(
                EvidenceItem(
                    type=ev_type,
                    description=description,
                    severity=severity_val,
                    confidence=conf,
                    source_module=source_mod,
                    location=loc,
                    timestamp_seconds=ts,
                )
            )

    # If raw_evidence was empty/absent but timestamps were provided, construct from timestamps:
    if not raw_evidence and isinstance(timestamps, list) and timestamps:
        for interval in timestamps:
            if not isinstance(interval, dict):
                continue
            reason = interval.get("reason", "")
            if not reason:
                logger.warning(
                    "Skipping video timestamp interval without a forensic reason: %s",
                    interval,
                )
                continue

            severity_val = _interval_severity(interval)
            conf = _interval_confidence(interval)
            if severity_val is None or conf is None:
                logger.warning(
                    "Skipping video timestamp interval with missing or invalid "
                    "severity/confidence metadata: %s",
                    interval,
                )
                continue

            ts = _finite_float(interval.get("start_time_seconds"))
            loc = _interval_location(interval)

            evidence_items.append(
                EvidenceItem(
                    type=interval.get("evidence_type", "sustained_deepfake_signal"),
                    description=reason,
                    severity=severity_val,
                    confidence=conf,
                    source_module="video",
                    location=loc,
                    timestamp_seconds=ts,
                )
            )

    return evidence_items


def adapt_video_result(
    raw_video_result: dict[str, Any],
    analysis_id: str = "VM-VIDEO",
) -> ModuleResult:
    """Adapt raw video forensics pipeline output into standardized ModuleResult."""
    pipeline_status = raw_video_result.get("status", "failed")
    if pipeline_status == "success":
        status = ModuleStatus.SUCCESS
    elif pipeline_status == "insufficient_evidence":
        status = ModuleStatus.UNAVAILABLE
    else:
        status = ModuleStatus.FAILED

    evidence_items = _build_video_evidence(raw_video_result)

    warnings = list(raw_video_result.get("warnings") or [])
    model_versions = dict(raw_video_result.get("model_versions") or {})
    if not model_versions:
        model_versions = {"detector": "not-reported-by-module"}
        warnings.append("Video pipeline did not report a model version.")
    elif "deepfake_detector" in model_versions and "detector" not in model_versions:
        model_versions["detector"] = model_versions.pop("deepfake_detector")

    errors = list(raw_video_result.get("errors") or [])
    if status == ModuleStatus.FAILED and not errors:
        errors = ["Video analysis failed."]

    return ModuleResult(
        analysis_id=analysis_id,
        media_type=MediaType.VIDEO,
        module="video",
        status=status,
        scores=raw_video_result.get("scores") or {},
        evidence=evidence_items,
        artifacts=list(raw_video_result.get("artifacts") or []),
        timestamps=list(raw_video_result.get("timestamps") or []),
        warnings=warnings,
        errors=errors,
        model_versions=model_versions,
    )


def _analyze_video(
    file_path: Path,
    analysis_id: str,
    workspace: Path | None = None,
) -> ModuleResult:
    raw: dict[str, Any] = analyze_video(str(file_path))
    return adapt_video_result(raw, analysis_id=analysis_id)


def _unavailable_result(
    analysis_id: str,
    media_type: MediaType,
    module: str,
    message: str,
) -> ModuleResult:
    return ModuleResult(
        analysis_id=analysis_id,
        media_type=media_type,
        module=module,
        status=ModuleStatus.UNAVAILABLE,
        scores={},
        evidence=[],
        artifacts=[],
        timestamps=[],
        warnings=[message],
        errors=[],
        model_versions={},
    )


def process_uploaded_media(
    contents: bytes,
    filename: str,
    media_type: str,
    analysis_id: str,
) -> dict[str, Any]:
    """Run the supported detector, normalize its result, and fuse it.

    The uploaded source and intermediate outputs live in a temporary
    analysis-specific directory and are removed after processing.
    """
    suffix = Path(filename).suffix.lower()

    with tempfile.TemporaryDirectory(prefix=f"{analysis_id}_") as temp_dir:
        workspace = Path(temp_dir)
        input_path = workspace / f"upload{suffix}"
        input_path.write_bytes(contents)

        if media_type == "image":
            try:
                module_result = _analyze_image(
                    input_path, analysis_id, workspace
                )
            except Exception as exc:
                module_result = _failed_result(
                    analysis_id, MediaType.IMAGE, "image", exc
                )

        elif media_type == "audio":
            try:
                module_result = _analyze_audio(
                    input_path, analysis_id, workspace
                )
            except Exception as exc:
                module_result = _failed_result(
                    analysis_id, MediaType.AUDIO, "audio", exc
                )

        elif media_type == "video":
            module_result = _unavailable_result(
                analysis_id,
                MediaType.VIDEO,
                "video",
                "Video orchestration is awaiting the confirmed video score mapping.",
            )

        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        fusion_result = run_fusion([module_result])

        if fusion_result["fusion"]["fused_score"] is not None:
            analysis_status = "completed"
        elif module_result.status == ModuleStatus.FAILED:
            analysis_status = "failed"
        else:
            analysis_status = "partial"

        return {
            "status": analysis_status,
            "module_results": [
                module_result.model_dump(mode="json")
            ],
            "fusion": fusion_result["fusion"],
            "risk_assessment": fusion_result["risk_assessment"],
        }