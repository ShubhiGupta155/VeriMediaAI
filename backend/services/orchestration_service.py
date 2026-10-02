import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from backend.schemas.analysis import MediaType, ModuleResult, ModuleStatus
from modules.audio.pipeline import run_audio_pipeline
from modules.fusion.pipeline import run_fusion
from modules.image.detector import detect_ai_generated


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