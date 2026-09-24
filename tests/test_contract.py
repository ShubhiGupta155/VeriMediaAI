import pytest
from pydantic import ValidationError

from backend.schemas.analysis import MediaType, ModuleResult, ModuleStatus
from backend.schemas.evidence import EvidenceItem


def valid_result() -> ModuleResult:
    return ModuleResult(
        analysis_id="VM-0001",
        media_type=MediaType.IMAGE,
        module="image",
        status=ModuleStatus.SUCCESS,
        scores={"suspicion_score": 0.72},
        evidence=[
            EvidenceItem(
                type="compression_anomaly",
                description="Inconsistent compression detected.",
                severity=0.65,
                confidence=0.81,
                source_module="image",
            )
        ],
        model_versions={"detector": "baseline-0.1"},
    )


def test_valid_module_result():
    result = valid_result()

    assert result.analysis_id == "VM-0001"
    assert result.scores["suspicion_score"] == 0.72
    assert result.status == ModuleStatus.SUCCESS


def test_score_above_one_is_rejected():
    with pytest.raises(ValidationError):
        ModuleResult(
            analysis_id="VM-0002",
            media_type=MediaType.IMAGE,
            module="image",
            status=ModuleStatus.SUCCESS,
            scores={"suspicion_score": 1.5},
            model_versions={"detector": "baseline-0.1"},
        )


def test_negative_score_is_rejected():
    with pytest.raises(ValidationError):
        ModuleResult(
            analysis_id="VM-0003",
            media_type=MediaType.IMAGE,
            module="image",
            status=ModuleStatus.SUCCESS,
            scores={"suspicion_score": -0.1},
            model_versions={"detector": "baseline-0.1"},
        )


def test_success_requires_model_version():
    with pytest.raises(ValidationError):
        ModuleResult(
            analysis_id="VM-0004",
            media_type=MediaType.IMAGE,
            module="image",
            status=ModuleStatus.SUCCESS,
            scores={"suspicion_score": 0.5},
        )


def test_failed_result_requires_error():
    with pytest.raises(ValidationError):
        ModuleResult(
            analysis_id="VM-0005",
            media_type=MediaType.IMAGE,
            module="image",
            status=ModuleStatus.FAILED,
        )


def test_failed_result_with_error_is_valid():
    result = ModuleResult(
        analysis_id="VM-0006",
        media_type=MediaType.VIDEO,
        module="video",
        status=ModuleStatus.FAILED,
        errors=["Video model could not load."],
    )

    assert result.status == ModuleStatus.FAILED
    assert result.errors[0] == "Video model could not load."


def test_valid_lipsync_module_result():
    result = ModuleResult(
        analysis_id="VM-LIPSYNC-0001",
        media_type=MediaType.LIPSYNC,
        module="lipsync",
        status=ModuleStatus.SUCCESS,
        scores={
            "sync_score": 0.35,
            "mismatch_probability": 0.72,
            "suspicion_score": 0.72,
        },
        evidence=[
            EvidenceItem(
                type="lip_sync_mismatch",
                description="Mouth movement is misaligned with the speech segment.",
                severity=0.72,
                confidence=0.84,
                source_module="lipsync",
                timestamp_seconds=12.5,
            )
        ],
        timestamps=[
            {
                "start_seconds": 12.0,
                "end_seconds": 14.0,
                "confidence": 0.84,
            }
        ],
        model_versions={"sync_model": "baseline-0.1"},
    )

    assert result.media_type == MediaType.LIPSYNC
    assert result.module == "lipsync"
    assert result.scores["mismatch_probability"] == 0.72