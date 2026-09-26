import pytest

from backend.schemas.analysis import MediaType, ModuleResult, ModuleStatus
from modules.fusion.pipeline import run_fusion
from modules.fusion.risk_engine import assess_risk
from modules.fusion.strategies import weighted_fusion
from modules.fusion.validator import validate_module_results


def make_result(
    module: str,
    score: float | None,
    status: ModuleStatus = ModuleStatus.SUCCESS,
) -> ModuleResult:
    scores = {} if score is None else {"suspicion_score": score}
    errors = ["Test module failure."] if status == ModuleStatus.FAILED else []
    model_versions = (
        {"test_model": "1.0"}
        if status in (ModuleStatus.SUCCESS, ModuleStatus.PARTIAL)
        else {}
    )

    return ModuleResult(
        analysis_id="VM-TEST-0001",
        media_type=MediaType.LIPSYNC if module == "lipsync" else MediaType.IMAGE,
        module=module,
        status=status,
        scores=scores,
        errors=errors,
        model_versions=model_versions,
    )


def test_weighted_fusion_normalizes_weights():
    results = [
        make_result("image", 0.70),
        make_result("video", 0.60),
    ]

    result = weighted_fusion(
        results,
        weights={"image": 0.40, "video": 0.40},
    )

    assert result["fused_score"] == pytest.approx(0.65)
    assert result["active_modules"] == ["image", "video"]


def test_unavailable_module_is_excluded_not_scored_as_zero():
    results = [
        make_result("image", 0.70),
        make_result("audio", None, ModuleStatus.UNAVAILABLE),
    ]

    eligible, excluded, _ = validate_module_results(results)
    result = weighted_fusion(
        eligible,
        weights={"image": 0.40, "audio": 0.20},
    )

    assert result["fused_score"] == pytest.approx(0.70)
    assert "audio" not in result["active_modules"]
    assert any(item["module"] == "audio" for item in excluded)


def test_failed_module_is_excluded():
    results = [
        make_result("image", 0.70),
        make_result("video", None, ModuleStatus.FAILED),
    ]

    eligible, excluded, _ = validate_module_results(results)

    assert [item.module for item in eligible] == ["image"]
    assert excluded[0]["module"] == "video"


def test_partial_result_with_score_can_participate():
    results = [
        make_result("image", 0.60, ModuleStatus.PARTIAL),
    ]

    eligible, excluded, warnings = validate_module_results(results)

    assert [item.module for item in eligible] == ["image"]
    assert excluded == []
    assert warnings


def test_result_without_suspicion_score_is_excluded():
    results = [make_result("image", None)]

    eligible, excluded, _ = validate_module_results(results)

    assert eligible == []
    assert excluded[0]["reason"] == "Result has no suspicion_score."


def test_single_usable_module_keeps_its_score():
    result = weighted_fusion(
        [make_result("image", 0.73)],
        weights={"image": 0.40},
    )

    assert result["fused_score"] == pytest.approx(0.73)


def test_no_usable_modules_returns_no_score():
    result = run_fusion(
        [
            make_result("audio", None, ModuleStatus.UNAVAILABLE),
        ],
        weights={"audio": 0.20},
    )

    assert result["fusion"]["fused_score"] is None
    assert result["risk_assessment"]["assessment"] == "Inconclusive"


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.00, "Low Concern"),
        (0.29, "Low Concern"),
        (0.30, "Inconclusive"),
        (0.59, "Inconclusive"),
        (0.60, "Elevated Concern"),
        (0.79, "Elevated Concern"),
        (0.80, "High Concern"),
        (1.00, "High Concern"),
    ],
)
def test_risk_thresholds(score, expected):
    result = assess_risk(score)

    assert result["assessment"] == expected


def test_conflicting_module_scores_are_inconclusive():
    result = assess_risk(
        0.50,
        module_scores={"image": 0.10, "video": 0.90},
    )

    assert result["assessment"] == "Inconclusive"
    assert result["conflicting_modules"] == ["image", "video"]
    assert result["human_review_recommended"] is True


def test_invalid_custom_weight_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_fusion(
            [make_result("image", 0.5)],
            weights={"image": -0.1},
        )


@pytest.mark.parametrize("invalid_score", [-0.01, 1.01])
def test_invalid_suspicion_score_is_rejected(invalid_score):
    result = make_result("image", 0.5).model_copy(
        update={"scores": {"suspicion_score": invalid_score}}
    )

    with pytest.raises(ValueError, match="Invalid suspicion_score"):
        weighted_fusion([result], weights={"image": 0.40})        