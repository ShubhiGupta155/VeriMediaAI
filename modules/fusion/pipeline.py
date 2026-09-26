from typing import Any

from backend.schemas.analysis import ModuleResult

from modules.fusion.risk_engine import assess_risk
from modules.fusion.strategies import DEFAULT_WEIGHTS, weighted_fusion
from modules.fusion.validator import validate_module_results


def run_fusion(
    results: list[ModuleResult | dict[str, Any]],
    weights: dict[str, float] | None = None,
    conflict_threshold: float = 0.50,
) -> dict[str, Any]:
    """Validate module results, fuse usable scores, and assess risk."""
    eligible, validation_exclusions, validation_warnings = (
        validate_module_results(results)
    )

    fusion = weighted_fusion(
        eligible,
        weights=DEFAULT_WEIGHTS if weights is None else weights,
    )

    all_exclusions = validation_exclusions + fusion["excluded_modules"]
    all_warnings = validation_warnings + fusion["warnings"]

    module_scores = {
        result.module: result.scores["suspicion_score"]
        for result in eligible
        if result.module in fusion["active_modules"]
    }

    risk = assess_risk(
        fusion["fused_score"],
        module_scores=module_scores,
        conflict_threshold=conflict_threshold,
    )

    return {
        "status": "success" if fusion["fused_score"] is not None else "partial",
        "fusion": {
            **fusion,
            "excluded_modules": all_exclusions,
            "warnings": all_warnings,
        },
        "risk_assessment": risk,
        "module_scores": module_scores,
        "thresholds_are_provisional": True,
    }