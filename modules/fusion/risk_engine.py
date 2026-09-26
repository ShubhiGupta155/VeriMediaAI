from typing import Any


RISK_THRESHOLDS = {
    "low_concern_max": 0.30,
    "inconclusive_max": 0.60,
    "elevated_concern_max": 0.80,
}


def assess_risk(
    fused_score: float | None,
    module_scores: dict[str, float] | None = None,
    conflict_threshold: float = 0.50,
) -> dict[str, Any]:
    """Map a fused score to a provisional risk category.

    If at least two module scores differ by the conflict threshold or more,
    the assessment is Inconclusive because the signals disagree substantially.
    """
    if fused_score is None:
        return {
            "overall_score": None,
            "assessment": "Inconclusive",
            "confidence": 0.0,
            "conflicting_modules": [],
            "human_review_recommended": True,
            "reason": "No usable module scores were available.",
            "thresholds_are_provisional": True,
        }

    if not 0.0 <= fused_score <= 1.0:
        raise ValueError("fused_score must be between 0.0 and 1.0")

    scores = module_scores or {}
    conflicting_modules: list[str] = []

    if len(scores) >= 2:
        highest_module = max(scores, key=scores.get)
        lowest_module = min(scores, key=scores.get)

        if scores[highest_module] - scores[lowest_module] >= conflict_threshold:
            conflicting_modules = [lowest_module, highest_module]

    if conflicting_modules:
        assessment = "Inconclusive"
        reason = (
            "Module scores disagree substantially; human review is recommended."
        )
    elif fused_score < RISK_THRESHOLDS["low_concern_max"]:
        assessment = "Low Concern"
        reason = "The fused suspicion score is in the low range."
    elif fused_score < RISK_THRESHOLDS["inconclusive_max"]:
        assessment = "Inconclusive"
        reason = "The fused suspicion score is in the intermediate range."
    elif fused_score < RISK_THRESHOLDS["elevated_concern_max"]:
        assessment = "Elevated Concern"
        reason = "The fused suspicion score is elevated."
    else:
        assessment = "High Concern"
        reason = "The fused suspicion score is in the high range."

    # This is a simple initial confidence estimate, not calibrated probability.
    confidence = max(0.0, min(1.0, abs(fused_score - 0.5) * 2.0))

    return {
        "overall_score": fused_score,
        "assessment": assessment,
        "confidence": confidence,
        "conflicting_modules": conflicting_modules,
        "human_review_recommended": (
            assessment in ("Inconclusive", "Elevated Concern", "High Concern")
        ),
        "reason": reason,
        "thresholds_are_provisional": True,
    }