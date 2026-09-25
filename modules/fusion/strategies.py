from typing import Any

from backend.schemas.analysis import ModuleResult


DEFAULT_WEIGHTS: dict[str, float] = {
    "image": 0.40,
    "video": 0.40,
    "audio": 0.20,
    "lipsync": 0.20,
}


def weighted_fusion(
    results: list[ModuleResult],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Combine suspicion scores using a normalized weighted average.

    Modules without a configured positive weight are excluded. The default
    weights are initial prototype values, not experimentally calibrated.
    """
    configured_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

    for module_name, weight in configured_weights.items():
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(
                f"Weight for '{module_name}' must be a non-negative number."
            )

    weighted_scores: list[tuple[str, float, float]] = []
    excluded: list[dict[str, str]] = []
    warnings: list[str] = []

    for result in results:
        module_name = result.module
        weight = configured_weights.get(module_name)

        if weight is None or weight == 0:
            excluded.append(
                {
                    "module": module_name,
                    "reason": "No positive fusion weight is configured.",
                }
            )
            continue

        score = result.scores.get("suspicion_score")
        if score is None:
            excluded.append(
                {
                    "module": module_name,
                    "reason": "Result has no suspicion_score.",
                }
            )
            continue

        weighted_scores.append((module_name, float(score), float(weight)))

    if not weighted_scores:
        return {
            "fused_score": None,
            "strategy_name": "weighted_average",
            "active_modules": [],
            "weights_used": {},
            "excluded_modules": excluded,
            "warnings": ["No modules had a usable score and positive weight."],
        }

    total_weight = sum(weight for _, _, weight in weighted_scores)
    score_sum = sum(score * weight for _, score, weight in weighted_scores)
    fused_score = score_sum / total_weight

    return {
        "fused_score": fused_score,
        "strategy_name": "weighted_average",
        "active_modules": [name for name, _, _ in weighted_scores],
        "weights_used": {name: weight for name, _, weight in weighted_scores},
        "excluded_modules": excluded,
        "warnings": warnings,
    }