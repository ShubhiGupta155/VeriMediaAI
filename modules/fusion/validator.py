from typing import Any

from pydantic import ValidationError

from backend.schemas.analysis import ModuleResult, ModuleStatus


def validate_module_results(
    results: list[ModuleResult | dict[str, Any]],
) -> tuple[list[ModuleResult], list[dict[str, str]], list[str]]:
    """Validate module results and separate scoreable from excluded results.

    Successful and partial results are scoreable only when they contain a
    suspicion_score. Failed, unavailable, invalid, or unscored results are
    excluded with a reason.
    """
    eligible: list[ModuleResult] = []
    excluded: list[dict[str, str]] = []
    warnings: list[str] = []

    for index, raw_result in enumerate(results):
        try:
            result = (
                raw_result
                if isinstance(raw_result, ModuleResult)
                else ModuleResult.model_validate(raw_result)
            )
        except (ValidationError, TypeError, ValueError) as exc:
            excluded.append(
                {
                    "module": f"result_{index}",
                    "reason": f"Invalid module result: {exc}",
                }
            )
            continue

        module_name = result.module

        if result.status in (ModuleStatus.FAILED, ModuleStatus.UNAVAILABLE):
            reason = (
                result.errors[0]
                if result.errors
                else f"Module status is {result.status.value}."
            )
            excluded.append({"module": module_name, "reason": reason})
            continue

        if "suspicion_score" not in result.scores:
            excluded.append(
                {
                    "module": module_name,
                    "reason": "Result has no suspicion_score.",
                }
            )
            continue

        eligible.append(result)

        if result.status == ModuleStatus.PARTIAL:
            warnings.append(f"{module_name} returned a partial result.")

    return eligible, excluded, warnings