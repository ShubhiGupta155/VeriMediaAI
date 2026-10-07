def calculate_sync_score(alignment_result):
    """
    Calculate an overall lip-sync consistency score.

    The score is based on the interval alignment result.
    """

    if not isinstance(alignment_result, dict):
        raise ValueError(
            "alignment_result must be a dictionary"
        )

    required_keys = [
        "alignment_score",
        "matched_intervals",
        "mismatch_intervals"
    ]

    for key in required_keys:
        if key not in alignment_result:
            raise ValueError(
                f"Missing required key: {key}"
            )

    alignment_score = float(
        alignment_result["alignment_score"]
    )

    if not 0.0 <= alignment_score <= 1.0:
        raise ValueError(
            "alignment_score must be between 0 and 1"
        )

    sync_score = alignment_score * 100.0

    if sync_score >= 80.0:
        status = "good"
    elif sync_score >= 50.0:
        status = "moderate"
    else:
        status = "poor"

    return {
        "sync_score": float(sync_score),
        "sync_status": status,
        "matched_intervals": int(
            alignment_result["matched_intervals"]
        ),
        "mismatch_intervals": alignment_result[
            "mismatch_intervals"
        ]
    }