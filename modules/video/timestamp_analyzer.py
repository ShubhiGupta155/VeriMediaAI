"""
Suspicious Timestamp Analyzer module for VeriMedia AI Video Forensics.

Converts temporal consistency analysis outputs and suspicious segment predictions
into formatted, explainable suspicious video timestamp intervals suitable for
forensic reporting and downstream multimodal fusion.
"""

from typing import Any, Dict, List, Optional


def extract_suspicious_timestamps(
    temporal_result: Dict[str, Any],
    min_severity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract and format suspicious timestamp intervals from temporal-analysis output.

    Parameters
    ----------
    temporal_result : dict
        Dictionary returned by `analyze_temporal_consistency()`.
        Expected keys include:
        - "suspicious_segments": list of segment dicts
        - "temporal_inconsistency_score": float
        - "status": str ("complete", "empty")
    min_severity : str, optional
        Optional filter for minimum severity level ("low", "medium", "high").
        If provided, only intervals with this severity or higher are returned.

    Returns
    -------
    list of dict
        Formatted timestamp intervals, each containing:
        - interval_index: int (0-based chronological index)
        - start_time_seconds: float
        - end_time_seconds: float
        - duration_seconds: float
        - start_frame: int
        - end_frame: int
        - frame_count: int
        - mean_fake_probability: float
        - peak_fake_probability: float
        - severity: str ("low", "medium", "high")
        - reason: str (Human-readable forensic rationale)
        - evidence_type: str ("sustained_deepfake_signal")

    Raises
    ------
    ValueError: If temporal_result is not a dictionary.
    """
    if not isinstance(temporal_result, dict):
        raise ValueError(f"temporal_result must be a dictionary, got {type(temporal_result).__name__}")

    raw_segments = temporal_result.get("suspicious_segments", [])
    if not isinstance(raw_segments, list):
        return []

    severity_order = {"low": 1, "medium": 2, "high": 3}
    min_rank = severity_order.get(min_severity.lower(), 1) if min_severity else 1

    extracted_intervals: List[Dict[str, Any]] = []

    for idx, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            continue

        start_t = float(seg.get("start_time_seconds", 0.0))
        end_t = float(seg.get("end_time_seconds", 0.0))
        duration = float(seg.get("duration_seconds", max(0.0, end_t - start_t)))
        start_f = int(seg.get("start_frame", 0))
        end_f = int(seg.get("end_frame", 0))
        count = int(seg.get("frame_count", max(1, end_f - start_f + 1)))
        mean_p = float(seg.get("mean_fake_probability", 0.0))
        peak_p = float(seg.get("peak_fake_probability", mean_p))
        severity = str(seg.get("severity", "medium")).lower()

        if severity not in severity_order:
            severity = "medium"

        # Apply minimum severity filter if requested
        if severity_order[severity] < min_rank:
            continue

        # Formulate human-readable forensic reason
        if severity == "high":
            reason = (
                f"Sustained elevated deepfake indicators between {start_t:.2f}s and {end_t:.2f}s "
                f"(frames {start_f}–{end_f}) with high mean fake probability of {mean_p:.2f} "
                f"and peak of {peak_p:.2f} across {count} analyzed frames."
            )
        elif severity == "medium":
            reason = (
                f"Elevated deepfake indicator crossing enter threshold between {start_t:.2f}s and {end_t:.2f}s "
                f"(frames {start_f}–{end_f}) with mean fake probability of {mean_p:.2f} across {count} frames."
            )
        else:
            reason = (
                f"Minor sustained elevation between {start_t:.2f}s and {end_t:.2f}s "
                f"(mean fake probability {mean_p:.2f})."
            )

        extracted_intervals.append({
            "interval_index": idx,
            "start_time_seconds": round(start_t, 3),
            "end_time_seconds": round(end_t, 3),
            "duration_seconds": round(duration, 3),
            "start_frame": start_f,
            "end_frame": end_f,
            "frame_count": count,
            "mean_fake_probability": round(mean_p, 4),
            "peak_fake_probability": round(peak_p, 4),
            "severity": severity,
            "reason": reason,
            "evidence_type": "sustained_deepfake_signal",
        })

    return extracted_intervals


def format_timestamp_evidence(
    intervals: List[Dict[str, Any]],
    inconsistency_score: Optional[float] = None,
) -> List[str]:
    """
    Format extracted timestamp intervals into concise evidence statements.

    Parameters
    ----------
    intervals : list of dict
        Output from `extract_suspicious_timestamps()`.
    inconsistency_score : float, optional
        Temporal inconsistency score from temporal analysis.

    Returns
    -------
    list of str
        Human-readable evidence strings ready for forensic reporting.
    """
    evidence: List[str] = []

    if not intervals:
        evidence.append("No suspicious timestamp intervals identified in video.")
    else:
        evidence.append(f"Identified {len(intervals)} suspicious timestamp interval(s).")
        for item in intervals:
            evidence.append(
                f"Interval [{item['start_time_seconds']:.2f}s - {item['end_time_seconds']:.2f}s] "
                f"(frames {item['start_frame']}..{item['end_frame']}): {item['reason']}"
            )

    if inconsistency_score is not None:
        if inconsistency_score >= 0.60:
            evidence.append(
                f"High temporal instability observed (inconsistency score {inconsistency_score:.2f}); "
                "predictions fluctuate sharply across consecutive sampled intervals."
            )
        elif inconsistency_score >= 0.35:
            evidence.append(
                f"Moderate temporal variation observed (inconsistency score {inconsistency_score:.2f})."
            )

    return evidence
