"""Temporal analysis module for video deepfake detection sequences.

Provides deterministic smoothing, temporal instability metrics, and hysteresis
segmentation over frame-level deepfake predictions.
"""

from typing import Any, Dict, List, Optional
import math


def analyze_temporal_consistency(
    predictions: List[Dict[str, Any]],
    enter_threshold: float = 0.65,
    recovery_threshold: float = 0.50,
    tau: float = 1.5,
    max_gap_seconds: float = 3.0,
    min_frames: int = 2,
    min_duration: float = 1.0,
) -> Dict[str, Any]:
    """Analyze temporal consistency of frame-level deepfake predictions.

    Args:
        predictions: List of dicts with keys 'frame_number', 'timestamp_seconds',
            'fake_probability', 'real_probability', and 'label'.
        enter_threshold: Smoothed fake_probability to trigger a suspicious segment.
        recovery_threshold: Smoothed fake_probability to exit a suspicious segment.
        tau: Characteristic time constant in seconds for time-aware EWMA smoothing.
        max_gap_seconds: Timestamp gap beyond which EWMA state is reset.
        min_frames: Minimum sampled frames to qualify as a suspicious segment.
        min_duration: Minimum duration in seconds to qualify as a suspicious segment.

    Returns:
        Structured dictionary containing aggregated metrics, smoothed probabilities,
        temporal instability indicators, suspicious segments, and evidence summary.

    Raises:
        ValueError: If input probabilities or threshold arguments are invalid.
    """
    if enter_threshold < recovery_threshold:
        raise ValueError(
            f"enter_threshold ({enter_threshold}) must be >= recovery_threshold ({recovery_threshold})"
        )
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}")
    if min_duration < 0:
        raise ValueError(f"min_duration must be non-negative, got {min_duration}")
    if max_gap_seconds <= 0:
        raise ValueError(f"max_gap_seconds must be greater than 0, got {max_gap_seconds}")

    # Handle empty input
    if not predictions:
        return {
            "total_predictions": 0,
            "global_mean_fake_prob": 0.0,
            "global_peak_fake_prob": 0.0,
            "temporal_variance": 0.0,
            "mean_absolute_derivative": 0.0,
            "temporal_inconsistency_score": 0.0,
            "is_temporally_suspicious": False,
            "smoothed_probabilities": [],
            "suspicious_segments": [],
            "evidence": ["No frame-level predictions provided for temporal analysis."],
            "status": "empty",
        }

    # Validate each record and collect clean records
    validated: List[Dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            raise ValueError(f"Prediction at index {idx} must be a dictionary, got {type(pred).__name__}")

        if "fake_probability" not in pred or "timestamp_seconds" not in pred:
            raise ValueError(f"Prediction at index {idx} missing required keys ('fake_probability', 'timestamp_seconds')")

        fake_p = float(pred["fake_probability"])
        if not (0.0 <= fake_p <= 1.0) or math.isnan(fake_p):
            raise ValueError(f"Invalid fake_probability {fake_p} at index {idx}; must be in [0.0, 1.0]")

        real_p = float(pred.get("real_probability", 1.0 - fake_p))
        if not (0.0 <= real_p <= 1.0) or math.isnan(real_p):
            raise ValueError(f"Invalid real_probability {real_p} at index {idx}; must be in [0.0, 1.0]")

        if abs((fake_p + real_p) - 1.0) > 1e-6:
            raise ValueError(
                f"Probabilities at index {idx} are not complementary: "
                f"fake_probability ({fake_p}) + real_probability ({real_p}) = {fake_p + real_p} != 1.0"
            )

        t_sec = float(pred["timestamp_seconds"])
        if t_sec < 0.0 or math.isnan(t_sec):
            raise ValueError(f"Invalid timestamp_seconds {t_sec} at index {idx}; must be >= 0.0")

        frame_num = int(pred.get("frame_number", idx))

        validated.append({
            "frame_number": frame_num,
            "timestamp_seconds": t_sec,
            "fake_probability": fake_p,
            "real_probability": real_p,
            "label": pred.get("label", "fake" if fake_p > 0.5 else "real"),
        })

    # Sort chronologically by timestamp, then frame number
    sorted_preds = sorted(validated, key=lambda x: (x["timestamp_seconds"], x["frame_number"]))
    n = len(sorted_preds)

    raw_fake = [p["fake_probability"] for p in sorted_preds]
    timestamps = [p["timestamp_seconds"] for p in sorted_preds]
    frame_numbers = [p["frame_number"] for p in sorted_preds]

    global_mean_fake = float(sum(raw_fake) / n)
    global_peak_fake = float(max(raw_fake))

    # Apply 3-point rolling median filter to suppress isolated single-frame spikes
    median_filtered: List[float] = []
    for i in range(n):
        if n < 3:
            # For 1 or 2 points, 3-point median window is degenerate; keep raw
            val = raw_fake[i]
        elif i == 0:
            # Symmetric reflection at boundary: [p0, p0, p1]
            window = [raw_fake[0], raw_fake[0], raw_fake[1]]
            val = float(sorted(window)[1])
        elif i == n - 1:
            # Symmetric reflection at boundary: [pn-2, pn-1, pn-1]
            window = [raw_fake[n - 2], raw_fake[n - 1], raw_fake[n - 1]]
            val = float(sorted(window)[1])
        else:
            window = [raw_fake[i - 1], raw_fake[i], raw_fake[i + 1]]
            val = float(sorted(window)[1])
        median_filtered.append(val)

    # Apply time-aware EWMA smoothing: alpha = 1 - exp(-delta_t / tau)
    # If delta_t > max_gap_seconds, reset state to current median filtered value
    smoothed: List[float] = []
    current_state = median_filtered[0]
    smoothed.append(current_state)

    for i in range(1, n):
        delta_t = timestamps[i] - timestamps[i - 1]
        if delta_t < 0:
            delta_t = 0.0

        if delta_t > max_gap_seconds:
            current_state = median_filtered[i]
        else:
            alpha = 1.0 - math.exp(-delta_t / tau) if tau > 0 else 1.0
            # Clamp alpha to [0.0, 1.0]
            alpha = max(0.0, min(1.0, alpha))
            current_state = alpha * median_filtered[i] + (1.0 - alpha) * current_state

        smoothed.append(float(current_state))

    # Calculate temporal statistics
    if n > 1:
        mean_s = sum(smoothed) / n
        temporal_var = float(sum((s - mean_s) ** 2 for s in smoothed) / n)

        diffs = [abs(smoothed[i] - smoothed[i - 1]) for i in range(1, n)]
        mean_abs_diff = float(sum(diffs) / len(diffs))

        # Heuristic temporal inconsistency score in [0.0, 1.0]:
        # Combines normalized variance (typical smooth sequences have var < 0.005)
        # and mean absolute frame-to-frame jump (typical smooth sequences have jump < 0.02)
        norm_var = min(1.0, temporal_var / 0.05)
        norm_diff = min(1.0, mean_abs_diff / 0.15)
        inconsistency_score = round(float(0.5 * norm_var + 0.5 * norm_diff), 4)
    else:
        temporal_var = 0.0
        mean_abs_diff = 0.0
        inconsistency_score = 0.0

    # Hysteresis segmentation
    # Enter when smoothed >= enter_threshold, stay until smoothed < recovery_threshold
    raw_segments: List[Dict[str, Any]] = []
    in_segment = False
    seg_start_idx = 0

    for i in range(n):
        val = smoothed[i]
        delta_t = (timestamps[i] - timestamps[i - 1]) if i > 0 else 0.0

        # Reset segment if there's a large gap
        if in_segment and delta_t > max_gap_seconds:
            raw_segments.append({
                "start_idx": seg_start_idx,
                "end_idx": i - 1,
            })
            in_segment = False

        if not in_segment:
            if val >= enter_threshold:
                in_segment = True
                seg_start_idx = i
        else:
            if val < recovery_threshold:
                raw_segments.append({
                    "start_idx": seg_start_idx,
                    "end_idx": i - 1,
                })
                in_segment = False

    if in_segment:
        raw_segments.append({
            "start_idx": seg_start_idx,
            "end_idx": n - 1,
        })

    # Filter segments by minimum frames OR minimum duration
    suspicious_segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        s_idx, e_idx = seg["start_idx"], seg["end_idx"]
        seg_frames = (e_idx - s_idx) + 1
        seg_duration = round(float(timestamps[e_idx] - timestamps[s_idx]), 3)

        if seg_frames >= min_frames or seg_duration >= min_duration:
            seg_smoothed = smoothed[s_idx : e_idx + 1]
            seg_raw = raw_fake[s_idx : e_idx + 1]

            mean_fake = round(float(sum(seg_smoothed) / len(seg_smoothed)), 4)
            peak_fake = round(float(max(seg_raw)), 4)

            # Heuristic severity
            if mean_fake >= 0.80 or peak_fake >= 0.90:
                severity = "high"
            elif mean_fake >= 0.65:
                severity = "medium"
            else:
                severity = "low"

            suspicious_segments.append({
                "start_frame": frame_numbers[s_idx],
                "end_frame": frame_numbers[e_idx],
                "start_time_seconds": round(float(timestamps[s_idx]), 3),
                "end_time_seconds": round(float(timestamps[e_idx]), 3),
                "duration_seconds": seg_duration,
                "frame_count": seg_frames,
                "mean_fake_probability": mean_fake,
                "peak_fake_probability": peak_fake,
                "severity": severity,
            })

    is_temporally_suspicious = bool(len(suspicious_segments) > 0 or global_mean_fake >= enter_threshold)

    # Evidence summary formulation
    evidence: List[str] = []
    if suspicious_segments:
        evidence.append(
            f"Detected {len(suspicious_segments)} suspicious temporal segment(s) with sustained elevated deepfake indicators."
        )
        for s in suspicious_segments:
            evidence.append(
                f"Suspicious interval {s['start_time_seconds']:.1f}s - {s['end_time_seconds']:.1f}s "
                f"(frames {s['start_frame']}..{s['end_frame']}): mean fake prob {s['mean_fake_probability']:.2f}, "
                f"peak {s['peak_fake_probability']:.2f} ({s['severity']} severity)."
            )
    else:
        evidence.append("No sustained suspicious temporal segments identified across analyzed frames.")

    if inconsistency_score >= 0.60:
        evidence.append(
            f"High temporal instability detected (inconsistency score {inconsistency_score:.2f}, "
            f"mean jump {mean_abs_diff:.2f}); frame-to-frame model outputs fluctuate significantly."
        )
    elif inconsistency_score >= 0.35:
        evidence.append(
            f"Moderate temporal variation observed (inconsistency score {inconsistency_score:.2f})."
        )
    else:
        evidence.append(
            f"Temporal progression is stable (inconsistency score {inconsistency_score:.2f})."
        )

    return {
        "total_predictions": n,
        "global_mean_fake_prob": round(global_mean_fake, 4),
        "global_peak_fake_prob": round(global_peak_fake, 4),
        "temporal_variance": round(temporal_var, 6),
        "mean_absolute_derivative": round(mean_abs_diff, 4),
        "temporal_inconsistency_score": inconsistency_score,
        "is_temporally_suspicious": is_temporally_suspicious,
        "smoothed_probabilities": [round(p, 4) for p in smoothed],
        "suspicious_segments": suspicious_segments,
        "evidence": evidence,
        "status": "complete",
    }
