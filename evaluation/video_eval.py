"""
Video Forensics Evaluation Module for VeriMedia AI.

Evaluates the video forensics pipeline across labeled video benchmarks
(e.g., FaceForensics++, DeepfakeDetection, or local validation splits).
Computes standard statistical metrics:
- Precision, Recall, F1-Score
- Accuracy
- Confusion Matrix (TP, FP, TN, FN)
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Execution time per video
"""

from pathlib import Path
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.video.pipeline import analyze_video
from modules.video.deepfake_detector import CHECKPOINT_ENV_VAR, load_deepfake_model

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Compute binary classification forensic metrics from ground truth and predictions.

    Convention:
    - 1: Fake / Manipulated
    - 0: Real / Authentic

    Returns dict containing TP, FP, TN, FN, Accuracy, Precision, Recall, F1, FPR, FNR.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} ground-truth labels vs {len(y_pred)} predictions")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "total_samples": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


def evaluate_video_dataset(
    dataset_dir: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.50,
    sample_interval: float = 1.0,
    max_frames_per_video: Optional[int] = 30,
) -> Dict[str, Any]:
    """
    Evaluate video dataset structured into real/ and fake/ subdirectories.

    Expected structure:
    dataset_dir/
      ├── real/
      │     ├── video1.mp4
      │     └── ...
      └── fake/ (or ai_generated/)
            ├── video2.mp4
            └── ...
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Video evaluation dataset directory not found: {root}")

    real_dir = root / "real"
    fake_dir = root / "fake"
    if not fake_dir.exists():
        fake_dir = root / "ai_generated"

    if not real_dir.exists() and not fake_dir.exists():
        raise ValueError(
            f"Dataset at {root} must contain 'real' and 'fake' (or 'ai_generated') directories."
        )

    # Pre-load detector once for batch evaluation efficiency
    resolved_model_path = model_path or os.environ.get(CHECKPOINT_ENV_VAR)
    cached_model = None
    if resolved_model_path:
        cached_model = load_deepfake_model(model_path=resolved_model_path)

    items: List[Dict[str, Any]] = []

    if real_dir.exists():
        for p in sorted(real_dir.iterdir()):
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                items.append({"path": p, "expected": "real", "label_int": 0})

    if fake_dir.exists():
        for p in sorted(fake_dir.iterdir()):
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                items.append({"path": p, "expected": "fake", "label_int": 1})

    if not items:
        return {
            "status": "empty",
            "message": f"No supported video files ({SUPPORTED_VIDEO_EXTENSIONS}) found in {root}",
            "metrics": {},
            "details": [],
        }

    y_true: List[int] = []
    y_pred: List[int] = []
    details: List[Dict[str, Any]] = []
    total_eval_time = 0.0

    for item in items:
        v_path = item["path"]
        expected_label = item["expected"]
        expected_int = item["label_int"]

        t0 = time.time()
        try:
            res = analyze_video(
                video_path=v_path,
                model=cached_model,
                sample_interval=sample_interval,
                max_frames=max_frames_per_video,
            )
            elapsed = time.time() - t0
            total_eval_time += elapsed

            if res["status"] == "success":
                score = res["scores"]["suspicion_score"]
                predicted_label = "fake" if score >= threshold else "real"
                pred_int = 1 if predicted_label == "fake" else 0

                y_true.append(expected_int)
                y_pred.append(pred_int)

                details.append({
                    "filename": v_path.name,
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "suspicion_score": score,
                    "status": "evaluated",
                    "elapsed_seconds": round(elapsed, 2),
                    "suspicious_segments_count": len(res.get("timestamps", [])),
                })
            else:
                details.append({
                    "filename": v_path.name,
                    "expected": expected_label,
                    "predicted": None,
                    "suspicion_score": None,
                    "status": res["status"],
                    "warnings": res.get("warnings", []),
                    "elapsed_seconds": round(elapsed, 2),
                })

        except Exception as e:
            details.append({
                "filename": v_path.name,
                "expected": expected_label,
                "predicted": None,
                "status": "error",
                "error": str(e),
            })

    metrics = compute_binary_metrics(y_true, y_pred) if y_true else {}

    return {
        "status": "complete" if y_true else "insufficient_evaluable_videos",
        "evaluated_count": len(y_true),
        "total_files": len(items),
        "total_time_seconds": round(total_eval_time, 2),
        "avg_time_per_video": round(total_eval_time / len(y_true), 2) if y_true else 0.0,
        "metrics": metrics,
        "details": details,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate video forensics pipeline on a dataset directory.")
    parser.add_argument("--dataset-dir", type=str, default="evaluation/video_dataset")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    args = parser.parse_args()

    print(f"\n==========================================")
    print("VERIMEDIA AI: VIDEO DETECTOR EVALUATION")
    print(f"==========================================")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Sampling interval: {args.sample_interval}s | Classification threshold: {args.threshold}")

    try:
        report = evaluate_video_dataset(
            dataset_dir=args.dataset_dir,
            model_path=args.model_path,
            threshold=args.threshold,
            sample_interval=args.sample_interval,
        )
        print("\nEvaluation Summary:")
        print(json.dumps(report.get("metrics", {}), indent=2))
        print(f"Status: {report['status']} (Evaluated {report.get('evaluated_count', 0)} of {report.get('total_files', 0)} files)")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")


if __name__ == "__main__":
    main()
