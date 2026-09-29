from pathlib import Path
import sys
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.image.detector import detect_ai_generated


REAL_DIR = PROJECT_ROOT / "evaluation" / "dataset" / "real"
AI_DIR = PROJECT_ROOT / "evaluation" / "dataset" / "ai_generated"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "image_evaluation.json"


def get_image_files(folder):
    """
    Validate a dataset directory and return supported image files.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
        NotADirectoryError: If the provided path is not a directory.
        ValueError: If the directory contains no valid image files.
    """

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {folder}"
        )

    image_files = sorted(
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        raise ValueError(
            f"No valid image files found in dataset directory: {folder}"
        )

    return image_files


def evaluate_folder(folder, expected_label):
    """
    Evaluate all supported images in a dataset folder.

    Returns:
        tuple:
            total number of successfully evaluated images
            number of correct predictions
            number of evaluation errors
            list of evaluation results
    """

    image_files = get_image_files(folder)

    total = 0
    correct = 0
    errors = 0
    results = []

    print(f"\nEvaluating: {expected_label}")
    print("-" * 50)

    for image_path in image_files:
        try:
            result = detect_ai_generated(image_path)

            # Count an image only after successful inference.
            total += 1

            predicted = result["predicted_label"]

            is_correct = predicted == expected_label

            if is_correct:
                correct += 1

            results.append(
                {
                    "filename": image_path.name,
                    "expected": expected_label,
                    "predicted": predicted,
                    "correct": is_correct,
                    "real_probability": result["real_probability"],
                    "ai_generated_probability": result[
                        "ai_generated_probability"
                    ],
                    "confidence": result["confidence"],
                }
            )

            print(
                f"{image_path.name} -> "
                f"predicted={predicted} | "
                f"confidence={result['confidence']:.4f} | "
                f"{'PASS' if is_correct else 'FAIL'}"
            )

        except Exception as e:
            errors += 1

            print(
                f"{image_path.name} -> ERROR: {e}"
            )

    return total, correct, errors, results


def calculate_metrics(results):
    """
    Calculate confusion matrix and classification metrics.
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:
        expected = result["expected"]
        predicted = result["predicted"]

        if expected == "ai_generated" and predicted == "ai_generated":
            true_positive += 1

        elif expected == "real" and predicted == "real":
            true_negative += 1

        elif expected == "real" and predicted == "ai_generated":
            false_positive += 1

        elif expected == "ai_generated" and predicted == "real":
            false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def print_error_analysis(results):
    """
    Print false-positive and false-negative analysis.
    """

    false_positives = [
        result
        for result in results
        if result["expected"] == "real"
        and result["predicted"] == "ai_generated"
    ]

    false_negatives = [
        result
        for result in results
        if result["expected"] == "ai_generated"
        and result["predicted"] == "real"
    ]

    print("\n" + "-" * 50)
    print("FALSE-POSITIVE ANALYSIS")
    print("-" * 50)

    if false_positives:
        for result in false_positives:
            print(
                f"{result['filename']} -> "
                f"expected={result['expected']} | "
                f"predicted={result['predicted']} | "
                f"real_probability="
                f"{result['real_probability']:.4f} | "
                f"ai_generated_probability="
                f"{result['ai_generated_probability']:.4f}"
            )
    else:
        print("No false positives.")

    print("\n" + "-" * 50)
    print("FALSE-NEGATIVE ANALYSIS")
    print("-" * 50)

    if false_negatives:
        for result in false_negatives:
            print(
                f"{result['filename']} -> "
                f"expected={result['expected']} | "
                f"predicted={result['predicted']} | "
                f"real_probability="
                f"{result['real_probability']:.4f} | "
                f"ai_generated_probability="
                f"{result['ai_generated_probability']:.4f}"
            )
    else:
        print("No false negatives.")


def save_evaluation_results(
    output_path,
    all_results,
    metrics,
    total,
    correct,
    errors,
    threshold_analysis,
    calibration_analysis,
):
    """
    Save complete image detector evaluation results
    to a JSON file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accuracy = correct / total if total > 0 else 0.0

    evaluation = {
        "summary": {
            "total_images": total,
            "correct_predictions": correct,
            "evaluation_errors": errors,
            "accuracy": accuracy,
        },
        "confusion_matrix": {
            "true_positive": metrics["true_positive"],
            "true_negative": metrics["true_negative"],
            "false_positive": metrics["false_positive"],
            "false_negative": metrics["false_negative"],
        },
        "classification_metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        },
        "threshold_analysis": threshold_analysis,
        "calibration_analysis": calibration_analysis,
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            evaluation,
            file,
            indent=4,
        )

    print(
        f"\nEvaluation results saved to: {output_path}"
    )


def calculate_threshold_metrics(results, threshold):
    """
    Calculate classification metrics using an AI-generated
    probability threshold.
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:
        ai_probability = result["ai_generated_probability"]

        predicted_ai = ai_probability >= threshold
        expected_ai = result["expected"] == "ai_generated"

        if expected_ai and predicted_ai:
            true_positive += 1

        elif not expected_ai and not predicted_ai:
            true_negative += 1

        elif not expected_ai and predicted_ai:
            false_positive += 1

        elif expected_ai and not predicted_ai:
            false_negative += 1

    total = len(results)

    accuracy = (
        (true_positive + true_negative) / total
        if total > 0
        else 0.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def print_threshold_analysis(results):
    """
    Print evaluation metrics for multiple AI-generated
    probability thresholds.
    """

    thresholds = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    print("\n" + "-" * 75)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("-" * 75)

    print(
        "Threshold | TP | TN | FP | FN | "
        "Accuracy | Precision | Recall | F1"
    )

    print("-" * 75)

    for threshold in thresholds:
        metrics = calculate_threshold_metrics(
            results,
            threshold,
        )

        print(
            f"{threshold:.1f}       | "
            f"{metrics['true_positive']:2} | "
            f"{metrics['true_negative']:2} | "
            f"{metrics['false_positive']:2} | "
            f"{metrics['false_negative']:2} | "
            f"{metrics['accuracy']:.2%}   | "
            f"{metrics['precision']:.2%}    | "
            f"{metrics['recall']:.2%} | "
            f"{metrics['f1_score']:.2%}"
        )


def calculate_calibration_bins(results, bin_count=10):
    """
    Calculate confidence calibration statistics.

    Predictions are grouped into confidence bins.
    Each bin reports the average confidence and
    observed accuracy.
    """

    bins = []

    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count

        bin_results = []

        for result in results:
            confidence = result["confidence"]

            if index == bin_count - 1:
                in_bin = (
                    lower <= confidence <= upper
                )
            else:
                in_bin = (
                    lower <= confidence < upper
                )

            if in_bin:
                bin_results.append(result)

        count = len(bin_results)

        if count == 0:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "average_confidence": 0.0,
                    "accuracy": 0.0,
                    "calibration_gap": 0.0,
                }
            )

            continue

        average_confidence = (
            sum(
                result["confidence"]
                for result in bin_results
            )
            / count
        )

        accuracy = (
            sum(
                result["correct"]
                for result in bin_results
            )
            / count
        )

        calibration_gap = abs(
            average_confidence - accuracy
        )

        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "average_confidence": average_confidence,
                "accuracy": accuracy,
                "calibration_gap": calibration_gap,
            }
        )

    return bins


def print_calibration_analysis(results):
    """
    Print confidence calibration statistics.
    """

    bins = calculate_calibration_bins(results)

    print("\n" + "-" * 75)
    print("CONFIDENCE CALIBRATION ANALYSIS")
    print("-" * 75)

    print(
        "Confidence | Count | Avg Confidence | "
        "Accuracy | Calibration Gap"
    )

    print("-" * 75)

    for calibration_bin in bins:
        if calibration_bin["count"] == 0:
            continue

        print(
            f"{calibration_bin['lower']:.1f}-"
            f"{calibration_bin['upper']:.1f}      | "
            f"{calibration_bin['count']:5} | "
            f"{calibration_bin['average_confidence']:.2%}         | "
            f"{calibration_bin['accuracy']:.2%}   | "
            f"{calibration_bin['calibration_gap']:.2%}"
        )


def print_class_accuracy(results):
    """
    Print accuracy separately for real and AI-generated images.
    """

    real_results = [
        result
        for result in results
        if result["expected"] == "real"
    ]

    ai_results = [
        result
        for result in results
        if result["expected"] == "ai_generated"
    ]

    real_correct = sum(
        result["correct"]
        for result in real_results
    )

    ai_correct = sum(
        result["correct"]
        for result in ai_results
    )

    print("\n" + "-" * 50)
    print("PER-CLASS ACCURACY")
    print("-" * 50)

    if real_results:
        real_accuracy = (
            real_correct / len(real_results)
        )

        print(
            f"Real accuracy:        "
            f"{real_correct}/{len(real_results)} "
            f"({real_accuracy:.2%})"
        )
    else:
        print("Real accuracy:        N/A")

    if ai_results:
        ai_accuracy = (
            ai_correct / len(ai_results)
        )

        print(
            f"AI-generated accuracy: "
            f"{ai_correct}/{len(ai_results)} "
            f"({ai_accuracy:.2%})"
        )
    else:
        print("AI-generated accuracy: N/A")


def main():
    try:
        (
            real_total,
            real_correct,
            real_errors,
            real_results,
        ) = evaluate_folder(
            REAL_DIR,
            "real",
        )

        (
            ai_total,
            ai_correct,
            ai_errors,
            ai_results,
        ) = evaluate_folder(
            AI_DIR,
            "ai_generated",
        )

    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as e:
        print(f"\nDataset error: {e}")

        print(
            "\nPlease make sure the evaluation dataset is "
            "properly configured."
        )

        return

    total = real_total + ai_total
    correct = real_correct + ai_correct
    errors = real_errors + ai_errors

    all_results = real_results + ai_results

    metrics = calculate_metrics(all_results)

    print("\n" + "=" * 50)
    print("IMAGE DETECTOR EVALUATION")
    print("=" * 50)

    print(
        f"Real images:        "
        f"{real_correct}/{real_total}"
    )

    print(
        f"AI-generated:       "
        f"{ai_correct}/{ai_total}"
    )

    print(
        f"Total correct:      "
        f"{correct}/{total}"
    )

    print(
        f"Evaluation errors:  "
        f"{errors}"
    )

    if total > 0:
        accuracy = correct / total

        print(
            f"Accuracy:           "
            f"{accuracy:.2%}"
        )
    else:
        print("Accuracy:           N/A")

    print("\n" + "-" * 50)
    print("CONFUSION MATRIX")
    print("-" * 50)

    print(
        f"True Positive (TP):   "
        f"{metrics['true_positive']}"
    )

    print(
        f"True Negative (TN):   "
        f"{metrics['true_negative']}"
    )

    print(
        f"False Positive (FP):  "
        f"{metrics['false_positive']}"
    )

    print(
        f"False Negative (FN):  "
        f"{metrics['false_negative']}"
    )

    print("\n" + "-" * 50)
    print("CLASSIFICATION METRICS")
    print("-" * 50)

    print(
        f"Precision:           "
        f"{metrics['precision']:.2%}"
    )

    print(
        f"Recall:              "
        f"{metrics['recall']:.2%}"
    )

    print(
        f"F1-score:            "
        f"{metrics['f1_score']:.2%}"
    )

    print_error_analysis(all_results)
    print_class_accuracy(all_results)

    print_threshold_analysis(all_results)

    calibration_analysis = calculate_calibration_bins(
        all_results
    )

    print_calibration_analysis(all_results)

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / "image_evaluation.json"
    )

def save_evaluation_results(
    output_path,
    all_results,
    metrics,
    total,
    correct,
    errors,
    threshold_analysis,
    calibration_analysis,
):
 if __name__ == "__main__":
    main()