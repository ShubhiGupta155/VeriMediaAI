from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.image.detector import detect_ai_generated


REAL_DIR = PROJECT_ROOT / "evaluation" / "dataset" / "real"
AI_DIR = PROJECT_ROOT / "evaluation" / "dataset" / "ai_generated"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


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
    Calculate confusion-matrix values and classification metrics.

    Positive class:
        ai_generated

    Negative class:
        real

    Returns:
        dict containing:
            true_positive
            true_negative
            false_positive
            false_negative
            precision
            recall
            f1_score
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

    precision_denominator = true_positive + false_positive

    if precision_denominator > 0:
        precision = true_positive / precision_denominator
    else:
        precision = 0.0

    recall_denominator = true_positive + false_negative

    if recall_denominator > 0:
        recall = true_positive / recall_denominator
    else:
        recall = 0.0

    f1_denominator = precision + recall

    if f1_denominator > 0:
        f1_score = (
            2 * precision * recall
            / f1_denominator
        )
    else:
        f1_score = 0.0

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
    Print detailed analysis of false-positive and false-negative cases.
    """

    false_positives = [
        result
        for result in results
        if (
            result["expected"] == "real"
            and result["predicted"] == "ai_generated"
        )
    ]

    false_negatives = [
        result
        for result in results
        if (
            result["expected"] == "ai_generated"
            and result["predicted"] == "real"
        )
    ]

    print("\n" + "-" * 50)
    print("FALSE-POSITIVE ANALYSIS")
    print("-" * 50)

    if false_positives:
        for result in false_positives:
            print(
                f"{result['filename']} -> "
                f"expected=real | "
                f"predicted=ai_generated | "
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
                f"expected=ai_generated | "
                f"predicted=real | "
                f"real_probability="
                f"{result['real_probability']:.4f} | "
                f"ai_generated_probability="
                f"{result['ai_generated_probability']:.4f}"
            )
    else:
        print("No false negatives.")


def main():
    try:
        (
            real_total,
            real_correct,
            real_errors,
            real_results,
        ) = evaluate_folder(
            REAL_DIR,
            "real"
        )

        (
            ai_total,
            ai_correct,
            ai_errors,
            ai_results,
        ) = evaluate_folder(
            AI_DIR,
            "ai_generated"
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
        real_accuracy = real_correct / len(real_results)

        print(
            f"Real accuracy:        "
            f"{real_correct}/{len(real_results)} "
            f"({real_accuracy:.2%})"
        )
    else:
        print("Real accuracy:        N/A")

    if ai_results:
        ai_accuracy = ai_correct / len(ai_results)

        print(
            f"AI-generated accuracy: "
            f"{ai_correct}/{len(ai_results)} "
            f"({ai_accuracy:.2%})"
        )
    else:
        print("AI-generated accuracy: N/A")

if __name__ == "__main__":
    main()
