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
            total number of images
            number of correct predictions
    """

    image_files = get_image_files(folder)

    total = 0
    correct = 0

    print(f"\nEvaluating: {expected_label}")
    print("-" * 50)

    for image_path in image_files:
        total += 1

        try:
            result = detect_ai_generated(image_path)

            predicted = result["predicted_label"]

            is_correct = predicted == expected_label

            if is_correct:
                correct += 1

            print(
                f"{image_path.name} -> "
                f"predicted={predicted} | "
                f"confidence={result['confidence']:.4f} | "
                f"{'PASS' if is_correct else 'FAIL'}"
            )

        except Exception as e:
            print(
                f"{image_path.name} -> ERROR: {e}"
            )

    return total, correct


def main():
    try:
        real_total, real_correct = evaluate_folder(
            REAL_DIR,
            "real"
        )

        ai_total, ai_correct = evaluate_folder(
            AI_DIR,
            "ai_generated"
        )

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"\nDataset error: {e}")
        print(
            "\nPlease make sure the evaluation dataset is "
            "properly configured."
        )
        return

    total = real_total + ai_total
    correct = real_correct + ai_correct

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

    if total > 0:
        accuracy = correct / total

        print(
            f"Accuracy:           "
            f"{accuracy:.2%}"
        )


if __name__ == "__main__":
    main()