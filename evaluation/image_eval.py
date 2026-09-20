from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.image.detector import detect_ai_generated


REAL_DIR = PROJECT_ROOT / "evaluation" / "dataset" / "real"
AI_DIR = PROJECT_ROOT / "evaluation" / "dataset" / "ai_generated"


def evaluate_folder(folder, expected_label):
    total = 0
    correct = 0

    print(f"\nEvaluating: {expected_label}")
    print("-" * 50)

    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".webp"
        }:
            continue

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
            print(f"{image_path.name} -> ERROR: {e}")

    return total, correct


def main():
    real_total, real_correct = evaluate_folder(
        REAL_DIR,
        "real"
    )

    ai_total, ai_correct = evaluate_folder(
        AI_DIR,
        "ai_generated"
    )

    total = real_total + ai_total
    correct = real_correct + ai_correct

    print("\n" + "=" * 50)
    print("IMAGE DETECTOR EVALUATION")
    print("=" * 50)

    print(f"Real images:        {real_correct}/{real_total}")
    print(f"AI-generated:       {ai_correct}/{ai_total}")
    print(f"Total correct:      {correct}/{total}")

    if total > 0:
        accuracy = correct / total
        print(f"Accuracy:           {accuracy:.2%}")


if __name__ == "__main__":
    main()