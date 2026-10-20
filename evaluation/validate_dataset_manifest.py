
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MANIFEST = ROOT / "dataset_manifest.csv"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_COLUMNS = {
    "filename",
    "label",
    "source",
    "generation_method",
    "notes",
}
VALID_LABELS = {"real", "ai_generated"}


def main():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if set(reader.fieldnames or []) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected manifest columns: {reader.fieldnames}"
            )

        rows = list(reader)

    filenames = [row["filename"] for row in rows]

    if any(not name for name in filenames):
        raise ValueError("Manifest contains an empty filename.")

    if len(filenames) != len(set(filenames)):
        raise ValueError("Manifest contains duplicate filenames.")

    for row in rows:
        label = row["label"]
        filename = row["filename"]

        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label for {filename}: {label}")

        if Path(filename).name != filename:
            raise ValueError(f"Filename must not contain a path: {filename}")

        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {filename}")

        expected_path = DATASET / label / filename
        if not expected_path.is_file():
            raise FileNotFoundError(
                f"Manifest entry does not match its label folder: {filename}"
            )

    actual_files = {
        (label, path.name)
        for label in VALID_LABELS
        for path in (DATASET / label).iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    manifest_files = {(row["label"], row["filename"]) for row in rows}

    missing_entries = sorted(actual_files - manifest_files)
    stale_entries = sorted(manifest_files - actual_files)

    if missing_entries or stale_entries:
        raise ValueError(
            f"Files missing from manifest: {missing_entries}\n"
            f"Manifest entries without matching files: {stale_entries}"
        )

    print(f"Manifest valid: {len(rows)} images")
    print(f"Real images: {sum(r['label'] == 'real' for r in rows)}")
    print(
        "AI-generated images: "
        f"{sum(r['label'] == 'ai_generated' for r in rows)}"
    )


if __name__ == "__main__":
    main()
