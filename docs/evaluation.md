
# VeriMedia AI Evaluation Guide

## Purpose

This guide documents how to validate the image evaluation dataset and
reproduce the image detector's baseline evaluation.

## Dataset

The current dataset contains 16 images:
- 10 real images obtained from WhatsApp, as documented in the project README
- 6 AI-generated images generated using ChatGPT, as documented in the project README

The dataset is organized under `evaluation/dataset/real/` and
`evaluation/dataset/ai_generated/`.

The file `evaluation/dataset_manifest.csv` records the filename, expected
label, source, generation method, and notes for each image. Unknown details
should remain unspecified rather than inferred from filenames.

## Validate the Dataset Manifest

From the project root, run:

```powershell
python evaluation/validate_dataset_manifest.py
```

The validator checks the manifest schema, supported image extensions,
duplicate filenames, valid labels, label-folder consistency, and coverage
of all supported images in the dataset directories.

Run validation after changing the dataset or its labels.

## Run the Image Evaluation

After manifest validation, run:

```powershell
python evaluation/image_eval.py
```

The script reports the classification results and saves the detailed JSON
report to `evaluation/results/image_evaluation.json`.

The generated report is ignored by Git under the repository's current
configuration. Preserve important baseline reports separately when needed.

## Current Baseline

The documented 16-image baseline is:
- Correct predictions: 11/16
- Accuracy: 68.75%
- Precision: 57.14%
- Recall: 66.67%
- F1-score: 61.54%
- False-positive rate: 30.00%
- False-negative rate: 33.33%
- ROC-AUC: 73.33%

These metrics describe this particular small dataset only.

## Limitations

The dataset is too small to support broad claims about real-world detector
accuracy. It does not establish performance across different image
generators, resolutions, compression levels, social-media transformations,
editing methods, or unseen distributions.

Model confidence is not definitive proof of authenticity. The manifest
records documented provenance but does not independently establish the
correctness of labels.

When expanding the dataset, preserve the existing baseline and document
the source and ground-truth basis for every added image.
