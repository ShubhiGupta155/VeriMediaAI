# Image Detector Evaluation

This directory contains the evaluation script and dataset used to evaluate the VeriMedia AI image detector.

## Dataset Structure

The evaluator expects the following directory structure:

```text
evaluation/
├── dataset/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   └── ai_generated/
│       ├── image1.png
│       ├── image2.png
│       └── ...
│
├── image_eval.py
└── README.md
```

## Dataset Source

### Real Images

The real-image dataset contains 10 images that were personally obtained from WhatsApp and saved locally to the desktop.

These images were placed in:

```text
evaluation/dataset/real/
```

### AI-Generated Images

The AI-generated dataset contains 6 images generated using ChatGPT.

These images were placed in:

```text
evaluation/dataset/ai_generated/
```

## Reproducing the Exact Evaluation Dataset

To reproduce the reported evaluation result, use the same 16-image evaluation set:

- 10 real images from the `real` directory
- 6 AI-generated images from the `ai_generated` directory
- Total: 16 images

The exact files used for the reported result are included in the repository under:

```text
evaluation/dataset/
```

If recreating the dataset manually, place 10 real images in `evaluation/dataset/real/` and 6 AI-generated images in `evaluation/dataset/ai_generated/`.

Supported image formats are:

```text
.jpg
.jpeg
.png
.webp
```

## Evaluation Protocol

The evaluation script:

1. Loads all supported images from the `real` dataset.
2. Loads all supported images from the `ai_generated` dataset.
3. Runs `detect_ai_generated()` on each image.
4. Compares the predicted label with the expected label.
5. Reports the number of correct predictions for each class.
6. Calculates overall accuracy from the successfully evaluated images.

For the reported benchmark, the expected labels are:

```text
real
ai_generated
```

## Setup and Running the Evaluation

From the project root, install the project dependencies according to the project's environment setup.

Then run:

```powershell
python evaluation/image_eval.py
```

The script requires the image detector module and its model dependencies to be available.

## Reported Evaluation Result

Using the 16-image evaluation set described above, the reported result was:

```text
Real images:        7/10
AI-generated:       4/6
Total correct:      11/16
Accuracy:           68.75%
```

This result represents the evaluation of this specific 16-image dataset and should not be interpreted as a general accuracy estimate for all real-world images.
