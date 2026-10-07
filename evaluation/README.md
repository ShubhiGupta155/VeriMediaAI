# Image Detector Evaluation

This directory contains the evaluation script and dataset used to evaluate the VeriMedia AI image detector.

## Dataset Structure

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
├── results/
│   └── image_evaluation.json
└── README.md
## Dataset Source
### Real Images
The real-image dataset contains 10 images that were personally obtained from WhatsApp and saved locally.
These images are stored in:
evaluation/dataset/real/

### AI-Generated Images
The AI-generated dataset contains 6 images generated using ChatGPT.
These images are stored in:
evaluation/dataset/ai_generated/

## Reproducing the Evaluation Dataset
The reported evaluation uses the same 16-image evaluation set:
- 10 real images
- 6 AI-generated images
- Total: 16 images
The exact evaluation files are included in the repository under:
evaluation/dataset/

If recreating the dataset manually, place:
- 10 real images in evaluation/dataset/real/
- 6 AI-generated images in evaluation/dataset/ai_generated/
Supported image formats are:
.jpg
.jpeg
.png
.webp

## Evaluation Protocol
The evaluation script:
1. Loads all supported images from the real-image dataset.
2. Loads all supported images from the AI-generated dataset.
3. Runs detect_ai_generated() on each image.
4. Compares the predicted label with the expected label.
5. Counts correct predictions and evaluation errors.
6. Calculates classification metrics.
7. Performs false-positive and false-negative analysis.
8. Calculates per-class accuracy.
9. Evaluates multiple confidence thresholds.
10. Performs confidence calibration analysis.
11. Saves the complete evaluation results as JSON.
The expected labels are:
real
ai_generated

## Evaluation Metrics
The evaluation reports the following classification information.
### Confusion Matrix
- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
### Classification Metrics
- Precision
- Recall
- F1-score
- Overall accuracy
### Per-Class Accuracy
The evaluation separately reports accuracy for:
- Real images
- AI-generated images
This helps identify whether the detector performs differently between the two classes.
## Error Analysis
The evaluation identifies:
- False positives: real images predicted as AI-generated
- False negatives: AI-generated images predicted as real
This analysis is important because overall accuracy alone does not show the types of mistakes made by the detector.
## Confidence Threshold Analysis
The evaluation tests multiple AI-generation confidence thresholds:
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9

For each threshold, the evaluation calculates the resulting classification performance.
This allows the effect of changing the decision threshold to be studied instead of assuming that one threshold is universally optimal.
The thresholds are analysis values and should not be interpreted as calibrated forensic probabilities.
## Confidence Calibration Analysis
The evaluation also groups predictions into confidence bins and compares model confidence with observed correctness.
This is used to investigate whether high-confidence predictions are actually more reliable on the evaluation dataset.
Poor calibration should be considered when interpreting the detector's confidence scores.
## Evaluation Limitations
The current evaluation dataset contains only 16 images:
- 10 real images
- 6 AI-generated images
Therefore, the reported results are specific to this evaluation dataset and should not be interpreted as a general real-world accuracy estimate.
The dataset is also not sufficient to establish performance across:
- different image-generation models
- different image resolutions
- different compression levels
- social-media transformations
- image editing/manipulation methods
- unseen real-world distributions
The evaluation should be expanded with a larger and more diverse dataset before making stronger performance claims.
The detector output represents a learned AI-generation indicator and should not be treated as definitive proof that an image is authentic or manipulated.
## Reported Baseline Result
Using the 16-image evaluation dataset, the reported baseline result was:
Real images:        7/10
AI-generated:       4/6
Total correct:      11/16
Accuracy:           68.75%

The baseline confusion counts were:
True Positive (TP):   4
True Negative (TN):   7
False Positive (FP):  3
False Negative (FN):  2

The baseline classification metrics were:
Precision:  57.14%
Recall:     66.67%
F1-score:   61.54%

These results are specific to the current 16-image evaluation dataset.
## Running the Evaluation
From the project root:
python evaluation/image_eval.py

The script evaluates the complete dataset and prints the evaluation results to the terminal.
The results are also saved to:
evaluation/results/image_evaluation.json

The JSON file contains:
- summary
- confusion matrix
- classification metrics
- threshold analysis
- calibration analysis
- per-image evaluation results
## Reproducibility
To reproduce the reported evaluation:
1. Use the same dataset files stored under evaluation/dataset/.
2. Use the project's configured Python environment and dependencies.
3. Run:
python evaluation/image_eval.py

The evaluation dataset and evaluation script are version-controlled so that the reported baseline can be reproduced from the repository.
## Interpretation
The evaluation is intended to establish a baseline for the VeriMedia AI image detector.
The results should be used to:
- identify detector strengths and weaknesses
- analyze false positives and false negatives
- study confidence thresholds
- investigate confidence calibration
- guide future dataset expansion and detector improvement
The current results should not be presented as a universal accuracy guarantee for AI-generated image detection.