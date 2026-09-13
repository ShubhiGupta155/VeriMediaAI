# Deepfake Detection Model Research & Baseline Specification

**Project:** VeriMedia AI — Video Forensics (Member 2)
**Scope:** Step 5B — Frame-Level Deepfake Detector
**Document Type:** Implementation-focused research rationale and baseline specification

---

## 1. Research Objective

Establish a verified, reproducible deepfake detection baseline for the video forensics pipeline. The detector operates on face crops extracted from video frames, outputting binary classification scores (real vs. fake) to feed downstream temporal and timestamp analysis modules.

---

## 2. Dataset Selection

| Dataset | Size | Manipulation Methods | Primary Advantage | Limitation | Selected Role |
|---|---|---|---|---|---|
| **FaceForensics++ (FF++)** | 1,000 sequences (1.8M+ frames) | DeepFakes, Face2Face, FaceSwap, NeuralTextures | Canonical benchmark; multi-compression splits (c0, c23, c40) | Older manipulation methods | **Primary Baseline** |
| **DFDC** | 128,154 clips (100k+ actors) | Diverse/unspecified generative methods | Massive scale, diverse lighting | Variable quality; unstructured access | Future cross-dataset test |
| **Celeb-DF (v2)** | 5,639 high-res videos | High-quality DeepFake synthesis | Reduced visual artifacts | Single manipulation type | Future evaluation |
| **DeeperForensics-1.0** | 60,000 videos | Real-world perturbation models | Perturbation benchmark | High dataset footprint | Out of scope |

**Selection Rationale:** FaceForensics++ is the standard reference benchmark across computer vision literature. It provides clean, standardized manipulation splits and published baseline weights directly comparable with peer-reviewed research.

---

## 3. Model Selection

| Model Family | Architecture | Input Resolution | Published Benchmark Performance | Key Advantage | Selected Role |
|---|---|---|---|---|---|
| **Xception (FF++)** | 71-layer depthwise separable CNN | 299×299 | ~99.26% binary acc on FF++ c23 (ICCV 2019 Table 3) | Canonical reference baseline; verified architecture | **Selected Baseline** |
| **EfficientNet-B4** | Compound scaling CNN | 256×256 | ~99.1% AUC on FF++ c23 (DeepfakeBench) | Slightly faster CPU inference | Phase 2 alternative |
| **MesoNet** | 4-layer compact CNN | 256×256 | ~85–90% on compressed video | Lightweight footprint | Lower accuracy |
| **Vision Transformer (ViT)** | Self-attention patch model | 224×224 | ~97–98% on benchmark data | Global context modeling | High compute / GPU dependent |
| **3D CNN / Temporal** | I3D / Spatio-temporal | Multi-frame clips | Variable | Models inter-frame flicker | Deferred to Step 5C |

**Selection Rationale:** FaceForensics++ Xception is selected as the primary baseline. It provides direct academic comparability, proven single-frame face-crop classification, and a verified structural specification without introducing complex temporal dependencies before Step 5C.

---

## 4. Exact Xception Baseline

The baseline uses the genuine Chollet / FaceForensics++ 71-layer Xception architecture (`modules/video/network/xception.py`), rejecting simplified ad-hoc CNN variants (`XceptionLite`):

* **Entry Flow:**
  * `conv1`: `Conv2d(3, 32, kernel=3, stride=2)` + `BatchNorm2d` + ReLU
  * `conv2`: `Conv2d(32, 64, kernel=3)` + `BatchNorm2d` + ReLU
  * `block1`–`block3`: 3 residual blocks expanding channels ($64 \to 128 \to 256 \to 728$) with stride 2 and $1 \times 1$ conv skip projections.
* **Middle Flow:**
  * Exactly 8 identical residual blocks (`block4` through `block11`), each operating at 728 channels with 3 repetitions of `SeparableConv2d` and stride 1.
* **Exit Flow:**
  * `block12`: Residual block expanding $728 \to 1024$ channels with stride 2.
  * `conv3`: `SeparableConv2d(1024, 1536, kernel=3)` + `BatchNorm2d` + ReLU.
  * `conv4`: `SeparableConv2d(1536, 2048, kernel=3)` + `BatchNorm2d`.
* **Classification Head:**
  * Global adaptive average pooling to $(1, 1)$, flattened to a **2048-dimensional** feature vector.
  * Linear classification head: `self.last_linear = nn.Linear(2048, 2)` outputting logits for real and fake classes.
* **Parameter Count:** Exactly **20,811,050** trainable parameters.

---

## 5. Preprocessing Contract

Implemented in `DeepfakeModel.preprocess()` matching the FaceForensics++ test specification:

1. **Resolution:** Bilinear interpolation to strictly **`299 × 299`** pixels (`cv2.INTER_LINEAR`).
2. **Color Space:** Convert OpenCV BGR to RGB (`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`).
3. **Floating Point Scaling:** Cast to float32 and scale from $[0, 255]$ to $[0.0, 1.0]$.
4. **Normalization:** Apply FaceForensics++ mean and standard deviation:
   $$\text{normalized} = \frac{\text{image} - 0.5}{0.5}$$
   Mapping pixel values strictly into the interval **$[-1.0, 1.0]$**. (ImageNet normalization is rejected).
5. **Tensor Format:** Transposed to $CHW$, unsqueezed with batch dimension to $(1, 3, 299, 299)$, and transferred to target device (`cpu` or `cuda`).

---

## 6. Checkpoint Strategy

* **Local Provisioning:** Pretrained model weights must be provided locally via file path.
* **No Automatic Downloading:** No weights are fetched automatically during runtime or inference.
* **No Random Production Fallback:** If the checkpoint is missing, a clear `FileNotFoundError` is raised. The system does not silently fall back to random weights for inference.
* **Strict Loading:** PyTorch `load_state_dict` is executed with default strict matching to catch architectural mismatches.
* **Backward Compatibility Shims:**
  * PyTorch 0.4 pointwise weight reshaping: Reshapes 2D pointwise conv weights (`[out, in]`) to 4D (`[out, in, 1, 1]`) per FF++ reference code.
  * Key normalization: Strips `model.` and `module.` prefixes and remaps `fc.` to `last_linear.`.
  * Accepts full serialized `nn.Module` objects as well as nested dictionary wrappers.
* **Status Notice:** The genuine FaceForensics++ checkpoint binary has **NOT yet been obtained or tested locally** (access requires formal terms-of-use submission to TUM).

---

## 7. Detector Interface

Implemented in `modules/video/deepfake_detector.py`:

### Public API
```python
load_deepfake_model(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None
) -> Optional[DeepfakeModel]

detect_deepfake(
    face_crop: np.ndarray,
    model: Optional[DeepfakeModel] = None,
    device: Optional[str] = None
) -> dict
```

### Output Schema
```python
{
    "label": "real" | "fake",
    "real_probability": float,       # Softmax score [0.0, 1.0], index 0
    "fake_probability": float,       # Softmax score [0.0, 1.0], index 1
    "confidence": float,             # max(real_prob, fake_prob)
    "model": "xception_ff++"
}
```
*Note on Confidence:* `confidence` represents the raw maximum softmax probability. It is an uncalibrated model score, not a certified forensic confidence estimate.

---

## 8. Integration

The module connects into the video forensics pipeline:
```
Video File
    ↓
frame_extractor.py     → Frame array (BGR), frame number, timestamp
    ↓
face_detector.py       → Face bounding boxes (x, y, w, h)
    ↓
[Crop Extraction]      → face_crop = frame[y:y+h, x:x+w]
    ↓
deepfake_detector.py   → Frame-level real/fake softmax probabilities
    ↓
temporal_analysis.py   → Time-series smoothing and anomaly detection (Step 5C)
    ↓
timestamp_analyzer.py  → Suspicious segment intervals (Step 5D)
    ↓
pipeline.py            → Orchestrated forensic report (Step 5E)
```

---

## 9. Testing Status

The automated test suite in `tests/test_video.py` reports:
* **65 total tests collected**
* **63 passed**
* **2 skipped:**
  1. `test_xception_cuda_execution_if_available`: Skipped because the execution environment lacks a CUDA GPU.
  2. `test_real_checkpoint_load_and_predict`: Skipped conditionally because the official FaceForensics++ checkpoint file is not yet locally present.

*Methodological Distinction:* Unit and integration tests instantiate the full Xception architecture with synthetic/temporary weights to verify tensor shapes, preprocessing math, device handling, and interface contracts. **These tests do NOT establish real-world deepfake detection accuracy.**

---

## 10. Limitations

1. **Single-Frame Spatial Scope:** Evaluates crops independently; cannot capture temporal incoherence, jitter, or frame boundary inconsistencies (handled downstream in Step 5C).
2. **Face Detector Dependency:** Occlusion, extreme angles, or severe blur that prevent face detection will bypass deepfake analysis.
3. **Cross-Dataset Generalization:** Published benchmarks indicate performance degrades on unseen manipulation techniques (e.g., modern diffusion models) and heavy compression ($c40$).
4. **Uncalibrated Probabilities:** Output probabilities reflect raw softmax distributions and require cautious interpretation in forensic contexts.

---

## 11. References

* **FaceForensics++:** Rössler et al., *"FaceForensics++: Learning to Detect Manipulated Facial Images"*, ICCV 2019. [arXiv:1901.08971](https://arxiv.org/abs/1901.08971).
* **Xception:** Chollet, F., *"Xception: Deep Learning with Depthwise Separable Convolutions"*, CVPR 2017. [arXiv:1610.02357](https://arxiv.org/abs/1610.02357).
* **FaceForensics Repository:** [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics).
* **DeepfakeBench:** Yan et al., *"DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection"*, NeurIPS 2023. [arXiv:2307.01426](https://arxiv.org/abs/2307.01426).

---

## 12. Current Status & Next Steps

* **Architecture:** Full 71-layer FaceForensics++ Xception implemented in `modules/video/network/xception.py`.
* **Detector:** Preprocessing, inference wrapper, and caching implemented in `modules/video/deepfake_detector.py`.
* **Verification:** 63 automated tests passing.
* **Pending:** Submission of FaceForensics++ access form to TUM, downloading of official c23 weights, and quantitative benchmark evaluation.
