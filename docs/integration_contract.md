# VeriMedia AI Integration Contract

## Purpose

Every forensic module must return a standardized `ModuleResult`.

This contract is used by:

- Image forensics
- Video forensics
- Audio forensics
- Lip-sync analysis
- Multimodal fusion

The backend uses this structure to validate, combine, store, and explain forensic results.

## Supported media types

The `media_type` field must use one of these values:

```text
image
video
audio
lipsync
multimodal
```

---

## Video Forensics Module Integration Contract (Member 2)

### 1. Overview
The video forensics module analyzes input video files using facial deepfake detection (FaceForensics++ full Xception) and time-aware temporal consistency smoothing.

### 2. Callable Interfaces & Signatures

#### A. Video Pipeline Orchestrator (Primary Backend Entrypoint)
```python
from modules.video.pipeline import analyze_video

def analyze_video(
    video_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    sample_interval: float = 1.0,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    enter_threshold: float = 0.65,
    recovery_threshold: float = 0.50,
    tau: float = 1.5,
    max_gap_seconds: float = 3.0,
    min_segment_frames: int = 2,
    min_segment_duration: float = 1.0,
    model: Optional[DeepfakeModel] = None,
) -> dict: ...
```

#### B. Individual Video Forensics Stages
1. **Video Metadata Extraction:**
   ```python
   from modules.video.metadata import extract_video_metadata
   def extract_video_metadata(video_path: str) -> dict: ...
   ```
   - Input: Local file path string.
   - Output: `{"filename": str, "path": str, "width": int, "height": int, "fps": float, "frame_count": int, "duration_seconds": float, "codec": str, "is_readable": bool}`

2. **Frame Extraction:**
   ```python
   from modules.video.frame_extractor import extract_frames
   def extract_frames(video_path: str, sample_interval: float = 1.0, max_frames: Optional[int] = None) -> list[dict]: ...
   ```
   - Input: `video_path` (`str`), `sample_interval` (seconds `> 0`), `max_frames` (optional `int > 0`).
   - Output: List of dicts, each: `{"frame": np.ndarray (BGR), "frame_number": int, "timestamp_seconds": float}`

3. **Face Detection:**
   ```python
   from modules.video.face_detector import detect_faces
   def detect_faces(frame: np.ndarray) -> list[dict]: ...
   ```
   - Input: NumPy ndarray (BGR or Grayscale).
   - Output: List of dicts, each: `{"bbox": {"x": int, "y": int, "width": int, "height": int}, "confidence": 1.0, "face_id": int}`

4. **Deepfake Model Loading & Inference:**
   ```python
   from modules.video.deepfake_detector import load_deepfake_model, detect_deepfake
   def load_deepfake_model(model_path: Optional[Union[str, Path]] = None, device: Optional[str] = None) -> Optional[DeepfakeModel]: ...
   def detect_deepfake(face_crop: np.ndarray, model: Optional[DeepfakeModel] = None, device: Optional[str] = None) -> dict: ...
   ```
   - Preprocessing: Auto-resizes face crop to `299x299`, converts to RGB, normalizes to `[-1.0, 1.0]`.
   - Output: `{"label": "fake"|"real", "real_probability": float, "fake_probability": float, "confidence": float, "model": "xception_ff++"}`

5. **Temporal Consistency Analysis:**
   ```python
   from modules.video.temporal_analysis import analyze_temporal_consistency
   def analyze_temporal_consistency(predictions: list[dict], enter_threshold: float = 0.65, recovery_threshold: float = 0.50, tau: float = 1.5, max_gap_seconds: float = 3.0, min_frames: int = 2, min_duration: float = 1.0) -> dict: ...
   ```
   - Input: List of frame prediction dicts. Requires complementary probabilities: `abs(fake_p + real_p - 1.0) <= 1e-6`.
   - Output: Comprehensive temporal summary including `global_mean_fake_prob`, `global_peak_fake_prob`, `temporal_inconsistency_score`, `is_temporally_suspicious`, `smoothed_probabilities`, `suspicious_segments`, and `evidence`.

6. **Suspicious Timestamp Extraction:**
   ```python
   from modules.video.timestamp_analyzer import extract_suspicious_timestamps, format_timestamp_evidence
   def extract_suspicious_timestamps(temporal_result: dict, min_severity: Optional[str] = None) -> list[dict]: ...
   def format_timestamp_evidence(intervals: list[dict], inconsistency_score: Optional[float] = None) -> list[str]: ...
   ```
   - Output: List of suspicious interval dicts formatted with timestamps, frames, probabilities, severity, and human-readable forensic reasons.

---

### 3. Video-Level Suspicion Score Contract

- **Primary Score Key:** `scores["suspicion_score"]`
- **Definition:** Derived directly from `temporal_summary["global_mean_fake_prob"]`.
- **Range:** `0.0` to `1.0`.
- **Direction:**
  - `1.0`: Maximum deepfake probability across sampled frames.
  - `0.0`: Completely authentic / authentic face probability.
- **Supporting Scores in `scores` dict:**
  - `confidence`: Average uncalibrated maximum softmax score across sampled frames.
  - `peak_fake_probability`: Highest frame-level fake probability observed.
  - `temporal_inconsistency_score`: `[0.0, 1.0]` sequence instability metric (kept separate; not combined with suspicion score).
- **Missing / Insufficient Evidence Contract:**
  - If a video contains no faces, or if no checkpoint is configured, `status` is set to `"insufficient_evidence"` and `scores` is returned as empty `{}`.
  - A fabricated score of `0.0` is **never** returned for absent evidence.

---

### 4. Checkpoint Configuration
- The pipeline checks `model_path` argument first.
- If `model_path` is `None`, it checks the `VERIMEDIA_XCEPTION_CHECKPOINT` environment variable.
- Supported checkpoint formats: Official FaceForensics++ Xception PyTorch state dictionaries (both raw weights and wrapper dicts with keys like `state_dict` or `model`).

---

### 5. Backend Adapter Mapping (`ModuleResult`)

To adapt the video pipeline output to the backend `ModuleResult` schema:

```python
from backend.services.orchestration_service import adapt_video_result
from modules.video.pipeline import analyze_video

raw_video_result = analyze_video(file_path)
module_result = adapt_video_result(raw_video_result, analysis_id=analysis_id)
```

The `adapt_video_result()` adapter function:
- Converts all entries in `raw_video_result["evidence"]` (structured dicts, forensic statements, or `EvidenceItem` objects) into typed `EvidenceItem` instances.
- Preserves exact descriptions, maps severity levels ("low", "medium", "high") to standard float values `[0.0, 1.0]`, preserves confidence and timestamps (or start/end interval bounds in `location`) wherever provided.
- Copies structured timestamp segments into `module_result.timestamps`.
- Maps pipeline status `"success"` → `ModuleStatus.SUCCESS`, `"insufficient_evidence"` → `ModuleStatus.UNAVAILABLE`, and other errors → `ModuleStatus.FAILED`.
- Handles empty evidence without errors (returning an empty `evidence` list).