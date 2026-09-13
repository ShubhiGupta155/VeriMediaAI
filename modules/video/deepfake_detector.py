"""Deepfake detection module using FaceForensics++ Xception."""

from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn

from modules.video.network.xception import Xception

_loaded_model: Optional["DeepfakeModel"] = None
_loaded_model_path: Optional[str] = None


class DeepfakeModel:
    """Wrapper for FaceForensics++ Xception deepfake detection model."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        model_type: str = "xception",
    ):
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"FaceForensics++ Xception checkpoint not found: {model_path}. "
                "Provide a local pretrained checkpoint path."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        try:
            checkpoint = torch.load(
                str(self.model_path), map_location=self.device, weights_only=False
            )
            if isinstance(checkpoint, nn.Module):
                self.model = checkpoint
                self.model.to(self.device).eval()
                return

            self.model = Xception(num_classes=2)
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                for k in ("state_dict", "model_state_dict", "model"):
                    if k in checkpoint and isinstance(checkpoint[k], dict):
                        state_dict = checkpoint[k]
                        break

            if not isinstance(state_dict, dict):
                raise ValueError(f"No valid state_dict in {self.model_path}")

            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                if k.startswith("model."):
                    k = k[6:]
                if k.startswith("fc."):
                    k = "last_linear." + k[3:]
                if "pointwise" in k and isinstance(v, torch.Tensor) and v.dim() == 2:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                cleaned[k] = v

            self.model.load_state_dict(cleaned)
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def preprocess(self, face_crop: np.ndarray) -> torch.Tensor:
        """Preprocess face crop to normalized 299x299 tensor in [-1.0, 1.0]."""
        if not isinstance(face_crop, np.ndarray):
            raise ValueError(f"Face crop must be numpy array, got {type(face_crop).__name__}")
        if face_crop.size == 0:
            raise ValueError("Face crop is empty")

        if len(face_crop.shape) == 2:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
        elif len(face_crop.shape) != 3 or face_crop.shape[2] != 3:
            raise ValueError(f"Expected shape (H, W, 3) or (H, W), got {face_crop.shape}")

        resized = cv2.resize(face_crop, (299, 299), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, face_crop: np.ndarray) -> dict:
        """
        Run inference on face crop.

        Confidence is the uncalibrated maximum softmax score, not a certified
        forensic certainty estimate.
        """
        tensor = self.preprocess(face_crop)
        with torch.no_grad():
            output = self.model(tensor)

        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        real_prob, fake_prob = float(probs[0]), float(probs[1])

        return {
            "label": "fake" if fake_prob > 0.5 else "real",
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "confidence": float(max(real_prob, fake_prob)),
            "model": "xception_ff++",
        }


def load_deepfake_model(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Optional[DeepfakeModel]:
    """Load and cache FaceForensics++ Xception deepfake detector."""
    global _loaded_model, _loaded_model_path

    if model_path is not None:
        path_str = str(model_path)
        if _loaded_model is not None and _loaded_model_path == path_str:
            if device and device != _loaded_model.device:
                _loaded_model.model.to(device)
                _loaded_model.device = device
            return _loaded_model

        model = DeepfakeModel(path_str, device=device)
        _loaded_model = model
        _loaded_model_path = path_str
        return model

    if _loaded_model is not None:
        if device and device != _loaded_model.device:
            _loaded_model.model.to(device)
            _loaded_model.device = device
        return _loaded_model

    return None


def detect_deepfake(
    face_crop: np.ndarray,
    model: Optional[DeepfakeModel] = None,
    device: Optional[str] = None,
) -> dict:
    """Run deepfake detection on face crop using cached or provided model."""
    if model is None:
        model = load_deepfake_model(device=device)

    if model is None:
        raise ValueError(
            "No deepfake model available. Please load a model with "
            "load_deepfake_model(model_path) before calling detect_deepfake()."
        )

    return model.predict(face_crop)
