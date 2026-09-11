import cv2
import numpy as np
from typing import Optional


# Global cascade classifier - loaded once for efficiency
_cascade_classifier: Optional[cv2.CascadeClassifier] = None


def _load_cascade_classifier() -> cv2.CascadeClassifier:
    """
    Load the Haar Cascade face classifier.

    Uses the pre-trained cascade that comes with OpenCV.

    Returns:
        Loaded CascadeClassifier instance.

    Raises:
        RuntimeError: If the cascade classifier fails to load.
    """
    global _cascade_classifier

    if _cascade_classifier is not None:
        return _cascade_classifier

    # Load the pre-trained Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)

    if classifier.empty():
        raise RuntimeError(
            f"Failed to load Haar Cascade classifier from {cascade_path}. "
            "OpenCV data files may not be properly installed."
        )

    _cascade_classifier = classifier
    return classifier


def detect_faces(frame: np.ndarray) -> list[dict]:
    """
    Detect faces in a video frame using OpenCV Haar Cascade.

    The detector converts the frame to grayscale, applies the Haar Cascade
    classifier, and returns bounding boxes for detected faces.

    Args:
        frame: Input frame as numpy array (BGR or grayscale).
               Expected shape: (height, width, 3) for BGR or (height, width) for grayscale.

    Returns:
        List of dictionaries, each containing:
        - bbox: Dictionary with keys (x, y, width, height) in pixel coordinates
        - confidence: Float confidence score (1.0 for Haar Cascade baseline)
        - face_id: Integer index of the face in the detected list

    Raises:
        ValueError: If the input frame is invalid.
    """

    # Validate input
    if not isinstance(frame, np.ndarray):
        raise ValueError(
            f"Input frame must be a numpy array, got {type(frame).__name__}"
        )

    if frame.size == 0:
        raise ValueError("Input frame is empty")

    if len(frame.shape) not in (2, 3):
        raise ValueError(
            f"Input frame must be 2D (grayscale) or 3D (color), "
            f"got shape {frame.shape}"
        )

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    # Load cascade classifier
    cascade = _load_cascade_classifier()

    # Detect faces using Haar Cascade
    # scaleFactor: how much the image size is reduced at each step
    # minNeighbors: how many neighbors each candidate rectangle should have
    # minSize: minimum face size
    faces = cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Convert detections to the standard output format
    detections = []
    for face_id, (x, y, w, h) in enumerate(faces):
        detections.append(
            {
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "confidence": 1.0,  # Haar Cascade doesn't provide confidence scores
                "face_id": face_id,
            }
        )

    return detections
