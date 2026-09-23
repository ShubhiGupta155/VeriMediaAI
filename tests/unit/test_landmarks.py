import cv2
import numpy as np

from modules.lipsync.landmarks import extract_lip_landmarks


def create_test_face(path):
    """Create a simple test image."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_missing_file_raises_error(tmp_path):
    image_path = tmp_path / "missing.jpg"

    try:
        extract_lip_landmarks(str(image_path))
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True


def test_invalid_image_raises_error(tmp_path):
    image_path = tmp_path / "invalid.jpg"
    image_path.write_text("not an image")

    try:
        extract_lip_landmarks(str(image_path))
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_no_face_returns_empty_list(tmp_path):
    image_path = tmp_path / "test.jpg"

    create_test_face(image_path)

    landmarks = extract_lip_landmarks(
        str(image_path)
    )

    assert isinstance(landmarks, list)
    assert landmarks == []