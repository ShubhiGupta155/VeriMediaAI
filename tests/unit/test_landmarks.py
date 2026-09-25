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

    for point in landmarks :
        assert isinstance(point, tuple)
        assert len(point) == 2
        assert all(isinstance(value, int) for value in point)
    
    assert landmarks == []
def test_successful_face_detection():
    image_path = "tests/fixtures/test_face.jpg"

    landmarks = extract_lip_landmarks(image_path)

    assert isinstance(landmarks, list)
    assert len(landmarks) == 4

    image = cv2.imread(image_path)

    height, width = image.shape[:2]

    for point in landmarks:
        assert isinstance(point, tuple)
        assert len(point) == 2

        x, y = point

        assert 0 <= x < width
        assert 0 <= y < height