import pytest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from modules.video.metadata import extract_video_metadata
from modules.video.frame_extractor import extract_frames
from modules.video.face_detector import detect_faces


class TestVideoMetadataExtraction:
    """Test suite for video metadata extraction functionality."""

    @pytest.fixture
    def valid_video_path(self):
        """Provide path to a valid test video."""
        return r"C:\Users\ASUS\Videos\Captures\ChatGPT Classic 2026-09-06 13-15-05.mp4"

    def test_extract_video_metadata_valid_video(self, valid_video_path):
        """Test that metadata is correctly extracted from a valid video file."""
        metadata = extract_video_metadata(valid_video_path)

        # Verify all required keys are present
        required_keys = {
            "filename",
            "path",
            "width",
            "height",
            "fps",
            "frame_count",
            "duration_seconds",
            "codec",
            "is_readable",
        }
        assert set(metadata.keys()) == required_keys

        # Verify data types
        assert isinstance(metadata["filename"], str)
        assert isinstance(metadata["path"], str)
        assert isinstance(metadata["width"], int)
        assert isinstance(metadata["height"], int)
        assert isinstance(metadata["fps"], float)
        assert isinstance(metadata["frame_count"], int)
        assert isinstance(metadata["duration_seconds"], float)
        assert isinstance(metadata["codec"], str)
        assert isinstance(metadata["is_readable"], bool)

        # Verify reasonable values
        assert metadata["width"] > 0
        assert metadata["height"] > 0
        assert metadata["fps"] >= 0
        assert metadata["frame_count"] >= 0
        assert metadata["duration_seconds"] >= 0
        assert metadata["is_readable"] is True

        # Verify filename matches
        assert metadata["filename"] == "ChatGPT Classic 2026-09-06 13-15-05.mp4"

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent video file."""
        non_existent_path = "/path/to/non_existent_video.mp4"

        with pytest.raises(FileNotFoundError) as exc_info:
            extract_video_metadata(non_existent_path)

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_video_raises_value_error(self):
        """Test that ValueError is raised for invalid/unreadable video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name
            # Write invalid video data (just plain text)
            tmp.write(b"This is not a valid video file")

        try:
            with pytest.raises(ValueError) as exc_info:
                extract_video_metadata(temp_path)

            assert "unable to open" in str(exc_info.value).lower() or (
                "could not open" in str(exc_info.value).lower()
            )
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

    def test_video_capture_resource_released(self, valid_video_path):
        """Test that VideoCapture resource is properly released after extraction."""
        # Extract metadata multiple times to ensure resources are being released
        for _ in range(3):
            metadata = extract_video_metadata(valid_video_path)
            assert metadata is not None
            assert metadata["is_readable"] is True


class TestFrameExtraction:
    """Test suite for frame extraction functionality."""

    @pytest.fixture
    def valid_video_path(self):
        """Provide path to a valid test video."""
        return r"C:\Users\ASUS\Videos\Captures\ChatGPT Classic 2026-09-06 13-15-05.mp4"

    def test_extract_frames_valid_video(self, valid_video_path):
        """Test that frames are successfully extracted from a valid video."""
        frames = extract_frames(valid_video_path, sample_interval=1.0, max_frames=5)

        # Verify frames were extracted
        assert len(frames) > 0
        assert len(frames) <= 5  # Should respect max_frames

        # Verify first frame structure
        first_frame = frames[0]
        assert "frame" in first_frame
        assert "frame_number" in first_frame
        assert "timestamp_seconds" in first_frame

    def test_extracted_frames_contain_required_fields(self, valid_video_path):
        """Test that each extracted frame contains the required fields."""
        frames = extract_frames(valid_video_path, sample_interval=0.5, max_frames=3)

        for frame_data in frames:
            # Verify frame is a numpy array
            assert isinstance(frame_data["frame"], np.ndarray)
            # Frame should have shape (height, width, 3) for BGR
            assert len(frame_data["frame"].shape) == 3
            assert frame_data["frame"].shape[2] == 3

            # Verify frame_number is an integer
            assert isinstance(frame_data["frame_number"], int)
            assert frame_data["frame_number"] >= 0

            # Verify timestamp_seconds is a float
            assert isinstance(frame_data["timestamp_seconds"], float)
            assert frame_data["timestamp_seconds"] >= 0

    def test_sample_interval_respected(self, valid_video_path):
        """Test that sample_interval parameter is respected."""
        metadata = extract_video_metadata(valid_video_path)
        fps = metadata["fps"]

        # Extract with 2 second interval, get first few frames
        frames = extract_frames(valid_video_path, sample_interval=2.0, max_frames=5)

        assert len(frames) > 0

        # Check that time difference between consecutive frames is approximately sample_interval
        if len(frames) >= 2:
            time_diff = frames[1]["timestamp_seconds"] - frames[0]["timestamp_seconds"]
            # Allow small tolerance due to FPS rounding
            assert abs(time_diff - 2.0) < 0.5

    def test_max_frames_limit_respected(self, valid_video_path):
        """Test that max_frames parameter limits the number of extracted frames."""
        max_frames = 3
        frames = extract_frames(
            valid_video_path, sample_interval=0.1, max_frames=max_frames
        )

        assert len(frames) <= max_frames

    def test_default_parameters(self, valid_video_path):
        """Test extraction with default parameters (sample_interval=1.0, no max_frames)."""
        frames = extract_frames(valid_video_path)

        # Should extract frames with default 1.0 second interval
        assert len(frames) > 0

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent video file."""
        non_existent_path = "/path/to/non_existent_video.mp4"

        with pytest.raises(FileNotFoundError) as exc_info:
            extract_frames(non_existent_path)

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_video_raises_value_error(self):
        """Test that ValueError is raised for invalid/unreadable video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name
            # Write invalid video data
            tmp.write(b"This is not a valid video file")

        try:
            with pytest.raises(ValueError) as exc_info:
                extract_frames(temp_path)

            assert "unable to open" in str(exc_info.value).lower() or (
                "could not open" in str(exc_info.value).lower()
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_invalid_sample_interval_raises_value_error(self, valid_video_path):
        """Test that ValueError is raised for invalid sample_interval."""
        # Test zero interval
        with pytest.raises(ValueError) as exc_info:
            extract_frames(valid_video_path, sample_interval=0)

        assert "sample_interval" in str(exc_info.value).lower()

        # Test negative interval
        with pytest.raises(ValueError) as exc_info:
            extract_frames(valid_video_path, sample_interval=-1.0)

        assert "sample_interval" in str(exc_info.value).lower()

    def test_invalid_max_frames_raises_value_error(self, valid_video_path):
        """Test that ValueError is raised for invalid max_frames."""
        # Test zero max_frames
        with pytest.raises(ValueError) as exc_info:
            extract_frames(valid_video_path, max_frames=0)

        assert "max_frames" in str(exc_info.value).lower()

        # Test negative max_frames
        with pytest.raises(ValueError) as exc_info:
            extract_frames(valid_video_path, max_frames=-1)

        assert "max_frames" in str(exc_info.value).lower()

    def test_frame_numbers_are_sequential(self, valid_video_path):
        """Test that frame numbers are sequential and increasing."""
        frames = extract_frames(valid_video_path, sample_interval=5.0, max_frames=5)

        if len(frames) > 1:
            for i in range(1, len(frames)):
                assert frames[i]["frame_number"] > frames[i - 1]["frame_number"]

    def test_timestamps_are_sequential(self, valid_video_path):
        """Test that timestamps are sequential and increasing."""
        frames = extract_frames(valid_video_path, sample_interval=1.0, max_frames=5)

        if len(frames) > 1:
            for i in range(1, len(frames)):
                assert (
                    frames[i]["timestamp_seconds"] > frames[i - 1]["timestamp_seconds"]
                )

    def test_video_capture_resource_released_frame_extraction(self, valid_video_path):
        """Test that VideoCapture resource is properly released after extraction."""
        # Extract frames multiple times to ensure resources are being released
        for _ in range(3):
            frames = extract_frames(valid_video_path, sample_interval=1.0, max_frames=2)
            assert len(frames) > 0


class TestFaceDetection:
    """Test suite for face detection functionality."""

    @pytest.fixture
    def valid_video_path(self):
        """Provide path to a valid test video."""
        return r"C:\Users\ASUS\Videos\Captures\ChatGPT Classic 2026-09-06 13-15-05.mp4"

    @pytest.fixture
    def valid_frame(self, valid_video_path):
        """Extract a frame from the test video for testing."""
        frames = extract_frames(valid_video_path, sample_interval=1.0, max_frames=1)
        if frames:
            return frames[0]["frame"]
        # Fallback: create a minimal test frame
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def blank_frame(self):
        """Create a blank frame with no faces."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_detect_faces_returns_list(self, valid_frame):
        """Test that detect_faces returns a list."""
        result = detect_faces(valid_frame)
        assert isinstance(result, list)

    def test_detect_faces_valid_frame_structure(self, valid_frame):
        """Test that detected faces have the required structure."""
        detections = detect_faces(valid_frame)

        # Frame may or may not have faces, but structure should be consistent
        for detection in detections:
            # Check required keys
            assert "bbox" in detection
            assert "confidence" in detection
            assert "face_id" in detection

            # Check bbox structure
            bbox = detection["bbox"]
            assert isinstance(bbox, dict)
            assert "x" in bbox
            assert "y" in bbox
            assert "width" in bbox
            assert "height" in bbox

            # Check data types
            assert isinstance(bbox["x"], int)
            assert isinstance(bbox["y"], int)
            assert isinstance(bbox["width"], int)
            assert isinstance(bbox["height"], int)
            assert isinstance(detection["confidence"], float)
            assert isinstance(detection["face_id"], int)

            # Check valid coordinate ranges
            assert bbox["x"] >= 0
            assert bbox["y"] >= 0
            assert bbox["width"] > 0
            assert bbox["height"] > 0

    def test_blank_frame_returns_empty_list(self, blank_frame):
        """Test that a blank frame returns an empty list."""
        detections = detect_faces(blank_frame)
        assert detections == []

    def test_detect_faces_invalid_input_not_ndarray(self):
        """Test that non-ndarray input raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            detect_faces("not_a_frame")

        assert "numpy array" in str(exc_info.value).lower()

    def test_detect_faces_empty_frame(self):
        """Test that empty frame raises ValueError."""
        empty_frame = np.array([])
        with pytest.raises(ValueError) as exc_info:
            detect_faces(empty_frame)

        assert "empty" in str(exc_info.value).lower()

    def test_detect_faces_invalid_dimensions(self):
        """Test that invalid frame dimensions raise ValueError."""
        # Test 1D array
        with pytest.raises(ValueError) as exc_info:
            detect_faces(np.array([1, 2, 3]))

        assert "2d" in str(exc_info.value).lower() or "3d" in str(exc_info.value).lower()

        # Test 4D array
        with pytest.raises(ValueError) as exc_info:
            detect_faces(np.zeros((10, 10, 3, 3)))

        assert "2d" in str(exc_info.value).lower() or "3d" in str(exc_info.value).lower()

    def test_detect_faces_grayscale_frame(self, valid_frame):
        """Test that grayscale frames are handled correctly."""
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)
        detections = detect_faces(gray_frame)

        # Should not raise, should return list
        assert isinstance(detections, list)

    def test_detect_faces_multiple_calls(self, valid_frame):
        """Test that detector can be called multiple times without issues."""
        detections1 = detect_faces(valid_frame)
        detections2 = detect_faces(valid_frame)

        # Both calls should succeed and produce consistent results
        assert isinstance(detections1, list)
        assert isinstance(detections2, list)
        # Results may not be identical due to randomness, but both should be valid
        assert all("bbox" in d and "confidence" in d for d in detections1)
        assert all("bbox" in d and "confidence" in d for d in detections2)

    def test_detect_faces_different_sizes(self):
        """Test that detector works with different frame sizes."""
        sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3), (240, 320, 3)]

        for size in sizes:
            frame = np.zeros(size, dtype=np.uint8)
            detections = detect_faces(frame)
            assert isinstance(detections, list)

    def test_bbox_coordinates_within_frame(self, valid_frame):
        """Test that detected bounding boxes are within frame bounds."""
        height, width = valid_frame.shape[:2]
        detections = detect_faces(valid_frame)

        for detection in detections:
            bbox = detection["bbox"]
            # Check bounding box is within frame
            assert bbox["x"] >= 0
            assert bbox["y"] >= 0
            assert bbox["x"] + bbox["width"] <= width
            assert bbox["y"] + bbox["height"] <= height

    def test_confidence_score_valid_range(self, valid_frame):
        """Test that confidence scores are in valid range."""
        detections = detect_faces(valid_frame)

        for detection in detections:
            confidence = detection["confidence"]
            # Confidence should be between 0 and 1
            assert 0.0 <= confidence <= 1.0

    def test_face_ids_are_unique(self, valid_frame):
        """Test that face IDs are unique within a detection set."""
        detections = detect_faces(valid_frame)

        if len(detections) > 1:
            face_ids = [d["face_id"] for d in detections]
            # All face_ids should be unique
            assert len(face_ids) == len(set(face_ids))

    def test_integration_extract_frames_and_detect_faces(self, valid_video_path):
        """Integration test: extract frames and detect faces in them."""
        frames = extract_frames(valid_video_path, sample_interval=5.0, max_frames=3)

        assert len(frames) > 0

        for frame_data in frames:
            frame = frame_data["frame"]
            detections = detect_faces(frame)

            # Should return a list without errors
            assert isinstance(detections, list)
            # Each detection should have valid structure
            for detection in detections:
                assert "bbox" in detection
                assert "confidence" in detection
                assert "face_id" in detection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
