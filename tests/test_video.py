import pytest
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

from modules.video.metadata import extract_video_metadata
from modules.video.frame_extractor import extract_frames
from modules.video.face_detector import detect_faces
from modules.video.deepfake_detector import (
    load_deepfake_model,
    detect_deepfake,
    DeepfakeModel,
)
from modules.video.network.xception import Xception


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


import os

# Optional environment variable pointing to a real FaceForensics++ checkpoint
LOCAL_FFPP_CHECKPOINT = os.getenv(
    "FFPP_CHECKPOINT_PATH",
    r"C:\Users\ASUS\VeriMediaAI\models\faceforensics_xception.pth",
)


@pytest.fixture
def dummy_model_path():
    """
    Create a temporary checkpoint using the real FaceForensics++ Xception architecture.

    NOTE ON TESTING METHODOLOGY:
    This fixture initializes the REAL Xception architecture with random weights.
    It rigorously tests architecture instantiation, forward pass shapes, device transfers,
    checkpoint serialization/deserialization shims, preprocessing math, and pipeline mechanics.
    It does NOT prove deepfake detection accuracy. Detection accuracy can only be validated
    with a real trained FaceForensics++ checkpoint evaluated on labeled benchmark datasets.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_xception_model.pth"

        # Instantiate the real full Xception model
        model = Xception(num_classes=2)
        torch.save(model.state_dict(), str(model_path))

        yield str(model_path)


class TestXceptionArchitecture:
    """Test suite for the FaceForensics++ full Xception network architecture."""

    def test_xception_instantiation(self):
        """Test that the full Xception model instantiates with correct class count."""
        model = Xception(num_classes=2)
        assert model is not None
        assert model.num_classes == 2
        assert hasattr(model, "last_linear")
        assert model.last_linear.out_features == 2
        assert model.last_linear.in_features == 2048

    def test_xception_parameter_count(self):
        """
        Verify that total parameter count matches full Xception (~20.81M parameters).
        This guarantees the network is not an under-parameterized toy model.
        """
        model = Xception(num_classes=2)
        total_params = sum(p.numel() for p in model.parameters())
        # Genuine Chollet/FF++ Xception has ~20.81M params (20,811,050)
        assert 20_800_000 <= total_params <= 21_500_000, (
            f"Unexpected parameter count: {total_params}. Expected ~20.81M"
        )

    def test_xception_forward_shape_single_sample(self):
        """Test forward pass with single 299x299 image tensor."""
        model = Xception(num_classes=2)
        model.eval()
        dummy_input = torch.randn(1, 3, 299, 299)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"

    def test_xception_forward_shape_batch(self):
        """Test forward pass with batch of 299x299 image tensors."""
        model = Xception(num_classes=2)
        model.eval()
        dummy_input = torch.randn(4, 3, 299, 299)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"

    def test_xception_cpu_execution(self):
        """Verify model forward pass executes on CPU."""
        model = Xception(num_classes=2).to("cpu")
        model.eval()
        x = torch.randn(1, 3, 299, 299, device="cpu")

        with torch.no_grad():
            out = model(x)

        assert out.device.type == "cpu"
        assert out.shape == (1, 2)

    def test_xception_cuda_execution_if_available(self):
        """Verify model forward pass executes on CUDA if a GPU is present."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available in current environment")

        model = Xception(num_classes=2).to("cuda")
        model.eval()
        x = torch.randn(1, 3, 299, 299, device="cuda")

        with torch.no_grad():
            out = model(x)

        assert out.device.type == "cuda"
        assert out.shape == (1, 2)


class TestFaceForensicsPreprocessing:
    """Test suite for FaceForensics++ verified preprocessing pipeline."""

    def test_preprocess_output_tensor_shape(self, dummy_model_path):
        """Verify arbitrary face crops are resized to (1, 3, 299, 299)."""
        model = load_deepfake_model(dummy_model_path)
        crop = np.random.randint(0, 256, (140, 180, 3), dtype=np.uint8)

        tensor = model.preprocess(crop)
        assert tensor.shape == (1, 3, 299, 299)
        assert tensor.dtype == torch.float32

    def test_preprocess_bgr_to_rgb_conversion(self, dummy_model_path):
        """
        Verify that OpenCV BGR input is correctly transformed to RGB.
        A pure blue pixel in BGR [255, 0, 0] must become RGB [0, 0, 255].
        """
        model = load_deepfake_model(dummy_model_path)
        # Create solid BGR image where B=255, G=0, R=0
        blue_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255

        tensor = model.preprocess(blue_bgr).cpu()

        # In RGB tensor: channel 0 is R (0), channel 2 is B (255)
        # Normalized: 0 -> -1.0, 255 -> +1.0
        r_channel_mean = tensor[0, 0, :, :].mean().item()
        b_channel_mean = tensor[0, 2, :, :].mean().item()

        assert abs(r_channel_mean - (-1.0)) < 1e-4, f"Expected R ~ -1.0, got {r_channel_mean}"
        assert abs(b_channel_mean - (1.0)) < 1e-4, f"Expected B ~ +1.0, got {b_channel_mean}"

    def test_preprocess_black_image_maps_to_minus_one(self, dummy_model_path):
        """Verify black pixels (0) normalize to -1.0 under (x - 0.5) / 0.5."""
        model = load_deepfake_model(dummy_model_path)
        black_crop = np.zeros((120, 120, 3), dtype=np.uint8)

        tensor = model.preprocess(black_crop).cpu()
        assert torch.allclose(tensor, torch.tensor(-1.0), atol=1e-4)

    def test_preprocess_white_image_maps_to_plus_one(self, dummy_model_path):
        """Verify white pixels (255) normalize to +1.0 under (x - 0.5) / 0.5."""
        model = load_deepfake_model(dummy_model_path)
        white_crop = np.full((120, 120, 3), 255, dtype=np.uint8)

        tensor = model.preprocess(white_crop).cpu()
        assert torch.allclose(tensor, torch.tensor(1.0), atol=1e-4)

    def test_preprocess_mid_gray_maps_near_zero(self, dummy_model_path):
        """Verify mid-gray pixels (127) normalize to ~ -0.0039."""
        model = load_deepfake_model(dummy_model_path)
        gray_crop = np.full((100, 100, 3), 127, dtype=np.uint8)

        tensor = model.preprocess(gray_crop).cpu()
        expected = (127.0 / 255.0 - 0.5) / 0.5
        assert torch.allclose(tensor, torch.tensor(expected, dtype=torch.float32), atol=1e-3)

    def test_preprocess_grayscale_conversion(self, dummy_model_path):
        """Verify 2D grayscale array is automatically converted to 3-channel BGR/RGB."""
        model = load_deepfake_model(dummy_model_path)
        gray_2d = np.full((80, 80), 128, dtype=np.uint8)

        tensor = model.preprocess(gray_2d)
        assert tensor.shape == (1, 3, 299, 299)

    def test_preprocess_various_resolutions(self, dummy_model_path):
        """Verify robustness across diverse input aspect ratios and resolutions."""
        model = load_deepfake_model(dummy_model_path)
        sizes = [(32, 32, 3), (64, 128, 3), (300, 200, 3), (512, 512, 3)]

        for h, w, c in sizes:
            crop = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
            tensor = model.preprocess(crop)
            assert tensor.shape == (1, 3, 299, 299)


class TestInputValidation:
    """Test suite for detector error handling and parameter validation."""

    def test_preprocess_none_raises_value_error(self, dummy_model_path):
        """Test that None input raises ValueError."""
        model = load_deepfake_model(dummy_model_path)
        with pytest.raises(ValueError) as exc_info:
            model.preprocess(None)
        assert "numpy array" in str(exc_info.value).lower()

    def test_preprocess_string_raises_value_error(self, dummy_model_path):
        """Test that string input raises ValueError."""
        model = load_deepfake_model(dummy_model_path)
        with pytest.raises(ValueError) as exc_info:
            model.preprocess("invalid_image_path")
        assert "numpy array" in str(exc_info.value).lower()

    def test_preprocess_empty_array_raises_value_error(self, dummy_model_path):
        """Test that empty NumPy array raises ValueError."""
        model = load_deepfake_model(dummy_model_path)
        with pytest.raises(ValueError) as exc_info:
            model.preprocess(np.array([]))
        assert "empty" in str(exc_info.value).lower()

    def test_preprocess_1d_array_raises_value_error(self, dummy_model_path):
        """Test that 1D array raises ValueError."""
        model = load_deepfake_model(dummy_model_path)
        with pytest.raises(ValueError) as exc_info:
            model.preprocess(np.array([1, 2, 3]))
        assert "shape" in str(exc_info.value).lower()

    def test_preprocess_4d_array_raises_value_error(self, dummy_model_path):
        """Test that 4D array raises ValueError."""
        model = load_deepfake_model(dummy_model_path)
        with pytest.raises(ValueError) as exc_info:
            model.preprocess(np.zeros((1, 50, 50, 3), dtype=np.uint8))
        assert "shape" in str(exc_info.value).lower()

    def test_preprocess_unsupported_channels_raises_value_error(self, dummy_model_path):
        """Test that 4-channel image (e.g. BGRA) raises ValueError."""
        model = load_deepfake_model(dummy_model_path)
        with pytest.raises(ValueError) as exc_info:
            model.preprocess(np.zeros((50, 50, 4), dtype=np.uint8))
        assert "shape" in str(exc_info.value).lower()


class TestCheckpointHandling:
    """Test suite for checkpoint loading, shims, and error handling."""

    def test_load_deepfake_model_file_not_found(self):
        """Test that missing checkpoint produces informative FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_deepfake_model("/path/to/nonexistent/ffpp_model.pth")

        err_msg = str(exc_info.value).lower()
        assert "faceforensics++ xception checkpoint not found" in err_msg or "not found" in err_msg

    def test_load_deepfake_model_corrupt_file(self):
        """Test that invalid checkpoint content raises RuntimeError."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"CORRUPTED_CHECKPOINT_DATA_NOT_A_VALID_PTH")

        try:
            with pytest.raises(RuntimeError) as exc_info:
                load_deepfake_model(tmp_path)
            assert "failed to load model" in str(exc_info.value).lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_raw_state_dict(self, dummy_model_path):
        """Test loading checkpoint saved as direct state_dict."""
        model = load_deepfake_model(dummy_model_path)
        assert model is not None
        assert isinstance(model, DeepfakeModel)
        assert model.model_type == "xception"

    def test_load_nested_state_dict_wrapper(self):
        """Test loading checkpoint wrapped in {'state_dict': ...} or {'model_state_dict': ...}."""
        real_model = Xception(num_classes=2)
        raw_sd = real_model.state_dict()

        for key_name in ["state_dict", "model_state_dict", "model"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = Path(tmpdir) / f"wrapped_{key_name}.pth"
                torch.save({key_name: raw_sd}, str(ckpt_path))

                loaded = load_deepfake_model(str(ckpt_path))
                assert loaded is not None
                assert isinstance(loaded, DeepfakeModel)

    def test_load_state_dict_with_model_prefix(self):
        """Test loading checkpoint where keys have 'model.' or 'module.' prefix."""
        real_model = Xception(num_classes=2)
        prefixed_sd = {f"model.{k}": v for k, v in real_model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "prefixed_model.pth"
            torch.save(prefixed_sd, str(ckpt_path))

            loaded = load_deepfake_model(str(ckpt_path))
            assert loaded is not None

    def test_load_state_dict_with_fc_key_remapping(self):
        """Test checkpoint with 'fc.' classification head key remapping to 'last_linear.'."""
        real_model = Xception(num_classes=2)
        sd = real_model.state_dict()

        # Rename last_linear -> fc
        rekeyed_sd = {}
        for k, v in sd.items():
            if k.startswith("last_linear."):
                rekeyed_sd["fc." + k[12:]] = v
            else:
                rekeyed_sd[k] = v

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "rekeyed_model.pth"
            torch.save(rekeyed_sd, str(ckpt_path))

            loaded = load_deepfake_model(str(ckpt_path))
            assert loaded is not None

    def test_load_state_dict_pointwise_2d_reshaping(self):
        """
        Test PyTorch 0.4 legacy compatibility where pointwise conv weights
        were serialized as 2D tensors [out_c, in_c] instead of 4D [out_c, in_c, 1, 1].
        """
        real_model = Xception(num_classes=2)
        sd = real_model.state_dict()

        # Flatten one pointwise weight to 2D
        legacy_sd = {}
        for k, v in sd.items():
            if "pointwise" in k and v.dim() == 4:
                legacy_sd[k] = v.squeeze(-1).squeeze(-1)  # 4D -> 2D
            else:
                legacy_sd[k] = v

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "legacy_04_model.pth"
            torch.save(legacy_sd, str(ckpt_path))

            loaded = load_deepfake_model(str(ckpt_path))
            assert loaded is not None

    def test_model_caching(self, dummy_model_path):
        """Verify model caching reuses the loaded instance for identical paths."""
        m1 = load_deepfake_model(dummy_model_path)
        m2 = load_deepfake_model(dummy_model_path)
        assert m1 is m2

    def test_model_cache_reset_on_different_path(self):
        """Verify changing checkpoint path instantiates a new model instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "m1.pth"
            p2 = Path(tmpdir) / "m2.pth"

            m = Xception(num_classes=2)
            torch.save(m.state_dict(), str(p1))
            torch.save(m.state_dict(), str(p2))

            loaded1 = load_deepfake_model(str(p1))
            loaded2 = load_deepfake_model(str(p2))
            assert loaded1 is not loaded2


class TestDeepfakeDetection:
    """Test suite for deepfake inference interface and output schema."""

    @pytest.fixture
    def valid_face_crop(self):
        """Valid synthetic face crop for interface verification."""
        return np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8)

    def test_detect_deepfake_no_model_raises_error(self, valid_face_crop):
        """Test that ValueError is raised when no model is loaded or cached."""
        import modules.video.deepfake_detector as detector_module
        detector_module._loaded_model = None
        detector_module._loaded_model_path = None

        with pytest.raises(ValueError) as exc_info:
            detect_deepfake(valid_face_crop)
        assert "no deepfake model available" in str(exc_info.value).lower()

    def test_detect_deepfake_output_schema(self, valid_face_crop, dummy_model_path):
        """Verify output dictionary schema conforms to downstream requirements."""
        model = load_deepfake_model(dummy_model_path)
        result = detect_deepfake(valid_face_crop, model=model)

        required_keys = {"label", "real_probability", "fake_probability", "confidence", "model"}
        assert set(result.keys()) == required_keys

        assert isinstance(result["label"], str)
        assert result["label"] in ["real", "fake"]
        assert isinstance(result["real_probability"], float)
        assert isinstance(result["fake_probability"], float)
        assert isinstance(result["confidence"], float)
        assert result["model"] == "xception_ff++"

    def test_detect_deepfake_probability_range_and_sum(self, valid_face_crop, dummy_model_path):
        """Verify probability values lie in [0, 1] and sum to ~1.0."""
        model = load_deepfake_model(dummy_model_path)
        result = detect_deepfake(valid_face_crop, model=model)

        r_prob = result["real_probability"]
        f_prob = result["fake_probability"]

        assert 0.0 <= r_prob <= 1.0
        assert 0.0 <= f_prob <= 1.0
        assert abs(r_prob + f_prob - 1.0) < 1e-3

    def test_detect_deepfake_confidence_is_max_probability(self, valid_face_crop, dummy_model_path):
        """Verify confidence field is strictly max(real_prob, fake_prob)."""
        model = load_deepfake_model(dummy_model_path)
        result = detect_deepfake(valid_face_crop, model=model)

        expected_conf = max(result["real_probability"], result["fake_probability"])
        assert abs(result["confidence"] - expected_conf) < 1e-6

    def test_detect_deepfake_blank_face_crop(self, dummy_model_path):
        """Verify solid black face crop completes inference without errors."""
        model = load_deepfake_model(dummy_model_path)
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        result = detect_deepfake(blank, model=model)
        assert isinstance(result, dict)
        assert 0.0 <= result["fake_probability"] <= 1.0

    def test_detect_deepfake_multiple_invocations(self, valid_face_crop, dummy_model_path):
        """Verify deterministic outputs over multiple consecutive calls."""
        model = load_deepfake_model(dummy_model_path)
        res1 = detect_deepfake(valid_face_crop, model=model)
        res2 = detect_deepfake(valid_face_crop, model=model)

        assert res1["label"] == res2["label"]
        assert abs(res1["fake_probability"] - res2["fake_probability"]) < 1e-6


class TestRealCheckpointConditional:
    """
    Conditional verification suite for a genuine FaceForensics++ pretrained checkpoint.

    This suite only executes when a real checkpoint is found on disk or specified via
    the FFPP_CHECKPOINT_PATH environment variable. It ensures real pretrained weights
    load without missing keys and produce valid probability distributions.
    """

    @pytest.mark.skipif(
        not Path(LOCAL_FFPP_CHECKPOINT).exists(),
        reason=(
            f"FaceForensics++ pretrained checkpoint not found at '{LOCAL_FFPP_CHECKPOINT}'. "
            "To execute this test, download the official checkpoint and set FFPP_CHECKPOINT_PATH."
        ),
    )
    def test_real_checkpoint_load_and_predict(self):
        """Verify actual FaceForensics++ pretrained weights perform inference."""
        model = load_deepfake_model(LOCAL_FFPP_CHECKPOINT)
        assert model is not None

        synthetic_face = np.random.randint(0, 256, (299, 299, 3), dtype=np.uint8)
        result = detect_deepfake(synthetic_face, model=model)

        assert result["label"] in ["real", "fake"]
        assert 0.0 <= result["fake_probability"] <= 1.0
        assert 0.0 <= result["real_probability"] <= 1.0
        assert abs(result["fake_probability"] + result["real_probability"] - 1.0) < 1e-3


class TestVideoPipelineIntegration:
    """Integration test suite: frame extraction -> face detection -> deepfake detection."""

    @pytest.fixture
    def valid_video_path(self):
        """Provide path to a valid test video."""
        return r"C:\Users\ASUS\Videos\Captures\ChatGPT Classic 2026-09-06 13-15-05.mp4"

    def test_integration_face_detector_to_deepfake(self, valid_video_path, dummy_model_path):
        """Verify complete pipeline from video frames and face crops to deepfake classification."""
        frames = extract_frames(valid_video_path, sample_interval=5.0, max_frames=2)
        model = load_deepfake_model(dummy_model_path)

        assert len(frames) > 0

        for frame_data in frames:
            frame = frame_data["frame"]
            detections = detect_faces(frame)

            for detection in detections:
                bbox = detection["bbox"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                face_crop = frame[y : y + h, x : x + w]

                if face_crop.size > 0:
                    result = detect_deepfake(face_crop, model=model)
                    assert isinstance(result, dict)
                    assert "label" in result
                    assert "real_probability" in result
                    assert "fake_probability" in result
                    assert "confidence" in result
                    assert result["model"] == "xception_ff++"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
