import cv2
from pathlib import Path
from typing import Optional


def extract_frames(
    video_path: str,
    sample_interval: float = 1.0,
    max_frames: Optional[int] = None,
) -> list[dict]:
    """
    Extract frames from a video at configurable time intervals.

    Frames are sampled based on the sample_interval parameter. For example,
    with sample_interval=1.0, frames are extracted at 0s, 1s, 2s, etc.

    Args:
        video_path: Path to the input video.
        sample_interval: Sampling interval in seconds (default: 1.0).
                        Must be > 0.
        max_frames: Maximum number of frames to extract. If None, no limit.
                   Must be > 0 if specified.

    Returns:
        List of dictionaries, each containing:
        - frame: numpy array (BGR format)
        - frame_number: Original frame index in the video
        - timestamp_seconds: Time in seconds from video start

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open the video, or if parameters are invalid.
    """

    # Validate parameters
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")

    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {max_frames}")

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if fps <= 0:
            raise ValueError(f"Invalid FPS retrieved from video: {fps}")

        # Calculate frame interval based on sample_interval in seconds
        frame_interval = max(1, int(sample_interval * fps))

        frames_list = []
        frame_count = 0
        frame_number = 0

        while True:
            ret, frame = capture.read()

            if not ret:
                # End of video reached
                break

            # Check if this frame should be extracted
            if frame_number % frame_interval == 0:
                timestamp_seconds = frame_number / fps

                frames_list.append(
                    {
                        "frame": frame,
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp_seconds,
                    }
                )

                frame_count += 1

                # Check if max_frames limit reached
                if max_frames is not None and frame_count >= max_frames:
                    break

            frame_number += 1

        return frames_list

    finally:
        capture.release()
