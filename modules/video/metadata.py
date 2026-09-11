import cv2
from pathlib import Path


def extract_video_metadata(video_path: str) -> dict:
    """
    Extract basic technical metadata from a video file.

    Args:
        video_path: Path to the input video.

    Returns:
        Dictionary containing video metadata.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open the video.
    """

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0.0

        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))

        codec = "".join(
            [
                chr(fourcc & 0xFF),
                chr((fourcc >> 8) & 0xFF),
                chr((fourcc >> 16) & 0xFF),
                chr((fourcc >> 24) & 0xFF),
            ]
        )

        return {
            "filename": path.name,
            "path": str(path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "codec": codec.strip(),
            "is_readable": True,
        }

    finally:
        capture.release()