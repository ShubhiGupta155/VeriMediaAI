
import os
import subprocess
import tempfile

import imageio_ffmpeg


def extract_audio_from_video(video_path, output_path=None):
    """
    Extract mono WAV audio at 16 kHz from a video.

    Returns:
        Path to the extracted WAV file.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    is_temporary = output_path is None

    if is_temporary:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav",
            prefix="verimedia_audio_",
        )
        os.close(fd)

    success = False

    try:
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        command = [
            ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            output_path,
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        if (
            not os.path.isfile(output_path)
            or os.path.getsize(output_path) == 0
        ):
            raise ValueError(
                "Audio extraction produced an empty file"
            )

        success = True
        return output_path

    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Audio extraction failed: {error.stderr[-1000:]}"
        ) from error

    finally:
        if (
            is_temporary
            and not success
            and os.path.exists(output_path)
        ):
            os.remove(output_path)
