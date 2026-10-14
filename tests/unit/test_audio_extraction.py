import os

import pytest

from modules.lipsync.audio_extraction import (
    extract_audio_from_video,
)


def test_extract_audio_from_video(tmp_path):
    video_path = "tests/fixtures/test_mouth_tracker.mp4"
    output_path = tmp_path / "extracted_audio.wav"

    result = extract_audio_from_video(
        video_path,
        str(output_path),
    )

    assert result == str(output_path)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_missing_video_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_audio_from_video(
            str(tmp_path / "missing.mp4")
        )