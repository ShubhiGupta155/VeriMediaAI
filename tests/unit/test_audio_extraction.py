import os
import subprocess

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


def test_extraction_failure_cleans_temporary_file(
    tmp_path, monkeypatch
):
    import modules.lipsync.audio_extraction as extraction

    created_paths = []
    real_mkstemp = extraction.tempfile.mkstemp

    def fake_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(
            *args, dir=tmp_path, **kwargs
        )
        created_paths.append(path)
        return fd, path

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, "ffmpeg", stderr="Simulated extraction failure"
        )

    monkeypatch.setattr(extraction.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(extraction.subprocess, "run", fake_run)
    monkeypatch.setattr(
        extraction.imageio_ffmpeg,
        "get_ffmpeg_exe",
        lambda: "fake-ffmpeg",
    )

    with pytest.raises(ValueError, match="Audio extraction failed"):
        extract_audio_from_video(
            "tests/fixtures/test_mouth_tracker.mp4"
        )

    assert len(created_paths) == 1
    assert not os.path.exists(created_paths[0])


def test_empty_output_cleans_temporary_file(
    tmp_path, monkeypatch
):
    import modules.lipsync.audio_extraction as extraction

    created_paths = []
    real_mkstemp = extraction.tempfile.mkstemp

    def fake_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(
            *args, dir=tmp_path, **kwargs
        )
        created_paths.append(path)
        return fd, path

    def fake_run(*args, **kwargs):
        # The temporary file exists but remains empty.
        return None

    monkeypatch.setattr(extraction.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(extraction.subprocess, "run", fake_run)
    monkeypatch.setattr(
        extraction.imageio_ffmpeg,
        "get_ffmpeg_exe",
        lambda: "fake-ffmpeg",
    )

    with pytest.raises(ValueError, match="empty file"):
        extract_audio_from_video(
            "tests/fixtures/test_mouth_tracker.mp4"
        )

    assert len(created_paths) == 1
    assert not os.path.exists(created_paths[0])
