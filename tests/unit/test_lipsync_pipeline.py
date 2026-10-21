
import os

import numpy as np
import soundfile as sf
import pytest

from modules.lipsync.pipeline import run_lipsync_pipeline


def test_missing_video_raises_error(tmp_path):
    video_path = tmp_path / "missing.mp4"
    audio_path = tmp_path / "test.wav"

    with pytest.raises(FileNotFoundError):
        run_lipsync_pipeline(
            str(video_path),
            str(audio_path),
        )


def test_missing_audio_raises_error(tmp_path):
    video_path = "tests/fixtures/test_mouth_tracker.mp4"
    audio_path = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        run_lipsync_pipeline(
            video_path,
            str(audio_path),
        )


def test_successful_lipsync_pipeline(tmp_path):
    video_path = "tests/fixtures/test_mouth_tracker.mp4"
    audio_path = tmp_path / "test_audio.wav"

    sample_rate = 16000

    audio = np.ones(
        sample_rate,
        dtype=np.float32,
    ) * 0.1

    sf.write(
        str(audio_path),
        audio,
        sample_rate,
    )

    result = run_lipsync_pipeline(
        video_path,
        str(audio_path),
    )

    assert isinstance(result, dict)
    assert result["module"] == "lipsync"
    assert result["status"] == "success"

    assert isinstance(result["frame_count"], int)
    assert isinstance(result["movement_frames"], int)
    assert isinstance(result["average_mouth_movement"], float)
    assert isinstance(result["alignment_score"], float)
    assert isinstance(result["sync_score"], float)

    assert result["sync_status"] in [
        "good",
        "moderate",
        "poor",
    ]

    assert result["tolerance"] == 0.2


def test_temporary_audio_cleaned_when_lipsync_processing_fails(
    tmp_path, monkeypatch
):
    import modules.lipsync.pipeline as pipeline

    video_path = "tests/fixtures/test_mouth_tracker.mp4"
    created_audio_paths = []

    def fake_extract_audio(video):
        audio_path = tmp_path / "temporary_audio.wav"
        audio_path.write_bytes(b"fake audio")
        created_audio_paths.append(str(audio_path))
        return str(audio_path)

    def fake_track_mouth(video):
        raise RuntimeError("Simulated mouth tracking failure")

    monkeypatch.setattr(
        pipeline,
        "extract_audio_from_video",
        fake_extract_audio,
    )
    monkeypatch.setattr(
        pipeline,
        "track_mouth_movement",
        fake_track_mouth,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated mouth tracking failure",
    ):
        pipeline.run_lipsync_pipeline(video_path)

    assert len(created_audio_paths) == 1
    assert not os.path.exists(created_audio_paths[0])
