import os

from modules.lipsync.mouth_tracker import (
    track_mouth_movement,
)

from modules.lipsync.audio_alignment import (
    detect_audio_activity,
    extract_mouth_intervals,
    calculate_audio_mouth_alignment,
)

from modules.lipsync.sync_score import (
    calculate_sync_score,
)


def run_lipsync_pipeline(
    video_path,
    audio_path,
    tolerance=0.2
):
    """
    Run the complete baseline lip-sync analysis pipeline.

    Pipeline stages:
    1. Track mouth movement from video
    2. Detect audio activity
    3. Convert mouth movement to intervals
    4. Compare audio and mouth activity
    5. Calculate final sync score
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    mouth_result = track_mouth_movement(
        video_path
    )

    audio_result = detect_audio_activity(
        audio_path
    )

    mouth_intervals = extract_mouth_intervals(
        mouth_result["movement_intervals"]
    )

    alignment_result = calculate_audio_mouth_alignment(
        audio_result["audio_activity_intervals"],
        mouth_intervals,
        tolerance=tolerance
    )

    sync_result = calculate_sync_score(
        alignment_result
    )

    return {
        "module": "lipsync",
        "status": "success",
        "frame_count": mouth_result["frame_count"],
        "movement_frames": mouth_result["movement_frames"],
        "average_mouth_movement": mouth_result[
            "average_mouth_movement"
        ],
        "audio_activity_segments": audio_result[
            "activity_segments"
        ],
        "alignment_score": alignment_result[
            "alignment_score"
        ],
        "matched_intervals": alignment_result[
            "matched_intervals"
        ],
        "mismatch_intervals": alignment_result[
            "mismatch_intervals"
        ],
        "sync_score": sync_result[
            "sync_score"
        ],
        "sync_status": sync_result[
            "sync_status"
        ],
        "tolerance": tolerance,
    }