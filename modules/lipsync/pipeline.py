
import os

from modules.lipsync.mouth_tracker import track_mouth_movement

from modules.lipsync.audio_alignment import (
    detect_audio_activity,
    extract_mouth_intervals,
    calculate_audio_mouth_alignment,
)

from modules.lipsync.audio_extraction import extract_audio_from_video

from modules.lipsync.sync_score import calculate_sync_score


def run_lipsync_pipeline(
    video_path,
    audio_path=None,
    tolerance=0.2
):
    """
    Run the baseline lip-sync analysis pipeline.

    If audio_path is omitted, extract audio from the video.
    An explicitly provided audio_path is also supported.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    extracted_audio_path = None

    if audio_path is None:
        extracted_audio_path = extract_audio_from_video(video_path)
        audio_path = extracted_audio_path
    elif not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    try:
        mouth_result = track_mouth_movement(video_path)

        audio_result = detect_audio_activity(audio_path)

        mouth_intervals = extract_mouth_intervals(
            mouth_result["movement_intervals"]
        )

        alignment_result = calculate_audio_mouth_alignment(
            audio_result["audio_activity_intervals"],
            mouth_intervals,
            tolerance=tolerance,
        )

        sync_result = calculate_sync_score(alignment_result)

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
            "alignment_score": alignment_result["alignment_score"],
            "matched_intervals": alignment_result["matched_intervals"],
            "mismatch_intervals": alignment_result["mismatch_intervals"],
            "sync_score": sync_result["sync_score"],
            "sync_status": sync_result["sync_status"],
            "tolerance": tolerance,
        }

    finally:
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            os.remove(extracted_audio_path)
