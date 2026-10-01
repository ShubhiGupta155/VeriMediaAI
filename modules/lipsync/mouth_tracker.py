import cv2

from modules.lipsync.landmarks import (
    _extract_lip_landmarks_from_image,
    
)


def calculate_mouth_movement(previous_landmarks, current_landmarks):
    """
    Calculate mouth movement between two consecutive frames.

    Returns:
        float: Average movement of the four mouth-region points.
    """

    if not previous_landmarks or not current_landmarks:
        return 0.0

    if len(previous_landmarks) != len(current_landmarks):
        return 0.0

    total_movement = 0.0

    for previous, current in zip(
        previous_landmarks,
        current_landmarks
    ):
        previous_x, previous_y = previous
        current_x, current_y = current

        distance = (
            (current_x - previous_x) ** 2 +
            (current_y - previous_y) ** 2
        ) ** 0.5

        total_movement += distance

    return total_movement / len(current_landmarks)


def track_mouth_movement(video_path):
    """
    Track approximate mouth movement across video frames.

    Returns:
        dict containing movement measurements and status.
    """

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video file: {video_path}"
        )

    previous_landmarks = None
    movements = []
    frame_count = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        frame_count += 1

        # Reuse the shared landmark extraction logic for each video frame.
        current_landmarks = _extract_lip_landmarks_from_image(
    frame
)
        

        if previous_landmarks and current_landmarks:
            movement = calculate_mouth_movement(
                previous_landmarks,
                current_landmarks
            )

            movements.append(movement)

        if current_landmarks:
            previous_landmarks = current_landmarks

    capture.release()

    average_movement = (
        sum(movements) / len(movements)
        if movements
        else 0.0
    )

    return {
        "status": "success",
        "frame_count": frame_count,
        "movement_frames": len(movements),
        "average_mouth_movement": average_movement
    }