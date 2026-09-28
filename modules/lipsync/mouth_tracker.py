import cv2

from modules.lipsync.landmarks import extract_lip_landmarks


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

        # Save current frame temporarily in memory.
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Use the existing landmark logic on the frame.
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        current_landmarks = []

        if len(faces) > 0:
            face_x, face_y, face_w, face_h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            mouth_x = face_x + int(face_w * 0.20)
            mouth_y = face_y + int(face_h * 0.58)
            mouth_w = int(face_w * 0.60)
            mouth_h = int(face_h * 0.25)

            current_landmarks = [
                (int(mouth_x), int(mouth_y)),
                (int(mouth_x + mouth_w), int(mouth_y)),
                (int(mouth_x), int(mouth_y + mouth_h)),
                (int(mouth_x + mouth_w), int(mouth_y + mouth_h))
            ]

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