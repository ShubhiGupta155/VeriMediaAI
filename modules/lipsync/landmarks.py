import os
import cv2


def extract_lip_landmarks(image_path):
    """
    Extract basic mouth-region landmarks from a face image.

    This compatibility implementation uses OpenCV's
    Haar Cascade face detector. It provides a baseline
    mouth-region representation for the lip-sync module.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"File not found: {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read the image file"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

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

    if len(faces) == 0:
        return []

    # Select the largest detected face.
    face_x, face_y, face_w, face_h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Approximate mouth region from the lower part of the face.
    mouth_x = face_x + int(face_w * 0.20)
    mouth_y = face_y + int(face_h * 0.58)
    mouth_w = int(face_w * 0.60)
    mouth_h = int(face_h * 0.25)

    # Return four corner points of the mouth region.
    
    lip_landmarks = [
    (int(mouth_x), int(mouth_y)),
    (int(mouth_x + mouth_w), int(mouth_y)),
    (int(mouth_x), int(mouth_y + mouth_h)),
    (int(mouth_x + mouth_w), int(mouth_y + mouth_h))
]
    return lip_landmarks


if __name__ == "__main__":

    input_image = "sample.jpg"

    try:
        landmarks = extract_lip_landmarks(
            input_image
        )

        print("Lip landmark detection successful!")
        print(
            f"Lip landmarks detected: {len(landmarks)}"
        )

    except Exception as error:
        print(f"Error: {error}")