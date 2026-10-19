import cv2
import os
import sys

# Project paths
AI_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_PATH = os.path.join(AI_ROOT, "src")

# Add src directly to Python path
sys.path.insert(0, SRC_PATH)

from detection.face_detector import FaceDetector
from privacy.face_blur import FaceBlurrer


# YuNet model
MODEL_PATH = os.path.join(
    AI_ROOT,
    "models",
    "face_detection",
    "face_detection_yunet_2023mar.onnx"
)


def main():

    print("=" * 50)
    print("SIH26245 - YuNet Face Privacy Test")
    print("=" * 50)

    # Check model
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] YuNet model not found:")
        print(MODEL_PATH)
        return

    # Load face detector
    face_detector = FaceDetector(MODEL_PATH)

    # Load face blurrer
    face_blurrer = FaceBlurrer()

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("[OK] Camera started.")
    print("[INFO] Press Q to quit.")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Face detection
        faces = face_detector.detect(frame)

        # Blur faces
        frame = face_blurrer.blur_faces(
            frame,
            faces
        )

        # Display face count
        cv2.putText(
            frame,
            f"Faces Detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SIH26245 - Face Privacy Test",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("[OK] Camera stopped.")
    print("[OK] Face privacy test completed.")


if __name__ == "__main__":
    main()