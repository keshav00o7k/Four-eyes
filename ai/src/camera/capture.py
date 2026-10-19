import cv2
from datetime import datetime
from pathlib import Path


class CameraCapture:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.camera = None

    def open(self):
        self.camera = cv2.VideoCapture(self.camera_source)

        if not self.camera.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {self.camera_source}"
            )

        print("[INFO] Camera opened successfully.")

    def capture_frame(self):
        if self.camera is None:
            raise RuntimeError("Camera is not open.")

        success, frame = self.camera.read()

        if not success:
            raise RuntimeError("Failed to capture frame.")

        return frame

    def save_frame(self, frame, output_dir="data/input"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"frame_{timestamp}.jpg"

        cv2.imwrite(str(filename), frame)

        print(f"[INFO] Frame saved: {filename}")

        return filename

    def release(self):
        if self.camera is not None:
            self.camera.release()

        cv2.destroyAllWindows()
        print("[INFO] Camera released.")


if __name__ == "__main__":
    camera = CameraCapture(0)

    try:
        camera.open()

        print("[INFO] Press SPACE to capture a frame.")
        print("[INFO] Press Q to quit.")

        while True:
            frame = camera.capture_frame()

            cv2.imshow("SIH26245 - Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                camera.save_frame(frame)

            elif key == ord("q"):
                break

    finally:
        camera.release()