import sys
from pathlib import Path

import cv2


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from ai.src.detection.object_detector import ObjectDetector
from ai.src.operability.operability import OperabilityAnalyzer


# ============================================================
# CONFIG
# ============================================================

CAMERA_SOURCE = 1
camera = cv2.VideoCapture(
    CAMERA_SOURCE,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    print(
        f"[ERROR] Could not open camera "
        f"{CAMERA_SOURCE}"
    )
    camera.release()
    exit()

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

print(
    f"[OK] Camera {CAMERA_SOURCE} opened using DirectShow."
)

MODEL_PATH = (
    "ai/models/object_detection/yolov8n.onnx"
)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("SIH26245 - MONITOR OPERABILITY TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.35,
        nms_threshold=0.50
    )

    # --------------------------------------------------------
    # Load operability analyzer
    # --------------------------------------------------------

    analyzer = OperabilityAnalyzer(
        monitor_v_mean_threshold=60.0,
        monitor_v_std_threshold=20.0
    )

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_SOURCE
    )

    if not camera.isOpened():

        print(
            "[ERROR] Could not open webcam."
        )

        return

    print("[OK] Webcam opened.")
    print()
    print("Show the monitor/camera scene.")
    print("Press Q to quit.")
    print("=" * 60)

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "[ERROR] Failed to capture frame."
                )

                break

            # ------------------------------------------------
            # YOLO detection
            # ------------------------------------------------

            detections = detector.detect(
                frame
            )

            # ------------------------------------------------
            # Monitor operability
            # ------------------------------------------------

            monitor_result = (
                analyzer.analyze_monitors(
                    frame,
                    detections
                )
            )

            # ------------------------------------------------
            # Draw monitor detections
            # ------------------------------------------------

            for detection in detections:

                if detection["class_name"] != "monitor":
                    continue

                x1, y1, x2, y2 = (
                    detection["bbox"]
                )

                result = analyzer.analyze_monitor(
                    frame,
                    detection["bbox"]
                )

                if result["on"]:

                    status = "ON"

                else:

                    status = "OFF"

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"monitor "
                    f"{detection['confidence']:.2f} "
                    f"{status}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(25, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

            # ------------------------------------------------
            # Display statistics
            # ------------------------------------------------

            total = monitor_result["total"]
            on = monitor_result["on"]

            off = total - on

            cv2.putText(
                frame,
                f"Monitors Total: {total}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Monitors ON: {on}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Monitors OFF: {off}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Show frame
            # ------------------------------------------------

            cv2.imshow(
                "SIH26245 - Monitor Operability",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print()
        print("[INFO] Test finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()