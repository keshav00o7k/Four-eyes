import cv2
import time

from ai.src.detection.object_detector import ObjectDetector
from ai.src.tracking.tracker import CentroidTracker


print("[TEST] test_detection.py started")


def main():

    # ==========================================
    # 1. LOAD OBJECT DETECTOR
    # ==========================================

    print("[TEST] Starting detector...")

    detector = ObjectDetector()

    # ==========================================
    # 2. LOAD TRACKER
    # ==========================================

    tracker = CentroidTracker(
        max_distance=100,
        max_disappeared=10
    )

    # ==========================================
    # 3. OPEN CAMERA
    # ==========================================

    print("[TEST] Opening camera...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    print("[INFO] Camera started.")
    print("[INFO] Press Q to quit.")

    # ==========================================
    # 4. FPS TIMER
    # ==========================================

    previous_time = time.perf_counter()

    # ==========================================
    # 5. MAIN LOOP
    # ==========================================

    while True:

        # --------------------------------------
        # Capture frame
        # --------------------------------------

        success, frame = camera.read()

        if not success:
            print("[ERROR] Failed to read camera.")
            break

        # --------------------------------------
        # Start processing timer
        # --------------------------------------

        start_time = time.perf_counter()

        # ======================================
        # 6. OBJECT DETECTION
        # ======================================

        detections = detector.detect(frame)

        # ======================================
        # 7. TRACK PEOPLE
        # ======================================

        tracked_objects = tracker.update(
            detections
        )

        # ======================================
        # 8. DRAW PERSON DETECTIONS
        # ======================================

        for detection in detections:

            # We only want people
            if detection["class_name"] != "person":
                continue

            x1, y1, x2, y2 = detection["bbox"]

            confidence = detection["confidence"]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Detection label
            label = f"person {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ======================================
        # 9. DRAW TRACKING IDs
        # ======================================

        for object_id, centroid in tracked_objects.items():

            cx, cy = centroid

            # Draw centroid
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (255, 0, 0),
                -1
            )

            # Draw temporary tracking ID
            cv2.putText(
                frame,
                f"ID {object_id}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # ======================================
        # 10. PERSON COUNT
        # ======================================

        person_count = len(
            tracked_objects
        )

        # ======================================
        # 11. CALCULATE PROCESSING TIME
        # ======================================

        end_time = time.perf_counter()

        processing_time = (
            end_time - start_time
        ) * 1000

        # ======================================
        # 12. CALCULATE FPS
        # ======================================

        current_time = time.perf_counter()

        frame_time = (
            current_time - previous_time
        )

        previous_time = current_time

        if frame_time > 0:
            fps = 1.0 / frame_time
        else:
            fps = 0.0

        # ======================================
        # 13. DISPLAY PERSON COUNT
        # ======================================

        cv2.putText(
            frame,
            f"Persons: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ======================================
        # 14. DISPLAY FPS
        # ======================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ======================================
        # 15. DISPLAY PROCESSING TIME
        # ======================================

        cv2.putText(
            frame,
            f"Processing: {processing_time:.1f} ms",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ======================================
        # 16. DISPLAY CAMERA WINDOW
        # ======================================

        cv2.imshow(
            "SIH26245 - Person Detection",
            frame
        )

        # ======================================
        # 17. QUIT
        # ======================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Q pressed. Exiting...")
            break
    # ==========================================
    # 18. RELEASE CAMERA
    # ==========================================

    print("[INFO] Q pressed. Stopping detection...")

    camera.release()

cv2.destroyAllWindows()

print("[INFO] Camera released.")
print("[INFO] Detection stopped.")

# ==============================================
# PROGRAM ENTRY POINT
# ==============================================

if __name__ == "__main__":
    main()