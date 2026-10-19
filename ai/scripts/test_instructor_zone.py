import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))

import cv2

from src.zones.instructor_zone import PolygonZone


zone = PolygonZone(
    "INSTRUCTOR",
    [
        (0, 0),
        (400, 0),
        (400, 300),
        (0, 300)
    ]
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("[INFO] Polygon zone test started.")
print("[INFO] Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Cannot read frame.")
        break

    detections = []

    # Temporary test bounding boxes
    # Remove these later when connecting YOLO.
    h, w = frame.shape[:2]

    detections.append(
        (int(w * 0.20), int(h * 0.20),
         int(w * 0.40), int(h * 0.70))
    )

    detections.append(
        (int(w * 0.60), int(h * 0.20),
         int(w * 0.80), int(h * 0.70))
    )

    zone.draw(frame)

    for bbox in detections:

        x1, y1, x2, y2 = bbox

        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)

        inside = zone.contains_person(bbox)

        if inside:
            label = "INSTRUCTOR"
        else:
            label = "STUDENT"

        color = (0, 0, 255) if inside else (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.circle(
            frame,
            (foot_x, foot_y),
            6,
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Polygon Instructor Zone", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("[OK] Test finished.")