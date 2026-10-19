import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))

import cv2

from src.tamper.tamper_detector import TamperDetector


# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()


# ---------------------------------------------------------
# TAMPER DETECTOR
# ---------------------------------------------------------

tamper_detector = TamperDetector()


print()
print("========================================")
print(" FINAL TAMPER DETECTION TEST")
print("========================================")
print()
print("C = Capture registration reference")
print("Q = Quit")
print()
print("First capture a normal camera reference.")
print()


reference_registered = False


while True:

    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Cannot read camera frame.")
        break

    display = frame.copy()

    # -----------------------------------------------------
    # REGISTER REFERENCE
    # -----------------------------------------------------

    if not reference_registered:

        cv2.putText(
            display,
            "PRESS C TO REGISTER CAMERA",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:

        result = tamper_detector.check(frame)

        # ---------------------------------------------
        # STATUS
        # ---------------------------------------------

        if result["tampered"]:

            status = "TAMPER DETECTED"
            color = (0, 0, 255)

        elif result["rawTampered"]:

            status = "CHECKING..."
            color = (0, 165, 255)

        else:

            status = "CAMERA OK"
            color = (0, 255, 0)

        cv2.putText(
            display,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        # ---------------------------------------------
        # DIAGNOSTICS
        # ---------------------------------------------

        y = 75

        lines = [
            f"Brightness: {result['brightness']}",
            f"Blur score: {result['blurScore']}",
            f"Gray std: {result['grayStd']}",
            f"Match ratio: {result['matchRatio']}",
            f"Good matches: {result['goodMatches']}",
            f"Consecutive: {result['consecutiveTampered']}"
        ]

        for line in lines:

            cv2.putText(
                display,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            y += 25

        # ---------------------------------------------
        # REASONS
        # ---------------------------------------------

        if result["reasons"]:

            cv2.putText(
                display,
                " | ".join(result["reasons"]),
                (20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

    cv2.imshow(
        "Final Tamper Detection",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------------------------------
    # REGISTER REFERENCE
    # -----------------------------------------------------

    if key == ord("c"):

        result = tamper_detector.register_reference(
            frame
        )

        reference_registered = True

        print()
        print("[OK] Camera reference registered.")
        print(result)
        print()

    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("[OK] Test finished.")