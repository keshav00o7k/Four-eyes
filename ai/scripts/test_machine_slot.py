
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ai" / "src"))

from operability.operability import OperabilityAnalyzer

CAMERA_SOURCE = 0
SLOT_ID = "lathe-1"

REFERENCE_PATH = (
    PROJECT_ROOT / "ai" / "datasets" /
    f"slot_reference_{SLOT_ID}.png"
)


def main():
    camera = cv2.VideoCapture(
        CAMERA_SOURCE,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print("[ERROR] Could not open camera.")
        camera.release()
        return

    analyzer = OperabilityAnalyzer()

    try:
        success, frame = camera.read()

        if not success:
            print("[ERROR] Could not capture registration frame.")
            return

        print("[INFO] Select the machine/slot rectangle.")
        print("[INFO] Select the machine and its immediate area.")
        print("[INFO] Press ENTER or SPACE to confirm selection.")

        roi = cv2.selectROI(
            "Register Machine Slot",
            frame,
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyWindow("Register Machine Slot")

        x, y, w, h = map(int, roi)

        if w <= 0 or h <= 0:
            print("[ERROR] No valid region selected.")
            return

        reference_crop = frame[y:y + h, x:x + w].copy()

        REFERENCE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not cv2.imwrite(
            str(REFERENCE_PATH),
            reference_crop
        ):
            print("[ERROR] Could not save reference crop.")
            return

        print(f"[OK] Reference registered: {REFERENCE_PATH}")
        print("[INFO] Press SPACE to evaluate one sample.")
        print("[INFO] Press Q to quit.")
        print("[INFO] Test with the same camera position and lighting.")

        while True:
            success, frame = camera.read()

            if not success:
                print("[ERROR] Frame capture failed.")
                break

            current_crop = frame[y:y + h, x:x + w]

            display = frame.copy()
            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Slot: {SLOT_ID}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Machine Slot Test", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                similarity = analyzer.calculate_edge_similarity(
                    current_crop,
                    reference_crop
                )

                result = analyzer.update_machine_presence(
                    SLOT_ID,
                    similarity
                )

                status = (
                    "PRESENT"
                    if result["present"]
                    else "MISSING"
                )

                print(
                    f"[SAMPLE] similarity={result['similarity']:.3f} | "
                    f"status={status} | "
                    f"missing_streak={result['missingStreak']}"
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
