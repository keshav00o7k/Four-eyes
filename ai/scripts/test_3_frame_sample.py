import sys
import os
import cv2
import time

# ============================================================
# ADD src DIRECTORY TO PYTHON PATH
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
)

# ============================================================
# IMPORT MODULES
# ============================================================

from detection.object_detector import ObjectDetector
from detection.face_detector import FaceDetector
from privacy.face_blur import FaceBlurrer
from zones.instructor_zone import PolygonZone
from tamper.tamper_detector import TamperDetector
from output.package import ObservationPackager


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

YOLO_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "object_detection",
    "yolov8n.onnx"
)

YUNET_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_detection",
    "face_detection_yunet_2023mar.onnx"
)


# ============================================================
# SETTINGS
# ============================================================

BURST_FRAMES = 3

# Architecture requirement:
# 2 seconds between sample frames
FRAME_INTERVAL = 2.0


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    frame,
    object_detector,
    face_detector,
    instructor_zone
):
    """
    Process one frame.

    Pipeline:

        Frame
          ↓
        YOLO
          ↓
        Person detection
          ↓
        Instructor exclusion
          ↓
        YuNet face detection
    """

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    detections = object_detector.detect(
        frame
    )

    # --------------------------------------------------------
    # PERSONS ONLY
    # --------------------------------------------------------

    person_detections = []

    for detection in detections:

        if detection["class_name"] != "person":
            continue

        person_detections.append(
            detection
        )

    # --------------------------------------------------------
    # INSTRUCTOR EXCLUSION
    # --------------------------------------------------------

    student_count = 0
    instructor_count = 0

    for detection in person_detections:

        x1, y1, x2, y2 = detection["bbox"]

        # Bottom-center of bounding box
        # = person's foot point

        foot_x = int(
            (x1 + x2) / 2
        )

        foot_y = int(y2)

        if instructor_zone.contains_point(
            foot_x,
            foot_y
        ):

            instructor_count += 1

        else:

            student_count += 1

    # --------------------------------------------------------
    # YUNET FACE DETECTION
    # --------------------------------------------------------

    faces = face_detector.detect(
        frame
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "persons": student_count,

        "excludedInstructor":
            instructor_count,

        "totalPersons":
            len(person_detections),

        "faces":
            faces,

        "detections":
            person_detections
    }


# ============================================================
# DRAW PROCESSED FRAME
# ============================================================

def draw_frame(
    frame,
    result,
    instructor_zone,
    face_blurrer
):

    display_frame = frame.copy()

    # ========================================================
    # FACE PRIVACY BLUR
    # ========================================================

    display_frame = face_blurrer.blur_faces(
        display_frame,
        result["faces"]
    )

    height, width = display_frame.shape[:2]

    # ========================================================
    # SAFETY PRIVACY BLUR
    #
    # Blur upper 25% of every person bounding box.
    # ========================================================

    for detection in result["detections"]:

        x1, y1, x2, y2 = detection["bbox"]

        person_height = y2 - y1

        upper_y2 = (
            y1 +
            int(person_height * 0.25)
        )

        x1_safe = max(
            0,
            int(x1)
        )

        y1_safe = max(
            0,
            int(y1)
        )

        x2_safe = min(
            width,
            int(x2)
        )

        y2_safe = min(
            height,
            int(upper_y2)
        )

        roi = display_frame[
            y1_safe:y2_safe,
            x1_safe:x2_safe
        ]

        if roi.size != 0:

            blurred = cv2.GaussianBlur(
                roi,
                (0, 0),
                sigmaX=20,
                sigmaY=20
            )

            display_frame[
                y1_safe:y2_safe,
                x1_safe:x2_safe
            ] = blurred

    # ========================================================
    # DRAW INSTRUCTOR POLYGON
    # ========================================================

    display_frame = instructor_zone.draw(
        display_frame
    )

    # ========================================================
    # DRAW PERSON DETECTIONS
    # ========================================================

    for detection in result["detections"]:

        x1, y1, x2, y2 = detection["bbox"]

        # ----------------------------------------------------
        # Foot point
        # ----------------------------------------------------

        foot_x = int(
            (x1 + x2) / 2
        )

        foot_y = int(y2)

        # ----------------------------------------------------
        # Determine role
        # ----------------------------------------------------

        if instructor_zone.contains_point(
            foot_x,
            foot_y
        ):

            role = "instructor"

        else:

            role = "student"

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            display_frame,
            (
                int(x1),
                int(y1)
            ),
            (
                int(x2),
                int(y2)
            ),
            (255, 0, 0),
            2
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{role} "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            display_frame,
            label,
            (
                int(x1),
                max(
                    20,
                    int(y1) - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        # ----------------------------------------------------
        # Foot point
        # ----------------------------------------------------

        cv2.circle(
            display_frame,
            (
                foot_x,
                foot_y
            ),
            5,
            (0, 255, 255),
            -1
        )

    return display_frame


# ============================================================
# RUN 3-FRAME BURST
# ============================================================

def run_burst(
    cap,
    object_detector,
    face_detector,
    face_blurrer,
    instructor_zone,
    packager
):

    persons_per_frame = []

    excluded_per_frame = []

    frame_results = []

    evidence_frame = None

    max_persons = -1

    max_frame_index = -1

    # ========================================================
    # START
    # ========================================================

    print("\n")

    print("=" * 60)
    print("STARTING 3-FRAME SAMPLE")
    print("=" * 60)

    # ========================================================
    # CAPTURE THREE FRAMES
    # ========================================================

    for i in range(BURST_FRAMES):

        print(
            f"\n[INFO] Capturing frame "
            f"{i + 1}/{BURST_FRAMES}..."
        )

        # ----------------------------------------------------
        # Capture
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "[ERROR] Could not capture frame."
            )

            continue

        raw_frame = frame.copy()

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        start_time = time.time()

        result = process_frame(
            raw_frame,
            object_detector,
            face_detector,
            instructor_zone
        )

        processing_time = (
            time.time() -
            start_time
        ) * 1000

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        persons = result["persons"]

        excluded = result[
            "excludedInstructor"
        ]

        persons_per_frame.append(
            persons
        )

        excluded_per_frame.append(
            excluded
        )

        frame_results.append(
            result
        )

        # ----------------------------------------------------
        # Select evidence frame
        #
        # The frame containing the maximum
        # number of students becomes evidence.
        # ----------------------------------------------------

        if persons > max_persons:

            max_persons = persons

            max_frame_index = i

            evidence_frame = (
                raw_frame.copy()
            )

        # ----------------------------------------------------
        # Draw current frame
        # ----------------------------------------------------

        display_frame = draw_frame(
            raw_frame,
            result,
            instructor_zone,
            face_blurrer
        )

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"SAMPLE FRAME: {i + 1}/3",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Students: {persons}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Instructor Excluded: {excluded}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Processing: "
            f"{processing_time:.1f} ms",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "AI Edge Auditor - 3 Frame Sample",
            display_frame
        )

        cv2.waitKey(1)

        # ----------------------------------------------------
        # Terminal
        # ----------------------------------------------------

        print(
            f"[FRAME {i + 1}] "
            f"students={persons}, "
            f"instructor={excluded}, "
            f"processing="
            f"{processing_time:.1f}ms"
        )

        # ----------------------------------------------------
        # Wait before next frame
        # ----------------------------------------------------

        if i < BURST_FRAMES - 1:

            print(
                "[INFO] Waiting 2 seconds..."
            )

            start_wait = time.time()

            while (
                time.time() - start_wait
                < FRAME_INTERVAL
            ):

                key = (
                    cv2.waitKey(50)
                    & 0xFF
                )

                if key == ord("q"):

                    print(
                        "[INFO] Burst cancelled."
                    )

                    return None

    # ========================================================
    # VALIDATION
    # ========================================================

    if not persons_per_frame:

        print(
            "[ERROR] No frames processed."
        )

        return None

    # ========================================================
    # MAX PERSON COUNT
    # ========================================================

    persons_sample = max(
        persons_per_frame
    )

    # ========================================================
    # EVIDENCE FRAME
    # ========================================================

    if evidence_frame is not None:

        evidence_result = (
            frame_results[
                max_frame_index
            ]
        )

        evidence_display = draw_frame(
            evidence_frame,
            evidence_result,
            instructor_zone,
            face_blurrer
        )

        cv2.putText(
            evidence_display,
            "EVIDENCE FRAME",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 0),
            2
        )

        cv2.putText(
            evidence_display,
            f"Persons Sample: "
            f"{persons_sample}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Edge Auditor - Evidence",
            evidence_display
        )

        cv2.waitKey(1500)

    # ========================================================
    # CREATE OBSERVATION PAYLOAD
    # ========================================================

    observation = (
        packager.create_sample_observation(
            persons=persons_sample,

            persons_per_frame=
                persons_per_frame,

            excluded_instructor=
                max(excluded_per_frame),

            evidence_frame=
                evidence_frame,

            monitors={
                "total": 0,
                "on": 0
            },

            seats=None,

            slots=[],

            tamper=None
        )
    )

    # ========================================================
    # PRINT OBSERVATION SUMMARY
    # ========================================================

    print("\n")

    print("=" * 60)
    print("OBSERVATION PAYLOAD")
    print("=" * 60)

    print(
        f"obsId: "
        f"{observation['obsId']}"
    )

    print(
        f"deviceId: "
        f"{observation['deviceId']}"
    )

    print(
        f"timestamp: "
        f"{observation['ts']}"
    )

    print(
        f"persons: "
        f"{observation['persons']}"
    )

    print(
        f"personsPerFrame: "
        f"{observation['personsPerFrame']}"
    )

    print(
        f"excludedInstructor: "
        f"{observation['excludedInstructor']}"
    )

    print(
        f"monitor total: "
        f"{observation['monitors']['total']}"
    )

    print(
        f"monitor ON: "
        f"{observation['monitors']['on']}"
    )

    print(
        f"seats: "
        f"{observation['seats']}"
    )

    print(
        f"slots: "
        f"{observation['slots']}"
    )

    print(
        f"tamper: "
        f"{observation['tamper']}"
    )

    print(
        f"evidence Base64 size: "
        f"{len(observation['evidence'])} "
        f"characters"
    )

    print(
        f"edgeVersion: "
        f"{observation['edgeVersion']}"
    )

    print("=" * 60)

    # ========================================================
    # JSON
    # ========================================================

    json_payload = packager.to_json(
        observation
    )

    print("\nJSON PAYLOAD:")
    print(json_payload)

    # ========================================================
    # RETURN
    # ========================================================

    return observation


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI EDGE AUDITOR - 3 FRAME SAMPLE TEST")
    print("=" * 60)

    # ========================================================
    # CAMERA
    # ========================================================

    print(
        "\n[INFO] Opening camera..."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "[ERROR] Camera could not be opened."
        )

        return

    print(
        "[OK] Camera opened."
    )

    # ========================================================
    # YOLO
    # ========================================================

    print(
        "[INFO] Loading YOLO..."
    )

    object_detector = ObjectDetector(
        model_path=YOLO_MODEL,
        confidence_threshold=0.40,
        nms_threshold=0.45
    )

    print(
        "[OK] YOLO loaded."
    )

    # ========================================================
    # YUNET
    # ========================================================

    print(
        "[INFO] Loading YuNet..."
    )

    face_detector = FaceDetector(
        model_path=YUNET_MODEL,
        confidence_threshold=0.60,
        nms_threshold=0.30
    )

    print(
        "[OK] YuNet loaded."
    )

    # ========================================================
    # PRIVACY
    # ========================================================

    face_blurrer = FaceBlurrer(
        margin_ratio=0.15
    )

    print(
        "[OK] Privacy blur loaded."
    )

    # ========================================================
    # TAMPER
    # ========================================================

    tamper_detector = TamperDetector()

    print(
        "[OK] Tamper detector loaded."
    )

    # ========================================================
    # OBSERVATION PACKAGER
    # ========================================================

    packager = ObservationPackager(
        device_id="D-101",
        edge_version="0.1.0",
        evidence_width=960,
        webp_quality=60
    )

    print(
        "[OK] Observation packager loaded."
    )

    # ========================================================
    # FIRST FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        print(
            "[ERROR] Could not read camera."
        )

        cap.release()

        return

    height, width = frame.shape[:2]

    print(
        f"[INFO] Resolution: "
        f"{width}x{height}"
    )

    # ========================================================
    # INSTRUCTOR POLYGON
    # ========================================================

    instructor_points = [

        (0, 0),

        (
            int(width * 0.30),
            0
        ),

        (
            int(width * 0.30),
            int(height * 0.45)
        ),

        (
            0,
            int(height * 0.45)
        )
    ]

    instructor_zone = PolygonZone(
        name="instructor",
        points=instructor_points
    )

    print(
        "[OK] Instructor polygon created."
    )

    # ========================================================
    # REFERENCE STATE
    # ========================================================

    reference_registered = False

    # ========================================================
    # CONTROLLER LOOP
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print(
        "C = Register camera reference"
    )
    print(
        "B = Run 3-frame sample"
    )
    print(
        "Q = Quit"
    )
    print("=" * 60)

    while True:

        # ----------------------------------------------------
        # Read frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            continue

        display_frame = frame.copy()

        # ====================================================
        # TAMPER STATUS
        # ====================================================

        if reference_registered:

            tamper_result = (
                tamper_detector.check(
                    frame
                )
            )

            if tamper_result[
                "tampered"
            ]:

                status = (
                    "CAMERA TAMPER DETECTED"
                )

                status_color = (
                    0,
                    0,
                    255
                )

            else:

                status = "CAMERA OK"

                status_color = (
                    0,
                    255,
                    0
                )

        else:

            status = (
                "PRESS C TO REGISTER CAMERA"
            )

            status_color = (
                0,
                255,
                255
            )

        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(
            display_frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            status_color,
            2
        )

        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(
            display_frame,
            "C: Register | "
            "B: 3-Frame Sample | "
            "Q: Quit",
            (
                20,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "AI Edge Auditor - Sample Controller",
            display_frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ====================================================
        # C = REGISTER REFERENCE
        # ====================================================

        if key == ord("c"):

            result = (
                tamper_detector
                .register_reference(
                    frame
                )
            )

            reference_registered = True

            tamper_detector.reset()

            print("\n")

            print(
                "=" * 60
            )

            print(
                "CAMERA REFERENCE REGISTERED"
            )

            print(
                "=" * 60
            )

            print(result)

        # ====================================================
        # B = 3 FRAME SAMPLE
        # ====================================================

        elif key == ord("b"):

            if not reference_registered:

                print("\n[WARNING]")
                print(
                    "Register camera reference "
                    "first using C."
                )

                continue

            result = run_burst(
                cap=cap,
                object_detector=
                    object_detector,
                face_detector=
                    face_detector,
                face_blurrer=
                    face_blurrer,
                instructor_zone=
                    instructor_zone,
                packager=
                    packager
            )

            if result is not None:

                print("\n")

                print(
                    "[SUCCESS] "
                    "Sample complete."
                )

        # ====================================================
        # Q = QUIT
        # ====================================================

        elif key == ord("q"):

            print(
                "\n[INFO] Exiting..."
            )

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[OK] Camera released."
    )

    print(
        "[OK] Pipeline stopped."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()