import sys
import time
from pathlib import Path

import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "ai" / "src"

# Allow imports from ai/src
sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PROJECT MODULES
# ============================================================

from detection.object_detector import ObjectDetector
from detection.face_detector import FaceDetector
from privacy.face_blur import FaceBlurrer
from tracking.tracker import CentroidTracker
from zones.instructor_zone import PolygonZone
from tamper.tamper_detector import TamperDetector
from operability.operability import OperabilityAnalyzer
from output.package import ObservationPackager
from storage.outbox import ObservationOutbox


# ============================================================
# MODEL PATHS
# ============================================================

YOLO_MODEL = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "object_detection"
    / "yolov8n.onnx"
)

YUNET_MODEL = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "face_detection"
    / "face_detection_yunet_2023mar.onnx"
)


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

CAMERA_SOURCE = 0

# Observation architecture
BURST_FRAMES = 3
FRAME_INTERVAL = 2.0


# ============================================================
# DEFAULT TAMPER RESULT
# ============================================================

def default_tamper_result():
    return {
        "tampered": False,
        "reasons": [],
        "brightness": 0.0,
        "blurScore": 0.0,
        "grayStd": 0.0,
        "matchRatio": 1.0,
        "goodMatches": 0,
        "consecutiveTampered": 0,
    }


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    frame,
    object_detector,
    face_detector,
    face_blurrer,
    instructor_zone,
    tamper_detector,
    operability,
    reference_registered,
    machine_zone,
):
    """
    Process one camera frame.

    Returns:
        dict containing all detection/analysis results.
    """

    original_frame = frame.copy()

    height, width = original_frame.shape[:2]

    # --------------------------------------------------------
    # 1. TAMPER DETECTION
    # --------------------------------------------------------

    if reference_registered:
        tamper_result = tamper_detector.check(
            original_frame
        )
    else:
        tamper_result = default_tamper_result()

    tampered = bool(
        tamper_result.get("tampered", False)
    )

    # --------------------------------------------------------
    # 2. OBJECT DETECTION
    # --------------------------------------------------------

    detections = object_detector.detect(
        original_frame
    )

    # Only persons
    person_detections = [
        detection
        for detection in detections
        if detection.get("class_name") == "person"
    ]

    # --------------------------------------------------------
    # 3. INSTRUCTOR EXCLUSION
    # --------------------------------------------------------

    person_results = []

    student_count = 0
    instructor_count = 0

    for detection in person_detections:

        x1, y1, x2, y2 = detection["bbox"]

        # Foot point = bottom-center of bounding box
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)

        is_instructor = instructor_zone.contains_point(
            foot_x,
            foot_y
        )

        if is_instructor:
            role = "instructor"
            instructor_count += 1
        else:
            role = "student"
            student_count += 1

        person_results.append(
            {
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "role": role,
                "foot": (foot_x, foot_y),
            }
        )

    # --------------------------------------------------------
    # 4. MONITOR ANALYSIS
    # --------------------------------------------------------

    monitor_result = operability.analyze_monitors(
        original_frame,
        detections
    )

    # --------------------------------------------------------
    # 5. MACHINE SLOT OCCUPANCY
    # --------------------------------------------------------

    occupied = None

    if machine_zone is not None:

        occupied = operability.is_slot_occupied(
            person_detections,
            machine_zone
        )

    # --------------------------------------------------------
    # 6. FACE DETECTION
    # --------------------------------------------------------

    faces = face_detector.detect(
        original_frame
    )

    # --------------------------------------------------------
    # 7. PRIVACY-SAFE EVIDENCE FRAME
    # --------------------------------------------------------

    privacy_frame = original_frame.copy()

    # Blur detected faces
    privacy_frame = face_blurrer.blur_faces(
        privacy_frame,
        faces
    )

    # --------------------------------------------------------
    # 8. SAFETY BLUR
    #
    # Blur upper 25% of every person box.
    # This protects privacy even when YuNet misses a face.
    # --------------------------------------------------------

    for person in person_results:

        x1, y1, x2, y2 = map(
            int,
            person["bbox"]
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        upper_y2 = min(
            y2,
            y1 + int((y2 - y1) * 0.25)
        )

        roi = privacy_frame[
            y1:upper_y2,
            x1:x2
        ]

        if roi.size > 0:

            privacy_frame[
                y1:upper_y2,
                x1:x2
            ] = cv2.GaussianBlur(
                roi,
                (0, 0),
                sigmaX=20,
                sigmaY=20
            )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "original_frame": original_frame,
        "privacy_frame": privacy_frame,

        "tamper_result": tamper_result,
        "tampered": tampered,

        "detections": detections,
        "person_detections": person_detections,
        "person_results": person_results,

        "student_count": student_count,
        "instructor_count": instructor_count,

        "monitor_result": monitor_result,

        "faces": faces,

        "occupied": occupied,
    }


# ============================================================
# DRAW LIVE DISPLAY
# ============================================================

def draw_display(
    result,
    instructor_zone,
    machine_bbox,
    machine_result,
    tracks,
    fps,
    processing_ms,
    reference_registered,
):
    """
    Draw privacy-safe live monitoring display.
    """

    display = result["privacy_frame"].copy()

    person_results = result["person_results"]
    monitor_result = result["monitor_result"]

    # --------------------------------------------------------
    # INSTRUCTOR ZONE
    # --------------------------------------------------------

    display = instructor_zone.draw(
        display
    )

    # --------------------------------------------------------
    # PERSON BOXES
    # --------------------------------------------------------

    for person in person_results:

        x1, y1, x2, y2 = map(
            int,
            person["bbox"]
        )

        if person["role"] == "instructor":
            color = (0, 165, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            display,
            (
                f"{person['role']} "
                f"{person['confidence']:.2f}"
            ),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # Foot point
        cv2.circle(
            display,
            person["foot"],
            4,
            (0, 255, 255),
            -1
        )

    # --------------------------------------------------------
    # MONITORS
    # --------------------------------------------------------

    for monitor in monitor_result.get(
        "details",
        []
    ):

        x1, y1, x2, y2 = monitor["bbox"]

        label = (
            f"{monitor['class']} "
            f"{'ON' if monitor['on'] else 'OFF'}"
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # MACHINE SLOT
    # --------------------------------------------------------

    if machine_bbox is not None:

        x, y, w, h = machine_bbox

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

        label = "Machine slot"

        if machine_result is not None:

            similarity = float(
                machine_result.get(
                    "similarity",
                    0.0
                )
            )

            present = bool(
                machine_result.get(
                    "present",
                    False
                )
            )

            label += (
                f" | {similarity:.3f}"
                f" | "
                f"{'PRESENT' if present else 'MISSING'}"
            )

        if result["occupied"] is not None:

            label += (
                " | OCCUPIED"
                if result["occupied"]
                else " | NOT OCCUPIED"
            )

        cv2.putText(
            display,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

    # --------------------------------------------------------
    # TRACK IDs  
    # --------------------------------------------------------

    for object_id, centroid in tracks.items():

        cv2.putText(
            display,
            f"ID {object_id}",
            tuple(
                map(int, centroid)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

    # --------------------------------------------------------
    # CAMERA STATUS
    # --------------------------------------------------------

    if result["tampered"]:

        status = "CAMERA TAMPER DETECTED"
        status_color = (0, 0, 255)

    elif not reference_registered:

        status = "REGISTER CAMERA - PRESS C"
        status_color = (0, 255, 255)

    else:

        status = "CAMERA OK"
        status_color = (0, 255, 0)

    cv2.putText(
        display,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    info = [
        f"Students: {result['student_count']}",
        f"Instructor: {result['instructor_count']}",
        f"Persons: {len(result['person_detections'])}",
        f"Faces: {len(result['faces'])}",
        f"Monitors: {monitor_result.get('total', 0)}",
        f"Monitors ON: {monitor_result.get('on', 0)}",
        f"FPS: {fps:.2f}",
        f"Processing: {processing_ms:.1f} ms",
        "C Camera | M Machine | E Evaluate",
        "S Sample | Q Quit",
    ]

    for index, text in enumerate(info):

        cv2.putText(
            display,
            text,
            (20, 60 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return display


# ============================================================
# CREATE OBSERVATION FROM 3 FRAME BURST
# ============================================================

def create_observation(
    burst_results,
    packager,
    machine_result,
    machine_bbox,
):
    """
    Convert a 3-frame burst into one observation payload.

    persons:
        MAX student count across the burst.

    evidence:
        Privacy-safe frame with maximum student count.
    """

    if not burst_results:
        return None

    # --------------------------------------------------------
    # PERSON COUNTS
    # --------------------------------------------------------

    persons_per_frame = [
        int(result["student_count"])
        for result in burst_results
    ]

    instructor_per_frame = [
        int(result["instructor_count"])
        for result in burst_results
    ]

    persons = max(
        persons_per_frame
    )

    excluded_instructor = max(
        instructor_per_frame
    )

    # --------------------------------------------------------
    # MAX-PERSON EVIDENCE FRAME
    # --------------------------------------------------------

    evidence_index = max(
        range(len(burst_results)),
        key=lambda i: (
            burst_results[i]["student_count"]
        )
    )

    evidence_result = (
        burst_results[evidence_index]
    )

    evidence_frame = (
        evidence_result["privacy_frame"]
    )

    # --------------------------------------------------------
    # MONITORS
    # --------------------------------------------------------

    monitor_result = (
        evidence_result["monitor_result"]
    )

    monitors = {
        "total": int(
            monitor_result.get(
                "total",
                0
            )
        ),
        "on": int(
            monitor_result.get(
                "on",
                0
            )
        ),
    }

    # --------------------------------------------------------
    # MACHINE SLOT
    # --------------------------------------------------------

    slots = []

    if machine_bbox is not None:

        x, y, w, h = machine_bbox

        slot = {
            "slotId": "machine-slot",
            "type": "machine",
            "bbox": [
                int(x),
                int(y),
                int(w),
                int(h),
            ],
        }

        # Machine reference analysis
        if machine_result is not None:

            slot["present"] = bool(
                machine_result.get(
                    "present",
                    False
                )
            )

            slot["similarity"] = float(
                machine_result.get(
                    "similarity",
                    0.0
                )
            )

            slot["missingStreak"] = int(
                machine_result.get(
                    "missingStreak",
                    0
                )
            )

        # Occupancy
        occupied = evidence_result.get(
            "occupied"
        )

        if occupied is not None:

            slot["occupied"] = bool(
                occupied
            )

        slots.append(slot)

    # --------------------------------------------------------
    # TAMPER
    # --------------------------------------------------------

    tamper = None

    tamper_result = (
        evidence_result["tamper_result"]
    )

    if tamper_result.get(
        "tampered",
        False
    ):

        tamper = {
            "tampered": True,
            "reasons": tamper_result.get(
                "reasons",
                []
            ),
        }

    # --------------------------------------------------------
    # CREATE FINAL PAYLOAD
    # --------------------------------------------------------

    payload = (
        packager.create_sample_observation(
            persons=persons,
            persons_per_frame=persons_per_frame,
            excluded_instructor=excluded_instructor,
            evidence_frame=evidence_frame,
            monitors=monitors,
            seats=None,
            slots=slots,
            tamper=tamper,
        )
    )

    return payload


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIH26245 - FULL EDGE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # VERIFY MODELS
    # --------------------------------------------------------

    if not YOLO_MODEL.exists():

        print(
            f"[ERROR] YOLO model not found:\n"
            f"{YOLO_MODEL}"
        )

        return

    if not YUNET_MODEL.exists():

        print(
            f"[ERROR] YuNet model not found:\n"
            f"{YUNET_MODEL}"
        )

        return

    print(
        f"[OK] YOLO model: {YOLO_MODEL}"
    )

    print(
        f"[OK] YuNet model: {YUNET_MODEL}"
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_SOURCE,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print(
            "[ERROR] Camera could not be opened."
        )

        cap.release()
        return

    print(
        "[OK] Camera opened."
    )

    # --------------------------------------------------------
    # AI MODULES
    # --------------------------------------------------------

    object_detector = ObjectDetector(
        model_path=str(YOLO_MODEL),
        confidence_threshold=0.40,
        nms_threshold=0.45,
    )

    face_detector = FaceDetector(
        model_path=str(YUNET_MODEL),
        confidence_threshold=0.60,
        nms_threshold=0.30,
    )

    face_blurrer = FaceBlurrer(
        margin_ratio=0.20
    )

    tracker = CentroidTracker(
        max_distance=100,
        max_disappeared=10,
    )

    tamper_detector = TamperDetector()

    operability = OperabilityAnalyzer()

    # --------------------------------------------------------
    # OBSERVATION PACKAGER
    # --------------------------------------------------------

    packager = ObservationPackager(
        device_id="D-101",
        edge_version="0.1.0",
        evidence_width=960,
        webp_quality=60,
    )

    # --------------------------------------------------------
    # SQLITE OUTBOX
    # --------------------------------------------------------

    outbox = ObservationOutbox()

    print(
        "[OK] Detection, privacy, tracking, "
        "tamper and operability modules loaded."
    )

    print(
        f"[OK] SQLite outbox:"
        f" {outbox.db_path}"
    )

    print(
        f"[INFO] Existing queue:"
        f" {outbox.count()}"
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    reference_registered = False

    instructor_zone = None

    machine_reference = None
    machine_bbox = None
    machine_zone = None
    machine_result = None

    fps = 0.0
    previous_time = time.time()

    # ========================================================
    # CONTROLS
    # ========================================================

    print()
    print("=" * 70)
    print("CONTROLS")
    print("=" * 70)
    print("C = Register camera tamper reference")
    print("M = Select/register machine slot")
    print("E = Evaluate machine sample")
    print("S = Capture 3-frame observation + save SQLite")
    print("Q = Quit")
    print("=" * 70)

    try:

        while True:

            loop_start = time.time()

            # ------------------------------------------------
            # CAMERA FRAME
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok:

                print(
                    "[WARNING] Camera frame failed."
                )

                time.sleep(0.1)
                continue

            height, width = frame.shape[:2]

            # ------------------------------------------------
            # INSTRUCTOR ZONE
            # ------------------------------------------------

            if instructor_zone is None:

                instructor_zone = PolygonZone(
                    name="instructor",
                    points=[
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
                        ),
                    ],
                )

                print(
                    "[OK] Instructor zone initialized."
                )

            # ------------------------------------------------
            # PROCESS FRAME
            # ------------------------------------------------

            result = process_frame(
                frame=frame,
                object_detector=object_detector,
                face_detector=face_detector,
                face_blurrer=face_blurrer,
                instructor_zone=instructor_zone,
                tamper_detector=tamper_detector,
                operability=operability,
                reference_registered=reference_registered,
                machine_zone=machine_zone,
            )

            # ------------------------------------------------
            # TRACK PERSONS
            # ------------------------------------------------

            tracks = tracker.update(
                result["person_detections"]
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            now = time.time()

            delta = now - previous_time

            if delta > 0:
                fps = 1.0 / delta

            previous_time = now

            processing_ms = (
                time.time() - loop_start
            ) * 1000

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display = draw_display(
                result=result,
                instructor_zone=instructor_zone,
                machine_bbox=machine_bbox,
                machine_result=machine_result,
                tracks=tracks,
                fps=fps,
                processing_ms=processing_ms,
                reference_registered=reference_registered,
            )

            cv2.imshow(
                "SIH26245 - Full Pipeline",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            # =================================================
            # C - CAMERA REFERENCE
            # =================================================

            if key == ord("c"):

                print()
                print(
                    "[ACTION] Registering camera reference..."
                )

                registration = (
                    tamper_detector.register_reference(
                        result["original_frame"]
                    )
                )

                reference_registered = True

                tamper_detector.reset()

                print(
                    "[OK] Camera reference registered."
                )

                print(registration)

            # =================================================
            # M - MACHINE REGISTRATION
            # =================================================

            elif key == ord("m"):

                print()
                print(
                    "[ACTION] Select machine region..."
                )

                cv2.imshow(
                    "Select machine slot",
                    result["original_frame"]
                )

                roi = cv2.selectROI(
                    "Select machine slot",
                    result["original_frame"],
                    showCrosshair=True,
                    fromCenter=False,
                )

                cv2.destroyWindow(
                    "Select machine slot"
                )

                x, y, w, h = map(
                    int,
                    roi
                )

                if w <= 0 or h <= 0:

                    print(
                        "[WARNING] Invalid machine region."
                    )

                    continue

                machine_bbox = (
                    x,
                    y,
                    w,
                    h
                )

                machine_reference = (
                    result["original_frame"][
                        y:y + h,
                        x:x + w
                    ].copy()
                )

                machine_zone = PolygonZone(
                    name="slot",
                    points=[
                        (x, y),
                        (x + w, y),
                        (x + w, y + h),
                        (x, y + h),
                    ],
                )

                operability.reset_machine_slot(
                    "machine-slot"
                )

                machine_result = None

                print(
                    "[OK] Machine reference registered."
                )

                print(
                    "[INFO] Press E to evaluate machine."
                )

            # =================================================
            # E - MACHINE EVALUATION
            # =================================================

            elif key == ord("e"):

                if (
                    machine_bbox is None
                    or machine_reference is None
                ):

                    print(
                        "[WARNING] "
                        "Press M and register machine first."
                    )

                    continue

                x, y, w, h = machine_bbox

                current_crop = (
                    result["original_frame"][
                        y:y + h,
                        x:x + w
                    ]
                )

                if current_crop.size == 0:

                    print(
                        "[ERROR] Machine crop is empty."
                    )

                    continue

                machine_result = (
                    operability.analyze_machine_slot(
                        "machine-slot",
                        current_crop,
                        machine_reference,
                    )
                )

                occupied = (
                    operability.is_slot_occupied(
                        result["person_detections"],
                        machine_zone,
                    )
                )

                result["occupied"] = occupied

                print()
                print("-" * 60)
                print("MACHINE SAMPLE")
                print("-" * 60)

                print(
                    f"Similarity: "
                    f"{machine_result['similarity']:.3f}"
                )

                print(
                    "Status: "
                    f"{'PRESENT' if machine_result['present'] else 'MISSING'}"
                )

                print(
                    f"Missing streak: "
                    f"{machine_result['missingStreak']}"
                )

                print(
                    f"Occupied: {occupied}"
                )

                print("-" * 60)

            # =================================================
            # S - 3 FRAME OBSERVATION
            # =================================================

            elif key == ord("s"):

                print()
                print("=" * 70)
                print("CAPTURING 3-FRAME OBSERVATION")
                print("=" * 70)

                burst_results = []

                # ------------------------------------------------
                # CAPTURE 3 FRAMES
                # ------------------------------------------------

                for frame_index in range(
                    BURST_FRAMES
                ):

                    print(
                        f"\n[FRAME "
                        f"{frame_index + 1}/"
                        f"{BURST_FRAMES}]"
                    )

                    # First frame = current frame
                    if frame_index == 0:

                        burst_frame = (
                            result["original_frame"].copy()
                        )

                    else:

                        print(
                            f"[WAIT] "
                            f"{FRAME_INTERVAL:.1f}s"
                        )

                        time.sleep(
                            FRAME_INTERVAL
                        )

                        ok_burst, burst_frame = (
                            cap.read()
                        )

                        if not ok_burst:

                            print(
                                "[ERROR] "
                                "Could not capture burst frame."
                            )

                            continue

                    # ------------------------------------------------
                    # PROCESS BURST FRAME
                    # ------------------------------------------------

                    burst_result = process_frame(
                        frame=burst_frame,
                        object_detector=object_detector,
                        face_detector=face_detector,
                        face_blurrer=face_blurrer,
                        instructor_zone=instructor_zone,
                        tamper_detector=tamper_detector,
                        operability=operability,
                        reference_registered=reference_registered,
                        machine_zone=machine_zone,
                    )

                    burst_results.append(
                        burst_result
                    )

                    print(
                        f"Students: "
                        f"{burst_result['student_count']}"
                    )

                    print(
                        f"Instructor: "
                        f"{burst_result['instructor_count']}"
                    )

                    print(
                        f"Monitors: "
                        f"{burst_result['monitor_result'].get('total', 0)}"
                    )

                    print(
                        f"Monitors ON: "
                        f"{burst_result['monitor_result'].get('on', 0)}"
                    )

                    print(
                        f"Faces: "
                        f"{len(burst_result['faces'])}"
                    )

                    if burst_result["tampered"]:

                        print(
                            "[WARNING] "
                            "TAMPER DETECTED"
                        )

                # ------------------------------------------------
                # VALIDATION
                # ------------------------------------------------

                if not burst_results:

                    print(
                        "[ERROR] "
                        "No frames captured."
                    )

                    continue

                # ------------------------------------------------
                # CREATE OBSERVATION
                # ------------------------------------------------

                observation = create_observation(
                    burst_results=burst_results,
                    packager=packager,
                    machine_result=machine_result,
                    machine_bbox=machine_bbox,
                )

                if observation is None:

                    print(
                        "[ERROR] "
                        "Observation creation failed."
                    )

                    continue

                # ------------------------------------------------
                # STORE SQLITE
                # ------------------------------------------------

                outbox.add(
                    observation
                )

                queue_count = outbox.count()

                print()
                print("=" * 70)
                print("OBSERVATION SAVED")
                print("=" * 70)

                print(
                    f"obsId: "
                    f"{observation['obsId']}"
                )

                print(
                    f"deviceId: "
                    f"{observation['deviceId']}"
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
                    f"monitors: "
                    f"{observation['monitors']}"
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
                    f"evidence size: "
                    f"{len(observation['evidence'])} chars"
                )

                print(
                    f"SQLite queue: "
                    f"{queue_count}"
                )

                print("=" * 70)
                print(
                    "[PASS] Observation stored in SQLite."
                )

            # =================================================
            # Q - QUIT
            # =================================================

            elif key == ord("q"):

                print()
                print(
                    "[INFO] Stopping pipeline..."
                )

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print()
        print(
            "[OK] Camera released."
        )

        print(
            f"[INFO] SQLite queue remaining: "
            f"{outbox.count()}"
        )

        print(
            "[OK] Pipeline stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()