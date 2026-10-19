import time
from pathlib import Path

import cv2

from ai.src.camera.capture import CameraCapture
from ai.src.detection.object_detector import ObjectDetector
from ai.src.detection.face_detector import FaceDetector
from ai.src.privacy.face_blur import FaceBlurrer
from ai.src.tamper.tamper_detector import TamperDetector
from ai.src.zones.instructor_zone import PolygonZone
from ai.src.output.package import ObservationPackager
from ai.src.storage.outbox import ObservationOutbox


class EdgePipeline:
    """
    Main Edge AI pipeline.

    Production flow:

        Camera
          ↓
        3-frame burst
          ↓
        Tamper check
          ↓
        YOLO detection
          ↓
        Instructor exclusion
          ↓
        MAX person count
          ↓
        Evidence frame
          ↓
        YuNet face detection
          ↓
        Privacy blur
          ↓
        WebP evidence
          ↓
        Observation payload
          ↓
        SQLite outbox
    """

    def __init__(
        self,
        camera_source=0,
        device_id="D-101",
        object_model="ai/models/object_detection/yolov8n.onnx",
        face_model="ai/models/face_detection/face_detection_yunet_2023mar.onnx",
        instructor_points=None,
        outbox_db=None,
        edge_version="0.1.0"
    ):

        self.device_id = device_id
        self.edge_version = edge_version

        # --------------------------------------------------
        # Camera
        # --------------------------------------------------

        self.camera = CameraCapture(
            camera_source
        )

        # --------------------------------------------------
        # AI models
        # --------------------------------------------------

        self.object_detector = ObjectDetector(
            model_path=object_model,
            confidence_threshold=0.35,
            nms_threshold=0.50
        )

        self.face_detector = FaceDetector(
            face_model
        )

        self.face_blurrer = FaceBlurrer(
            margin_ratio=0.20
        )

        # --------------------------------------------------
        # Tamper detector
        # --------------------------------------------------

        self.tamper_detector = TamperDetector()

        # --------------------------------------------------
        # Instructor zone
        # --------------------------------------------------

        if instructor_points is None:

            # Example default zone.
            # Replace through remote configuration later.
            instructor_points = [
                (100, 100),
                (500, 100),
                (500, 250),
                (100, 250)
            ]

        self.instructor_zone = PolygonZone(
            name="instructor",
            points=instructor_points
        )

        # --------------------------------------------------
        # Observation package
        # --------------------------------------------------

        self.packager = ObservationPackager(
            device_id=device_id,
            edge_version=edge_version
        )

        # --------------------------------------------------
        # SQLite Outbox
        # --------------------------------------------------

        if outbox_db is None:

            project_root = Path(
                __file__
            ).resolve().parents[2]

            outbox_db = (
                project_root
                / "data"
                / "edge_outbox.db"
            )

        self.outbox = ObservationOutbox(
            db_path=outbox_db
        )

        self.running = False

    # ======================================================
    # CAMERA
    # ======================================================

    def open(self):

        self.camera.open()

        print("[OK] Edge pipeline camera ready.")

    def release(self):

        self.camera.release()

        print("[OK] Edge pipeline released.")

    # ======================================================
    # CAPTURE BURST
    # ======================================================

    def capture_burst(
        self,
        frame_gap_seconds=2
    ):

        frames = []

        for index in range(3):

            frame = self.camera.capture_frame()

            frames.append(frame)

            print(
                f"[CAPTURE] Frame {index + 1}/3"
            )

            if index < 2:

                time.sleep(
                    frame_gap_seconds
                )

        return frames

    # ======================================================
    # PERSON PROCESSING
    # ======================================================

    def process_persons(
        self,
        detections
    ):

        total_persons = 0
        excluded_instructor = 0

        for detection in detections:

            if detection["class_name"] != "person":
                continue

            total_persons += 1

            bbox = detection["bbox"]

            if self.instructor_zone.contains_person(
                bbox
            ):

                excluded_instructor += 1

        students = (
            total_persons
            - excluded_instructor
        )

        return {
            "persons": students,
            "excludedInstructor": excluded_instructor
        }

    # ======================================================
    # EVIDENCE FRAME
    # ======================================================

    def create_evidence(
        self,
        frame,
        detections
    ):

        evidence = frame.copy()

        # ----------------------------------------------
        # Draw instructor zone
        # ----------------------------------------------

        self.instructor_zone.draw(
            evidence
        )

        # ----------------------------------------------
        # Draw detections
        # ----------------------------------------------

        for detection in detections:

            class_name = detection[
                "class_name"
            ]

            confidence = detection[
                "confidence"
            ]

            x1, y1, x2, y2 = detection[
                "bbox"
            ]

            # Person color / display
            if class_name == "person":

                cv2.rectangle(
                    evidence,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    evidence,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # ----------------------------------------------
        # YuNet face detection
        # ----------------------------------------------

        faces = self.face_detector.detect(
            evidence
        )

        # ----------------------------------------------
        # Face privacy blur
        # ----------------------------------------------

        evidence = self.face_blurrer.blur_faces(
            evidence,
            faces
        )

        # ----------------------------------------------
        # Safety net:
        # Blur upper 25% of EVERY person box.
        # ----------------------------------------------

        for detection in detections:

            if detection["class_name"] != "person":
                continue

            x1, y1, x2, y2 = detection[
                "bbox"
            ]

            person_height = y2 - y1

            head_y2 = int(
                y1 + person_height * 0.25
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(
                evidence.shape[1],
                x2
            )
            head_y2 = min(
                evidence.shape[0],
                head_y2
            )

            roi = evidence[
                y1:head_y2,
                x1:x2
            ]

            if roi.size == 0:
                continue

            blurred = cv2.GaussianBlur(
                roi,
                (0, 0),
                sigmaX=20,
                sigmaY=20
            )

            evidence[
                y1:head_y2,
                x1:x2
            ] = blurred

        return evidence, faces

    # ======================================================
    # RUN ONE SAMPLE
    # ======================================================

    def run_sample(
        self,
        frame_gap_seconds=2
    ):

        print("\n")
        print("=" * 60)
        print("EDGE AI SAMPLE")
        print("=" * 60)

        # --------------------------------------------------
        # 1. Capture 3 frames
        # --------------------------------------------------

        frames = self.capture_burst(
            frame_gap_seconds
        )

        # --------------------------------------------------
        # 2. Tamper check on first frame
        # --------------------------------------------------

        tamper_result = self.tamper_detector.check(
            frames[0]
        )

        print(
            f"[TAMPER] {tamper_result}"
        )

        if tamper_result["tampered"]:

            print(
                "[WARNING] Camera tampered."
            )

            payload = self.packager.create_payload(
                persons=0,
                persons_per_frame=[0, 0, 0],
                excluded_instructor=0,
                monitors={
                    "total": 0,
                    "on": 0
                },
                seats=None,
                slots=[],
                tamper={
                    "type": (
                        tamper_result["reasons"][0]
                        if tamper_result["reasons"]
                        else "unknown"
                    ),
                    "score": tamper_result[
                        "matchRatio"
                    ]
                },
                evidence=None,
                kind="tamper"
            )

            self.outbox.add(
                payload
            )

            return payload

        # --------------------------------------------------
        # 3. YOLO on all frames
        # --------------------------------------------------

        all_detections = []

        persons_per_frame = []

        excluded_instructors = []

        for index, frame in enumerate(frames):

            detections = (
                self.object_detector.detect(
                    frame
                )
            )

            all_detections.append(
                detections
            )

            person_result = (
                self.process_persons(
                    detections
                )
            )

            persons_per_frame.append(
                person_result["persons"]
            )

            excluded_instructors.append(
                person_result[
                    "excludedInstructor"
                ]
            )

            print(
                f"[YOLO] Frame {index + 1}: "
                f"persons={person_result['persons']} "
                f"excluded="
                f"{person_result['excludedInstructor']}"
            )

        # --------------------------------------------------
        # 4. Maximum person count
        # --------------------------------------------------

        persons_sample = max(
            persons_per_frame
        )

        # Spec says evidence frame is the frame
        # with maximum number of people.
        evidence_index = (
            persons_per_frame.index(
                persons_sample
            )
        )

        evidence_frame = frames[
            evidence_index
        ]

        evidence_detections = all_detections[
            evidence_index
        ]

        excluded_instructor = (
            excluded_instructors[
                evidence_index
            ]
        )

        print(
            f"[RESULT] personsPerFrame="
            f"{persons_per_frame}"
        )

        print(
            f"[RESULT] persons="
            f"{persons_sample}"
        )

        print(
            f"[RESULT] evidenceFrame="
            f"{evidence_index + 1}"
        )

        # --------------------------------------------------
        # 5. Evidence + privacy
        # --------------------------------------------------

        evidence, faces = (
            self.create_evidence(
                evidence_frame,
                evidence_detections
            )
        )

        print(
            f"[PRIVACY] Faces detected="
            f"{len(faces)}"
        )

        # --------------------------------------------------
        # 6. Build observation
        # --------------------------------------------------

        payload = self.packager.create_payload(
            persons=persons_sample,
            persons_per_frame=persons_per_frame,
            excluded_instructor=excluded_instructor,
            monitors={
                "total": 0,
                "on": 0
            },
            seats=None,
            slots=[],
            tamper=None,
            evidence=evidence,
            kind="sample"
        )

        # --------------------------------------------------
        # 7. SQLite outbox
        # --------------------------------------------------

        self.outbox.add(
            payload
        )

        print(
            f"[OUTBOX] Observation stored: "
            f"{payload['obsId']}"
        )

        print(
            f"[OUTBOX] Queue length: "
            f"{self.outbox.count()}"
        )

        print("=" * 60)

        return payload