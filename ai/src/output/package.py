import base64
import json
import uuid
from datetime import datetime, timezone

import cv2


class ObservationPackager:
    """
    Builds the edge observation payload.

    The payload follows the project architecture:
    - obsId
    - deviceId
    - timestamp
    - sample counts
    - monitor information
    - seating information
    - machine slots
    - tamper information
    - privacy-safe evidence image
    - edge version
    """

    def __init__(
        self,
        device_id="D-101",
        edge_version="0.1.0",
        evidence_width=960,
        webp_quality=60
    ):
        self.device_id = device_id
        self.edge_version = edge_version
        self.evidence_width = evidence_width
        self.webp_quality = webp_quality

    # ========================================================
    # RESIZE EVIDENCE
    # ========================================================

    def _resize_evidence(self, frame):

        height, width = frame.shape[:2]

        if width <= self.evidence_width:
            return frame

        scale = self.evidence_width / float(width)

        new_width = self.evidence_width
        new_height = int(height * scale)

        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # ========================================================
    # ENCODE WEBP
    # ========================================================

    def encode_evidence(self, frame):

        frame = self._resize_evidence(frame)

        success, encoded = cv2.imencode(
            ".webp",
            frame,
            [
                cv2.IMWRITE_WEBP_QUALITY,
                self.webp_quality
            ]
        )

        if not success:
            raise RuntimeError(
                "Failed to encode evidence as WebP."
            )

        evidence_base64 = base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8")

        return evidence_base64

    # ========================================================
    # CREATE OBSERVATION
    # ========================================================

    def create_sample_observation(
        self,
        persons,
        persons_per_frame,
        excluded_instructor,
        evidence_frame,
        monitors=None,
        seats=None,
        slots=None,
        tamper=None
    ):

        if monitors is None:
            monitors = {
                "total": 0,
                "on": 0
            }

        if slots is None:
            slots = []

        # ----------------------------------------------------
        # Encode privacy-safe evidence
        # ----------------------------------------------------

        evidence = self.encode_evidence(
            evidence_frame
        )

        # ----------------------------------------------------
        # UTC timestamp
        # ----------------------------------------------------

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        # ----------------------------------------------------
        # Observation ID
        # ----------------------------------------------------

        obs_id = str(
            uuid.uuid4()
        )

        # ----------------------------------------------------
        # Payload
        # ----------------------------------------------------

        payload = {
            "obsId": obs_id,
            "deviceId": self.device_id,
            "ts": timestamp,
            "kind": "sample",
            "persons": int(persons),
            "personsPerFrame": [
                int(x)
                for x in persons_per_frame
            ],
            "excludedInstructor": int(
                excluded_instructor
            ),
            "monitors": monitors,
            "seats": seats,
            "slots": slots,
            "tamper": tamper,
            "evidence": evidence,
            "edgeVersion": self.edge_version
        }

        return payload

    # ========================================================
    # JSON STRING
    # ========================================================

    def to_json(self, payload):

        return json.dumps(
            payload,
            separators=(",", ":")
        )