
import cv2
import numpy as np


class OperabilityAnalyzer:
    def __init__(
        self,
        screen_brightness_threshold=60,
        screen_std_threshold=20,
        machine_similarity_threshold=0.35,
        missing_samples_required=3,
    ):
        self.screen_brightness_threshold = screen_brightness_threshold
        self.screen_std_threshold = screen_std_threshold
        self.machine_similarity_threshold = machine_similarity_threshold
        self.missing_samples_required = missing_samples_required
        self._slot_similarity_state = {}

    # ---------------------------------------------------------
    # SAFE BOUNDING-BOX CROP
    # ---------------------------------------------------------

    @staticmethod
    def _crop_bbox(frame, bbox):
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    # ---------------------------------------------------------
    # MONITOR / LAPTOP SCREEN ON-OFF
    # ---------------------------------------------------------

    def analyze_monitor(self, frame, bbox):
        crop = self._crop_bbox(frame, bbox)

        if crop is None:
            return {
                "on": False,
                "vMean": 0.0,
                "vStd": 0.0,
                "valid": False,
            }

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]

        v_mean = float(np.mean(value))
        v_std = float(np.std(value))

        return {
            "on": bool(
                v_mean > self.screen_brightness_threshold
                and v_std > self.screen_std_threshold
            ),
            "vMean": round(v_mean, 2),
            "vStd": round(v_std, 2),
            "valid": True,
        }

    def is_monitor_on(self, frame, bbox):
        return self.analyze_monitor(frame, bbox)["on"]

    def analyze_monitors(self, frame, detections):
        total = 0
        on_count = 0
        details = []

        for detection in detections:
            name = detection.get(
                "class_name",
                detection.get("class", ""),
            )

            if str(name).lower() not in {
                "monitor", "tv", "laptop"
            }:
                continue

            bbox = detection.get("bbox", detection.get("box"))
            if bbox is None:
                continue

            result = self.analyze_monitor(frame, bbox)
            if not result["valid"]:
                continue

            total += 1
            on_count += int(result["on"])

            details.append({
                "class": str(name).lower(),
                "bbox": list(map(int, bbox)),
                **result,
            })

        return {
            "total": total,
            "on": on_count,
            "details": details,
        }

    # ---------------------------------------------------------
    # PERSON OCCUPANCY INSIDE A SLOT POLYGON
    # ---------------------------------------------------------

    @staticmethod
    def is_slot_occupied(person_detections, slot_zone):
        """
        slot_zone must expose contains_point(x, y).
        Person boxes must use XYXY coordinates.
        """
        for person in person_detections:
            bbox = person.get("bbox", person.get("box"))

            if bbox is None or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(float, bbox)
            foot_x = (x1 + x2) / 2.0
            foot_y = y2

            if slot_zone.contains_point(foot_x, foot_y):
                return True

        return False

    # ---------------------------------------------------------
    # MACHINE EDGE SIMILARITY
    # ---------------------------------------------------------

    @staticmethod
    def calculate_edge_similarity(current_crop, reference_crop):
        """
        Compare current slot with the registered reference.

        Returns edge-map IoU in the range 0.0 to 1.0:
          closer to 1 = more similar
          closer to 0 = more different

        Small dilation tolerates minor edge shifts.
        Keep the camera fixed and use the same slot region.
        """
        if current_crop is None or reference_crop is None:
            raise ValueError("Both machine crops are required.")

        if current_crop.size == 0 or reference_crop.size == 0:
            raise ValueError("Machine crops cannot be empty.")

        def to_gray(image):
            if image.ndim == 2:
                return image
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_gray = to_gray(current_crop)
        reference_gray = to_gray(reference_crop)

        reference_gray = cv2.resize(
            reference_gray,
            (current_gray.shape[1], current_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        # Slight blur reduces sensitivity to sensor noise.
        current_gray = cv2.GaussianBlur(
            current_gray, (3, 3), 0
        )
        reference_gray = cv2.GaussianBlur(
            reference_gray, (3, 3), 0
        )

        current_edges = cv2.Canny(current_gray, 50, 150)
        reference_edges = cv2.Canny(reference_gray, 50, 150)

        # Tolerate very small edge shifts.
        kernel = np.ones((3, 3), dtype=np.uint8)
        current_edges = cv2.dilate(
            current_edges, kernel, iterations=1
        )
        reference_edges = cv2.dilate(
            reference_edges, kernel, iterations=1
        )

        current_mask = current_edges > 0
        reference_mask = reference_edges > 0

        intersection = int(
            np.count_nonzero(current_mask & reference_mask)
        )
        union = int(
            np.count_nonzero(current_mask | reference_mask)
        )

        if union == 0:
            # Both edge maps are blank. Distinguish similar
            # blank regions from substantially different crops.
            pixel_difference = float(
                np.mean(
                    cv2.absdiff(current_gray, reference_gray)
                ) / 255.0
            )
            return float(np.clip(1.0 - pixel_difference, 0.0, 1.0))

        return float(intersection / union)

    # ---------------------------------------------------------
    # THREE-SAMPLE MACHINE MISSING CONFIRMATION
    # ---------------------------------------------------------

    def update_machine_presence(
        self,
        slot_id,
        similarity_score,
        threshold=None,
    ):
        if threshold is None:
            threshold = self.machine_similarity_threshold

        score = float(similarity_score)

        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                "similarity_score must be between 0.0 and 1.0."
            )

        state = self._slot_similarity_state.setdefault(
            slot_id,
            {
                "missing_streak": 0,
                "present": True,
            },
        )

        if score < threshold:
            state["missing_streak"] += 1

            if state["missing_streak"] >= self.missing_samples_required:
                state["present"] = False
        else:
            state["missing_streak"] = 0
            state["present"] = True

        return {
            "id": slot_id,
            "present": bool(state["present"]),
            "similarity": round(score, 3),
            "missingStreak": state["missing_streak"],
        }

    def analyze_machine_slot(
        self,
        slot_id,
        current_crop,
        reference_crop,
    ):
        score = self.calculate_edge_similarity(
            current_crop,
            reference_crop,
        )

        return self.update_machine_presence(
            slot_id,
            score,
        )

    def reset_machine_slot(self, slot_id):
        """Reset state after re-registering a machine slot."""
        self._slot_similarity_state.pop(slot_id, None)
