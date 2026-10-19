import cv2
import numpy as np


class TamperDetector:
    """
    Final tamper detector for the AI Edge Auditor.

    Checks:
        1. Dark camera
        2. Covered / blank camera
        3. Camera moved / redirected
        4. Two-consecutive-sample debounce
    """

    def __init__(
        self,
        target_width=320,
        dark_threshold=25.0,
        blur_threshold=15.0,
        std_threshold=12.0,
        moved_threshold=0.25,
        required_consecutive=2
    ):
        self.target_width = target_width

        self.dark_threshold = dark_threshold
        self.blur_threshold = blur_threshold
        self.std_threshold = std_threshold
        self.moved_threshold = moved_threshold

        self.required_consecutive = required_consecutive
        self.consecutive_tampered = 0

        # Registration reference
        self.reference_edges = None
        self.reference_keypoints = None
        self.reference_descriptors = None

        # ORB
        self.orb = cv2.ORB_create(
            nfeatures=1000
        )

        # BFMatcher for ORB binary descriptors
        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False
        )

    # ---------------------------------------------------------
    # PREPROCESS
    # ---------------------------------------------------------

    def _resize(self, frame):
        """
        Resize frame to 320px width while maintaining aspect ratio.
        """

        height, width = frame.shape[:2]

        if width == self.target_width:
            return frame.copy()

        scale = self.target_width / float(width)

        new_width = self.target_width
        new_height = int(height * scale)

        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------
    # REFERENCE
    # ---------------------------------------------------------

    def register_reference(self, frame):
        """
        Capture the camera's registration reference.

        This should normally be called when the device/camera
        is registered.

        Reference is kept as processed image data.
        """

        small = self._resize(frame)

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        keypoints, descriptors = self.orb.detectAndCompute(
            edges,
            None
        )

        self.reference_edges = edges
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors

        descriptor_count = 0

        if descriptors is not None:
            descriptor_count = len(descriptors)

        return {
            "registered": True,
            "width": small.shape[1],
            "height": small.shape[0],
            "keypoints": descriptor_count
        }

    # ---------------------------------------------------------
    # DARK CHECK
    # ---------------------------------------------------------

    def _check_dark(self, gray):
        """
        Dark camera:
            gray mean < 25
        """

        mean_brightness = float(np.mean(gray))

        is_dark = mean_brightness < self.dark_threshold

        return is_dark, mean_brightness

    # ---------------------------------------------------------
    # COVERED CHECK
    # ---------------------------------------------------------

    def _check_covered(self, gray):
        """
        Covered / blank camera:

            Laplacian variance < 15
            AND
            gray standard deviation < 12
        """

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        blur_score = float(
            laplacian.var()
        )

        gray_std = float(
            np.std(gray)
        )

        is_covered = (
            blur_score < self.blur_threshold
            and gray_std < self.std_threshold
        )

        return is_covered, blur_score, gray_std

    # ---------------------------------------------------------
    # MOVED / REDIRECTED CHECK
    # ---------------------------------------------------------

    def _check_moved(self, edges):
        """
        Compare current camera view with registration reference.

        Uses:
            Canny edges
            ORB features
            Hamming matching
            Lowe ratio test

        Good-match ratio < 0.25
            => camera moved / redirected
        """

        if self.reference_descriptors is None:
            return False, 1.0, 0

        keypoints, descriptors = self.orb.detectAndCompute(
            edges,
            None
        )

        if descriptors is None or len(descriptors) == 0:
            return True, 0.0, 0

        if len(self.reference_descriptors) < 2:
            return False, 1.0, 0

        matches = self.matcher.knnMatch(
            self.reference_descriptors,
            descriptors,
            k=2
        )

        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        reference_count = len(
            self.reference_descriptors
        )

        good_match_ratio = (
            len(good_matches) / reference_count
        )

        moved = (
            good_match_ratio < self.moved_threshold
        )

        return (
            moved,
            good_match_ratio,
            len(good_matches)
        )

    # ---------------------------------------------------------
    # MAIN CHECK
    # ---------------------------------------------------------

    def check(self, frame):
        """
        Analyze one sample.

        Returns detailed diagnostics plus
        debounced final tamper status.
        """

        small = self._resize(frame)

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        # -------------------------
        # Dark
        # -------------------------

        dark, brightness = self._check_dark(
            gray
        )

        # -------------------------
        # Covered
        # -------------------------

        covered, blur_score, gray_std = (
            self._check_covered(gray)
        )

        # -------------------------
        # Moved
        # -------------------------

        moved, match_ratio, good_matches = (
            self._check_moved(edges)
        )

        # -------------------------
        # Raw tamper result
        # -------------------------

        raw_tampered = (
            dark
            or covered
            or moved
        )

        reasons = []

        if dark:
            reasons.append(
                "camera_dark"
            )

        if covered:
            reasons.append(
                "camera_covered_or_blank"
            )

        if moved:
            reasons.append(
                "camera_moved_or_redirected"
            )

        # -------------------------
        # Debounce
        # -------------------------

        if raw_tampered:
            self.consecutive_tampered += 1
        else:
            self.consecutive_tampered = 0

        tampered = (
            self.consecutive_tampered
            >= self.required_consecutive
        )

        return {
            "tampered": tampered,
            "rawTampered": raw_tampered,

            "reasons": reasons,

            "brightness": round(
                brightness,
                2
            ),

            "blurScore": round(
                blur_score,
                2
            ),

            "grayStd": round(
                gray_std,
                2
            ),

            "matchRatio": round(
                match_ratio,
                3
            ),

            "goodMatches": good_matches,

            "consecutiveTampered": (
                self.consecutive_tampered
            )
        }

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self):
        """
        Reset debounce state.
        """

        self.consecutive_tampered = 0