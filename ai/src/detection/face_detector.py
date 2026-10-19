import cv2


class FaceDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    ):
        print("[INFO] Loading YuNet face detection model...")

        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            confidence_threshold,
            nms_threshold,
            top_k,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU
        )

        self.current_size = None

        print("[OK] YuNet face detector loaded.")

    def _update_input_size(self, frame):
        height, width = frame.shape[:2]
        size = (width, height)

        if self.current_size != size:
            self.detector.setInputSize(size)
            self.current_size = size

    def detect(self, frame):
        """
        Detect faces in a frame.

        Returns:
            list of dictionaries containing:
            x, y, w, h, confidence
        """

        self._update_input_size(frame)

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        detections = []

        for face in faces:
            x, y, w, h = face[:4]
            confidence = float(face[14])

            detections.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "confidence": confidence
            })

        return detections