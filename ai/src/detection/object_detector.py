import cv2
import numpy as np
import onnxruntime as ort


class ObjectDetector:

    # COCO classes required by SIH26245 architecture
    COCO_CLASSES = {
        0: "person",
        13: "bench",
        56: "chair",
        60: "dining_table",
        62: "monitor",
        63: "laptop",
        66: "keyboard",
    }

    def __init__(
        self,
        model_path="ai/models/object_detection/yolov8n.onnx",
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.50,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        print("[INFO] Loading ONNX object detection model...")

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape

        print("[OK] Object detector loaded.")
        print(f"[INFO] Model: {self.model_path}")
        print(f"[INFO] Input: {self.input_name}")
        print(f"[INFO] Input shape: {input_shape}")
        print("[INFO] Provider: CPUExecutionProvider")
        print(f"[INFO] Confidence threshold: {self.confidence_threshold}")
        print(f"[INFO] NMS IoU threshold: {self.nms_threshold}")

    # ---------------------------------------------------------
    # LETTERBOX
    # ---------------------------------------------------------

    def letterbox(self, image):

        original_height, original_width = image.shape[:2]

        scale = min(
            self.input_size / original_width,
            self.input_size / original_height
        )

        new_width = int(round(original_width * scale))
        new_height = int(round(original_height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (
                self.input_size,
                self.input_size,
                3
            ),
            114,
            dtype=np.uint8
        )

        padding_x = (self.input_size - new_width) // 2
        padding_y = (self.input_size - new_height) // 2

        canvas[
            padding_y:padding_y + new_height,
            padding_x:padding_x + new_width
        ] = resized

        return (
            canvas,
            scale,
            padding_x,
            padding_y
        )

    # ---------------------------------------------------------
    # PREPROCESS
    # ---------------------------------------------------------

    def preprocess(self, frame):

        image, scale, pad_x, pad_y = self.letterbox(frame)

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # uint8 -> float32 and normalize
        image = image.astype(
            np.float32
        ) / 255.0

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # CHW -> NCHW
        image = np.expand_dims(
            image,
            axis=0
        )

        return (
            image,
            scale,
            pad_x,
            pad_y
        )

    # ---------------------------------------------------------
    # POSTPROCESS
    # ---------------------------------------------------------

    def postprocess(
        self,
        output,
        original_frame,
        scale,
        pad_x,
        pad_y
    ):
        """
        YOLOv8 output:
            [1, 84, 8400]

        After removing batch:
            [84, 8400]

        Transpose:
            [8400, 84]

        Each prediction:
            [cx, cy, width, height, class scores...]
        """

        predictions = output[0]

        # [84, 8400] -> [8400, 84]
        predictions = np.transpose(
            predictions,
            (1, 0)
        )

        boxes = []
        scores = []
        class_ids = []

        frame_height, frame_width = original_frame.shape[:2]

        for prediction in predictions:

            # Bounding box
            cx, cy, width, height = prediction[:4]

            # Class confidence scores
            class_scores = prediction[4:]

            if len(class_scores) == 0:
                continue

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

            # Confidence filtering
            if confidence < self.confidence_threshold:
                continue

            # -------------------------------------------------
            # Convert center format -> corner format
            # -------------------------------------------------

            x1 = cx - width / 2
            y1 = cy - height / 2

            x2 = cx + width / 2
            y2 = cy + height / 2

            # -------------------------------------------------
            # Remove letterbox padding
            # and map back to original image
            # -------------------------------------------------

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale

            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # -------------------------------------------------
            # Clip to image boundaries
            # -------------------------------------------------

            x1 = max(
                0,
                min(x1, frame_width - 1)
            )

            y1 = max(
                0,
                min(y1, frame_height - 1)
            )

            x2 = max(
                0,
                min(x2, frame_width - 1)
            )

            y2 = max(
                0,
                min(y2, frame_height - 1)
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            boxes.append(
                [
                    int(x1),
                    int(y1),
                    int(box_width),
                    int(box_height)
                ]
            )

            scores.append(confidence)
            class_ids.append(class_id)

        # No detections
        if not boxes:
            return []

        # -----------------------------------------------------
        # NMS
        # -----------------------------------------------------

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_threshold,
            self.nms_threshold
        )

        detections = []

        if len(indices) == 0:
            return detections

        indices = np.array(
            indices
        ).flatten()

        for index in indices:

            x, y, box_width, box_height = boxes[index]

            class_id = class_ids[index]

            detection = {
                "class_id": class_id,

                "class_name": self.get_class_name(
                    class_id
                ),

                "confidence": round(
                    float(scores[index]),
                    4
                ),

                "bbox": [
                    x,
                    y,
                    x + box_width,
                    y + box_height
                ]
            }

            detections.append(detection)

        return detections

    # ---------------------------------------------------------
    # CLASS NAME
    # ---------------------------------------------------------

    def get_class_name(self, class_id):

        return self.COCO_CLASSES.get(
            class_id,
            f"class_{class_id}"
        )

    # ---------------------------------------------------------
    # DETECT
    # ---------------------------------------------------------

    def detect(self, frame):

        if frame is None:
            raise ValueError(
                "Input frame is None."
            )

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Input frame must be a NumPy array."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Input frame must have shape HxWx3."
            )

        # Preprocess
        input_tensor, scale, pad_x, pad_y = (
            self.preprocess(frame)
        )

        # ONNX inference
        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor
            }
        )

        # Postprocess
        detections = self.postprocess(
            outputs[0],
            frame,
            scale,
            pad_x,
            pad_y
        )

        return detections

    # ---------------------------------------------------------
    # HELPER: FILTER CLASS
    # ---------------------------------------------------------

    def get_detections_by_class(
        self,
        detections,
        class_name
    ):

        return [
            detection
            for detection in detections
            if detection["class_name"] == class_name
        ]

    # ---------------------------------------------------------
    # HELPER: COUNT CLASS
    # ---------------------------------------------------------

    def count_class(
        self,
        detections,
        class_name
    ):

        return sum(
            1
            for detection in detections
            if detection["class_name"] == class_name
        )


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SIH26245 - OBJECT DETECTOR TEST")
    print("=" * 60)

    detector = ObjectDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Could not open webcam.")
        raise SystemExit(1)

    print("[INFO] Webcam opened.")
    print("[INFO] Press Q to quit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("[ERROR] Failed to read frame.")
                break

            detections = detector.detect(frame)

            # Draw detections
            for detection in detections:

                x1, y1, x2, y2 = detection["bbox"]

                class_name = detection["class_name"]
                confidence = detection["confidence"]

                cv2.rectangle(
                    frame,
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
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Counts
            person_count = detector.count_class(
                detections,
                "person"
            )

            monitor_count = detector.count_class(
                detections,
                "monitor"
            )

            laptop_count = detector.count_class(
                detections,
                "laptop"
            )

            chair_count = detector.count_class(
                detections,
                "chair"
            )

            cv2.putText(
                frame,
                f"Persons: {person_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Monitors: {monitor_count}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Laptops: {laptop_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Chairs: {chair_count}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "SIH26245 - Object Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("[INFO] Object detector test finished.")