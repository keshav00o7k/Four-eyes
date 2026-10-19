import cv2


class FaceBlurrer:
    def __init__(self, margin_ratio=0.20):
        self.margin_ratio = margin_ratio

    def blur_faces(self, frame, faces):
        """
        Blur all detected faces.

        Raw faces are not saved.
        """

        height, width = frame.shape[:2]

        for face in faces:

            x = face["x"]
            y = face["y"]
            w = face["w"]
            h = face["h"]

            # Expand bounding box slightly
            margin_x = int(w * self.margin_ratio)
            margin_y = int(h * self.margin_ratio)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)

            x2 = min(width, x + w + margin_x)
            y2 = min(height, y + h + margin_y)

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Strong Gaussian blur
            blurred = cv2.GaussianBlur(
                roi,
                (0, 0),
                sigmaX=25,
                sigmaY=25
            )

            frame[y1:y2, x1:x2] = blurred

        return frame