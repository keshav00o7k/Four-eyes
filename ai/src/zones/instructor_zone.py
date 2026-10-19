import cv2
import numpy as np


class PolygonZone:
    """
    Polygon-based zone.

    Supported zone types:
        instructor
        ignore
        slot:<itemId>
    """

    def __init__(self, name, points):
        self.name = name
        self.points = [(int(x), int(y)) for x, y in points]

    def contains_point(self, x, y):
        polygon = np.array(self.points, dtype=np.int32)

        result = cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        )

        return result >= 0

    def contains_person(self, bbox):
        """
        bbox = (x1, y1, x2, y2)

        Uses bottom-center of the person bounding box
        as the foot point.
        """
        x1, y1, x2, y2 = bbox

        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)

        return self.contains_point(foot_x, foot_y)

    def draw(self, frame):
        polygon = np.array(self.points, dtype=np.int32)

        cv2.polylines(
            frame,
            [polygon],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2
        )

        label_x, label_y = self.points[0]

        cv2.putText(
            frame,
            self.name,
            (label_x, max(25, label_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        return frame