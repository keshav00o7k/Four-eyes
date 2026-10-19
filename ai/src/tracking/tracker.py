import math


class CentroidTracker:

    def __init__(self, max_distance=100, max_disappeared=10):

        self.next_object_id = 0

        self.objects = {}
        self.disappeared = {}

        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid):

        object_id = self.next_object_id

        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0

        self.next_object_id += 1

    def deregister(self, object_id):

        if object_id in self.objects:
            del self.objects[object_id]

        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, detections):

        # Only use person detections
        person_boxes = [
            detection["bbox"]
            for detection in detections
            if detection["class_name"] == "person"
        ]

        # No detections
        if len(person_boxes) == 0:

            for object_id in list(self.disappeared.keys()):

                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):
                    self.deregister(object_id)

            return self.objects.copy()

        input_centroids = []

        for box in person_boxes:

            x1, y1, x2, y2 = box

            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            input_centroids.append(
                (centroid_x, centroid_y)
            )

        # No currently tracked objects
        if len(self.objects) == 0:

            for centroid in input_centroids:
                self.register(centroid)

            return self.objects.copy()

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        used_objects = set()
        used_inputs = set()

        distances = []

        for object_id, object_centroid in zip(
            object_ids,
            object_centroids
        ):

            row = []

            for input_centroid in input_centroids:

                distance = math.sqrt(
                    (
                        object_centroid[0]
                        - input_centroid[0]
                    ) ** 2
                    +
                    (
                        object_centroid[1]
                        - input_centroid[1]
                    ) ** 2
                )

                row.append(distance)

            distances.append(row)

        # Find nearest matches
        candidates = []

        for object_index, object_id in enumerate(object_ids):

            for input_index in range(
                len(input_centroids)
            ):

                candidates.append(
                    (
                        distances[object_index][input_index],
                        object_index,
                        input_index
                    )
                )

        candidates.sort(
            key=lambda x: x[0]
        )

        for distance, object_index, input_index in candidates:

            if object_index in used_objects:
                continue

            if input_index in used_inputs:
                continue

            if distance > self.max_distance:
                continue

            object_id = object_ids[object_index]

            self.objects[object_id] = (
                input_centroids[input_index]
            )

            self.disappeared[object_id] = 0

            used_objects.add(object_index)
            used_inputs.add(input_index)

        # Handle unmatched tracked objects
        for object_index, object_id in enumerate(object_ids):

            if object_index not in used_objects:

                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):
                    self.deregister(object_id)

        # Register new detections
        for input_index, centroid in enumerate(
            input_centroids
        ):

            if input_index not in used_inputs:
                self.register(centroid)

        return self.objects.copy()