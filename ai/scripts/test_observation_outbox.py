
import sys
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "ai" / "src"

sys.path.insert(0, str(SRC_DIR))

from output.package import ObservationPackager
from storage.outbox import ObservationOutbox


def main():
    print("=" * 55)
    print("OBSERVATION + SQLITE OUTBOX INTEGRATION TEST")
    print("=" * 55)

    packager = ObservationPackager(
        device_id="D-101",
        edge_version="0.1.0"
    )

    # Synthetic image for testing only.
    # Replace with privacy-blurred camera evidence in production.
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    observation = packager.create_sample_observation(
        persons=2,
        persons_per_frame=[1, 2, 2],
        excluded_instructor=1,
        evidence_frame=test_frame,
        monitors={"total": 4, "on": 3},
        seats=None,
        slots=[
            {
                "id": "machine-slot",
                "present": True,
                "occupied": False
            }
        ],
        tamper=None
    )

    # Use a unique ID so the test does not collide with real records.
    observation["obsId"] = str(uuid.uuid4())

    outbox = ObservationOutbox()
    before = outbox.count()

    outbox.add(observation)

    after = outbox.count()
    pending = outbox.get_oldest(limit=20)

    matching = next(
        (
            item for item in pending
            if item["obsId"] == observation["obsId"]
        ),
        None
    )

    print(f"Observation ID: {observation['obsId']}")
    print(f"Queue before: {before}")
    print(f"Queue after:  {after}")

    if matching is None:
        print("[FAIL] Observation was not found in the pending queue.")
        return

    print("[PASS] Observation stored in SQLite.")
    print("[PASS] Pending observation retrieved.")

    # Verify the stored JSON can be decoded.
    import json
    stored_payload = json.loads(matching["body"])

    assert stored_payload["persons"] == 2
    assert stored_payload["personsPerFrame"] == [1, 2, 2]
    assert stored_payload["deviceId"] == "D-101"
    assert stored_payload["evidence"]

    print("[PASS] JSON payload verified.")
    print("[PASS] Evidence field is present.")

    # Remove only this synthetic test record.
    outbox.delete(observation["obsId"])

    print(f"Queue after test cleanup: {outbox.count()}")
    print("[PASS] Synthetic test record removed.")
    print("=" * 55)


if __name__ == "__main__":
    main()
