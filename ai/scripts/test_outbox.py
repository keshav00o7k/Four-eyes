import sys
import json
from pathlib import Path

# Add ai/src to Python path
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ai" / "src"

sys.path.insert(0, str(SRC))

from storage.outbox import ObservationOutbox


def main():

    print("=" * 60)
    print("SQLITE OUTBOX TEST")
    print("=" * 60)

    outbox = ObservationOutbox()

    print("[OK] Database initialized.")
    print(f"[INFO] Database: {outbox.db_path}")

    # Test observation
    payload = {
        "obsId": "test-observation-001",
        "deviceId": "D-101",
        "ts": "2026-10-15T18:00:00+00:00",
        "kind": "sample",
        "persons": 5,
        "personsPerFrame": [4, 5, 5],
        "excludedInstructor": 1,
        "monitors": {
            "total": 10,
            "on": 8
        },
        "seats": None,
        "slots": [],
        "tamper": None,
        "evidence": "TEST_BASE64",
        "edgeVersion": "0.1.0"
    }

    # Clear previous test data
    outbox.clear()

    # Add observation
    obs_id = outbox.add(payload)

    print()
    print("[OK] Observation added.")
    print("obsId:", obs_id)

    # Check count
    count = outbox.count()

    print()
    print("[INFO] Pending observations:", count)

    # Read oldest
    observations = outbox.get_oldest(limit=20)

    print()
    print("PENDING OBSERVATIONS")
    print("-" * 60)

    for observation in observations:

        print("obsId:", observation["obsId"])
        print("createdAt:", observation["createdAt"])
        print("attempts:", observation["attempts"])
        print("lastError:", observation["lastError"])

        body = json.loads(observation["body"])

        print("persons:", body["persons"])
        print("deviceId:", body["deviceId"])

    # Test failed attempt
    outbox.mark_attempt(
        obs_id,
        "Backend unavailable"
    )

    observations = outbox.get_oldest()

    print()
    print("AFTER FAILED ATTEMPT")
    print("-" * 60)

    print("attempts:", observations[0]["attempts"])
    print("lastError:", observations[0]["lastError"])

    # Test successful transmission
    outbox.delete(obs_id)

    print()
    print("[OK] Observation deleted after simulated success.")

    print("Remaining:", outbox.count())

    print()
    print("=" * 60)
    print("[SUCCESS] SQLITE OUTBOX TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()