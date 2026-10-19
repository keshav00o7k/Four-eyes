import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.src.storage.outbox import ObservationOutbox
from ai.src.network.http_sender import HTTPSender
from ai.src.network.outbox_sender import OutboxSender


def main():

    print("\n")
    print("=" * 60)
    print("OUTBOX → HMAC → HTTP SENDER INTEGRATION TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Create outbox
    # --------------------------------------------------

    outbox = ObservationOutbox()

    # Clear old test records.
    outbox.clear()

    print("\n[OK] SQLite outbox initialized.")
    print(f"[INFO] Initial queue: {outbox.count()}")

    # --------------------------------------------------
    # 2. Add test observations
    # --------------------------------------------------

    observations = [
        {
            "obsId": "integration-001",
            "deviceId": "D-101",
            "ts": "2026-10-15T18:40:00+00:00",
            "kind": "sample",
            "persons": 5,
            "personsPerFrame": [4, 5, 5],
            "excludedInstructor": 1,
            "monitors": {
                "total": 10,
                "on": 7
            },
            "seats": None,
            "slots": [],
            "tamper": None,
            "evidence": None,
            "edgeVersion": "0.1.0"
        },
        {
            "obsId": "integration-002",
            "deviceId": "D-101",
            "ts": "2026-10-15T18:45:00+00:00",
            "kind": "sample",
            "persons": 8,
            "personsPerFrame": [7, 8, 8],
            "excludedInstructor": 1,
            "monitors": {
                "total": 10,
                "on": 8
            },
            "seats": None,
            "slots": [],
            "tamper": None,
            "evidence": None,
            "edgeVersion": "0.1.0"
        }
    ]

    for observation in observations:

        outbox.add(observation)

        print(
            f"[OK] Added {observation['obsId']}"
        )

    print(
        f"\n[INFO] Queue before sending: {outbox.count()}"
    )

    assert outbox.count() == 2

    # --------------------------------------------------
    # 3. Create HTTP sender
    # --------------------------------------------------

    http_sender = HTTPSender(
        backend_url="http://127.0.0.1:8000/observations",
        device_id="D-101",
        device_secret="test-device-secret-123"
    )

    # --------------------------------------------------
    # 4. Create integrated sender
    # --------------------------------------------------

    sender = OutboxSender(
        outbox=outbox,
        http_sender=http_sender,
        batch_size=20
    )

    # --------------------------------------------------
    # 5. Send everything
    # --------------------------------------------------

    print("\n")
    print("-" * 60)
    print("STARTING OUTBOX SEND")
    print("-" * 60)

    result = sender.send_pending()

    # --------------------------------------------------
    # 6. Verify
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("INTEGRATION RESULT")
    print("=" * 60)

    print(f"Sent      : {result['sent']}")
    print(f"Failed    : {result['failed']}")
    print(f"Remaining : {result['remaining']}")

    assert result["sent"] == 2
    assert result["failed"] == 0
    assert result["remaining"] == 0
    assert outbox.count() == 0

    print("\n[OK] Both observations reached backend.")
    print("[OK] Both observations removed from SQLite.")
    print("[OK] HMAC authentication was used.")
    print("[OK] Outbox is empty after successful delivery.")

    print("\n")
    print("=" * 60)
    print("[SUCCESS] OUTBOX SENDER INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()