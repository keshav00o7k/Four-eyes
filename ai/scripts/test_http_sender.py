import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.src.network.http_sender import HTTPSender


def main():

    print("\nHTTP SENDER TEST")
    print("=" * 60)

    sender = HTTPSender(
        backend_url="http://127.0.0.1:8000/observations",
        device_id="D-101",
        device_secret="test-device-secret-123"
    )

    payload = {
        "obsId": "http-test-001",
        "deviceId": "D-101",
        "ts": "2026-10-15T18:30:00+00:00",
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
    }

    print("\nSending observation...")
    print(f"obsId: {payload['obsId']}")

    result = sender.send_observation(payload)

    print("\nRESULT")
    print("-" * 60)

    print(f"Success : {result['success']}")
    print(f"Status  : {result['status']}")
    print(f"Error   : {result['error']}")

    if result.get("response"):
        print(f"Response: {result['response']}")

    print("\n" + "=" * 60)

    if result["success"]:
        print("[SUCCESS] HTTP SENDER TEST PASSED")
    else:
        print("[FAILED] HTTP sender did not receive success response.")

    print("=" * 60)


if __name__ == "__main__":
    main()