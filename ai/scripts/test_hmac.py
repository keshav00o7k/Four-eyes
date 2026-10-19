import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.src.security.hmac_signer import HMACSigner


def main():
    print("\nHMAC AUTHENTICATION TEST")
    print("=" * 60)

    # Test-only secret. Never use this in production.
    secret = "test-device-secret-123"

    device_id = "D-101"
    timestamp = "2026-10-15T18:30:00+00:00"

    body = '{"obsId":"test-001","deviceId":"D-101","persons":5}'

    signer = HMACSigner(secret)

    # --------------------------------------------------
    # TEST 1: Generate signature
    # --------------------------------------------------

    headers = signer.sign(
        body=body,
        device_id=device_id,
        timestamp=timestamp
    )

    print("\n[TEST 1] Generated Headers")
    print("-" * 60)

    for key, value in headers.items():
        print(f"{key}: {value}")

    assert "X-Device-Id" in headers
    assert "X-Timestamp" in headers
    assert "X-Signature" in headers

    assert len(headers["X-Signature"]) == 64

    print("[OK] Signature generated.")
    print("[OK] Signature length = 64 hexadecimal characters.")

    # --------------------------------------------------
    # TEST 2: Same input -> same signature
    # --------------------------------------------------

    headers_again = signer.sign(
        body=body,
        device_id=device_id,
        timestamp=timestamp
    )

    print("\n[TEST 2] Deterministic Signature")
    print("-" * 60)

    assert headers["X-Signature"] == headers_again["X-Signature"]

    print("[OK] Same input produced same signature.")

    # --------------------------------------------------
    # TEST 3: Body modification -> signature changes
    # --------------------------------------------------

    modified_body = '{"obsId":"test-001","deviceId":"D-101","persons":6}'

    modified_headers = signer.sign(
        body=modified_body,
        device_id=device_id,
        timestamp=timestamp
    )

    print("\n[TEST 3] Body Tampering Detection")
    print("-" * 60)

    assert headers["X-Signature"] != modified_headers["X-Signature"]

    print("[OK] Modified body produced different signature.")

    # --------------------------------------------------
    # TEST 4: Wrong secret -> signature changes
    # --------------------------------------------------

    wrong_signer = HMACSigner("wrong-secret")

    wrong_headers = wrong_signer.sign(
        body=body,
        device_id=device_id,
        timestamp=timestamp
    )

    print("\n[TEST 4] Wrong Secret Detection")
    print("-" * 60)

    assert headers["X-Signature"] != wrong_headers["X-Signature"]

    print("[OK] Wrong secret produced different signature.")

    # --------------------------------------------------
    # TEST 5: Timestamp modification -> signature changes
    # --------------------------------------------------

    modified_timestamp = "2026-10-15T18:31:00+00:00"

    timestamp_headers = signer.sign(
        body=body,
        device_id=device_id,
        timestamp=modified_timestamp
    )

    print("\n[TEST 5] Timestamp Tampering Detection")
    print("-" * 60)

    assert headers["X-Signature"] != timestamp_headers["X-Signature"]

    print("[OK] Modified timestamp produced different signature.")

    print("\n" + "=" * 60)
    print("[SUCCESS] HMAC AUTHENTICATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()