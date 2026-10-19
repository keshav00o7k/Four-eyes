import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.src.storage.outbox import ObservationOutbox


def main():

    print("\n")
    print("=" * 60)
    print("RETRY / BACKOFF TEST")
    print("=" * 60)

    outbox = ObservationOutbox()

    outbox.clear()

    payload = {
        "obsId": "retry-test-001",
        "deviceId": "D-101",
        "kind": "sample",
        "persons": 5
    }

    outbox.add(payload)

    print("\n[OK] Test observation added.")

    # --------------------------------------------------
    # Attempt 1
    # --------------------------------------------------

    outbox.mark_attempt(
        "retry-test-001",
        "Backend unavailable"
    )

    record = outbox.get_oldest(limit=1)

    # It should NOT be immediately available because
    # retry is scheduled 10 seconds later.

    print("\n[ATTEMPT 1]")
    print("-" * 60)

    print(f"Attempts: {record[0]['attempts'] if record else 1}")

    assert record == []

    print("[OK] Retry delayed after attempt 1.")

    # --------------------------------------------------
    # Inspect database directly to verify metadata.
    # --------------------------------------------------

    import sqlite3

    with sqlite3.connect(outbox.db_path) as connection:

        row = connection.execute(
            """
            SELECT attempts, lastError, nextRetryAt
            FROM outbox
            WHERE obsId = ?
            """,
            ("retry-test-001",)
        ).fetchone()

    attempts, last_error, next_retry = row

    print("\nRETRY METADATA")
    print("-" * 60)

    print(f"Attempts   : {attempts}")
    print(f"Last error : {last_error}")
    print(f"Next retry : {next_retry}")

    assert attempts == 1
    assert last_error == "Backend unavailable"
    assert next_retry is not None

    print("[OK] Retry metadata recorded.")

    # --------------------------------------------------
    # Simulate retry time passing.
    # --------------------------------------------------

    with sqlite3.connect(outbox.db_path) as connection:

        old_time = (
            datetime.now(timezone.utc)
            - timedelta(seconds=1)
        ).isoformat()

        connection.execute(
            """
            UPDATE outbox
            SET nextRetryAt = ?
            WHERE obsId = ?
            """,
            (
                old_time,
                "retry-test-001"
            )
        )

        connection.commit()

    pending = outbox.get_oldest(limit=1)

    print("\nSIMULATED RETRY")
    print("-" * 60)

    assert len(pending) == 1

    print(
        "[OK] Observation became eligible for retry."
    )

    # --------------------------------------------------
    # Test 7-day cleanup
    # --------------------------------------------------

    stale_payload = {
        "obsId": "stale-test-001",
        "deviceId": "D-101",
        "kind": "sample",
        "persons": 3
    }

    outbox.add(stale_payload)

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(days=8)
    ).isoformat()

    with sqlite3.connect(outbox.db_path) as connection:

        connection.execute(
            """
            UPDATE outbox
            SET createdAt = ?
            WHERE obsId = ?
            """,
            (
                stale_time,
                "stale-test-001"
            )
        )

        connection.commit()

    deleted = outbox.drop_older_than(days=7)

    print("\n7-DAY CLEANUP")
    print("-" * 60)

    print(f"Deleted: {deleted}")

    assert deleted >= 1

    print(
        "[OK] Observations older than 7 days removed."
    )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    outbox.clear()

    print("\n")
    print("=" * 60)
    print("[SUCCESS] RETRY / BACKOFF TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()