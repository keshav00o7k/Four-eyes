import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path


class ObservationOutbox:

    RETRY_DELAYS = [
        10,       # attempt 1 -> 10 seconds
        30,       # attempt 2 -> 30 seconds
        120,      # attempt 3 -> 2 minutes
        600       # attempt 4+ -> 10 minutes
    ]

    def __init__(self, db_path=None):

        if db_path is None:
            project_root = Path(__file__).resolve().parents[2]
            db_path = project_root / "data" / "edge_outbox.db"

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _initialize(self):

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    obsId TEXT PRIMARY KEY,
                    body TEXT NOT NULL,
                    createdAt TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lastError TEXT,
                    nextRetryAt TEXT
                )
                """
            )

            # Upgrade an older database created before
            # nextRetryAt existed.
            columns = connection.execute(
                "PRAGMA table_info(outbox)"
            ).fetchall()

            column_names = {
                column[1]
                for column in columns
            }

            if "nextRetryAt" not in column_names:
                connection.execute(
                    "ALTER TABLE outbox ADD COLUMN nextRetryAt TEXT"
                )

            connection.commit()

    def add(self, payload):

        obs_id = payload["obsId"]

        body = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False
        )

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        with self._connect() as connection:

            connection.execute(
                """
                INSERT OR IGNORE INTO outbox
                (
                    obsId,
                    body,
                    createdAt,
                    attempts,
                    lastError,
                    nextRetryAt
                )
                VALUES (?, ?, ?, 0, NULL, NULL)
                """,
                (
                    obs_id,
                    body,
                    created_at
                )
            )

            connection.commit()

    def get_oldest(self, limit=20):

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT
                    obsId,
                    body,
                    createdAt,
                    attempts,
                    lastError,
                    nextRetryAt
                FROM outbox
                WHERE
                    nextRetryAt IS NULL
                    OR nextRetryAt <= ?
                ORDER BY createdAt ASC
                LIMIT ?
                """,
                (now, limit)
            ).fetchall()

        observations = []

        for row in rows:

            observations.append({
                "obsId": row[0],
                "body": row[1],
                "createdAt": row[2],
                "attempts": row[3],
                "lastError": row[4],
                "nextRetryAt": row[5]
            })

        return observations

    def mark_attempt(self, obs_id, error=None):

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT attempts
                FROM outbox
                WHERE obsId = ?
                """,
                (obs_id,)
            ).fetchone()

            if row is None:
                return

            attempts = row[0] + 1

            # attempts:
            # 1 -> 10 sec
            # 2 -> 30 sec
            # 3 -> 120 sec
            # 4+ -> 600 sec

            index = min(
                attempts - 1,
                len(self.RETRY_DELAYS) - 1
            )

            delay_seconds = self.RETRY_DELAYS[index]

            next_retry = (
                datetime.now(timezone.utc)
                + timedelta(seconds=delay_seconds)
            ).isoformat()

            connection.execute(
                """
                UPDATE outbox
                SET
                    attempts = ?,
                    lastError = ?,
                    nextRetryAt = ?
                WHERE obsId = ?
                """,
                (
                    attempts,
                    error,
                    next_retry,
                    obs_id
                )
            )

            connection.commit()

    def delete(self, obs_id):

        with self._connect() as connection:

            connection.execute(
                """
                DELETE FROM outbox
                WHERE obsId = ?
                """,
                (obs_id,)
            )

            connection.commit()

    def count(self):

        with self._connect() as connection:

            row = connection.execute(
                "SELECT COUNT(*) FROM outbox"
            ).fetchone()

        return row[0]

    def clear(self):

        with self._connect() as connection:

            connection.execute(
                "DELETE FROM outbox"
            )

            connection.commit()

    def drop_older_than(self, days=7):

        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=days)
        ).isoformat()

        with self._connect() as connection:

            cursor = connection.execute(
                """
                DELETE FROM outbox
                WHERE createdAt < ?
                """,
                (cutoff,)
            )

            deleted = cursor.rowcount

            connection.commit()

        return deleted