import json

from storage.outbox import ObservationOutbox
from network.http_sender import HTTPSender


class OutboxSender:

    def __init__(
        self,
        outbox,
        http_sender,
        batch_size=20
    ):
        self.outbox = outbox
        self.http_sender = http_sender
        self.batch_size = batch_size

    def send_pending(self):

        pending = self.outbox.get_oldest(
            limit=self.batch_size
        )

        if not pending:

            return {
                "sent": 0,
                "failed": 0,
                "remaining": self.outbox.count()
            }

        sent = 0
        failed = 0

        print(
            f"[INFO] Pending observations: {len(pending)}"
        )

        for observation in pending:

            obs_id = observation["obsId"]

            print(
                f"\n[SENDING] {obs_id}"
            )

            try:

                payload = json.loads(
                    observation["body"]
                )

            except json.JSONDecodeError as error:

                self.outbox.mark_attempt(
                    obs_id,
                    error=f"Invalid JSON: {error}"
                )

                failed += 1

                print(
                    f"[FAIL] Invalid JSON: {error}"
                )

                continue

            result = self.http_sender.send_observation(
                payload
            )

            if result["success"]:

                self.outbox.delete(obs_id)

                sent += 1

                print(
                    f"[OK] {obs_id} sent."
                )

                print(
                    f"[OK] HTTP {result['status']}"
                )

            else:

                error = (
                    result["error"]
                    or f"HTTP {result['status']}"
                )

                self.outbox.mark_attempt(
                    obs_id,
                    error=error
                )

                failed += 1

                print(
                    f"[FAIL] {obs_id}"
                )

                print(
                    f"[FAIL] {error}"
                )

        return {
            "sent": sent,
            "failed": failed,
            "remaining": self.outbox.count()
        }

    def cleanup_old_observations(self):

        deleted = self.outbox.drop_older_than(
            days=7
        )

        print(
            f"[CLEANUP] Removed {deleted} "
            f"observations older than 7 days."
        )

        return deleted