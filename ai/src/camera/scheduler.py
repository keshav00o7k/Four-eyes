import time
from datetime import datetime


class FrameScheduler:
    def __init__(self, interval_seconds=300):
        """
        Default:
        300 seconds = 5 minutes
        """
        self.interval_seconds = interval_seconds

    def should_capture(self, last_capture_time):
        if last_capture_time is None:
            return True

        elapsed = (
            datetime.now() - last_capture_time
        ).total_seconds()

        return elapsed >= self.interval_seconds

    def wait_until_next_capture(self):
        print(
            f"[INFO] Waiting {self.interval_seconds} seconds "
            "until next capture..."
        )

        time.sleep(self.interval_seconds)