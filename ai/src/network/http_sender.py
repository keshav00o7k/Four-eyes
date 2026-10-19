
import json
import time
import urllib.request
import urllib.error

from security.hmac_signer import HMACSigner


class HTTPSender:
    """Send observations using HMAC authentication."""

    def __init__(
        self,
        backend_url,
        device_id,
        device_secret,
        timeout=10
    ):
        self.backend_url = backend_url.rstrip("/")
        self.device_id = device_id
        self.timeout = timeout
        self.signer = HMACSigner(device_secret)

    @staticmethod
    def _timestamp():
        # Architecture requires Unix seconds, not ISO datetime.
        return str(int(time.time()))

    def send_observation(self, payload):
        # Sign the same compact JSON bytes that will be transmitted.
        body = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False
        )
        body_bytes = body.encode("utf-8")
        timestamp = self._timestamp()

        headers = self.signer.sign(
            body=body,
            device_id=self.device_id,
            timestamp=timestamp
        )
        headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self.backend_url,
            data=body_bytes,
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout
            ) as response:
                status = response.status
                response_body = response.read().decode(
                    "utf-8",
                    errors="replace"
                )

                return {
                    "success": status in (200, 409),
                    "status": status,
                    "response": response_body,
                    "error": None
                }

        except urllib.error.HTTPError as error:
            response_body = error.read().decode(
                "utf-8",
                errors="replace"
            )

            return {
                "success": error.code == 409,
                "status": error.code,
                "response": response_body,
                "error": (
                    None if error.code == 409
                    else f"HTTP {error.code}"
                )
            }

        except urllib.error.URLError as error:
            return {
                "success": False,
                "status": None,
                "response": None,
                "error": str(error.reason)
            }

        except Exception as error:
            return {
                "success": False,
                "status": None,
                "response": None,
                "error": str(error)
            }
