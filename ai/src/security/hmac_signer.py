import hashlib
import hmac


class HMACSigner:
    """
    Creates HMAC-SHA256 authentication headers for edge requests.

    Architecture:
        bodyHash = SHA256(rawBody)
        toSign = deviceId + "\\n" + timestamp + "\\n" + bodyHash
        signature = HMAC-SHA256(deviceSecret, toSign)
    """

    def __init__(self, device_secret):
        if not device_secret:
            raise ValueError("Device secret cannot be empty")

        if isinstance(device_secret, str):
            device_secret = device_secret.encode("utf-8")

        self.device_secret = device_secret

    def sign(self, body, device_id, timestamp):
        """
        Generate authentication headers.

        Args:
            body: Raw request body as str or bytes
            device_id: Device identifier
            timestamp: Request timestamp string

        Returns:
            Dictionary containing required X-* headers.
        """

        if isinstance(body, str):
            raw_body = body.encode("utf-8")
        elif isinstance(body, bytes):
            raw_body = body
        else:
            raise TypeError("body must be str or bytes")

        body_hash = hashlib.sha256(raw_body).hexdigest()

        to_sign = (
            f"{device_id}\n"
            f"{timestamp}\n"
            f"{body_hash}"
        ).encode("utf-8")

        signature = hmac.new(
            self.device_secret,
            to_sign,
            hashlib.sha256
        ).hexdigest()

        return {
            "X-Device-Id": device_id,
            "X-Timestamp": timestamp,
            "X-Signature": signature
        }