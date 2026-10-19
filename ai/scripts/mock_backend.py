from http.server import BaseHTTPRequestHandler, HTTPServer
import json


HOST = "127.0.0.1"
PORT = 8000

DEVICE_SECRET = "test-device-secret-123"


class MockBackendHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        print("\n" + "=" * 60)
        print("MOCK BACKEND RECEIVED REQUEST")
        print("=" * 60)

        # --------------------------------------------------
        # Read headers
        # --------------------------------------------------

        device_id = self.headers.get("X-Device-Id")
        timestamp = self.headers.get("X-Timestamp")
        signature = self.headers.get("X-Signature")

        print(f"Device ID : {device_id}")
        print(f"Timestamp : {timestamp}")
        print(f"Signature : {signature}")

        # --------------------------------------------------
        # Read body
        # --------------------------------------------------

        content_length = int(
            self.headers.get("Content-Length", 0)
        )

        raw_body = self.rfile.read(content_length)

        print(f"Body size : {len(raw_body)} bytes")

        try:
            payload = json.loads(raw_body.decode("utf-8"))

            print("\nPayload:")
            print(json.dumps(payload, indent=2))

        except Exception:
            print("[ERROR] Invalid JSON")

            self.send_response(400)
            self.end_headers()
            return

        # --------------------------------------------------
        # Mock successful response
        # --------------------------------------------------

        response = {
            "ok": True,
            "message": "Observation received",
            "obsId": payload.get("obsId")
        }

        response_bytes = json.dumps(response).encode("utf-8")

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(response_bytes))
        )

        self.end_headers()

        self.wfile.write(response_bytes)

        print("\n[OK] Mock backend accepted observation.")

    def log_message(self, format, *args):
        # Disable default HTTP server logging.
        pass


def main():

    server = HTTPServer(
        (HOST, PORT),
        MockBackendHandler
    )

    print("=" * 60)
    print("MOCK BACKEND")
    print("=" * 60)
    print(f"Listening on: http://{HOST}:{PORT}")
    print("POST endpoint: /observations")
    print("\nPress CTRL+C to stop.")
    print("=" * 60)

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")

    finally:
        server.server_close()


if __name__ == "__main__":
    main()