"""Camera-side half of the chip collection. Runs on the machine with the camera.

It serves the camera over HTTP on the given port:
    /                 live view in a browser, with the current instructions on top
    /frame            the latest full-size image as a JPEG
    /status?text=...  set the instructions shown on the live view, lines separated by |

It prints "ready" once the camera works, then a line every second, so that it stops
by itself when the SSH connection that started it goes away. It only needs OpenCV.
"""

from __future__ import annotations

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

# The page reconnects by itself when the stream stops, for example on a restart.
PAGE = b"""<!doctype html><title>Chip camera</title>
<body style="margin:0;background:#111"><img id="v" style="width:100%">
<script>
const v = document.getElementById("v");
const connect = () => { v.src = "/stream?" + Date.now(); };
v.onerror = () => setTimeout(connect, 1000);
connect();
</script></body>"""


class Shared:
    """What the camera loop and the HTTP handlers exchange."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame = None
        self.preview = b""
        self.lines: list[str] = []


def draw_preview(frame, lines: list[str]) -> bytes:
    view = cv2.resize(frame, (960, 540))
    for number, line in enumerate(lines):
        position = (10, 24 + 24 * number)
        cv2.putText(view, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5)
        cv2.putText(view, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def serve(shared: Shared, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, content_type: str, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/frame":
                with shared.condition:
                    # Wait for the next image so that the picture is taken after the request.
                    shared.condition.wait(timeout=2)
                    frame = shared.frame
                jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                self.reply("image/jpeg", jpeg)
            elif url.path == "/status":
                text = parse_qs(url.query).get("text", [""])[0]
                shared.lines = text.split("|")
                self.reply("text/plain", b"ok")
            elif url.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with shared.condition:
                            shared.condition.wait()
                            jpeg = shared.preview
                        header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                        self.wfile.write(header.encode() + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.reply("text/html", PAGE)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main() -> None:
    port = int(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "/dev/video0"
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ok, frame = camera.read()
    if not ok:
        sys.exit(f"cannot open camera {device}")

    shared = Shared()
    shared.frame = frame
    serve(shared, port)
    print("ready", flush=True)
    last_heartbeat = time.monotonic()
    frame_number = 0
    while True:
        ok, frame = camera.read()
        if not ok:
            continue
        frame_number += 1
        preview = draw_preview(frame, shared.lines) if frame_number % 3 == 0 else None
        with shared.condition:
            shared.frame = frame
            if preview is not None:
                shared.preview = preview
            shared.condition.notify_all()
        if time.monotonic() - last_heartbeat >= 1:
            print("alive", flush=True)
            last_heartbeat = time.monotonic()


if __name__ == "__main__":
    main()
