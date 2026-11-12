"""Serve what the camera sees to the brain, over HTTP.

Examples:
    uv run python -m pokervision.serve --picture ../data/cards/test/000_Jd_4c.jpg
    uv run python -m pokervision.serve --camera /dev/video0

GET /cards answers with the cards in view, from left to right in the picture.
The format is shown in protocol/cards.json at the root of the repository. A
card name is a rank (A23456789TJQK) followed by a suit (cdhs). `doubt` is empty
when the reading can be trusted. The answer is a 503 with an `error` field when
no picture is available.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from pokervision.read_cards import DEFAULT_WEIGHTS, CardReader, CardReading

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8770
DEFAULT_ZONE = "robot_hole"

FrameSource = Callable[[], "np.ndarray | None"]


def cards_message(readings: list[CardReading], zone: str) -> dict:
    cards = [
        {"name": reading.name, "confidence": round(reading.confidence, 3), "doubt": reading.doubt}
        for reading in readings
    ]
    return {"version": PROTOCOL_VERSION, "zone": zone, "cards": cards}


class PictureSource:
    """Reads the same file again at each request, so that it can be swapped while serving."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self) -> np.ndarray | None:
        return cv2.imread(str(self.path))


class CameraSource:
    """Keeps the latest picture of a camera. A thread reads it continuously,
    because the camera driver would otherwise hand out old buffered pictures."""

    def __init__(self, device: str, width: int = 1920, height: int = 1080) -> None:
        self._camera = cv2.VideoCapture(int(device) if device.isdigit() else device)
        self._camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._camera.isOpened():
            raise SystemExit(f"Cannot open camera {device}.")
        self._frame: np.ndarray | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            ok, frame = self._camera.read()
            self._frame = frame if ok else None

    def __call__(self) -> np.ndarray | None:
        return self._frame


def make_server(source: FrameSource, reader: CardReader, port: int, zone: str = DEFAULT_ZONE) -> ThreadingHTTPServer:
    # The network is not safe to call from several threads at once.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path != "/cards":
                self.reply(404, {"error": "unknown path, try /cards"})
                return
            frame = source()
            if frame is None:
                self.reply(503, {"error": "no picture available"})
                return
            with lock:
                readings = reader.read(frame)
            self.reply(200, cards_message(readings, zone))

        def log_message(self, *args) -> None:
            pass

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the cards seen by the camera.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", help="camera device or number")
    group.add_argument("--picture", type=Path, help="serve the reading of a picture file instead of a camera")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--zone", default=DEFAULT_ZONE, help="what this camera looks at")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    source = CameraSource(args.camera) if args.camera else PictureSource(args.picture)
    server = make_server(source, CardReader(args.weights), args.port, args.zone)
    print(f"Serving on port {args.port}: GET /cards", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
