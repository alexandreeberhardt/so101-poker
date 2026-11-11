"""Camera-side half of the data collection. Runs on the machine with the camera.

It watches the camera and sends one JPEG on standard output each time two
cards have been placed and the picture has stopped moving. Clearing the glass
re-arms it for the next pair. Each image is sent as a 4-byte big-endian length
followed by the JPEG bytes. It only needs OpenCV and numpy.

With a preview port, it also serves a live annotated view of the camera at
http://<host>:<port>/ so the setup can be adjusted while it runs.
"""

from __future__ import annotations

import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BRIGHTNESS_THRESHOLDS = (160, 175, 190, 205, 220)
DETECTION_SCALE = 2
EROSION_KERNEL = np.ones((7, 7), np.uint8)
EROSION_MARGIN = 6
MIN_AREA_SHARE = 0.05
MAX_AREA_SHARE = 0.16
MIN_ASPECT = 1.2
MAX_ASPECT = 1.7
MIN_FILL = 0.85
STILL_PIXELS = 25
DROPOUT_SECONDS = 0.4
STILL_SECONDS = 1.0
CLEAR_SECONDS = 0.3


def _card_shapes(mask: np.ndarray) -> list[tuple]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = mask.shape[0] * mask.shape[1]
    shapes = []
    for contour in contours:
        rectangle = cv2.minAreaRect(contour)
        width, height = rectangle[1]
        if min(width, height) == 0:
            continue
        rectangle_area = width * height
        aspect = max(width, height) / min(width, height)
        # The hull ignores the gaps that dark court-card artwork opens in the border.
        fill = cv2.contourArea(cv2.convexHull(contour)) / rectangle_area
        if not MIN_AREA_SHARE <= rectangle_area / frame_area <= MAX_AREA_SHARE:
            continue
        if MIN_ASPECT <= aspect <= MAX_ASPECT and fill >= MIN_FILL:
            shapes.append(rectangle)
    return shapes


def _encloses_another(rectangle: tuple, others: list[tuple]) -> bool:
    area = rectangle[1][0] * rectangle[1][1]
    for other in others:
        other_area = other[1][0] * other[1][1]
        inside = cv2.pointPolygonTest(cv2.boxPoints(rectangle), other[0], False) >= 0
        if other is not rectangle and inside and area > 1.5 * other_area:
            return True
    return False


def find_cards(frame: np.ndarray) -> list[tuple]:
    """Rotated rectangles of the bright card-shaped regions in a BGR frame.

    Several brightness thresholds are tried because a reflection next to a
    card can merge with it at one threshold and separate at another.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1 / DETECTION_SCALE, fy=1 / DETECTION_SCALE, interpolation=cv2.INTER_AREA)
    cards: list[tuple] = []
    for threshold in BRIGHTNESS_THRESHOLDS:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        # Eroding separates cards that touch and cuts thin bridges to reflections.
        mask = cv2.erode(mask, EROSION_KERNEL)
        for rectangle in _card_shapes(mask):
            center = np.array(rectangle[0])
            short_side = min(rectangle[1])
            known = any(np.linalg.norm(center - np.array(other[0])) < short_side / 2 for other in cards)
            if not known:
                cards.append(rectangle)
    # Two cards lying close together also form one bright card-shaped region.
    cards = [card for card in cards if not _encloses_another(card, cards)]
    scale = DETECTION_SCALE
    margin = EROSION_MARGIN
    return [
        ((x * scale, y * scale), ((width + margin) * scale, (height + margin) * scale), angle)
        for (x, y), (width, height), angle in cards
    ]


# The page reconnects by itself when the stream stops, for example on a restart.
PREVIEW_PAGE = b"""<!doctype html><title>Card camera</title>
<body style="margin:0;background:#111"><img id="v" style="width:100%">
<script>
const v = document.getElementById("v");
const connect = () => { v.src = "/stream?" + Date.now(); };
v.onerror = () => setTimeout(connect, 1000);
connect();
</script></body>"""


class Preview:
    """Live MJPEG stream of annotated frames, viewable in a browser."""

    def __init__(self, port: int) -> None:
        self.jpeg = b""
        self.condition = threading.Condition()
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if not self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PREVIEW_PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with preview.condition:
                            preview.condition.wait()
                            jpeg = preview.jpeg
                        header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                        self.wfile.write(header.encode() + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args) -> None:
                pass

        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def update(self, frame: np.ndarray, cards: list[tuple], status: str) -> None:
        view = frame.copy()
        for rectangle in cards:
            corners = cv2.boxPoints(rectangle).astype(np.int32)
            cv2.polylines(view, [corners], True, (0, 220, 0), 8)
        view = cv2.resize(view, (960, 540))
        color = (0, 200, 0) if len(cards) == 2 else (0, 0, 255)
        cv2.rectangle(view, (0, 496), (960, 540), (0, 0, 0), -1)
        cv2.putText(view, status, (12, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        with self.condition:
            self.jpeg = jpeg
            self.condition.notify_all()


def main() -> None:
    target = int(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "/dev/video0"
    preview_port = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    preview = Preview(preview_port) if preview_port else None
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not camera.isOpened():
        sys.exit(f"cannot open camera {device}")

    sent = 0
    armed = True
    frame_number = 0
    reference = None
    still_since = 0.0
    last_pair_time = 0.0
    clear_since = None
    while sent < target:
        ok, frame = camera.read()
        if not ok:
            continue
        now = time.monotonic()
        cards = find_cards(frame)
        card_count = len(cards)
        # Stillness is judged on the card positions, not on raw pixels, because
        # flickering lamps change the whole picture from one frame to the next.
        # A card missed for a few frames does not count as movement either.
        if card_count == 2:
            centers = sorted(rectangle[0] for rectangle in cards)
            last_pair_time = now
            shifted = reference is None or any(
                abs(x - old_x) + abs(y - old_y) > STILL_PIXELS
                for (x, y), (old_x, old_y) in zip(centers, reference)
            )
            if shifted:
                reference = centers
                still_since = now
        elif now - last_pair_time > DROPOUT_SECONDS:
            reference = None
        frame_number += 1
        if preview and frame_number % 3 == 0:
            state = "ready" if armed else "clear the glass"
            preview.update(frame, cards, f"cards seen: {card_count}   {state}   taken: {sent}/{target}")
        if card_count == 0:
            if clear_since is None:
                clear_since = now
            if not armed and now - clear_since >= CLEAR_SECONDS:
                armed = True
                print("armed", file=sys.stderr, flush=True)
        else:
            clear_since = None

        still = reference is not None and now - still_since >= STILL_SECONDS
        if armed and still and card_count == 2:
            encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            sys.stdout.buffer.write(struct.pack(">I", len(encoded)) + encoded)
            sys.stdout.buffer.flush()
            sent += 1
            armed = False
            print(f"sent {sent}", file=sys.stderr, flush=True)
    camera.release()


if __name__ == "__main__":
    main()
