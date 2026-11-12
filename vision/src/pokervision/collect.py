"""Collect labelled card images from a remote camera over SSH.

The deck is sorted beforehand (clubs, diamonds, hearts, spades, each from ace
to king) and shown two cards at a time, first card on the player's left. The
camera looks up through the glass, so that card is on the right of the image.
The camera side takes a picture each time two new cards are in place, so the
position in the sequence gives the labels. Labels use image coordinates. Each
picture is announced aloud on macOS.

Example: uv run python -m pokervision.collect --host <jetson> --out ../data/cards/pass1
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

SORTED_DECK = [rank + suit for suit in "cdhs" for rank in "A23456789TJQK"]
REMOTE_SCRIPT = Path(__file__).with_name("collect_remote.py")


def read_exactly(stream, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def announce(text: str) -> None:
    if shutil.which("say"):
        subprocess.Popen(["say", text])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labelled card images over SSH.")
    parser.add_argument("--host", required=True, help="SSH host with the camera")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--count", type=int, default=26, help="number of pictures to take")
    parser.add_argument("--first", type=int, default=0, help="position in the sorted deck of the first card")
    parser.add_argument("--device", default="/dev/video0", help="camera device on the host")
    parser.add_argument("--preview-port", type=int, default=0, help="serve a live view on this port of the host")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = args.out / "labels.csv"
    if not labels.exists():
        labels.write_text("file,left,right\n")

    command = ["ssh", "-o", "BatchMode=yes", args.host, "python3", "-u", "-", str(args.count), args.device, str(args.preview_port)]
    with REMOTE_SCRIPT.open("rb") as script:
        process = subprocess.Popen(command, stdin=script, stdout=subprocess.PIPE)
    received = 0
    while received < args.count:
        header = read_exactly(process.stdout, 4)
        if not header:
            break
        image = read_exactly(process.stdout, struct.unpack(">I", header)[0])
        if not image:
            break
        position = args.first + 2 * received
        right = SORTED_DECK[position % len(SORTED_DECK)]
        left = SORTED_DECK[(position + 1) % len(SORTED_DECK)]
        name = f"{position // 2:03d}_{left}_{right}.jpg"
        (args.out / name).write_bytes(image)
        with labels.open("a") as handle:
            handle.write(f"{name},{left},{right}\n")
        received += 1
        print(f"{received}/{args.count} {name}", flush=True)
        announce(str(received))
    process.wait()
    if received < args.count:
        sys.exit(f"stopped after {received} of {args.count} pictures")
    announce("done")


if __name__ == "__main__":
    main()
