"""Capture images from a webcam to build the dataset.

Example: uv run python -m pokervision.capture --camera 0 --out data/cards
Space saves the displayed image, q quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def next_index(folder: Path) -> int:
    numbers = [int(path.stem) for path in folder.glob("*.png") if path.stem.isdigit()]
    return max(numbers, default=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture images from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--out", type=Path, default=Path("data/captures"), help="output folder")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}.")

    args.out.mkdir(parents=True, exist_ok=True)
    index = next_index(args.out)
    print("Space: save. q: quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            raise SystemExit("The camera is no longer returning images.")
        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            path = args.out / f"{index:05d}.png"
            cv2.imwrite(str(path), frame)
            print(f"{path} ({frame.shape[1]}x{frame.shape[0]})")
            index += 1
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
