"""Guided capture of a poker chip dataset, with the ground truth written at capture time.

The script builds a list of scenes (which stacks to put on the table), shows them one
by one in the terminal and on the live view of the camera, and saves every picture with
the exact content of the scene in labels.jsonl. The counts can therefore be trusted
later, when the chips are gone. The camera is on a remote host reached over SSH.

Example:
    uv run python -m pokervision.collect_chips --host 10.50.0.4 --out ../data/chips \
        --chips white:1:100,red:5:50,green:25:50,blue:50:50,black:100:50

--chips is colour:value:how many chips of that colour are in the case. It is only needed
the first time: the plan is stored in plan.json and the session resumes where it stopped.

Keys: space saves, v starts or stops a burst, n next scene, b previous scene, q quits.
"""

from __future__ import annotations

import argparse
import json
import random
import select
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.parse
import urllib.request
from pathlib import Path

STACK_HEIGHTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
SHOTS_PER_SCENE = 4
BURST_PERIOD_SECONDS = 0.5
REMOTE_SCRIPT = Path(__file__).with_name("collect_chips_remote.py")


def parse_chips(text: str) -> list[dict]:
    chips = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise SystemExit(f"Bad chip description '{item}', expected colour:value:count.")
        chips.append({"colour": parts[0], "value": float(parts[1]), "count": int(parts[2])})
    return chips


def empty_scene(phase: str, hint: str) -> dict:
    return {"phase": phase, "hint": hint, "stacks": [], "loose": {}, "pile": {}}


def reference_scenes(chips: list[dict]) -> list[dict]:
    scenes = []
    for chip in chips:
        scene = empty_scene("reference", "One chip alone, close up. Shoot both faces and the edge.")
        scene["stacks"] = [[[chip["colour"], 1]]]
        scenes.append(scene)
    return scenes


def background_scenes() -> list[dict]:
    hint = "No chips. Empty table, then cards only, then a hand over the table."
    return [empty_scene("background", hint)]


def pure_stack_scenes(chips: list[dict]) -> list[dict]:
    """One stack per colour; over the scenes every colour goes through every height."""
    scenes = []
    for scene_number in range(len(STACK_HEIGHTS)):
        scene = empty_scene("stacks", "One stack per colour, apart from each other.")
        for colour_number, chip in enumerate(chips):
            position = (scene_number + 3 * colour_number) % len(STACK_HEIGHTS)
            height = min(STACK_HEIGHTS[position], chip["count"])
            scene["stacks"].append([[chip["colour"], height]])
        scenes.append(scene)
    return scenes


def take(remaining: dict[str, int], rng: random.Random, most: int) -> tuple[str, int] | None:
    """Pick a colour that still has chips and a number of chips, and remove them."""
    colours = [colour for colour, count in remaining.items() if count > 0]
    if not colours:
        return None
    colour = rng.choice(colours)
    count = rng.randint(1, min(most, remaining[colour]))
    remaining[colour] -= count
    return colour, count


def add_counts(target: dict[str, int], picked: tuple[str, int] | None) -> None:
    if picked is not None:
        colour, count = picked
        target[colour] = target.get(colour, 0) + count


def mixed_scene(chips: list[dict], rng: random.Random) -> dict:
    scene = empty_scene("mixed", "Stacks as listed (bottom to top). Loose chips flat, scattered.")
    remaining = {chip["colour"]: chip["count"] for chip in chips}
    for _ in range(rng.randint(2, 6)):
        stack = []
        for _ in range(rng.choice([1, 1, 2, 3])):
            picked = take(remaining, rng, most=8)
            if picked is None:
                break
            if stack and stack[-1][0] == picked[0]:
                stack[-1][1] += picked[1]
            else:
                stack.append(list(picked))
        if stack:
            scene["stacks"].append(stack)
    if rng.random() < 0.5:
        for _ in range(rng.randint(1, 3)):
            add_counts(scene["loose"], take(remaining, rng, most=4))
    return scene


def pile_scene(chips: list[dict], rng: random.Random) -> dict:
    scene = empty_scene("pile", "Pile: one messy heap like a pot, chips overlapping.")
    remaining = {chip["colour"]: chip["count"] for chip in chips}
    for _ in range(rng.randint(2, 4)):
        add_counts(scene["pile"], take(remaining, rng, most=12))
    for _ in range(rng.randint(0, 2)):
        picked = take(remaining, rng, most=10)
        if picked is not None:
            scene["stacks"].append([list(picked)])
    return scene


def build_plan(chips: list[dict], mixed: int, piles: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    scenes = reference_scenes(chips) + background_scenes() + pure_stack_scenes(chips)
    scenes += [mixed_scene(chips, rng) for _ in range(mixed)]
    scenes += [pile_scene(chips, rng) for _ in range(piles)]
    return scenes


def describe_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{count} {colour}" for colour, count in counts.items())


def describe(scene: dict) -> list[str]:
    lines = [scene["hint"]]
    for number, stack in enumerate(scene["stacks"], start=1):
        runs = " + ".join(f"{count} {colour}" for colour, count in stack)
        lines.append(f"stack {number}: {runs}")
    if scene["loose"]:
        lines.append(f"loose: {describe_counts(scene['loose'])}")
    if scene["pile"]:
        lines.append(f"pile: {describe_counts(scene['pile'])}")
    return lines


def load_or_create_plan(out: Path, args: argparse.Namespace) -> dict:
    plan_path = out / "plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    if args.chips is None:
        raise SystemExit("First run: describe the case with --chips colour:value:count,...")
    chips = parse_chips(args.chips)
    plan = {"chips": chips, "scenes": build_plan(chips, args.mixed, args.piles, args.seed)}
    plan_path.write_text(json.dumps(plan, indent=1))
    return plan


def next_index(folder: Path) -> int:
    numbers = [int(path.stem) for path in folder.glob("*.jpg") if path.stem.isdigit()]
    return max(numbers, default=-1) + 1


def shots_per_scene(labels_path: Path, scene_count: int) -> list[int]:
    shots = [0] * scene_count
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            shots[json.loads(line)["scene"]] += 1
    return shots


def first_unfinished(shots: list[int]) -> int:
    for number, count in enumerate(shots):
        if count < SHOTS_PER_SCENE:
            return number
    return len(shots) - 1


def save(image: bytes, out: Path, index: int, scene_number: int, scene: dict, burst: bool) -> None:
    name = f"{index:05d}.jpg"
    (out / name).write_bytes(image)
    label = {"file": name, "scene": scene_number, "burst": burst}
    label.update({key: scene[key] for key in ("phase", "stacks", "loose", "pile")})
    with (out / "labels.jsonl").open("a") as labels:
        labels.write(json.dumps(label) + "\n")


def fetch(base_url: str, path: str) -> bytes:
    with urllib.request.urlopen(base_url + path, timeout=10) as response:
        return response.read()


def start_camera(host: str, port: int, device: str) -> subprocess.Popen:
    """Start the camera server on the host and wait until it answers."""
    command = ["ssh", "-o", "BatchMode=yes", host, "python3", "-u", "-", str(port), device]
    with REMOTE_SCRIPT.open("rb") as script:
        # The camera driver floods the error output with JPEG warnings.
        process = subprocess.Popen(command, stdin=script, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if process.stdout.readline().strip() != b"ready":
        process.wait()
        raise SystemExit(f"The camera {device} did not start on {host}. Is another collection using it?")
    # Nobody reads the heartbeat lines: drain them so that the pipe never fills up.
    threading.Thread(target=process.stdout.read, daemon=True).start()
    return process


def read_key(timeout: float) -> str:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else ""


def show(base_url: str, scenes: list[dict], scene_number: int, shots: int, burst: bool) -> None:
    """Put the current instructions on the live view."""
    scene = scenes[scene_number]
    title = f"scene {scene_number + 1}/{len(scenes)} [{scene['phase']}]  shots {shots}/{SHOTS_PER_SCENE}"
    if burst:
        title += "  BURST"
    text = "|".join([title] + describe(scene))
    path = "/status?" + urllib.parse.urlencode({"text": text})
    # Sent in the background: on a slow link it would delay the next picture.
    threading.Thread(target=fetch, args=(base_url, path), daemon=True).start()


def announce_scene(scenes: list[dict], scene_number: int) -> None:
    scene = scenes[scene_number]
    print(f"\nscene {scene_number + 1}/{len(scenes)} [{scene['phase']}]")
    for line in describe(scene):
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided capture of a chip dataset.")
    parser.add_argument("--host", required=True, help="SSH host with the camera")
    parser.add_argument("--port", type=int, default=8000, help="port of the live view on the host")
    parser.add_argument("--device", default="/dev/video0", help="camera device on the host")
    parser.add_argument("--out", type=Path, default=Path("data/chips"), help="output folder")
    parser.add_argument("--chips", help="colour:value:count,... (first run only)")
    parser.add_argument("--mixed", type=int, default=24, help="number of mixed scenes")
    parser.add_argument("--piles", type=int, default=8, help="number of pile scenes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenes = load_or_create_plan(args.out, args)["scenes"]
    shots = shots_per_scene(args.out / "labels.jsonl", len(scenes))
    scene_number = first_unfinished(shots)
    index = next_index(args.out)

    camera = start_camera(args.host, args.port, args.device)
    base_url = f"http://{args.host}:{args.port}"
    print(f"Live view: {base_url}/")
    print("Space: save. v: burst on/off. n: next scene. b: previous scene. q: quit.")
    announce_scene(scenes, scene_number)

    burst = False
    last_burst_save = 0.0
    terminal_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    try:
        show(base_url, scenes, scene_number, shots[scene_number], burst)
        while camera.poll() is None:
            key = read_key(0.05)
            previous_state = (scene_number, burst)
            if key == "q":
                break
            if key == "v":
                burst = not burst
            if key == "n" and scene_number < len(scenes) - 1:
                scene_number += 1
                burst = False
            if key == "b" and scene_number > 0:
                scene_number -= 1
                burst = False
            if scene_number != previous_state[0]:
                announce_scene(scenes, scene_number)

            burst_due = burst and time.monotonic() - last_burst_save >= BURST_PERIOD_SECONDS
            saved = key == " " or burst_due
            if saved:
                image = fetch(base_url, "/frame")
                save(image, args.out, index, scene_number, scenes[scene_number], burst)
                index += 1
                shots[scene_number] += 1
                last_burst_save = time.monotonic()
                print(f"  saved {index - 1:05d}.jpg ({shots[scene_number]}/{SHOTS_PER_SCENE})")
            if saved or (scene_number, burst) != previous_state:
                show(base_url, scenes, scene_number, shots[scene_number], burst)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal_settings)
        camera.terminate()
    print(f"{sum(shots)} images in {args.out}, {sum(1 for count in shots if count)} scenes done.")


if __name__ == "__main__":
    main()
