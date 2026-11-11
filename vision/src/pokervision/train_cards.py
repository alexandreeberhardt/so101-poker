"""Train the card corner reader and measure it on photos kept aside.

Example: uv run python -m pokervision.train_cards --data ../data/cards --out ../models/cards.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from pokervision.card_model import RANKS, SUITS, CornerNet, load_corners, to_tensor

TEST_SUFFIX = "test"


def augment(corner: np.ndarray, rng: random.Random) -> np.ndarray:
    """A randomly shifted, turned, relit and blurred copy of a corner."""
    height, width = corner.shape[:2]
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.88, 1.12)
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    transform[0, 2] += rng.uniform(-7, 7)
    transform[1, 2] += rng.uniform(-9, 9)
    fill = tuple(rng.randint(60, 255) for _ in range(3))
    image = cv2.warpAffine(corner, transform, (width, height), borderValue=fill)

    image = image.astype(np.float32)
    gain = rng.uniform(0.6, 1.25)
    offset = rng.uniform(-25, 25)
    tint = np.array([rng.uniform(0.92, 1.08) for _ in range(3)], dtype=np.float32)
    image = image * gain * tint + offset
    if rng.random() < 0.3:
        # A soft bright patch, like a lamp reflected in the glass.
        center = (rng.uniform(0, width), rng.uniform(0, height))
        ys, xs = np.mgrid[0:height, 0:width]
        distance = ((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / rng.uniform(300, 1500)
        image += rng.uniform(40, 120) * np.exp(-distance)[:, :, None]
    image += np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(0, 6), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        size = rng.choice((3, 5))
        image = cv2.GaussianBlur(image, (size, size), 0)
    return image


def targets(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = torch.tensor([RANKS.index(label[0]) for label in labels])
    suits = torch.tensor([SUITS.index(label[1]) for label in labels])
    return ranks, suits


@torch.no_grad()
def evaluate(model: CornerNet, samples: list) -> dict:
    """Accuracy per corner and per card (two corners voting) on plain images."""
    model.eval()
    corner_hits = 0
    card_hits = 0
    mistakes = []
    for first, second, label in samples:
        rank_scores, suit_scores = model(to_tensor([first, second]))
        rank_probabilities = rank_scores.softmax(dim=1)
        suit_probabilities = suit_scores.softmax(dim=1)
        for index in range(2):
            name = RANKS[int(rank_probabilities[index].argmax())] + SUITS[int(suit_probabilities[index].argmax())]
            corner_hits += name == label
        name = RANKS[int(rank_probabilities.mean(dim=0).argmax())] + SUITS[int(suit_probabilities.mean(dim=0).argmax())]
        card_hits += name == label
        if name != label:
            mistakes.append((label, name))
    return {
        "corner_accuracy": corner_hits / (2 * len(samples)),
        "card_accuracy": card_hits / len(samples),
        "mistakes": mistakes,
    }


def train(samples: list, epochs: int, seed: int) -> CornerNet:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = CornerNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, total_steps=epochs)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    corners = [(corner, label) for first, second, label in samples for corner in (first, second)]
    for _ in range(epochs):
        model.train()
        rng.shuffle(corners)
        for start in range(0, len(corners), 64):
            batch = corners[start : start + 64]
            images = to_tensor([augment(corner, rng) for corner, _ in batch])
            ranks, suits = targets([label for _, label in batch])
            rank_scores, suit_scores = model(images)
            loss = loss_function(rank_scores, ranks) + loss_function(suit_scores, suits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the card corner reader.")
    parser.add_argument("--data", type=Path, default=Path("../data/cards"))
    parser.add_argument("--out", type=Path, default=Path("../models/cards.pt"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Folders whose name ends in "test" are kept aside; every other one is used to train.
    folders = sorted(path.parent for path in args.data.glob("*/labels.csv"))
    test_folders = [folder for folder in folders if folder.name.endswith(TEST_SUFFIX)]
    train_folders = [folder for folder in folders if folder not in test_folders]
    train_samples = [sample for folder in train_folders for sample in load_corners(folder)]
    print(f"{len(train_samples)} training cards from {', '.join(folder.name for folder in train_folders)}")

    model = train(train_samples, args.epochs, args.seed)
    sets = [("train", train_samples)] + [(folder.name, load_corners(folder)) for folder in test_folders]
    for name, samples in sets:
        result = evaluate(model, samples)
        print(
            f"{name}: {result['card_accuracy']:.1%} of {len(samples)} cards read correctly, "
            f"{result['corner_accuracy']:.1%} of single corners"
        )
        for expected, got in result["mistakes"]:
            print(f"  {expected} read as {got}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
