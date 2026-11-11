"""Read the cards in a picture: their names, and which readings to doubt.

Examples:
    uv run python -m pokervision.read_cards picture.jpg
    uv run python -m pokervision.read_cards --check ../data/cards

A reading is doubtful when the network is not confident, when the two corners
of the card disagree, or when the same card shows up twice in the picture.

The confidence threshold comes from a measurement on three decks. On held-out
cards of decks the network was trained on, confidence never fell below 0.87
(60 cards, all read correctly). Leaving each deck out of training in turn, 46 of
the 276 cards were misread, and 41 of those 46 had a confidence below 0.8. Half
of the correct readings of an unknown deck are flagged too, which is wanted: an
unknown deck should be shown to the robot before a game.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from pokervision.card_model import RANKS, SUITS, CornerNet, to_tensor
from pokervision.cards import corner_crops, straighten
from pokervision.collect_remote import find_cards

DEFAULT_WEIGHTS = Path("../models/cards.pt")
MIN_CONFIDENCE = 0.8
HELD_OUT_SUFFIX = "test"


@dataclass(frozen=True)
class CardReading:
    name: str
    confidence: float
    doubt: str
    centre: tuple[float, float]

    @property
    def doubtful(self) -> bool:
        return bool(self.doubt)

    def __str__(self) -> str:
        text = f"{self.name} ({self.confidence:.2f})"
        return f"{text} doubtful: {self.doubt}" if self.doubt else text


class CardReader:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, min_confidence: float = MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence
        self.model = CornerNet()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def read(self, frame: np.ndarray) -> list[CardReading]:
        """Readings of the cards found in a BGR picture, from left to right."""
        rectangles = sorted(find_cards(frame), key=lambda rectangle: rectangle[0][0])
        if not rectangles:
            return []
        corners = [corner for rectangle in rectangles for corner in corner_crops(straighten(frame, rectangle))]
        rank_scores, suit_scores = self.model(to_tensor(corners))
        # Two corners per card: shape (cards, corners, classes).
        ranks = rank_scores.softmax(dim=1).reshape(len(rectangles), 2, len(RANKS))
        suits = suit_scores.softmax(dim=1).reshape(len(rectangles), 2, len(SUITS))

        names = []
        confidences = []
        corners_agree = []
        for card_ranks, card_suits in zip(ranks, suits):
            rank = int(card_ranks.mean(dim=0).argmax())
            suit = int(card_suits.mean(dim=0).argmax())
            names.append(RANKS[rank] + SUITS[suit])
            confidences.append(float(card_ranks.mean(dim=0)[rank] * card_suits.mean(dim=0)[suit]))
            first_corner = (int(card_ranks[0].argmax()), int(card_suits[0].argmax()))
            second_corner = (int(card_ranks[1].argmax()), int(card_suits[1].argmax()))
            corners_agree.append(first_corner == second_corner)

        readings = []
        for index, rectangle in enumerate(rectangles):
            if names.count(names[index]) > 1:
                doubt = "same card twice"
            elif not corners_agree[index]:
                doubt = "corners disagree"
            elif confidences[index] < self.min_confidence:
                doubt = "low confidence"
            else:
                doubt = ""
            readings.append(CardReading(names[index], confidences[index], doubt, rectangle[0]))
        return readings


def check(reader: CardReader, data: Path) -> None:
    """Compare the readings with the labels of every labelled folder under `data`."""
    for labels in sorted(data.glob("*/labels.csv")):
        folder = labels.parent
        expected_count = found = right = flagged = flagged_wrong = 0
        mistakes = []
        with labels.open() as handle:
            for row in csv.DictReader(handle):
                expected = [row["left"], row["right"]]
                readings = reader.read(cv2.imread(str(folder / row["file"])))
                expected_count += len(expected)
                found += len(readings)
                if len(readings) != len(expected):
                    mistakes.append(f"{row['file']}: {len(readings)} cards found instead of {len(expected)}")
                    continue
                for label, reading in zip(expected, readings):
                    right += reading.name == label
                    flagged += reading.doubtful
                    if reading.name != label:
                        flagged_wrong += reading.doubtful
                        mistakes.append(f"{row['file']}: {label} read as {reading}")
        kind = "held out" if folder.name.endswith(HELD_OUT_SUFFIX) else "used for training"
        print(
            f"{folder.name} ({kind}): {found}/{expected_count} cards found, {right} read correctly, "
            f"{flagged} flagged as doubtful ({flagged_wrong} of them misread)"
        )
        for mistake in mistakes:
            print(f"  {mistake}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the cards in pictures.")
    parser.add_argument("pictures", type=Path, nargs="*", help="pictures to read")
    parser.add_argument("--check", type=Path, help="folder of labelled folders to compare the readings with")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    args = parser.parse_args()
    if not args.pictures and args.check is None:
        parser.error("give pictures to read, or --check with a folder of labelled folders")

    reader = CardReader(args.weights, args.min_confidence)
    for picture in args.pictures:
        frame = cv2.imread(str(picture))
        if frame is None:
            print(f"{picture}: cannot be read as a picture")
            continue
        readings = reader.read(frame)
        print(f"{picture}: " + (", ".join(str(reading) for reading in readings) or "no card found"))
    if args.check is not None:
        check(reader, args.check)


if __name__ == "__main__":
    main()
