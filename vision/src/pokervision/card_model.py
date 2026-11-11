"""Small network that reads the rank and the suit from a card corner."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from pokervision.cards import corner_crops, extract_cards

RANKS = "A23456789TJQK"
SUITS = "cdhs"


class CornerNet(nn.Module):
    """Input: a batch of RGB corners, 3 x 112 x 64, values in [0, 1]."""

    def __init__(self) -> None:
        super().__init__()

        def block(inputs: int, outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),
            block(16, 32),
            block(32, 64),
            block(64, 96),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
        )
        self.rank_head = nn.Linear(96, len(RANKS))
        self.suit_head = nn.Linear(96, len(SUITS))

    def forward(self, corners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(corners)
        return self.rank_head(features), self.suit_head(features)


def to_tensor(corners: list[np.ndarray]) -> torch.Tensor:
    """BGR uint8 corners -> float tensor batch in RGB, channels first."""
    batch = np.stack([cv2.cvtColor(corner, cv2.COLOR_BGR2RGB) for corner in corners])
    return torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255


def load_corners(folder: Path) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(first corner, second corner, card name) for every card in a labelled folder."""
    samples = []
    with (folder / "labels.csv").open() as handle:
        for row in csv.DictReader(handle):
            cards = extract_cards(cv2.imread(str(folder / row["file"])))
            if len(cards) != 2:
                continue
            for card, label in zip(cards, (row["left"], row["right"])):
                first, second = corner_crops(card)
                samples.append((first, second, label))
    return samples


@torch.no_grad()
def read_card(model: CornerNet, card: np.ndarray) -> tuple[str, float]:
    """Name of a straightened card and the confidence of the reading.

    The two corners vote together: their scores are averaged.
    """
    model.eval()
    rank_scores, suit_scores = model(to_tensor(corner_crops(card)))
    rank_probabilities = rank_scores.softmax(dim=1).mean(dim=0)
    suit_probabilities = suit_scores.softmax(dim=1).mean(dim=0)
    rank = int(rank_probabilities.argmax())
    suit = int(suit_probabilities.argmax())
    confidence = float(rank_probabilities[rank] * suit_probabilities[suit])
    return RANKS[rank] + SUITS[suit], confidence
