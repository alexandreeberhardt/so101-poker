"""The 169 preflop hand classes and their all-in equities against each other.

A class is stored as 13 * a + b. Pairs have a == b, suited hands have the
higher rank first (a > b) and offsuit hands the lower rank first (a < b).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from pokerbot.strategy.evaluator import evaluate

RANKS = "23456789TJQKA"
CLASS_COUNT = 169
DECK_SIZE = 52
DEFAULT_CACHE = Path("models/preflop_equity.npz")


def class_of(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Class of two-card hands given as arrays of card integers."""
    first_rank, second_rank = first // 4, second // 4
    high = np.maximum(first_rank, second_rank)
    low = np.minimum(first_rank, second_rank)
    suited = first % 4 == second % 4
    return np.where(suited, 13 * high + low, 13 * low + high)


def class_name(index: int) -> str:
    a, b = divmod(index, 13)
    if a == b:
        return RANKS[a] * 2
    if a > b:
        return RANKS[a] + RANKS[b] + "s"
    return RANKS[b] + RANKS[a] + "o"


def representative(index: int) -> tuple[int, int]:
    """One concrete hand of the class. Others are equivalent by suit symmetry."""
    a, b = divmod(index, 13)
    if a == b:
        return a * 4, a * 4 + 1
    if a > b:
        return a * 4, b * 4
    return b * 4, a * 4 + 1


def combo_count(index: int) -> int:
    a, b = divmod(index, 13)
    if a == b:
        return 6
    return 4 if a > b else 12


def compute_equities(samples: int = 300, seed: int = 0, chunk: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo equities and matchup weights.

    Returns `equity[i, j]`, the share of the pot class i wins on average when
    all-in against class j, and `weight[i, j]`, the probability that the first
    player holds class i and the second class j. `samples` boards are drawn
    for each concrete opposing hand.
    """
    rng = np.random.default_rng(seed)
    equity_sum = np.zeros((CLASS_COUNT, CLASS_COUNT))
    weight = np.zeros((CLASS_COUNT, CLASS_COUNT))
    for index in range(CLASS_COUNT):
        own = representative(index)
        others = [card for card in range(DECK_SIZE) if card not in own]
        opponents = np.array(list(combinations(others, 2)))
        opponent_classes = class_of(opponents[:, 0], opponents[:, 1])
        weight[index] = np.bincount(opponent_classes, minlength=CLASS_COUNT) * combo_count(index)

        wins = np.zeros(CLASS_COUNT)
        done = 0
        while done < samples:
            repeats = min(chunk, samples - done)
            done += repeats
            hands = np.repeat(opponents, repeats, axis=0)
            rows = np.arange(len(hands))
            keys = rng.random((len(hands), DECK_SIZE), dtype=np.float32)
            keys[:, own[0]] = 2.0
            keys[:, own[1]] = 2.0
            keys[rows, hands[:, 0]] = 2.0
            keys[rows, hands[:, 1]] = 2.0
            boards = np.argpartition(keys, 5, axis=1)[:, :5]
            own_cards = np.broadcast_to(np.array(own), (len(hands), 2))
            own_scores = evaluate(np.hstack([own_cards, boards]))
            opponent_scores = evaluate(np.hstack([hands, boards]))
            result = (own_scores > opponent_scores) + 0.5 * (own_scores == opponent_scores)
            wins += np.bincount(np.repeat(opponent_classes, repeats), weights=result, minlength=CLASS_COUNT)
        counts = np.bincount(opponent_classes, minlength=CLASS_COUNT) * samples
        equity_sum[index] = wins / np.maximum(counts, 1)

    # equity[i, j] and 1 - equity[j, i] estimate the same number: average them.
    equity = (equity_sum + 1 - equity_sum.T) / 2
    weight /= weight.sum()
    return equity, weight


def load_equities(path: Path = DEFAULT_CACHE, samples: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Equities and weights from the cache file, computed first if missing."""
    if path.exists():
        data = np.load(path)
        return data["equity"], data["weight"]
    equity, weight = compute_equities(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, equity=equity, weight=weight)
    return equity, weight
