"""Chance of winning a hand against an unknown opponent hand."""

from __future__ import annotations

import numpy as np

from pokerbot.strategy.evaluator import card_index, evaluate

DECK_SIZE = 52
BOARD_SIZE = 5


def equity_vs_random(
    hole: tuple[str, ...],
    board: list[str] | tuple[str, ...] = (),
    samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Share of the pot `hole` wins on average against a random hand.

    The opponent's cards and the missing board cards are drawn at random.
    A tie counts for half.
    """
    rng = rng or np.random.default_rng()
    own = [card_index(card) for card in hole]
    known_board = [card_index(card) for card in board]
    seen = set(own) | set(known_board)
    remaining = np.array([card for card in range(DECK_SIZE) if card not in seen])
    drawn_count = 2 + BOARD_SIZE - len(known_board)

    keys = rng.random((samples, len(remaining)))
    drawn = remaining[np.argpartition(keys, drawn_count, axis=1)[:, :drawn_count]]
    opponent = drawn[:, :2]
    full_board = np.hstack([np.tile(known_board, (samples, 1)).astype(np.int64), drawn[:, 2:]])
    own_scores = evaluate(np.hstack([np.tile(own, (samples, 1)), full_board]))
    opponent_scores = evaluate(np.hstack([opponent, full_board]))
    wins = (own_scores > opponent_scores) + 0.5 * (own_scores == opponent_scores)
    return float(wins.mean())
