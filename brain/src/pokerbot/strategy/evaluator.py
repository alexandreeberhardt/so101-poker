"""Fast seven-card hand evaluator, vectorized with numpy.

Cards are integers from 0 to 51: rank = card // 4 (0 is a deuce, 12 an ace),
suit = card % 4. `evaluate` scores many hands at once; a higher score wins.
"""

from __future__ import annotations

import numpy as np

RANK_COUNT = 13
MASK_COUNT = 1 << RANK_COUNT
CATEGORY_BASE = RANK_COUNT**5
WHEEL_MASK = 0b1000000001111

(HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH) = range(9)


def _build_tables() -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Lookup tables indexed by a 13-bit mask of the ranks present."""
    highest_bit = np.zeros(MASK_COUNT, dtype=np.int64)
    straight_high = np.full(MASK_COUNT, -1, dtype=np.int64)
    top_codes = {count: np.zeros(MASK_COUNT, dtype=np.int64) for count in (2, 3, 5)}
    for mask in range(1, MASK_COUNT):
        ranks = [rank for rank in range(RANK_COUNT - 1, -1, -1) if mask >> rank & 1]
        highest_bit[mask] = ranks[0]
        for count, table in top_codes.items():
            code = 0
            for position in range(count):
                digit = ranks[position] if position < len(ranks) else 0
                code = code * RANK_COUNT + digit
            table[mask] = code
        for high in range(RANK_COUNT - 1, 3, -1):
            run = 0b11111 << (high - 4)
            if mask & run == run:
                straight_high[mask] = high
                break
        else:
            if mask & WHEEL_MASK == WHEEL_MASK:
                straight_high[mask] = 3
    return highest_bit, straight_high, top_codes


_HIGHEST_BIT, _STRAIGHT_HIGH, _TOP = _build_tables()
_BITS = 1 << np.arange(RANK_COUNT, dtype=np.int64)


def evaluate(cards: np.ndarray) -> np.ndarray:
    """Scores of hands given as an integer array of shape (hands, 7)."""
    cards = np.asarray(cards, dtype=np.int64)
    ranks = cards // 4
    suits = cards % 4
    rows = np.arange(cards.shape[0])

    rank_counts = np.zeros((cards.shape[0], RANK_COUNT), dtype=np.int64)
    suit_counts = np.zeros((cards.shape[0], 4), dtype=np.int64)
    for column in range(cards.shape[1]):
        rank_counts[rows, ranks[:, column]] += 1
        suit_counts[rows, suits[:, column]] += 1

    present = (rank_counts > 0) @ _BITS
    pairs = (rank_counts == 2) @ _BITS
    trips = (rank_counts == 3) @ _BITS
    quads = (rank_counts == 4) @ _BITS

    flush_suit = suit_counts.argmax(axis=1)
    has_flush = suit_counts.max(axis=1) >= 5
    in_flush_suit = suits == flush_suit[:, None]
    flush_mask = ((1 << ranks) * in_flush_suit).sum(axis=1)

    straight = _STRAIGHT_HIGH[present]
    straight_flush = np.where(has_flush, _STRAIGHT_HIGH[flush_mask], -1)

    quad_rank = _HIGHEST_BIT[quads]
    quad_kicker = _HIGHEST_BIT[present & ~(1 << quad_rank)]

    trip_rank = _HIGHEST_BIT[trips]
    full_house_pairs = (trips & ~(1 << trip_rank)) | pairs
    has_full_house = (trips > 0) & (full_house_pairs > 0)
    full_house_pair = _HIGHEST_BIT[full_house_pairs]
    trip_kickers = _TOP[2][present & ~(1 << trip_rank)]

    first_pair = _HIGHEST_BIT[pairs]
    other_pairs = pairs & ~(1 << first_pair)
    second_pair = _HIGHEST_BIT[other_pairs]
    two_pair_kicker = _HIGHEST_BIT[present & ~(1 << first_pair) & ~(1 << second_pair)]
    pair_kickers = _TOP[3][present & ~(1 << first_pair)]

    conditions = [
        straight_flush >= 0,
        quads > 0,
        has_full_house,
        has_flush,
        straight >= 0,
        trips > 0,
        other_pairs > 0,
        pairs > 0,
    ]
    scores = [
        STRAIGHT_FLUSH * CATEGORY_BASE + straight_flush,
        QUADS * CATEGORY_BASE + quad_rank * RANK_COUNT + quad_kicker,
        FULL_HOUSE * CATEGORY_BASE + trip_rank * RANK_COUNT + full_house_pair,
        FLUSH * CATEGORY_BASE + _TOP[5][flush_mask],
        STRAIGHT * CATEGORY_BASE + straight,
        TRIPS * CATEGORY_BASE + trip_rank * RANK_COUNT**2 + trip_kickers,
        TWO_PAIR * CATEGORY_BASE + first_pair * RANK_COUNT**2 + second_pair * RANK_COUNT + two_pair_kicker,
        PAIR * CATEGORY_BASE + first_pair * RANK_COUNT**3 + pair_kickers,
    ]
    return np.select(conditions, scores, default=HIGH_CARD * CATEGORY_BASE + _TOP[5][present])


def card_index(card: str) -> int:
    """'As' -> 51. Ranks 23456789TJQKA, suits cdhs."""
    return "23456789TJQKA".index(card[0]) * 4 + "cdhs".index(card[1])
