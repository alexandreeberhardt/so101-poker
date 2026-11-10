import random

import numpy as np
from pokerkit import StandardHighHand

from pokerbot.strategy.evaluator import CATEGORY_BASE, card_index, evaluate

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [rank + suit for rank in RANKS for suit in SUITS]


def score(cards: str) -> int:
    names = [cards[i : i + 2] for i in range(0, len(cards), 2)]
    return int(evaluate(np.array([[card_index(name) for name in names]]))[0])


def test_categories_are_ordered():
    hands = [
        "AcKd9h7s5c3d2h",  # high card
        "AcAd9h7s5c3d2h",  # pair
        "AcAd9h9s5c3d2h",  # two pair
        "AcAdAh7s5c3d2h",  # trips
        "Ac2d3h4s5c9dKh",  # wheel straight
        "AcKc9c7c5c3d2h",  # flush
        "AcAdAh7s7c3d2h",  # full house
        "AcAdAhAs5c3d2h",  # quads
        "Ac2c3c4c5c9dKh",  # straight flush
    ]
    scores = [score(hand) for hand in hands]
    assert scores == sorted(scores)
    assert [value // CATEGORY_BASE for value in scores] == list(range(9))


def test_kickers_break_ties():
    assert score("AcAdKh7s5c3d2h") > score("AsAhQh7d5d3c2s")
    assert score("AcAdKh7s5c3d2h") == score("AsAhKd7d5d3c2s")


def test_three_pairs_use_best_kicker():
    assert score("AcAdKhKs5c5dQh") > score("AsAhKdKc5h5sJh")


def test_two_trips_make_a_full_house():
    assert score("AcAdAhKsKcKd2h") // CATEGORY_BASE == 6


def test_six_high_straight_beats_wheel():
    assert score("2c3d4h5s6cKdKh") > score("Ac2d3h4s5cKdKh")


def test_agrees_with_pokerkit_on_random_hands():
    rng = random.Random(0)
    ours = []
    reference = []
    for _ in range(3000):
        cards = rng.sample(DECK, 9)
        board = "".join(cards[4:])
        hands = ["".join(cards[0:2]), "".join(cards[2:4])]
        ours.append([score(hand + board) for hand in hands])
        reference.append([StandardHighHand.from_game(hand, board) for hand in hands])
    for (first, second), (ref_first, ref_second) in zip(ours, reference):
        assert (first > second) == (ref_first > ref_second)
        assert (first == second) == (ref_first == ref_second)
