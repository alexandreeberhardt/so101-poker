import numpy as np
import pytest

from pokerbot.strategy.equity import equity_vs_random


def test_preflop_equities_match_known_values():
    rng = np.random.default_rng(0)
    assert equity_vs_random(("As", "Ah"), samples=20000, rng=rng) == pytest.approx(0.85, abs=0.02)
    assert equity_vs_random(("7c", "2d"), samples=20000, rng=rng) == pytest.approx(0.35, abs=0.02)


def test_unbeatable_hand_always_wins():
    rng = np.random.default_rng(0)
    board = ["Ts", "Js", "Qs", "2d", "3c"]
    assert equity_vs_random(("As", "Ks"), board, rng=rng) == 1.0


def test_board_that_plays_for_both_is_a_tie():
    rng = np.random.default_rng(0)
    board = ["Ts", "Js", "Qs", "Ks", "As"]
    assert equity_vs_random(("2c", "3d"), board, rng=rng) == 0.5


def test_equity_falls_when_the_board_misses():
    rng = np.random.default_rng(0)
    before = equity_vs_random(("Ah", "Kh"), samples=8000, rng=rng)
    after = equity_vs_random(("Ah", "Kh"), ["7c", "8c", "9c"], samples=8000, rng=rng)
    assert after < before
