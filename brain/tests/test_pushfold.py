import numpy as np
import pytest

from pokerbot.strategy.preflop import (
    CLASS_COUNT,
    class_name,
    class_of,
    combo_count,
    representative,
)
from pokerbot.strategy.pushfold import range_share, solve

NAMES = {class_name(index): index for index in range(CLASS_COUNT)}


def test_classes_cover_every_hand():
    assert len(NAMES) == CLASS_COUNT
    assert sum(combo_count(index) for index in range(CLASS_COUNT)) == 1326
    for index in range(CLASS_COUNT):
        first, second = representative(index)
        assert class_of(np.array([first]), np.array([second]))[0] == index


def test_equities_match_known_matchups(tables):
    equity, weight = tables
    assert weight.sum() == pytest.approx(1.0)
    assert weight[NAMES["AA"]].sum() == pytest.approx(6 / 1326)
    assert equity[NAMES["AA"], NAMES["KK"]] == pytest.approx(0.82, abs=0.06)
    assert equity[NAMES["AKo"], NAMES["22"]] == pytest.approx(0.47, abs=0.06)
    assert np.allclose(equity + equity.T, 1.0)


def test_solution_is_an_equilibrium(tables):
    equity, weight = tables
    solution = solve(equity, weight, 10)
    assert solution.exploitability < 1e-3
    assert solution.push[NAMES["AA"]] == pytest.approx(1.0)
    assert solution.call[NAMES["AA"]] == pytest.approx(1.0)
    assert solution.call[NAMES["72o"]] == pytest.approx(0.0, abs=1e-6)


def test_ranges_tighten_as_stacks_grow(tables):
    equity, weight = tables
    short = solve(equity, weight, 5)
    deep = solve(equity, weight, 15)
    assert range_share(weight, short.push, 1) > range_share(weight, deep.push, 1)
    assert range_share(weight, short.call, 0) > range_share(weight, deep.call, 0)
