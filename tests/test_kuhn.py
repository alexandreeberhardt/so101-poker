import pytest

from pokerbot.strategy.kuhn import KuhnSolver, expected_value, exploitability


def test_cfr_converges_to_equilibrium():
    solver = KuhnSolver()
    solver.train(5000)
    strategy = solver.average_strategy()
    assert exploitability(strategy) < 0.005
    assert expected_value(strategy) == pytest.approx(-1 / 18, abs=0.005)


def test_equilibrium_never_calls_with_jack_nor_folds_king():
    solver = KuhnSolver()
    solver.train(5000)
    strategy = solver.average_strategy()
    assert strategy["0b"][1] < 0.02
    assert strategy["2b"][1] > 0.98


def test_uniform_strategy_is_exploitable():
    solver = KuhnSolver()
    solver.train(1)
    uniform = {infoset: [0.5, 0.5] for infoset in solver.average_strategy()}
    assert exploitability(uniform) > 0.3
