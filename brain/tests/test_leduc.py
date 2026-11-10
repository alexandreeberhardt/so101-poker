import pytest

from pokerbot.strategy.leduc import (
    LeducSolver,
    contributions,
    exploitability,
    legal_actions,
    payoff_for_first_player,
)
from pokerbot.strategy.leduc_rl import PolicyGradientAgent

JACK, QUEEN, KING = 0, 2, 4


def test_at_most_two_raises_per_round():
    assert legal_actions("") == "cr"
    assert legal_actions("r") == "fcr"
    assert legal_actions("rr") == "fc"


def test_contributions_use_round_raise_sizes():
    assert contributions("cc", "") == [1, 1]
    assert contributions("rc", "rrc") == [11, 11]


def test_fold_loses_what_was_paid():
    assert payoff_for_first_player((KING, JACK), None, "rf", "") == 1
    assert payoff_for_first_player((KING, JACK), None, "crf", "") == -1


def test_pair_with_public_card_beats_higher_rank():
    assert payoff_for_first_player((JACK, KING), JACK + 1, "cc", "cc") == 1
    assert payoff_for_first_player((QUEEN, KING), JACK, "cc", "cc") == -1
    assert payoff_for_first_player((KING, KING + 1), JACK, "rc", "cc") == 0


def test_uniform_random_exploitability_matches_known_value():
    assert exploitability({}) == pytest.approx(2.3736, abs=1e-3)


def test_cfr_reduces_exploitability():
    solver = LeducSolver()
    solver.train(200)
    assert exploitability(solver.average_strategy()) < 0.06


def test_policy_gradient_agent_learns_a_valid_strategy():
    agent = PolicyGradientAgent(seed=1)
    agent.train(2000)
    strategy = agent.strategy()
    assert strategy
    for probabilities in strategy.values():
        assert sum(probabilities.values()) == pytest.approx(1.0)
