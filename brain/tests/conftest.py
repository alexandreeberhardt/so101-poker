import pytest

from pokerbot.strategy.preflop import compute_equities


@pytest.fixture(scope="session")
def tables():
    """Preflop equities and matchup weights from a small sample: quick, a little noisy."""
    return compute_equities(samples=10)
