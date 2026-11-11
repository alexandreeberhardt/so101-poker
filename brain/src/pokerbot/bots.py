"""Opponents for the simulated game.

A bot only looks at what a player at the table could see: its own hole cards,
the board, the bets and the stacks.
"""

from __future__ import annotations

import random

import numpy as np

from pokerbot.engine import Action, Hand, Legal
from pokerbot.strategy.equity import equity_vs_random
from pokerbot.strategy.evaluator import card_index
from pokerbot.strategy.preflop import class_of, load_equities
from pokerbot.strategy.pushfold import PushFoldSolution, solve


class RandomBot:
    """Plays without looking at any card. Only useful to exercise the engine."""

    def act(self, hand: Hand, seat: int, rng: random.Random) -> Action:
        legal = hand.legal()
        roll = rng.random()
        if legal.can_raise and roll < 0.25:
            target = min(legal.max_raise_to, legal.min_raise_to + hand.pot)
            return Action.raise_to(target)
        if legal.can_fold and roll > 0.85:
            return Action.fold()
        return Action.check_call()


def all_in(legal: Legal) -> Action:
    """Put every chip in: a raise when one is allowed, otherwise a call."""
    if legal.can_raise:
        return Action.raise_to(legal.max_raise_to)
    return Action.check_call()


class PushFoldBot:
    """Short-stack bot built on the push-or-fold equilibrium.

    On the button it goes all-in or folds, and in the big blind it calls an
    all-in or folds, with the frequencies of the equilibrium for the current
    stack depth. That strategy is only sound up to about ten big blinds.

    The opponent is free to play outside of that small game, so three simple
    rules cover the rest. None of them is an equilibrium.
    - A raise that is not all-in is answered as if it were: all-in with the
      hands that would call a push, fold with the others.
    - Against a limp, the bot goes all-in with the same hands and checks the others.
    - After the flop it compares its chance of beating a random hand with the
      price it is offered.
    """

    STACK_STEP = 0.5
    BET_EQUITY = 0.65
    RAISE_EQUITY = 0.80
    CALL_MARGIN = 0.10

    def __init__(self, equity: np.ndarray | None = None, weight: np.ndarray | None = None) -> None:
        if equity is None or weight is None:
            equity, weight = load_equities()
        self._equity = equity
        self._weight = weight
        self._solutions: dict[float, PushFoldSolution] = {}

    def solution(self, stack: float) -> PushFoldSolution:
        """Equilibrium for a stack in big blinds, rounded to half a big blind."""
        rounded = max(1.0, round(stack / self.STACK_STEP) * self.STACK_STEP)
        if rounded not in self._solutions:
            self._solutions[rounded] = solve(self._equity, self._weight, rounded)
        return self._solutions[rounded]

    def act(self, hand: Hand, seat: int, rng: random.Random) -> Action:
        legal = hand.legal()
        if hand.board:
            return self._after_the_flop(hand, seat, legal, rng)
        return self._before_the_flop(hand, seat, legal, rng)

    def _before_the_flop(self, hand: Hand, seat: int, legal: Legal, rng: random.Random) -> Action:
        stack = min(hand.starting_stacks) / hand.big_blind
        solution = self.solution(stack)
        first, second = (card_index(card) for card in hand.hole[seat])
        hand_class = int(class_of(np.array([first]), np.array([second]))[0])
        opens = seat == hand.button and not any(event[0] == "action" for event in hand.history)

        if opens:
            if rng.random() < solution.push[hand_class]:
                return all_in(legal)
            return Action.fold() if legal.can_fold else Action.check_call()

        continues = rng.random() < solution.call[hand_class]
        if legal.call_amount == 0:
            return all_in(legal) if continues else Action.check_call()
        return all_in(legal) if continues else Action.fold()

    def _after_the_flop(self, hand: Hand, seat: int, legal: Legal, rng: random.Random) -> Action:
        generator = np.random.default_rng(rng.getrandbits(32))
        equity = equity_vs_random(hand.hole[seat], hand.board, rng=generator)
        if legal.call_amount == 0:
            return all_in(legal) if equity >= self.BET_EQUITY else Action.check_call()
        price = legal.call_amount / (hand.pot + legal.call_amount)
        if equity < price + self.CALL_MARGIN:
            return Action.fold()
        return all_in(legal) if equity >= self.RAISE_EQUITY else Action.check_call()
