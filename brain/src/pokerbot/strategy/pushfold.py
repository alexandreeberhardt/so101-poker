"""Exact push-or-fold strategy for short stacks in heads-up Hold'em.

The button posts the small blind (0.5) and either goes all-in or folds. The
big blind (1) then calls or folds. `stack` is the smaller of the two stacks,
in big blinds, before the blinds are posted. All payoffs are in big blinds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from pokerbot.strategy.preflop import RANKS, load_equities

SMALL_BLIND = 0.5
BIG_BLIND = 1.0


@dataclass
class PushFoldSolution:
    stack: float
    push: np.ndarray
    call: np.ndarray
    button_value: float
    exploitability: float


def _button_values(equity, weight, stack, call):
    """Value of pushing and of folding for each button class, weighted by chance."""
    called = weight * call[None, :]
    not_called = weight * (1 - call[None, :])
    push = not_called.sum(axis=1) * BIG_BLIND + (called * (2 * equity - 1)).sum(axis=1) * stack
    fold = -SMALL_BLIND * weight.sum(axis=1)
    return push, fold


def _big_blind_values(equity, weight, stack, push):
    """Value of calling and of folding a push for each big blind class."""
    pushed = weight * push[:, None]
    call = (pushed * (1 - 2 * equity)).sum(axis=0) * stack
    fold = -BIG_BLIND * pushed.sum(axis=0)
    return call, fold


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Probability of the first action for each class."""
    positive = np.maximum(regrets, 0)
    total = positive.sum(axis=1)
    return np.where(total > 0, positive[:, 0] / np.maximum(total, 1e-12), 0.5)


def solve(equity: np.ndarray, weight: np.ndarray, stack: float, iterations: int = 5000) -> PushFoldSolution:
    """CFR+ with linear averaging on the push-or-fold game."""
    classes = equity.shape[0]
    button_regrets = np.zeros((classes, 2))
    big_blind_regrets = np.zeros((classes, 2))
    push_sum = np.zeros(classes)
    call_sum = np.zeros(classes)
    for iteration in range(1, iterations + 1):
        push = _regret_matching(button_regrets)
        call = _regret_matching(big_blind_regrets)

        push_value, fold_value = _button_values(equity, weight, stack, call)
        node_value = push * push_value + (1 - push) * fold_value
        button_regrets[:, 0] += push_value - node_value
        button_regrets[:, 1] += fold_value - node_value

        call_value, fold_call_value = _big_blind_values(equity, weight, stack, push)
        node_value = call * call_value + (1 - call) * fold_call_value
        big_blind_regrets[:, 0] += call_value - node_value
        big_blind_regrets[:, 1] += fold_call_value - node_value

        np.maximum(button_regrets, 0, out=button_regrets)
        np.maximum(big_blind_regrets, 0, out=big_blind_regrets)
        push_sum += iteration * push
        call_sum += iteration * call

    weight_total = iterations * (iterations + 1) / 2
    push = push_sum / weight_total
    call = call_sum / weight_total
    return PushFoldSolution(
        stack=stack,
        push=push,
        call=call,
        button_value=button_value(equity, weight, stack, push, call),
        exploitability=exploitability(equity, weight, stack, push, call),
    )


def button_value(equity, weight, stack, push, call) -> float:
    """Average payoff per hand of the button, in big blinds."""
    push_value, fold_value = _button_values(equity, weight, stack, call)
    return float((push * push_value + (1 - push) * fold_value).sum())


def exploitability(equity, weight, stack, push, call) -> float:
    """Average gain per hand of a perfect opponent, in big blinds."""
    push_value, fold_value = _button_values(equity, weight, stack, call)
    best_button = np.maximum(push_value, fold_value).sum()
    call_value, fold_call_value = _big_blind_values(equity, weight, stack, push)
    folded_buttons = (weight.sum(axis=1) * (1 - push)).sum() * SMALL_BLIND
    best_big_blind = np.maximum(call_value, fold_call_value).sum() + folded_buttons
    return float(best_button + best_big_blind) / 2


def range_share(weight: np.ndarray, frequencies: np.ndarray, axis: int) -> float:
    """Share of all dealt hands that take the action."""
    return float((weight.sum(axis=axis) * frequencies).sum())


def chart(frequencies: np.ndarray) -> str:
    """13x13 grid, aces first. Suited hands above the diagonal, offsuit below."""
    lines = ["    " + "   ".join(reversed(RANKS))]
    for row in range(12, -1, -1):
        # Class 13 * row + column is suited when row > column, offsuit when row < column.
        cells = [f"{round(frequencies[13 * row + column] * 100):3d}" for column in range(12, -1, -1)]
        lines.append(f"{RANKS[row]}  " + " ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve heads-up push-or-fold.")
    parser.add_argument("stacks", type=float, nargs="*", default=[10.0], help="stack sizes in big blinds")
    args = parser.parse_args()
    equity, weight = load_equities()
    for stack in args.stacks:
        solution = solve(equity, weight, stack)
        print(f"\nStack {stack:g} big blinds")
        print(f"  Button pushes {range_share(weight, solution.push, 1):.1%} of hands")
        print(f"  Big blind calls {range_share(weight, solution.call, 0):.1%} of hands")
        print(f"  Button value {solution.button_value:+.4f} bb/hand, exploitability {solution.exploitability:.5f}")
        print("\n  Push % (suited: row rank above column rank; offsuit: below)")
        print(chart(solution.push))
        print("\n  Call %")
        print(chart(solution.call))


if __name__ == "__main__":
    main()
