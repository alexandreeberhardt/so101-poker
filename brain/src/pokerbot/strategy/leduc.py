"""Leduc poker: rules, a CFR solver and exact exploitability.

Six cards (two each of Jack, Queen, King). Each player antes 1 and gets one
private card. Two betting rounds with at most two raises each; raises are 2
in the first round and 4 in the second. A public card is revealed between the
rounds. A private card that pairs the public card wins, otherwise the higher
rank wins.

Actions: 'f' fold, 'c' check or call, 'r' bet or raise. Player 0 speaks first
in both rounds.
"""

from __future__ import annotations

RANKS = "JQK"
DECK = tuple(range(6))
RAISE_SIZES = (2, 4)
MAX_RAISES = 2
CHANCE = -1
TERMINAL = -2

Strategy = dict[str, dict[str, float]]


def rank(card: int) -> int:
    return card // 2


def legal_actions(round_history: str) -> str:
    actions = ""
    if round_history.endswith("r"):
        actions += "f"
    actions += "c"
    if round_history.count("r") < MAX_RAISES:
        actions += "r"
    return actions


def round_is_over(round_history: str) -> bool:
    return len(round_history) >= 2 and round_history.endswith("c")


def node_type(first_round: str, second_round: str, public: int | None) -> int:
    """Player to act (0 or 1), CHANCE or TERMINAL."""
    if first_round.endswith("f") or second_round.endswith("f"):
        return TERMINAL
    if not round_is_over(first_round):
        return len(first_round) % 2
    if public is None:
        return CHANCE
    if not round_is_over(second_round):
        return len(second_round) % 2
    return TERMINAL


def contributions(first_round: str, second_round: str) -> list[int]:
    total = [1, 1]
    for history, size in zip((first_round, second_round), RAISE_SIZES):
        bets = [0, 0]
        for index, action in enumerate(history):
            player = index % 2
            if action == "r":
                bets[player] = max(bets) + size
            elif action == "c":
                bets[player] = max(bets)
        total[0] += bets[0]
        total[1] += bets[1]
    return total


def payoff_for_first_player(
    cards: tuple[int, int], public: int | None, first_round: str, second_round: str
) -> int:
    paid = contributions(first_round, second_round)
    for history in (first_round, second_round):
        if history.endswith("f"):
            folder = (len(history) - 1) % 2
            return paid[1] if folder == 1 else -paid[0]
    strengths = []
    for card in cards:
        pairs_board = rank(card) == rank(public)
        strengths.append((pairs_board, rank(card)))
    if strengths[0] > strengths[1]:
        return paid[1]
    if strengths[1] > strengths[0]:
        return -paid[0]
    return 0


def infoset_key(card: int, public: int | None, first_round: str, second_round: str) -> str:
    board = "" if public is None else RANKS[rank(public)]
    return f"{RANKS[rank(card)]}{board}:{first_round}/{second_round}"


def next_histories(first_round: str, second_round: str, public: int | None, action: str) -> tuple[str, str]:
    if public is None:
        return first_round + action, second_round
    return first_round, second_round + action


class LeducSolver:
    """Vanilla CFR with every chance outcome enumerated at each iteration."""

    def __init__(self) -> None:
        self.regret_sum: dict[str, dict[str, float]] = {}
        self.strategy_sum: dict[str, dict[str, float]] = {}

    def current_strategy(self, infoset: str, actions: str) -> dict[str, float]:
        regrets = self.regret_sum.setdefault(infoset, dict.fromkeys(actions, 0.0))
        positive = {action: max(regret, 0.0) for action, regret in regrets.items()}
        total = sum(positive.values())
        if total == 0:
            return dict.fromkeys(actions, 1 / len(actions))
        return {action: value / total for action, value in positive.items()}

    def average_strategy(self) -> Strategy:
        average = {}
        for infoset, sums in self.strategy_sum.items():
            total = sum(sums.values())
            if total == 0:
                average[infoset] = dict.fromkeys(sums, 1 / len(sums))
            else:
                average[infoset] = {action: value / total for action, value in sums.items()}
        return average

    def train(self, iterations: int) -> None:
        deal_probability = 1 / (len(DECK) * (len(DECK) - 1))
        for _ in range(iterations):
            for first in DECK:
                for second in DECK:
                    if first != second:
                        self._cfr((first, second), None, "", "", 1.0, 1.0, deal_probability)

    def _cfr(
        self,
        cards: tuple[int, int],
        public: int | None,
        first_round: str,
        second_round: str,
        reach0: float,
        reach1: float,
        chance_reach: float,
    ) -> float:
        """Value of the node for player 0."""
        node = node_type(first_round, second_round, public)
        if node == TERMINAL:
            return payoff_for_first_player(cards, public, first_round, second_round)
        if node == CHANCE:
            remaining = [card for card in DECK if card not in cards]
            share = 1 / len(remaining)
            return sum(
                share * self._cfr(cards, card, first_round, second_round, reach0, reach1, chance_reach * share)
                for card in remaining
            )

        player = node
        history = second_round if public is not None else first_round
        actions = legal_actions(history)
        infoset = infoset_key(cards[player], public, first_round, second_round)
        strategy = self.current_strategy(infoset, actions)
        action_values = {}
        for action in actions:
            new_first, new_second = next_histories(first_round, second_round, public, action)
            if player == 0:
                value = self._cfr(cards, public, new_first, new_second, reach0 * strategy[action], reach1, chance_reach)
            else:
                value = self._cfr(cards, public, new_first, new_second, reach0, reach1 * strategy[action], chance_reach)
            action_values[action] = value
        node_value = sum(strategy[action] * action_values[action] for action in actions)

        own_reach = reach0 if player == 0 else reach1
        opponent_reach = reach1 if player == 0 else reach0
        sign = 1 if player == 0 else -1
        regrets = self.regret_sum[infoset]
        sums = self.strategy_sum.setdefault(infoset, dict.fromkeys(actions, 0.0))
        for action in actions:
            regrets[action] += chance_reach * opponent_reach * sign * (action_values[action] - node_value)
            sums[action] += own_reach * strategy[action]
        return node_value


def action_probabilities(strategy: Strategy, infoset: str, actions: str) -> dict[str, float]:
    """Probabilities from `strategy`, uniform for situations it never saw."""
    known = strategy.get(infoset)
    if known is None:
        return dict.fromkeys(actions, 1 / len(actions))
    return known


def expected_value(strategy: Strategy) -> float:
    """Average payoff per hand of player 0 when both players follow `strategy`."""

    def walk(cards: tuple[int, int], public: int | None, first_round: str, second_round: str) -> float:
        node = node_type(first_round, second_round, public)
        if node == TERMINAL:
            return payoff_for_first_player(cards, public, first_round, second_round)
        if node == CHANCE:
            remaining = [card for card in DECK if card not in cards]
            return sum(walk(cards, card, first_round, second_round) for card in remaining) / len(remaining)
        history = second_round if public is not None else first_round
        actions = legal_actions(history)
        infoset = infoset_key(cards[node], public, first_round, second_round)
        probabilities = action_probabilities(strategy, infoset, actions)
        total = 0.0
        for action in actions:
            new_first, new_second = next_histories(first_round, second_round, public, action)
            total += probabilities[action] * walk(cards, public, new_first, new_second)
        return total

    deals = [(first, second) for first in DECK for second in DECK if first != second]
    return sum(walk(cards, None, "", "") for cards in deals) / len(deals)


def best_response_value(strategy: Strategy, player: int) -> float:
    """Best average payoff per hand `player` can get against `strategy`."""

    def walk(
        own_card: int,
        public: int | None,
        first_round: str,
        second_round: str,
        opponent_reach: dict[int, float],
    ) -> float:
        """Payoff summed over opponent cards, weighted by their reach."""
        node = node_type(first_round, second_round, public)
        if node == TERMINAL:
            total = 0.0
            for opponent_card, reach in opponent_reach.items():
                cards = (own_card, opponent_card) if player == 0 else (opponent_card, own_card)
                payoff = payoff_for_first_player(cards, public, first_round, second_round)
                total += reach * (payoff if player == 0 else -payoff)
            return total
        if node == CHANCE:
            total = 0.0
            for card in DECK:
                if card == own_card:
                    continue
                # Each of the 4 cards left after both private cards is equally likely.
                narrowed = {
                    opponent_card: reach / (len(DECK) - 2)
                    for opponent_card, reach in opponent_reach.items()
                    if opponent_card != card
                }
                total += walk(own_card, card, first_round, second_round, narrowed)
            return total

        history = second_round if public is not None else first_round
        actions = legal_actions(history)
        if node == player:
            values = []
            for action in actions:
                new_first, new_second = next_histories(first_round, second_round, public, action)
                values.append(walk(own_card, public, new_first, new_second, opponent_reach))
            return max(values)
        total = 0.0
        for action in actions:
            new_first, new_second = next_histories(first_round, second_round, public, action)
            narrowed = {}
            for opponent_card, reach in opponent_reach.items():
                infoset = infoset_key(opponent_card, public, first_round, second_round)
                probability = action_probabilities(strategy, infoset, actions)[action]
                narrowed[opponent_card] = reach * probability
            total += walk(own_card, public, new_first, new_second, narrowed)
        return total

    deal_probability = 1 / (len(DECK) * (len(DECK) - 1))
    total = 0.0
    for own_card in DECK:
        opponent_reach = {card: deal_probability for card in DECK if card != own_card}
        total += walk(own_card, None, "", "", opponent_reach)
    return total


def exploitability(strategy: Strategy) -> float:
    """Average payoff per hand of a perfect opponent. Zero at equilibrium."""
    return (best_response_value(strategy, 0) + best_response_value(strategy, 1)) / 2


def main() -> None:
    solver = LeducSolver()
    done = 0
    for target in (10, 100, 500, 2000):
        solver.train(target - done)
        done = target
        strategy = solver.average_strategy()
        print(f"{target} iterations: exploitability {exploitability(strategy):.4f}")
    strategy = solver.average_strategy()
    print(f"Situations learned: {len(strategy)}")
    print(f"Value for the first player: {expected_value(strategy):+.4f} (theory: about -0.0856)")


if __name__ == "__main__":
    main()
