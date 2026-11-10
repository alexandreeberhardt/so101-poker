"""CFR sur le Kuhn poker, pour valider l'algorithme sur un jeu a solution connue.

Trois cartes (0 = Valet, 1 = Dame, 2 = Roi), une carte par joueur, une ante de 1.
Actions : 'p' (passer : checker ou se coucher) et 'b' (miser 1 ou suivre).
A l'equilibre, le premier joueur perd 1/18 par main en moyenne.
"""

from __future__ import annotations

import itertools

ACTIONS = "pb"
DEALS = list(itertools.permutations(range(3), 2))
TERMINALS = ("pp", "bp", "bb", "pbp", "pbb")
INFOSET_HISTORIES = {0: ("", "pb"), 1: ("p", "b")}


def player_to_act(history: str) -> int:
    return len(history) % 2


def payoff_for_first_player(cards: tuple[int, int], history: str) -> int:
    if history == "bp":
        return 1
    if history == "pbp":
        return -1
    stake = 1 if history == "pp" else 2
    return stake if cards[0] > cards[1] else -stake


class KuhnSolver:
    def __init__(self) -> None:
        self.regret_sum: dict[str, list[float]] = {}
        self.strategy_sum: dict[str, list[float]] = {}

    def current_strategy(self, infoset: str) -> list[float]:
        regrets = self.regret_sum.setdefault(infoset, [0.0, 0.0])
        positive = [max(regret, 0.0) for regret in regrets]
        total = sum(positive)
        if total == 0:
            return [0.5, 0.5]
        return [value / total for value in positive]

    def average_strategy(self) -> dict[str, list[float]]:
        average = {}
        for infoset, sums in self.strategy_sum.items():
            total = sum(sums)
            average[infoset] = [value / total for value in sums] if total else [0.5, 0.5]
        return average

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            for cards in DEALS:
                self._cfr(cards, "", 1.0, 1.0)

    def _cfr(self, cards: tuple[int, int], history: str, reach0: float, reach1: float) -> float:
        """Valeur de la position pour le premier joueur."""
        if history in TERMINALS:
            return payoff_for_first_player(cards, history)
        player = player_to_act(history)
        infoset = f"{cards[player]}{history}"
        strategy = self.current_strategy(infoset)
        action_values = []
        for index, action in enumerate(ACTIONS):
            if player == 0:
                value = self._cfr(cards, history + action, reach0 * strategy[index], reach1)
            else:
                value = self._cfr(cards, history + action, reach0, reach1 * strategy[index])
            action_values.append(value)
        node_value = sum(p * v for p, v in zip(strategy, action_values))

        own_reach = reach0 if player == 0 else reach1
        opponent_reach = reach1 if player == 0 else reach0
        sign = 1 if player == 0 else -1
        regrets = self.regret_sum[infoset]
        sums = self.strategy_sum.setdefault(infoset, [0.0, 0.0])
        for index in range(2):
            regrets[index] += opponent_reach * sign * (action_values[index] - node_value)
            sums[index] += own_reach * strategy[index]
        return node_value


def expected_value(strategy: dict[str, list[float]]) -> float:
    """Gain moyen par main du premier joueur quand les deux suivent `strategy`."""

    def walk(cards: tuple[int, int], history: str) -> float:
        if history in TERMINALS:
            return payoff_for_first_player(cards, history)
        probabilities = strategy[f"{cards[player_to_act(history)]}{history}"]
        return sum(
            probability * walk(cards, history + action)
            for probability, action in zip(probabilities, ACTIONS)
        )

    return sum(walk(cards, "") for cards in DEALS) / len(DEALS)


def best_response_value(strategy: dict[str, list[float]], player: int) -> float:
    """Meilleur gain possible de `player` contre `strategy`, par force brute."""
    infosets = [f"{card}{history}" for card in range(3) for history in INFOSET_HISTORIES[player]]
    best = float("-inf")
    for choices in itertools.product(range(2), repeat=len(infosets)):
        candidate = dict(strategy)
        for infoset, choice in zip(infosets, choices):
            candidate[infoset] = [1.0, 0.0] if choice == 0 else [0.0, 1.0]
        value = expected_value(candidate)
        best = max(best, value if player == 0 else -value)
    return best


def exploitability(strategy: dict[str, list[float]]) -> float:
    """Gain moyen par main d'un adversaire parfait. Vaut 0 a l'equilibre."""
    return (best_response_value(strategy, 0) + best_response_value(strategy, 1)) / 2


def main() -> None:
    solver = KuhnSolver()
    names = "VDR"
    done = 0
    for target in (10, 100, 1000, 10000):
        solver.train(target - done)
        done = target
        strategy = solver.average_strategy()
        print(f"{target} iterations : exploitabilite {exploitability(strategy):.5f}")
    strategy = solver.average_strategy()
    print(f"Valeur pour le premier joueur : {expected_value(strategy):+.4f} (theorie : {-1 / 18:+.4f})")
    print("Probabilite de miser ou suivre, par situation :")
    for infoset in sorted(strategy, key=lambda key: (len(key), key)):
        situation = infoset[1:] or "debut"
        print(f"  {names[int(infoset[0])]} apres '{situation}' : {strategy[infoset][1]:.2f}")


if __name__ == "__main__":
    main()
