"""Self-play reinforcement learning on Leduc poker, to compare with CFR.

A tabular policy-gradient agent (REINFORCE with a per-situation baseline)
plays both seats against itself. Its exploitability is measured exactly with
the same best-response code as the CFR solver.
"""

from __future__ import annotations

import math
import random

from pokerbot.strategy.leduc import (
    CHANCE,
    DECK,
    TERMINAL,
    Strategy,
    exploitability,
    infoset_key,
    legal_actions,
    next_histories,
    node_type,
    payoff_for_first_player,
)


class PolicyGradientAgent:
    def __init__(self, learning_rate: float = 0.02, seed: int = 0) -> None:
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)
        self.logits: dict[str, dict[str, float]] = {}
        self.baseline: dict[str, float] = {}

    def probabilities(self, infoset: str, actions: str) -> dict[str, float]:
        logits = self.logits.setdefault(infoset, dict.fromkeys(actions, 0.0))
        highest = max(logits.values())
        weights = {action: math.exp(value - highest) for action, value in logits.items()}
        total = sum(weights.values())
        return {action: weight / total for action, weight in weights.items()}

    def strategy(self) -> Strategy:
        return {infoset: self.probabilities(infoset, "".join(logits)) for infoset, logits in self.logits.items()}

    def play_episode(self) -> None:
        cards = tuple(self.rng.sample(DECK, 2))
        public = None
        first_round = ""
        second_round = ""
        decisions = []
        while True:
            node = node_type(first_round, second_round, public)
            if node == TERMINAL:
                break
            if node == CHANCE:
                public = self.rng.choice([card for card in DECK if card not in cards])
                continue
            history = second_round if public is not None else first_round
            actions = legal_actions(history)
            infoset = infoset_key(cards[node], public, first_round, second_round)
            probabilities = self.probabilities(infoset, actions)
            action = self.rng.choices(actions, weights=[probabilities[a] for a in actions])[0]
            decisions.append((node, infoset, action, probabilities))
            first_round, second_round = next_histories(first_round, second_round, public, action)

        payoff = payoff_for_first_player(cards, public, first_round, second_round)
        for player, infoset, action, probabilities in decisions:
            reward = payoff if player == 0 else -payoff
            baseline = self.baseline.get(infoset, 0.0)
            advantage = reward - baseline
            self.baseline[infoset] = baseline + 0.01 * advantage
            logits = self.logits[infoset]
            for candidate, probability in probabilities.items():
                indicator = 1.0 if candidate == action else 0.0
                logits[candidate] += self.learning_rate * advantage * (indicator - probability)

    def train(self, episodes: int) -> None:
        for _ in range(episodes):
            self.play_episode()


def main() -> None:
    agent = PolicyGradientAgent()
    done = 0
    for target in (10_000, 50_000, 200_000, 500_000, 1_000_000, 2_000_000):
        agent.train(target - done)
        done = target
        print(f"{target:>9} hands: exploitability {exploitability(agent.strategy()):.4f}")


if __name__ == "__main__":
    main()
