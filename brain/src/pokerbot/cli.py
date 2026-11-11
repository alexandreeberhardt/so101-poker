"""Command-line game against a bot.

The match is a freeze-out: it lasts until one player has all the chips. Stacks
start short (ten big blinds by default) because the default bot plays the
push-or-fold equilibrium, which is only sound for short stacks.
"""

from __future__ import annotations

import argparse
import random

from pokerbot.bots import PushFoldBot, RandomBot
from pokerbot.engine import Action, Dealer, Hand, IllegalMove, Kind, Legal, Pending
from pokerbot.strategy.preflop import DEFAULT_CACHE

HUMAN = 0
BOT = 1
BIG_BLIND = 2


def ask_human(hand: Hand) -> Action:
    legal = hand.legal()
    options = []
    if legal.can_fold:
        options.append("f = fold")
    options.append("c = check" if legal.call_amount == 0 else f"c = call {legal.call_amount}")
    if legal.can_raise:
        options.append(f"a number = raise to that total ({legal.min_raise_to}-{legal.max_raise_to})")
    while True:
        answer = input("  " + ", ".join(options) + " > ").strip().lower()
        if answer == "f":
            return Action.fold()
        if answer == "c":
            return Action.check_call()
        if answer.isdigit():
            return Action.raise_to(int(answer))
        print("  Input not understood.")


def describe(action: Action, legal: Legal) -> str:
    if action.kind is Kind.FOLD:
        return "folds"
    if action.kind is Kind.CHECK_CALL:
        return "checks" if legal.call_amount == 0 else f"calls {legal.call_amount}"
    suffix = " (all-in)" if action.amount == legal.max_raise_to else ""
    return f"raises to {action.amount}{suffix}"


def show_table(hand: Hand) -> None:
    board = " ".join(hand.board) or "-"
    print(
        f"  Board: {board} | pot {hand.pot} | "
        f"you {hand.stacks[HUMAN]} (bet {hand.bets[HUMAN]}) | "
        f"bot {hand.stacks[BOT]} (bet {hand.bets[BOT]})"
    )


def play_hand(stacks: tuple[int, int], button: int, dealer: Dealer, bot, rng: random.Random) -> Hand:
    hand = Hand(stacks, button, big_blind=BIG_BLIND)
    dealer.advance(hand)
    print(f"\nNew hand. Button: {'you' if button == HUMAN else 'bot'}.")
    print(f"  Your hole cards: {' '.join(hand.hole[HUMAN])}")
    while hand.pending is not Pending.DONE:
        show_table(hand)
        if hand.to_act == HUMAN:
            try:
                hand.act(ask_human(hand))
            except IllegalMove as error:
                print(f"  Rejected: {error}")
                continue
        else:
            legal = hand.legal()
            action = bot.act(hand, BOT, rng)
            print(f"  Bot {describe(action, legal)}.")
            hand.act(action)
        dealer.advance(hand)
    if any(event[0] == "show" for event in hand.history):
        print(f"  Final board: {' '.join(hand.board)} | bot: {' '.join(hand.hole[BOT])}")
    print(f"  Result: {hand.payoffs[HUMAN]:+d} for you.")
    return hand


def make_bot(name: str):
    if name == "random":
        return RandomBot()
    if not DEFAULT_CACHE.exists():
        print("Computing the preflop equities, about a minute. This only happens once.")
    return PushFoldBot()


def main() -> None:
    parser = argparse.ArgumentParser(description="Heads-up No-Limit Hold'em against a bot.")
    parser.add_argument("--bot", choices=("pushfold", "random"), default="pushfold")
    parser.add_argument("--stack", type=int, default=10, help="starting stack of each player, in big blinds")
    args = parser.parse_args()

    rng = random.Random()
    dealer = Dealer(rng)
    bot = make_bot(args.bot)
    stacks = (args.stack * BIG_BLIND, args.stack * BIG_BLIND)
    button = HUMAN
    print(f"Blinds 1 and {BIG_BLIND}. Each player starts with {stacks[HUMAN]} chips.")
    try:
        while min(stacks) > 0:
            stacks = play_hand(stacks, button, dealer, bot, rng).stacks
            button = 1 - button
    except (EOFError, KeyboardInterrupt):
        print()
    print(f"Game over. You: {stacks[HUMAN]}, bot: {stacks[BOT]}.")
