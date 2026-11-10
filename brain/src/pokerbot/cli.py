"""Command-line game against an opponent that plays at random."""

import random

from pokerbot.engine import Action, Dealer, Hand, IllegalMove, Pending

HUMAN = 0
BOT = 1


def bot_action(hand: Hand, rng: random.Random) -> Action:
    legal = hand.legal()
    roll = rng.random()
    if legal.can_raise and roll < 0.25:
        target = min(legal.max_raise_to, legal.min_raise_to + hand.pot)
        return Action.raise_to(target)
    if legal.can_fold and roll > 0.85:
        return Action.fold()
    return Action.check_call()


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


def show_table(hand: Hand) -> None:
    board = " ".join(hand.board) or "-"
    print(
        f"  Board: {board} | pot {hand.pot} | "
        f"you {hand.stacks[HUMAN]} (bet {hand.bets[HUMAN]}) | "
        f"bot {hand.stacks[BOT]} (bet {hand.bets[BOT]})"
    )


def play_hand(stacks: tuple[int, int], button: int, dealer: Dealer, rng: random.Random) -> Hand:
    hand = Hand(stacks, button)
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
            action = bot_action(hand, rng)
            print(f"  Bot plays: {action.kind.value} {action.amount or ''}")
            hand.act(action)
        dealer.advance(hand)
    if any(event[0] == "show" for event in hand.history):
        print(f"  Final board: {' '.join(hand.board)} | bot: {' '.join(hand.hole[BOT])}")
    print(f"  Result: {hand.payoffs[HUMAN]:+d} for you.")
    return hand


def main() -> None:
    rng = random.Random()
    dealer = Dealer(rng)
    stacks = (200, 200)
    button = HUMAN
    try:
        while min(stacks) > 0:
            stacks = play_hand(stacks, button, dealer, rng).stacks
            button = 1 - button
    except (EOFError, KeyboardInterrupt):
        print()
    print(f"Game over. You: {stacks[HUMAN]}, bot: {stacks[BOT]}.")
