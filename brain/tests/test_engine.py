import random

import pytest

from pokerbot.engine import Action, Dealer, Hand, IllegalMove, Pending


def start(stacks=(200, 200), button=0, hole0="AcKd", hole1="7h2s"):
    hand = Hand(stacks, button)
    hand.deal_hole(0, hole0)
    hand.deal_hole(1, hole1)
    return hand


def test_blinds_and_order_of_play():
    hand = start(button=0)
    assert hand.bets == (1, 2)
    assert hand.stacks == (199, 198)
    assert hand.to_act == 0
    hand.act(Action.check_call())
    hand.act(Action.check_call())
    assert hand.pending is Pending.BOARD
    assert hand.board_cards_needed == 3
    hand.deal_board("2c3d4h")
    assert hand.to_act == 1


def test_button_on_other_seat():
    hand = start(button=1)
    assert hand.bets == (2, 1)
    assert hand.to_act == 1


def test_fold_gives_pot_to_opponent():
    hand = start(button=0)
    hand.act(Action.raise_to(6))
    hand.act(Action.fold())
    assert hand.pending is Pending.DONE
    assert hand.payoffs == (2, -2)


def test_min_raise_rule():
    hand = start(button=0)
    assert hand.legal().min_raise_to == 4
    hand.act(Action.raise_to(10))
    legal = hand.legal()
    assert legal.call_amount == 8
    assert legal.min_raise_to == 18
    assert legal.max_raise_to == 200
    with pytest.raises(IllegalMove):
        hand.act(Action.raise_to(17))


def test_cannot_fold_when_checking_is_free():
    hand = start(button=0)
    hand.act(Action.check_call())
    with pytest.raises(IllegalMove):
        hand.act(Action.fold())


def test_duplicate_card_is_rejected():
    hand = Hand((200, 200), 0)
    hand.deal_hole(0, "AcKd")
    with pytest.raises(IllegalMove):
        hand.deal_hole(1, "AcQd")
    hand.deal_hole(1, "7h2s")
    hand.act(Action.check_call())
    hand.act(Action.check_call())
    with pytest.raises(IllegalMove):
        hand.deal_board("Ac3d4h")


def test_showdown_best_hand_wins():
    hand = start(button=0, hole0="AcAd", hole1="KhKs")
    hand.act(Action.raise_to(6))
    hand.act(Action.check_call())
    hand.deal_board("2c7d9h")
    for street_card in ("Js", "3s"):
        hand.act(Action.check_call())
        hand.act(Action.check_call())
        if hand.pending is Pending.BOARD:
            hand.deal_board(street_card)
    hand.act(Action.check_call())
    hand.act(Action.check_call())
    while hand.pending is Pending.SHOWDOWN:
        hand.show(hand.to_act)
    assert hand.payoffs == (6, -6)


def test_hidden_opponent_hand_revealed_at_showdown():
    hand = Hand((200, 200), 0)
    hand.deal_hole(0, "AcKd")
    hand.deal_hole(1, None)
    hand.act(Action.check_call())
    hand.act(Action.check_call())
    hand.deal_board("2c7d9h")
    for street_card in ("Js", "3s", None):
        hand.act(Action.check_call())
        hand.act(Action.check_call())
        if street_card:
            hand.deal_board(street_card)
    assert hand.pending is Pending.SHOWDOWN
    while hand.pending is Pending.SHOWDOWN:
        seat = hand.to_act
        if seat == 1:
            with pytest.raises(IllegalMove):
                hand.show(1)
            hand.show(1, "9s9c")
        else:
            hand.show(0)
    assert hand.payoffs == (-2, 2)


def test_all_in_shows_cards_then_runs_out_board():
    hand = start(stacks=(50, 200), button=0, hole0="AcAd", hole1="KhKs")
    hand.act(Action.raise_to(50))
    assert hand.legal().can_raise is False
    hand.act(Action.check_call())
    assert hand.pending is Pending.SHOWDOWN
    while hand.pending is Pending.SHOWDOWN:
        hand.show(hand.to_act)
    hand.deal_board("2c7d9h")
    hand.deal_board("Js")
    hand.deal_board("3s")
    assert hand.pending is Pending.DONE
    assert hand.payoffs == (50, -50)
    assert hand.stacks == (100, 150)


def random_action(hand, rng):
    legal = hand.legal()
    choices = [Action.check_call()]
    if legal.can_fold:
        choices.append(Action.fold())
    if legal.can_raise:
        amount = rng.randint(legal.min_raise_to, legal.max_raise_to)
        choices.append(Action.raise_to(amount))
        choices.append(Action.raise_to(legal.min_raise_to))
    return rng.choice(choices)


def test_random_matches_conserve_chips():
    rng = random.Random(0)
    dealer = Dealer(rng)
    for _ in range(300):
        stacks = (rng.randint(2, 300), rng.randint(2, 300))
        total = sum(stacks)
        button = rng.randint(0, 1)
        while min(stacks) > 0:
            hand = Hand(stacks, button)
            dealer.advance(hand)
            while hand.pending is not Pending.DONE:
                assert sum(hand.stacks) + hand.pot == total
                hand.act(random_action(hand, rng))
                dealer.advance(hand)
            stacks = hand.stacks
            assert sum(stacks) == total
            assert sum(hand.payoffs) == 0
            assert min(stacks) >= 0
            button = 1 - button
