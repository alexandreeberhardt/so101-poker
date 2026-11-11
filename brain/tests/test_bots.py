import random

import pytest

from pokerbot.bots import PushFoldBot, RandomBot
from pokerbot.engine import Action, Dealer, Hand, Kind, Pending

HUMAN = 0
BOT = 1
ACES = "AsAh"
TRASH = "3c2d"


@pytest.fixture(scope="module")
def bot(tables):
    equity, weight = tables
    return PushFoldBot(equity, weight)


def start(button, bot_cards, human_cards="9c4d", stacks=(20, 20)):
    hand = Hand(stacks, button)
    hand.deal_hole(HUMAN, human_cards)
    hand.deal_hole(BOT, bot_cards)
    return hand


def act(bot, hand, seed=0):
    return bot.act(hand, BOT, random.Random(seed))


def test_button_pushes_aces_and_folds_trash(bot):
    assert act(bot, start(BOT, ACES)) == Action.raise_to(20)
    assert act(bot, start(BOT, TRASH)) == Action.fold()


def test_push_is_capped_by_the_bot_stack(bot):
    hand = start(BOT, ACES, stacks=(40, 12))
    assert act(bot, hand) == Action.raise_to(12)


def test_big_blind_calls_a_push_with_aces_and_folds_trash(bot):
    for cards, expected in ((ACES, Action.check_call()), (TRASH, Action.fold())):
        hand = start(HUMAN, cards)
        hand.act(Action.raise_to(20))
        assert act(bot, hand) == expected


def test_small_raise_is_answered_like_a_push(bot):
    for cards, expected in ((ACES, Action.raise_to(20)), (TRASH, Action.fold())):
        hand = start(HUMAN, cards)
        hand.act(Action.raise_to(5))
        assert act(bot, hand) == expected


def test_limp_is_punished_with_aces_and_checked_with_trash(bot):
    for cards, expected in ((ACES, Action.raise_to(20)), (TRASH, Action.check_call())):
        hand = start(HUMAN, cards)
        hand.act(Action.check_call())
        assert act(bot, hand) == expected


def flop_after_a_limp(bot_cards, flop):
    hand = start(HUMAN, bot_cards, human_cards="9c4d")
    hand.act(Action.check_call())
    hand.act(Action.check_call())
    hand.deal_board(flop)
    assert hand.to_act == BOT
    return hand


def test_after_the_flop_a_strong_hand_goes_all_in(bot):
    hand = flop_after_a_limp("7h7d", "7c2s2h")
    action = act(bot, hand)
    assert action.kind is Kind.RAISE_TO
    assert action.amount == hand.legal().max_raise_to


def test_after_the_flop_a_weak_hand_checks_then_folds_to_a_big_bet(bot):
    hand = flop_after_a_limp("3c2d", "AhKsQs")
    assert act(bot, hand) == Action.check_call()
    hand.act(Action.check_call())
    hand.act(Action.raise_to(18))
    assert act(bot, hand) == Action.fold()


def test_solutions_are_cached_per_stack_depth(bot):
    assert bot.solution(10.1) is bot.solution(9.9)
    assert bot.solution(10) is not bot.solution(8)


def play_match(first, second, rng, hands, stacks=(20, 20)):
    """Total result of seat 0 over `hands` hands, stacks reset every hand."""
    dealer = Dealer(rng)
    bots = (first, second)
    total = 0
    for number in range(hands):
        hand = Hand(stacks, button=number % 2)
        dealer.advance(hand)
        while hand.pending is not Pending.DONE:
            seat = hand.to_act
            hand.act(bots[seat].act(hand, seat, rng))
            dealer.advance(hand)
        assert sum(hand.stacks) == sum(stacks)
        total += hand.payoffs[0]
    return total


def test_never_plays_an_illegal_action_against_a_random_player(bot):
    rng = random.Random(1)
    for stacks in ((20, 20), (7, 33), (33, 7), (3, 40), (40, 2)):
        play_match(RandomBot(), bot, rng, hands=150, stacks=stacks)


class AlwaysFold:
    def act(self, hand, seat, rng):
        return Action.fold() if hand.legal().can_fold else Action.check_call()


class AlwaysPush:
    def act(self, hand, seat, rng):
        legal = hand.legal()
        return Action.raise_to(legal.max_raise_to) if legal.can_raise else Action.check_call()


def test_wins_against_players_who_always_fold_or_always_push(bot):
    rng = random.Random(2)
    assert play_match(bot, AlwaysFold(), rng, hands=400) > 0
    assert play_match(bot, AlwaysPush(), rng, hands=3000) > 0
