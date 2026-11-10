"""Heads-up No-Limit Texas Hold'em engine, built on top of PokerKit.

A hand moves forward step by step. `Hand.pending` tells what the engine is
waiting for: hole cards, an action, board cards, a showdown, or nothing.
Cards can come from a random draw (simulation) or from the camera (real
table). An unknown opponent hand is dealt with `None` and revealed at
showdown.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from pokerkit import Automation, NoLimitTexasHoldem

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = tuple(rank + suit for rank in RANKS for suit in SUITS)

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
    Automation.RUNOUT_COUNT_SELECTION,
)


class IllegalMove(Exception):
    """Action or event rejected by the rules."""


class Pending(Enum):
    HOLE = "hole"
    ACTION = "action"
    BOARD = "board"
    SHOWDOWN = "showdown"
    DONE = "done"


class Kind(Enum):
    FOLD = "fold"
    CHECK_CALL = "check_call"
    RAISE_TO = "raise_to"


@dataclass(frozen=True)
class Action:
    """`amount` is the player's total bet on the current betting round."""

    kind: Kind
    amount: int = 0

    @staticmethod
    def fold() -> Action:
        return Action(Kind.FOLD)

    @staticmethod
    def check_call() -> Action:
        return Action(Kind.CHECK_CALL)

    @staticmethod
    def raise_to(amount: int) -> Action:
        return Action(Kind.RAISE_TO, amount)


@dataclass(frozen=True)
class Legal:
    can_fold: bool
    call_amount: int
    min_raise_to: int | None
    max_raise_to: int | None

    @property
    def can_raise(self) -> bool:
        return self.min_raise_to is not None


def parse_cards(text: str) -> tuple[str, ...]:
    """'AcKd' or 'Ac Kd' -> ('Ac', 'Kd')."""
    compact = text.replace(" ", "")
    cards = tuple(compact[i : i + 2] for i in range(0, len(compact), 2))
    for card in cards:
        if card not in FULL_DECK:
            raise IllegalMove(f"invalid card: {card!r}")
    return cards


class Hand:
    """One hand. Seats 0 and 1 are fixed; `button` posts the small blind."""

    def __init__(
        self,
        stacks: tuple[int, int],
        button: int,
        small_blind: int = 1,
        big_blind: int = 2,
    ) -> None:
        if button not in (0, 1):
            raise ValueError("button must be 0 or 1")
        if min(stacks) <= 0:
            raise ValueError("each player must have chips")
        self.button = button
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stacks = tuple(stacks)
        self.hole: list[tuple[str, ...] | None] = [None, None]
        self.hole_dealt = [False, False]
        self.board: list[str] = []
        self.history: list[tuple] = []
        # PokerKit puts the big blind at index 0 and the button at index 1.
        pk_stacks = (stacks[self._seat(0)], stacks[self._seat(1)])
        self._state = NoLimitTexasHoldem.create_state(
            _AUTOMATIONS,
            True,
            0,
            (small_blind, big_blind),
            big_blind,
            pk_stacks,
            2,
        )

    def _pk(self, seat: int) -> int:
        return 1 if seat == self.button else 0

    def _seat(self, pk_index: int) -> int:
        return self.button if pk_index == 1 else 1 - self.button

    @property
    def pending(self) -> Pending:
        state = self._state
        if not state.status:
            return Pending.DONE
        if state.can_deal_hole():
            return Pending.HOLE
        if state.actor_index is not None:
            return Pending.ACTION
        if state.showdown_index is not None:
            return Pending.SHOWDOWN
        if state.can_deal_board():
            return Pending.BOARD
        raise RuntimeError("unexpected PokerKit state")

    def _burn(self) -> None:
        # The burn card stays unknown: it is not removed from the visible deck.
        while self._state.can_burn_card():
            self._state.burn_card("??")

    @property
    def to_act(self) -> int | None:
        """Seat expected next: to act or to show its cards."""
        state = self._state
        pending = self.pending
        if pending is Pending.ACTION:
            return self._seat(state.actor_index)
        if pending is Pending.SHOWDOWN:
            return self._seat(state.showdown_index)
        if pending is Pending.HOLE:
            return self._seat(state.hole_dealee_index)
        return None

    @property
    def board_cards_needed(self) -> int:
        if self.pending is not Pending.BOARD:
            return 0
        return 3 if not self.board else 1

    @property
    def stacks(self) -> tuple[int, int]:
        pk_stacks = self._state.stacks
        return (pk_stacks[self._pk(0)], pk_stacks[self._pk(1)])

    @property
    def bets(self) -> tuple[int, int]:
        pk_bets = self._state.bets
        return (pk_bets[self._pk(0)], pk_bets[self._pk(1)])

    @property
    def pot(self) -> int:
        """Total in the middle, including the bets of the current round."""
        return self._state.total_pot_amount

    @property
    def payoffs(self) -> tuple[int, int]:
        """Net winnings of each seat, valid once the hand is over."""
        stacks = self.stacks
        return (
            stacks[0] - self.starting_stacks[0],
            stacks[1] - self.starting_stacks[1],
        )

    def known_cards(self) -> set[str]:
        cards = set(self.board)
        for hole in self.hole:
            if hole:
                cards.update(hole)
        return cards

    def _check_new_cards(self, cards: tuple[str, ...], count: int) -> None:
        if len(cards) != count:
            raise IllegalMove(f"expected {count} card(s), got {len(cards)}")
        if len(set(cards)) != len(cards) or set(cards) & self.known_cards():
            raise IllegalMove(f"duplicate card: {cards}")

    def deal_hole(self, seat: int, cards: str | None) -> None:
        """`cards=None` for a hidden hand (the opponent at the real table)."""
        if self.pending is not Pending.HOLE:
            raise IllegalMove("no hole cards expected")
        if self.hole_dealt[seat]:
            raise IllegalMove(f"seat {seat} already has its cards")
        parsed = None
        if cards is not None:
            parsed = parse_cards(cards)
            self._check_new_cards(parsed, 2)
        # PokerKit enforces a dealing order; wait until both hands are known.
        self.hole[seat] = parsed
        self.hole_dealt[seat] = True
        self.history.append(("hole", seat, parsed))
        if all(self.hole_dealt):
            while self._state.can_deal_hole():
                dealee = self._seat(self._state.hole_dealee_index)
                hole = self.hole[dealee]
                self._state.deal_hole("".join(hole) if hole else "????")

    def legal(self) -> Legal:
        if self.pending is not Pending.ACTION:
            raise IllegalMove("no action expected")
        state = self._state
        call_amount = state.checking_or_calling_amount
        can_raise = state.can_complete_bet_or_raise_to()
        return Legal(
            can_fold=state.can_fold(),
            call_amount=call_amount,
            min_raise_to=state.min_completion_betting_or_raising_to_amount if can_raise else None,
            max_raise_to=state.max_completion_betting_or_raising_to_amount if can_raise else None,
        )

    def act(self, action: Action) -> None:
        legal = self.legal()
        seat = self.to_act
        state = self._state
        if action.kind is Kind.FOLD:
            if not legal.can_fold:
                raise IllegalMove("cannot fold with no bet to call")
            state.fold()
        elif action.kind is Kind.CHECK_CALL:
            state.check_or_call()
        else:
            if not legal.can_raise:
                raise IllegalMove("cannot raise")
            if not legal.min_raise_to <= action.amount <= legal.max_raise_to:
                raise IllegalMove(
                    f"raise to {action.amount} outside "
                    f"[{legal.min_raise_to}, {legal.max_raise_to}]"
                )
            state.complete_bet_or_raise_to(action.amount)
        self._burn()
        self.history.append(("action", seat, action.kind.value, action.amount))

    def deal_board(self, cards: str) -> None:
        if self.pending is not Pending.BOARD:
            raise IllegalMove("no board cards expected")
        parsed = parse_cards(cards)
        self._check_new_cards(parsed, self.board_cards_needed)
        self._state.deal_board("".join(parsed))
        self._burn()
        self.board.extend(parsed)
        self.history.append(("board", parsed))

    def show(self, seat: int, cards: str | None = None) -> None:
        """Show the seat's hand. `cards` is required if it was hidden."""
        if self.pending is not Pending.SHOWDOWN or self.to_act != seat:
            raise IllegalMove(f"it is not seat {seat}'s turn to show")
        if self.hole[seat] is None:
            if cards is None:
                raise IllegalMove("unknown cards: they must be provided")
            parsed = parse_cards(cards)
            self._check_new_cards(parsed, 2)
            self.hole[seat] = parsed
            self._state.show_or_muck_hole_cards("".join(parsed))
        else:
            if cards is not None and parse_cards(cards) != self.hole[seat]:
                raise IllegalMove("cards differ from the ones dealt")
            self._state.show_or_muck_hole_cards(True)
        self._burn()
        self.history.append(("show", seat, self.hole[seat]))

    def muck(self, seat: int) -> None:
        """The seat mucks its hand without showing it and gives up the pot."""
        if self.pending is not Pending.SHOWDOWN or self.to_act != seat:
            raise IllegalMove(f"it is not seat {seat}'s turn to show")
        self._state.show_or_muck_hole_cards(False)
        self._burn()
        self.history.append(("muck", seat))


class Dealer:
    """Simulated dealer: draws cards from a shuffled deck."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def advance(self, hand: Hand) -> None:
        """Advance the hand until the next action or until it is over."""
        while hand.pending not in (Pending.ACTION, Pending.DONE):
            if hand.pending is Pending.HOLE:
                for seat in (0, 1):
                    hand.deal_hole(seat, "".join(self._draw(hand, 2)))
            elif hand.pending is Pending.BOARD:
                hand.deal_board("".join(self._draw(hand, hand.board_cards_needed)))
            else:
                hand.show(hand.to_act)

    def _draw(self, hand: Hand, count: int) -> list[str]:
        remaining = sorted(set(FULL_DECK) - hand.known_cards())
        return self.rng.sample(remaining, count)
