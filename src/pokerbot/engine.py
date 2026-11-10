"""Moteur de Texas Hold'em No-Limit en tete-a-tete, au-dessus de PokerKit.

Une main avance par etapes. `Hand.pending` indique ce que le moteur attend :
des cartes privees, une action, des cartes du board, un abattage, ou rien.
Les cartes peuvent venir d'un tirage aleatoire (simulation) ou de la camera
(table reelle). Une main adverse inconnue se distribue avec `None` et se
revele a l'abattage.
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
    """Action ou evenement refuse par les regles."""


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
    """`amount` est la mise totale du joueur sur le tour d'encheres en cours."""

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
    """'AcKd' ou 'Ac Kd' -> ('Ac', 'Kd')."""
    compact = text.replace(" ", "")
    cards = tuple(compact[i : i + 2] for i in range(0, len(compact), 2))
    for card in cards:
        if card not in FULL_DECK:
            raise IllegalMove(f"carte invalide : {card!r}")
    return cards


class Hand:
    """Une main. Les sieges 0 et 1 sont fixes ; `button` poste la petite blinde."""

    def __init__(
        self,
        stacks: tuple[int, int],
        button: int,
        small_blind: int = 1,
        big_blind: int = 2,
    ) -> None:
        if button not in (0, 1):
            raise ValueError("button doit valoir 0 ou 1")
        if min(stacks) <= 0:
            raise ValueError("chaque joueur doit avoir des jetons")
        self.button = button
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stacks = tuple(stacks)
        self.hole: list[tuple[str, ...] | None] = [None, None]
        self.hole_dealt = [False, False]
        self.board: list[str] = []
        self.history: list[tuple] = []
        # PokerKit place la grosse blinde a l'indice 0 et le bouton a l'indice 1.
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
        raise RuntimeError("etat PokerKit inattendu")

    def _burn(self) -> None:
        # La carte brulee reste inconnue : on ne la retire pas du paquet visible.
        while self._state.can_burn_card():
            self._state.burn_card("??")

    @property
    def to_act(self) -> int | None:
        """Siege attendu : pour une action ou pour montrer ses cartes."""
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
        """Total au milieu, mises du tour en cours comprises."""
        return self._state.total_pot_amount

    @property
    def payoffs(self) -> tuple[int, int]:
        """Gain net de chaque siege, valable quand la main est finie."""
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
            raise IllegalMove(f"{count} carte(s) attendue(s), {len(cards)} recue(s)")
        if len(set(cards)) != len(cards) or set(cards) & self.known_cards():
            raise IllegalMove(f"carte en double : {cards}")

    def deal_hole(self, seat: int, cards: str | None) -> None:
        """`cards=None` pour une main cachee (l'adversaire sur la table reelle)."""
        if self.pending is not Pending.HOLE:
            raise IllegalMove("aucune carte privee attendue")
        if self.hole_dealt[seat]:
            raise IllegalMove(f"le siege {seat} a deja ses cartes")
        parsed = None
        if cards is not None:
            parsed = parse_cards(cards)
            self._check_new_cards(parsed, 2)
        # PokerKit impose un ordre de distribution ; on attend d'avoir les deux mains.
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
            raise IllegalMove("aucune action attendue")
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
                raise IllegalMove("impossible de se coucher sans mise a suivre")
            state.fold()
        elif action.kind is Kind.CHECK_CALL:
            state.check_or_call()
        else:
            if not legal.can_raise:
                raise IllegalMove("relance impossible")
            if not legal.min_raise_to <= action.amount <= legal.max_raise_to:
                raise IllegalMove(
                    f"relance a {action.amount} hors de "
                    f"[{legal.min_raise_to}, {legal.max_raise_to}]"
                )
            state.complete_bet_or_raise_to(action.amount)
        self._burn()
        self.history.append(("action", seat, action.kind.value, action.amount))

    def deal_board(self, cards: str) -> None:
        if self.pending is not Pending.BOARD:
            raise IllegalMove("aucune carte du board attendue")
        parsed = parse_cards(cards)
        self._check_new_cards(parsed, self.board_cards_needed)
        self._state.deal_board("".join(parsed))
        self._burn()
        self.board.extend(parsed)
        self.history.append(("board", parsed))

    def show(self, seat: int, cards: str | None = None) -> None:
        """Montre la main du siege. `cards` est requis si elle etait cachee."""
        if self.pending is not Pending.SHOWDOWN or self.to_act != seat:
            raise IllegalMove(f"ce n'est pas au siege {seat} de montrer")
        if self.hole[seat] is None:
            if cards is None:
                raise IllegalMove("cartes inconnues : il faut les fournir")
            parsed = parse_cards(cards)
            self._check_new_cards(parsed, 2)
            self.hole[seat] = parsed
            self._state.show_or_muck_hole_cards("".join(parsed))
        else:
            if cards is not None and parse_cards(cards) != self.hole[seat]:
                raise IllegalMove("cartes differentes de celles distribuees")
            self._state.show_or_muck_hole_cards(True)
        self._burn()
        self.history.append(("show", seat, self.hole[seat]))

    def muck(self, seat: int) -> None:
        """Le siege jette sa main sans la montrer et abandonne le pot."""
        if self.pending is not Pending.SHOWDOWN or self.to_act != seat:
            raise IllegalMove(f"ce n'est pas au siege {seat} de montrer")
        self._state.show_or_muck_hole_cards(False)
        self._burn()
        self.history.append(("muck", seat))


class Dealer:
    """Donneur simule : tire les cartes d'un paquet melange."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def advance(self, hand: Hand) -> None:
        """Fait avancer la main jusqu'a la prochaine action ou jusqu'a la fin."""
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
