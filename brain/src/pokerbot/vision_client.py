"""Ask the vision service which cards it sees.

The message format is shown in protocol/cards.json at the root of the
repository. A reading is used only when the vision trusts every card of it,
and only once it has come back unchanged several times in a row.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from pokerbot.engine import FULL_DECK

PROTOCOL_VERSION = 1


class VisionError(Exception):
    """The vision service cannot be reached or sent something unusable."""


@dataclass(frozen=True)
class SeenCard:
    name: str
    confidence: float
    doubt: str


def parse_cards_message(message: dict) -> list[SeenCard]:
    """Cards of a message, from left to right in the picture."""
    if not isinstance(message, dict) or message.get("version") != PROTOCOL_VERSION:
        raise VisionError(f"unsupported message: {message!r}")
    cards = []
    try:
        for entry in message["cards"]:
            card = SeenCard(str(entry["name"]), float(entry["confidence"]), str(entry["doubt"]))
            if card.name not in FULL_DECK:
                raise VisionError(f"unknown card name: {card.name!r}")
            cards.append(card)
    except (KeyError, TypeError, ValueError) as error:
        raise VisionError(f"malformed message: {message!r}") from error
    return cards


def fetch_cards(url: str, timeout: float = 2.0) -> list[SeenCard]:
    """Cards currently seen by the vision service at `url`, such as http://host:8770."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/cards", timeout=timeout) as response:
            message = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise VisionError(f"no answer from the vision service at {url}: {error}") from error
    return parse_cards_message(message)


def trusted_cards(cards: list[SeenCard], count: int) -> tuple[str, ...] | None:
    """Card names in a fixed order, or None unless there are exactly `count`
    different cards and the vision doubts none of them."""
    names = tuple(sorted(card.name for card in cards))
    if len(cards) != count or len(set(names)) != count:
        return None
    if any(card.doubt for card in cards):
        return None
    return names


def wait_for_cards(
    fetch: Callable[[], list[SeenCard]],
    count: int = 2,
    repeats: int = 3,
    interval: float = 0.2,
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Wait until `count` trusted cards have been seen `repeats` times in a row.

    Returns the cards as one string, such as '4cJd', ready for the engine.
    """
    deadline = clock() + timeout
    previous = None
    streak = 0
    last_problem = "nothing seen yet"
    while True:
        try:
            cards = fetch()
            names = trusted_cards(cards, count)
            if names is None:
                last_problem = "saw " + (", ".join(f"{c.name} {c.doubt}".strip() for c in cards) or "no card")
        except VisionError as error:
            names = None
            last_problem = str(error)
        streak = streak + 1 if names is not None and names == previous else int(names is not None)
        previous = names
        if streak >= repeats:
            return "".join(names)
        if clock() >= deadline:
            raise VisionError(f"no stable reading of {count} cards after {timeout:g} s ({last_problem})")
        sleep(interval)
