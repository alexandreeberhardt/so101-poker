import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pokerbot.engine import Hand, Pending
from pokerbot.vision_client import SeenCard, VisionError, fetch_cards, parse_cards_message, trusted_cards, wait_for_cards

EXAMPLE = json.loads((Path(__file__).parents[2] / "protocol" / "cards.json").read_text())
JACK = SeenCard("Jd", 0.98, "")
FOUR = SeenCard("4c", 0.95, "")
SHAKY = SeenCard("4c", 0.41, "low confidence")


def test_shared_example_is_understood():
    assert parse_cards_message(EXAMPLE) == [JACK, FOUR]


@pytest.mark.parametrize(
    "message",
    [
        {**EXAMPLE, "version": 2},
        {"version": 1},
        {"version": 1, "cards": [{"name": "Jd"}]},
        {"version": 1, "cards": [{"name": "1x", "confidence": 0.9, "doubt": ""}]},
        {"version": 1, "cards": "Jd"},
        [],
    ],
)
def test_unusable_messages_are_rejected(message):
    with pytest.raises(VisionError):
        parse_cards_message(message)


def test_only_a_full_and_undoubted_reading_is_trusted():
    assert trusted_cards([JACK, FOUR], 2) == ("4c", "Jd")
    assert trusted_cards([FOUR, JACK], 2) == ("4c", "Jd")
    assert trusted_cards([JACK], 2) is None
    assert trusted_cards([JACK, SHAKY], 2) is None
    assert trusted_cards([JACK, JACK], 2) is None
    assert trusted_cards([JACK, FOUR, SeenCard("9s", 0.9, "")], 2) is None


class Script:
    """Answers of the vision, one per call; the last one repeats."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0
        self.now = 0.0

    def fetch(self):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return answer

    def sleep(self, seconds):
        self.now += seconds

    def clock(self):
        return self.now

    def wait(self, **options):
        return wait_for_cards(self.fetch, sleep=self.sleep, clock=self.clock, **options)


def test_waits_for_three_identical_readings():
    script = Script([], [JACK], [JACK, SHAKY], [JACK, FOUR])
    assert script.wait() == "4cJd"
    assert script.calls == 6


def test_a_change_of_cards_restarts_the_count():
    other = [JACK, SeenCard("9s", 0.9, "")]
    script = Script([JACK, FOUR], [JACK, FOUR], other, other, other)
    assert script.wait() == "9sJd"
    assert script.calls == 5


def test_a_failed_request_restarts_the_count():
    script = Script([JACK, FOUR], [JACK, FOUR], VisionError("down"), [JACK, FOUR])
    assert script.wait() == "4cJd"
    assert script.calls == 6


def test_gives_up_after_the_timeout_and_says_what_it_saw():
    script = Script([JACK, SHAKY])
    with pytest.raises(VisionError, match="low confidence"):
        script.wait(timeout=1.0)
    assert script.now == pytest.approx(1.0, abs=0.21)


def serve(status, body):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_cards_fetched_over_http_can_be_dealt_to_the_engine():
    server, url = serve(200, json.dumps(EXAMPLE).encode())
    try:
        cards = wait_for_cards(lambda: fetch_cards(url), interval=0)
    finally:
        server.shutdown()
    hand = Hand((20, 20), button=0)
    hand.deal_hole(0, None)
    hand.deal_hole(1, cards)
    assert hand.hole[1] == ("4c", "Jd")
    assert hand.pending is Pending.ACTION


@pytest.mark.parametrize("status, body", [(503, b'{"error": "no picture available"}'), (200, b"not json")])
def test_http_failures_become_vision_errors(status, body):
    server, url = serve(status, body)
    try:
        with pytest.raises(VisionError):
            fetch_cards(url)
    finally:
        server.shutdown()


def test_unreachable_service_is_a_vision_error():
    with pytest.raises(VisionError):
        fetch_cards("http://127.0.0.1:9", timeout=0.5)
