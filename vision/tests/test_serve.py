import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from conftest import draw_card

from pokervision.read_cards import CardReader, CardReading
from pokervision.serve import cards_message, make_server

EXAMPLE = Path(__file__).parents[2] / "protocol" / "cards.json"


@pytest.fixture
def reader(untrained_weights):
    return CardReader(untrained_weights)


def serve(source, reader):
    server = make_server(source, reader, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_message_reproduces_the_shared_example():
    readings = [CardReading("Jd", 0.98, "", (500.0, 400.0)), CardReading("4c", 0.95, "", (1200.0, 400.0))]
    assert cards_message(readings, "robot_hole") == json.loads(EXAMPLE.read_text())


def test_server_answers_with_one_entry_per_card(table, reader):
    draw_card(table, (500, 540))
    draw_card(table, (1400, 540))
    server, url = serve(lambda: table, reader)
    try:
        with urllib.request.urlopen(url + "/cards", timeout=5) as response:
            message = json.loads(response.read())
    finally:
        server.shutdown()
    assert message["version"] == 1
    assert len(message["cards"]) == 2
    assert set(message["cards"][0]) == {"name", "confidence", "doubt"}


def test_server_says_when_there_is_no_picture(reader):
    server, url = serve(lambda: None, reader)
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/cards", timeout=5)
    finally:
        server.shutdown()
    assert error.value.code == 503
    assert "error" in json.loads(error.value.read())


def test_unknown_path_is_a_404(table, reader):
    server, url = serve(lambda: table, reader)
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/other", timeout=5)
    finally:
        server.shutdown()
    assert error.value.code == 404
