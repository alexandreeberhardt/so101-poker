from pathlib import Path

import cv2
import pytest
import torch
from conftest import draw_card

from pokervision.card_model import RANKS, SUITS, CornerNet
from pokervision.read_cards import DEFAULT_WEIGHTS, MIN_CONFIDENCE, CardReader

REAL_DATA = Path("../data/cards")


@pytest.fixture
def untrained_weights(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "cards.pt"
    torch.save(CornerNet().state_dict(), path)
    return path


def test_no_card_gives_no_reading(table, untrained_weights):
    assert CardReader(untrained_weights).read(table) == []


def test_one_reading_per_card_from_left_to_right(table, untrained_weights):
    draw_card(table, (1400, 500))
    draw_card(table, (500, 600), angle=20)
    readings = CardReader(untrained_weights).read(table)
    assert len(readings) == 2
    assert readings[0].centre[0] < readings[1].centre[0]
    for reading in readings:
        assert reading.name[0] in RANKS and reading.name[1] in SUITS
        assert 0 <= reading.confidence <= 1


def test_an_untrained_network_is_not_trusted(table, untrained_weights):
    draw_card(table, (960, 540))
    (reading,) = CardReader(untrained_weights).read(table)
    assert reading.confidence < MIN_CONFIDENCE
    assert reading.doubt == "low confidence"
    assert reading.doubtful


def test_threshold_decides_what_is_doubtful(table, untrained_weights):
    draw_card(table, (960, 540))
    (reading,) = CardReader(untrained_weights, min_confidence=0.0).read(table)
    assert not reading.doubtful
    assert str(reading) == f"{reading.name} ({reading.confidence:.2f})"


def test_the_same_card_twice_is_doubtful(table, untrained_weights):
    draw_card(table, (500, 540))
    draw_card(table, (1400, 540))
    readings = CardReader(untrained_weights, min_confidence=0.0).read(table)
    assert readings[0].name == readings[1].name
    assert [reading.doubt for reading in readings] == ["same card twice", "same card twice"]


@pytest.mark.skipif(not (DEFAULT_WEIGHTS.exists() and REAL_DATA.exists()), reason="needs the pictures and the trained weights")
def test_held_out_pictures_are_read_without_mistake():
    reader = CardReader()
    for folder in sorted(REAL_DATA.glob("*test")):
        for line in (folder / "labels.csv").read_text().splitlines()[1:]:
            name, left, right = line.split(",")
            readings = reader.read(cv2.imread(str(folder / name)))
            assert [reading.name for reading in readings] == [left, right], f"{folder.name}/{name}"
            assert not any(reading.doubtful for reading in readings), f"{folder.name}/{name}"
