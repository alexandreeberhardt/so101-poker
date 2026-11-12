import cv2
import pytest
from conftest import draw_card

from pokervision.cards import (
    CARD_HEIGHT,
    CARD_WIDTH,
    CORNER_HEIGHT,
    CORNER_WIDTH,
    corner_crops,
    extract_cards,
    straighten,
)
from pokervision.collect_remote import find_cards


def centres(frame):
    return sorted((round(x), round(y)) for (x, y), _, _ in find_cards(frame))


def test_empty_table_has_no_card(table):
    assert find_cards(table) == []


def test_two_separate_cards_are_found(table):
    draw_card(table, (600, 500))
    draw_card(table, (1300, 520))
    found = centres(table)
    assert len(found) == 2
    assert found[0] == pytest.approx((600, 500), abs=6)
    assert found[1] == pytest.approx((1300, 520), abs=6)


def test_cards_that_almost_touch_stay_two_cards(table):
    draw_card(table, (760, 550))
    draw_card(table, (760 + 360 + 12, 550))
    assert len(find_cards(cv2.GaussianBlur(table, (15, 15), 0))) == 2


@pytest.mark.parametrize("angle", [-40, -15, 20, 35, 90])
def test_tilted_card_is_found(table, angle):
    draw_card(table, (960, 540), angle)
    assert len(find_cards(table)) == 1


def test_shapes_that_are_not_cards_are_ignored(table):
    draw_card(table, (400, 500), size=(80, 700))
    draw_card(table, (1200, 500), size=(120, 170))
    draw_card(table, (1600, 300), size=(400, 400))
    assert find_cards(table) == []


def test_straightened_card_is_upright_whatever_its_angle(table):
    for angle in (0, 30, 90, -60):
        frame = table.copy()
        draw_card(frame, (960, 540), angle)
        (rectangle,) = find_cards(frame)
        card = straighten(frame, rectangle)
        assert card.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
        # The middle of the card is white: the long side was laid out vertically.
        assert card[40:-40, 30:-30].mean() > 235


def test_a_mark_in_a_card_corner_lands_in_one_of_the_two_corner_crops(table):
    draw_card(table, (960, 540))
    left, top = 960 - 180, 540 - 250
    cv2.rectangle(table, (left + 25, top + 30), (left + 60, top + 110), (0, 0, 0), -1)
    (card,) = extract_cards(table)
    darkness = [crop.shape == (CORNER_HEIGHT, CORNER_WIDTH, 3) and (crop < 60).mean() for crop in corner_crops(card)]
    assert max(darkness) > 0.05
    assert min(darkness) < 0.01


def test_cards_come_out_from_left_to_right(table):
    draw_card(table, (1400, 500))
    draw_card(table, (500, 600))
    cv2.circle(table, (500, 600), 60, (0, 0, 255), -1)
    first, second = extract_cards(table)

    def red_share(card):
        return float(((card[..., 2] > 200) & (card[..., 1] < 80)).mean())

    assert red_share(first) > 0.05
    assert red_share(second) == 0
