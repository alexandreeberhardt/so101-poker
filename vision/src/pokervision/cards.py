"""Locate the cards in a picture, straighten them and cut out their corners.

Every card carries its rank and suit in two opposite corners. A straightened
card is turned both ways so that each of the two corners is read upright,
which removes the need to know which way up the card lies.
"""

from __future__ import annotations

import cv2
import numpy as np

from pokervision.collect_remote import find_cards

CARD_WIDTH = 250
CARD_HEIGHT = 350
CORNER_WIDTH = 64
CORNER_HEIGHT = 112


def _ordered_corners(rectangle: tuple) -> np.ndarray:
    """Corners of a rotated rectangle, clockwise, starting on a short side."""
    points = cv2.boxPoints(rectangle)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    first_side = np.linalg.norm(points[1] - points[0])
    second_side = np.linalg.norm(points[2] - points[1])
    if first_side > second_side:
        points = np.roll(points, -1, axis=0)
    return points.astype(np.float32)


def straighten(frame: np.ndarray, rectangle: tuple) -> np.ndarray:
    """The card inside `rectangle`, as an upright CARD_WIDTH x CARD_HEIGHT image."""
    target = np.array(
        [[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(_ordered_corners(rectangle), target)
    return cv2.warpPerspective(frame, transform, (CARD_WIDTH, CARD_HEIGHT))


def corner_crops(card: np.ndarray) -> list[np.ndarray]:
    """The two index corners of a straightened card, both upright."""
    turned = cv2.rotate(card, cv2.ROTATE_180)
    return [image[:CORNER_HEIGHT, :CORNER_WIDTH] for image in (card, turned)]


def extract_cards(frame: np.ndarray) -> list[np.ndarray]:
    """Straightened cards found in a BGR frame, ordered from left to right."""
    rectangles = sorted(find_cards(frame), key=lambda rectangle: rectangle[0][0])
    return [straighten(frame, rectangle) for rectangle in rectangles]
