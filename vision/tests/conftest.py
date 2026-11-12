import cv2
import numpy as np
import pytest
import torch

from pokervision.card_model import CornerNet

FRAME_SIZE = (1080, 1920)
TABLE_GREY = 110
CARD_WHITE = 245
CARD_SIZE = (360, 500)


def draw_card(
    frame: np.ndarray, centre: tuple[float, float], angle: float = 0.0, size: tuple[int, int] = CARD_SIZE
) -> None:
    """Draw a plain white card on the frame."""
    corners = cv2.boxPoints((centre, size, angle)).astype(np.int32)
    cv2.fillConvexPoly(frame, corners, (CARD_WHITE, CARD_WHITE, CARD_WHITE))


@pytest.fixture
def table() -> np.ndarray:
    """An empty grey table, as a BGR picture."""
    return np.full((*FRAME_SIZE, 3), TABLE_GREY, np.uint8)


@pytest.fixture
def untrained_weights(tmp_path):
    """Weights file of a network that has learned nothing."""
    torch.manual_seed(0)
    path = tmp_path / "cards.pt"
    torch.save(CornerNet().state_dict(), path)
    return path
