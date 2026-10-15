from pathlib import Path

import cv2
import numpy as np
import pytest

from robotics_toolbox.maze.detect import detect_hoop, hoop_fully_visible

BLUE_BGR = (150, 70, 10)


def scene(center, radius=175, distractors=True):
    img = np.full((1200, 1920, 3), 225, np.uint8)
    cv2.circle(img, center, radius, BLUE_BGR, 7)
    if distractors:
        cv2.circle(img, (center[0] + 330, center[1] - 100), 70, BLUE_BGR, -1)  # blue blob
        cv2.rectangle(img, (center[0] - 6, center[1] - radius - 170), (center[0] + 6, center[1] - radius), BLUE_BGR, -1)  # handle
    return img


def test_ring_center_is_found_despite_other_blue_objects():
    c = detect_hoop(scene((900, 650)))
    assert c.center == pytest.approx([900, 650], abs=1.0)
    assert c.radius == pytest.approx(175, abs=6.0)
    assert hoop_fully_visible(c)


def test_ring_cut_by_the_image_border_is_rejected():
    c = detect_hoop(scene((90, 650), distractors=False))
    assert c is None or not hoop_fully_visible(c)


def test_ring_partly_hidden_is_rejected():
    img = scene((900, 650), distractors=False)
    cv2.rectangle(img, (900 - 200, 650 - 200), (900 + 200, 650 + 10), (225, 225, 225), -1)  # hide the upper half
    c = detect_hoop(img)
    assert c is None or not hoop_fully_visible(c)


def test_real_photo():
    path = Path(__file__).resolve().parents[1] / "data" / "images" / "01_hoop_above_base.png"
    c = detect_hoop(cv2.imread(str(path)))
    assert c.center == pytest.approx([878, 665], abs=3.0)
    assert hoop_fully_visible(c)


def test_dark_robot_barrel_is_not_taken_for_the_ring():
    img = np.full((1200, 1920, 3), 225, np.uint8)
    cv2.circle(img, (1500, 1000), 200, (95, 75, 60), -1)  # dark bluish-grey wrist barrel, V < 100
    assert detect_hoop(img) is None
