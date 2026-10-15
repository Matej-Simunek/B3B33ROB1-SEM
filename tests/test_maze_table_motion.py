import cv2
import numpy as np
import pytest

from robotics_toolbox.maze.aruco_base import ARUCO_DICT
from robotics_toolbox.maze.pipeline import estimate_table_motion

AXIS = (1000.0, 700.0)  # image px, y down (as OpenCV sees it)


def render(angle_deg: float) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    canvas = np.full((1200, 1920), 255, np.uint8)
    for marker_id, (cx, cy) in {1: (1097, 511), 2: (1388, 835)}.items():
        m = cv2.aruco.generateImageMarker(d, marker_id, 158)
        canvas[cy - 79: cy + 79, cx - 79: cx + 79] = m
    rot = cv2.getRotationMatrix2D(AXIS, angle_deg, 1.0)  # + = counter-clockwise on screen
    out = cv2.warpAffine(canvas, rot, (1920, 1200), borderValue=255)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("omega_deg", [2.0, -2.0, 0.0])
def test_estimates_axis_and_direction(omega_deg):
    frames = [(t, render(omega_deg * t)) for t in np.arange(0.0, 10.0, 1.0)]
    motion = estimate_table_motion(frames)
    assert np.rad2deg(motion.omega) == pytest.approx(omega_deg, abs=0.1)
    assert (motion.direction == "stationary") == (omega_deg == 0.0)
    if omega_deg != 0.0:
        assert motion.center == pytest.approx([AXIS[0], -AXIS[1]], abs=3.0)
        pos, _ = motion.marker_at(1, 12.0)
        expected = np.array([1097 - AXIS[0], -(511 - AXIS[1])])
        a = np.deg2rad(omega_deg * 12.0)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        assert pos == pytest.approx(np.array([AXIS[0], -AXIS[1]]) + rot @ expected, abs=4.0)
