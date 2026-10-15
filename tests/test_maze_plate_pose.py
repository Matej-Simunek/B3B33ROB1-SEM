import numpy as np
import pytest

from robotics_toolbox.maze.aruco_base import Marker
from robotics_toolbox.maze.constants import PLATFORM_HEIGHT
from robotics_toolbox.maze.plate_pose import PlateLayout, PlateTracker, marker_world_observation
from robotics_toolbox.maze.table_motion import _rot, fit_table_motion
from tests.maze_helpers import true_camera

AXIS = np.array([0.50, 0.05])
OFFSETS = {1: np.array([-0.0375, 0.0375]) + np.array([0.01, 0.0]), 2: np.array([0.0375, -0.0375]) + np.array([0.01, 0.0])}


def corners_pixels(camera, center, yaw, size=0.04):
    s = size / 2
    local = np.array([[-s, s], [s, s], [s, -s], [-s, -s]])
    xy = center + local @ _rot(yaw).T
    return camera.project(np.column_stack([xy, np.full(4, PLATFORM_HEIGHT)]))


def scene(camera, t, omega=np.deg2rad(3.0), plate_yaw0=0.3, markers=(1, 2)):
    angle = omega * t
    out = []
    for i in markers:
        center = AXIS + _rot(angle) @ (OFFSETS[i] + [0.0, 0.0])
        out.append(Marker(i, corners_pixels(camera, center, plate_yaw0 + angle)))
    return out


def test_world_observation_matches_truth():
    cam = true_camera()
    (m,) = scene(cam, 0.0, markers=(1,))
    center, yaw = marker_world_observation(m, cam)
    assert center == pytest.approx(AXIS + OFFSETS[1], abs=1e-6)
    assert yaw == pytest.approx(0.3, abs=1e-6)


def test_tracker_predicts_future_plate_pose_even_with_one_marker_visible():
    cam, omega = true_camera(), np.deg2rad(-3.0)
    frames = [(t, {m.id: marker_world_observation(m, cam) for m in scene(cam, t, omega)}) for t in np.arange(0, 10, 0.5)]
    layout = PlateLayout.learn([o for _, o in frames])
    motion = fit_table_motion(frames)
    assert motion.omega == pytest.approx(omega, abs=1e-4)
    assert motion.center == pytest.approx(AXIS, abs=2e-3)
    tracker = PlateTracker(motion, layout)

    t = 25.0
    truth = {m.id: marker_world_observation(m, cam) for m in scene(cam, t, omega)}
    expected = layout.plate_pose(truth)
    assert tracker.pose(t) == pytest.approx(expected, abs=2e-3)
    # one visible marker gives the same plate pose
    assert layout.plate_pose({1: truth[1]}) == pytest.approx(expected, abs=1e-3)
    assert layout.plate_pose({2: truth[2]}) == pytest.approx(expected, abs=1e-3)
