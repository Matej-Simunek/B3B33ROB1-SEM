"""Pose of the maze base plate in the robot frame, from ArUco markers + camera model.

Observations are expressed in the marker plane of the robot frame (x, y [m], yaw [rad], CCW seen from above),
so the table motion fit (table_motion.py) works directly in robot coordinates.

Plate frame: origin at the midpoint between markers 1 and 2 (z = marker plane), x along the TL->TR edge of
marker 1, z up.
"""
from dataclasses import dataclass, field

import numpy as np

from robotics_toolbox.maze.aruco_base import Marker
from robotics_toolbox.maze.camera_model import CameraModel
from robotics_toolbox.maze.constants import PLATFORM_HEIGHT
from robotics_toolbox.maze.table_motion import TableMotion, _rot


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def marker_world_observation(marker: Marker, camera: CameraModel) -> tuple[np.ndarray, float]:
    """Center (x, y) and yaw of a marker in the robot frame at the height of the marker plane."""
    c = camera.backproject(marker.corners, PLATFORM_HEIGHT)[:, :2]
    edge = c[1] - c[0]
    return c.mean(axis=0), float(np.arctan2(edge[1], edge[0]))


@dataclass
class PlateLayout:
    """Where the plate origin is, expressed in each marker's own frame, and the marker yaw offsets."""

    origin_in_marker: dict = field(default_factory=lambda: {1: np.array([0.0375, -0.0375]), 2: np.array([-0.0375, 0.0375])})
    yaw_offset: dict = field(default_factory=lambda: {1: 0.0, 2: 0.0})  # marker yaw minus plate yaw

    @classmethod
    def learn(cls, observations: list[dict]) -> "PlateLayout":
        """Estimate the layout from frames in which markers 1 and 2 are both visible (median over frames)."""
        both = [o for o in observations if 1 in o and 2 in o]
        if not both:
            return cls()
        off1, phi2 = [], []
        for o in both:
            (c1, y1), (c2, y2) = o[1], o[2]
            off1.append(_rot(-y1) @ (c2 - c1) / 2)
            phi2.append(_wrap(y2 - y1))
        o1 = np.median(off1, axis=0)
        ph = float(np.median(phi2))
        return cls({1: o1, 2: -(_rot(-ph) @ o1)}, {1: 0.0, 2: ph})

    def plate_pose(self, obs: dict) -> np.ndarray:
        """T_robot_plate from the visible markers (averages over them)."""
        origins, yaws = [], []
        for i, (c, yaw) in obs.items():
            plate_yaw = yaw - self.yaw_offset[i]
            origins.append(c + _rot(plate_yaw) @ self.origin_in_marker[i])
            yaws.append(plate_yaw)
        yaw = float(np.angle(np.mean(np.exp(1j * np.array(yaws)))))
        pose = np.eye(4)
        pose[:2, :2] = _rot(yaw)
        pose[:2, 3] = np.mean(origins, axis=0)
        pose[2, 3] = PLATFORM_HEIGHT
        return pose


class PlateTracker:
    """Predicts the plate pose on the rotating table."""

    def __init__(self, motion: TableMotion, layout: PlateLayout):
        self.motion, self.layout = motion, layout

    def pose(self, t: float) -> np.ndarray:
        obs = {i: self.motion.marker_at(i, t) for i in self.motion.marker_offsets}
        return self.layout.plate_pose(obs)

    def axis_xy(self) -> np.ndarray:
        return self.motion.center


PUZZLE_FOOT_OFFSET = np.array([0.0, 0.0])  # puzzle foot in the plate frame (xy) [m]; tune from real images


def puzzle_frame(T_robot_plate: np.ndarray, foot_offset=PUZZLE_FOOT_OFFSET) -> np.ndarray:
    """T_robot_puzzle: the puzzle foot is PUZZLE_RAISE above the marker plane."""
    from robotics_toolbox.maze.constants import PUZZLE_RAISE

    local = np.eye(4)
    local[:3, 3] = [foot_offset[0], foot_offset[1], PUZZLE_RAISE]
    return T_robot_plate @ local
