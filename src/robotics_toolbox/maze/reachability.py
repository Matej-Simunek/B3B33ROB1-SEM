"""Offline reachability check for CRS93 using its analytical IK (no robot connection)."""
import numpy as np
from ctu_crs import CRS93

# Flange pointing down (z axis of the flange towards the table).
R_DOWN = np.diag([1.0, -1.0, -1.0])


def flange_pose(position: np.ndarray, yaw: float = 0.0, tilt: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ct, st = np.cos(tilt), np.sin(tilt)
    ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ R_DOWN @ ry
    pose[:3, 3] = position
    return pose


def reachable_solutions(robot: CRS93, position, yaws=np.linspace(-np.pi, np.pi, 24, endpoint=False)):
    """Joint solutions within limits for a downward-pointing flange at `position` (robot base frame)."""
    sols = []
    for yaw in yaws:
        for q in robot.ik(flange_pose(np.asarray(position, float), yaw)):
            if robot.in_limits(q):
                sols.append(q)
    return sols


def is_reachable(robot: CRS93, position) -> bool:
    return len(reachable_solutions(robot, position)) > 0
