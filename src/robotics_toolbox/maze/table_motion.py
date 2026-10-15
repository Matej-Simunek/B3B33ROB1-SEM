"""Estimate the rotation of the table from ArUco markers tracked over time.

Coordinates are any right-handed 2D frame (image with y flipped up, or the robot xy plane): angles are
positive counter-clockwise. The pipeline uses the robot xy plane (see plate_pose.py).

Model: the plate rotates rigidly about an axis c. For marker i observed at time t_k
    p_ik = c + R(theta_k) d_i
where theta_k is the measured marker yaw change since the first observation, c the axis and d_i the
marker offset from the axis at the start. This is linear in (c, d_i) so even a short arc is enough.
"""
from dataclasses import dataclass

import numpy as np

from robotics_toolbox.maze.aruco_base import Marker


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def marker_observation(marker: Marker) -> tuple[np.ndarray, float]:
    """Center (x, y-up) and yaw of a marker in the image."""
    corners = marker.corners * np.array([1.0, -1.0])
    edge = corners[1] - corners[0]
    return corners.mean(axis=0), float(np.arctan2(edge[1], edge[0]))


@dataclass
class TableMotion:
    center: np.ndarray  # rotation axis in the image (x, y-up) [px]
    omega: float  # [rad/s], + = counter-clockwise seen by the camera
    t_ref: float
    marker_offsets: dict  # id -> offset from the axis at t_ref [px]
    marker_yaw_ref: dict  # id -> marker yaw at t_ref [rad]
    rms_px: float  # position residual
    yaw_residual_rad: float
    n_frames: int

    @property
    def rotating(self) -> bool:
        return abs(self.omega) > np.deg2rad(0.2) and self.yaw_residual_rad < abs(self.omega) * 2

    @property
    def direction(self) -> str:
        if not self.rotating:
            return "stationary"
        return "left (counter-clockwise from camera)" if self.omega > 0 else "right (clockwise from camera)"

    def marker_at(self, marker_id: int, t: float) -> tuple[np.ndarray, float]:
        """Predicted center (x, y-up) and yaw of a marker at time t."""
        a = self.omega * (t - self.t_ref)
        return self.center + _rot(a) @ self.marker_offsets[marker_id], self.marker_yaw_ref[marker_id] + a


def fit_table_motion(observations: list[tuple[float, dict]]) -> TableMotion:
    """observations: [(t, {marker_id: (center_xy_up, yaw)}), ...] sorted by time."""
    ids = sorted({i for _, obs in observations for i in obs})
    first = {i: next(o[i] for _, o in observations if i in o) for i in ids}
    t_ref = next(t for t, o in observations if ids[0] in o) if ids else 0.0

    # unwrapped yaw change per marker, then omega from a line fit through all markers
    ts, dths, rows = [], [], []
    for i in ids:
        pts = [(t, o[i]) for t, o in observations if i in o]
        yaw = np.unwrap([p[1][1] for p in pts])
        for (t, (c, _)), y in zip(pts, yaw):
            ts.append(t)
            dths.append(y - yaw[0])
            rows.append((i, t, c, y - yaw[0]))
    ts, dths = np.array(ts), np.array(dths)
    if len(ts) < 3:
        raise ValueError("not enough marker observations")
    omega = float(np.polyfit(ts, dths, 1)[0]) if np.ptp(ts) > 0 else 0.0
    yaw_res = float(np.std(dths - np.polyval(np.polyfit(ts, dths, 1), ts)))

    # linear least squares for c and the offsets d_i (use the measured yaw change per marker)
    n = len(ids)
    A, b = [], []
    for i, t, c, dth in rows:
        r = _rot(dth)
        for axis in range(2):
            row = np.zeros(2 + 2 * n)
            row[axis] = 1.0
            row[2 + 2 * ids.index(i): 4 + 2 * ids.index(i)] = r[axis]
            A.append(row)
            b.append(c[axis])
    x, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    rms = float(np.sqrt(np.mean((np.array(A) @ x - np.array(b)) ** 2)))
    center = x[:2]
    offsets = {i: x[2 + 2 * k: 4 + 2 * k] for k, i in enumerate(ids)}
    # offsets are at each marker's own first observation; shift to the common reference time
    t_first = {i: next(t for t, o in observations if i in o) for i in ids}
    offsets = {i: _rot(omega * (t_ref - t_first[i])) @ offsets[i] for i in ids}
    yaw_ref = {i: first[i][1] + omega * (t_ref - t_first[i]) for i in ids}
    return TableMotion(center, omega, t_ref, offsets, yaw_ref, rms, yaw_res, len(observations))
