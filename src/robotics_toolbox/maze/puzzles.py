"""Puzzle models (docs/ROB1-2025-hw/drawings/puzzles.pdf) and the hoop path along them.

Puzzle frame: origin in the foot of the rod, z up, the long arm of the bent puzzles points to +x.
Only centerlines of the 10x10 mm rods are modelled (the hoop is much wider than the rod).
Puzzle E (30/45 degree spatial bends) is not modelled.
"""
import numpy as np

from robotics_toolbox.maze.constants import END_CLEARANCE, PLATFORM_HEIGHT, PUZZLE_BASE_HEIGHT

ROD_WIDTH = 0.010
APPROACH = 0.06  # start of the hoop path beyond the free end of the rod
END_Z = PLATFORM_HEIGHT + END_CLEARANCE - PUZZLE_BASE_HEIGHT  # final hoop height in the puzzle frame (0.001 m)
ORIENTATION_SIGMA = 0.015  # [m] smoothing length of the hoop orientation at kinks
Y_SIGN_C = 1.0  # direction of the second bend of puzzle C (not verified, see PLAN.md)


def _arc(center, radius, a0, a1, n=24):
    a = np.linspace(a0, a1, n)
    return np.stack([center[0] + radius * np.cos(a), np.zeros(n), center[1] + radius * np.sin(a)], axis=1)


def _keypoints() -> dict[str, np.ndarray]:
    p = lambda *pts: np.array(pts, dtype=float)  # noqa: E731
    d_arc = _arc((0.035, 0.105), 0.05, np.pi, np.pi / 2)
    return {
        "A": p((0, 0, 0), (0, 0, 0.200)),
        "B": p((0, 0, 0), (0, 0, 0.080), (0.070, 0, 0.150), (0.070, 0, 0.200)),
        "C": p((0, 0, 0), (0, 0, 0.050), (-0.050, 0, 0.100), (-0.050, 0.050 * Y_SIGN_C, 0.150),
               (-0.050, 0.050 * Y_SIGN_C, 0.200)),
        "D": np.vstack([p((0, 0, 0), (0, 0, 0.030), (-0.015, 0, 0.045), (-0.015, 0, 0.105)), d_arc[1:],
                        p((0.085, 0, 0.155))]),
    }


PUZZLES = tuple(_keypoints())
# A, B, C can be threaded with the ring staying horizontal (rod slopes are <= 45 deg, the ring is 62 mm wide for a
# 10 mm rod); only D has a horizontal arm at its free end and needs the ring to turn (the task variant with rotation).
HORIZONTAL_HOOP = {"A", "B", "C"}


def centerline(name: str, step: float = 0.002) -> np.ndarray:
    """Densely sampled rod centerline from the foot to the free end, (N, 3) [m]."""
    pts = _keypoints()[name]
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(out)


def rod_length(name: str) -> float:
    return float(np.sum(np.linalg.norm(np.diff(centerline(name), axis=0), axis=1)))


def _smooth_directions(dirs: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing of unit vectors along the path (ends are padded by their own direction)."""
    half = int(3 * sigma_samples) + 1
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.vstack([np.repeat(dirs[:1], half, axis=0), dirs, np.repeat(dirs[-1:], half, axis=0)])
    out = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def hoop_path(name: str, handle_hint=(0.0, 1.0, 0.0), step: float = 0.004, horizontal: bool | None = None) -> list[np.ndarray]:
    """Poses T_puzzle_hoop of the ring center from beyond the free end down to the foot.

    Hoop frame: z = ring normal = direction of travel, x = from ring center towards the handle,
    kept as close to `handle_hint` as possible (rotation-minimizing frame).
    """
    line = centerline(name, step)
    tangent_out = line[-1] - line[-2]
    tangent_out /= np.linalg.norm(tangent_out)
    start = line[-1] + APPROACH * tangent_out
    positions = np.vstack([start, line[::-1]])
    # resample uniformly along the path
    seg = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    grid = np.arange(0.0, s[-1], step)
    positions = np.stack([np.interp(grid, s, positions[:, k]) for k in range(3)], axis=1)
    positions[:, 2] = np.maximum(positions[:, 2], END_Z)
    positions = positions[positions[:, 2] > END_Z + 1e-9] if len(positions) > 1 else positions
    positions = np.vstack([positions, [line[0][0], line[0][1], END_Z]])

    if horizontal is None:
        horizontal = name in HORIZONTAL_HOOP
    if horizontal:  # ring plane stays horizontal, only its center follows the rod
        z_axes = np.tile([0.0, 0.0, -1.0], (len(positions), 1))
    else:
        z_axes = np.gradient(positions, axis=0)
        z_axes /= np.linalg.norm(z_axes, axis=1, keepdims=True)
        z_axes = _smooth_directions(z_axes, ORIENTATION_SIGMA / step)
    poses, x_prev = [], np.asarray(handle_hint, float)
    for pos, z in zip(positions, z_axes):
        x = x_prev - np.dot(x_prev, z) * z
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        x /= np.linalg.norm(x)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
        pose[:3, 3] = pos
        poses.append(pose)
        x_prev = x
    return poses
