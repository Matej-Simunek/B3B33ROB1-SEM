"""Pinhole camera with a pose in the robot base frame (no lens distortion; the lens is long, ~24 deg HFOV)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robotics_toolbox.maze.constants import IMAGE_SIZE


@dataclass
class CameraModel:
    f: float  # focal length [px]
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # world -> camera translation
    cx: float = IMAGE_SIZE[0] / 2
    cy: float = IMAGE_SIZE[1] / 2

    @classmethod
    def from_params(cls, params: np.ndarray) -> "CameraModel":
        """params = [rvec(3), tvec(3), f]"""
        return cls(float(params[6]), Rotation.from_rotvec(params[:3]).as_matrix(), np.asarray(params[3:6], float))

    def to_params(self) -> np.ndarray:
        return np.concatenate([Rotation.from_matrix(self.R).as_rotvec(), self.t, [self.f]])

    @property
    def center(self) -> np.ndarray:
        """Camera center in the world frame."""
        return -self.R.T @ self.t

    def project(self, points: np.ndarray) -> np.ndarray:
        pc = np.atleast_2d(points) @ self.R.T + self.t
        return np.stack([self.f * pc[:, 0] / pc[:, 2] + self.cx, self.f * pc[:, 1] / pc[:, 2] + self.cy], axis=1)

    def backproject(self, pixels: np.ndarray, z: float) -> np.ndarray:
        """World points on the plane Z=z that project to the given pixels."""
        pixels = np.atleast_2d(pixels).astype(float)
        rays_c = np.stack([(pixels[:, 0] - self.cx) / self.f, (pixels[:, 1] - self.cy) / self.f,
                           np.ones(len(pixels))], axis=1)
        rays_w = rays_c @ self.R  # R^T applied to row vectors
        c = self.center
        lam = (z - c[2]) / rays_w[:, 2]
        return c + lam[:, None] * rays_w

    def save(self, path: Path, **extra) -> None:
        data = {"f": self.f, "R": self.R.tolist(), "t": self.t.tolist(), "cx": self.cx, "cy": self.cy, **extra}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> tuple["CameraModel", dict]:
        data = json.loads(Path(path).read_text())
        cam = cls(data["f"], np.array(data["R"]), np.array(data["t"]), data["cx"], data["cy"])
        return cam, {k: v for k, v in data.items() if k not in ("f", "R", "t", "cx", "cy")}
