"""Camera -> robot calibration from the hoop seen at known flange poses.

Samples: (q, pixel) where pixel is the center of the hoop ring in the image and q the joint configuration
(forward kinematics gives the flange pose). Unknowns: camera pose, focal length and the hoop center in the
flange frame (TCP translation). Weak priors keep the problem well posed (camera height, TCP height).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robotics_toolbox.maze.camera_model import CameraModel
from robotics_toolbox.maze.constants import CAMERA_HEIGHT

TCP_GUESS = np.array([0.15, 0.0, 0.06])  # ring center in the flange frame [m]; refined by the calibration
F_GUESS = 4500.0


@dataclass
class CalibrationResult:
    camera: CameraModel
    tcp: np.ndarray
    rms_px: float
    max_px: float


def _ring_centers(robot, qs, tcp) -> np.ndarray:
    return np.array([(robot.fk(q) @ np.append(tcp, 1.0))[:3] for q in qs])


def _initial_guesses(points_world: np.ndarray):
    center_xy = points_world[:, :2].mean(axis=0)
    for psi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        # camera looks down: x_cam = world x, y_cam = -world y, z_cam = -world z, then yaw psi
        r = Rotation.from_euler("x", np.pi).as_matrix() @ Rotation.from_euler("z", psi).as_matrix()
        c = np.array([*center_xy, CAMERA_HEIGHT])
        yield np.concatenate([Rotation.from_matrix(r).as_rotvec(), -r @ c, [F_GUESS]])


def fit_calibration(robot, qs, pixels, tcp0=TCP_GUESS, height_sigma=0.02, tcp_z_sigma=0.03) -> CalibrationResult:
    qs, pixels = np.asarray(qs, float), np.asarray(pixels, float)
    assert len(qs) == len(pixels) >= 8, "need at least 8 samples"

    def residuals(x):
        cam, tcp = CameraModel.from_params(x[:7]), x[7:10]
        err = (cam.project(_ring_centers(robot, qs, tcp)) - pixels).ravel()
        priors = [(cam.center[2] - CAMERA_HEIGHT) / height_sigma * 1.0, (tcp[2] - tcp0[2]) / tcp_z_sigma]
        return np.concatenate([err, priors])

    best = None
    for guess in _initial_guesses(_ring_centers(robot, qs, tcp0)):
        sol = least_squares(residuals, np.concatenate([guess, tcp0]), x_scale=np.r_[np.ones(3), np.ones(3), 1000, np.ones(3)])
        if best is None or sol.cost < best.cost:
            best = sol
    cam, tcp = CameraModel.from_params(best.x[:7]), best.x[7:10]
    err = np.linalg.norm(cam.project(_ring_centers(robot, qs, tcp)) - pixels, axis=1)
    return CalibrationResult(cam, tcp, float(np.sqrt(np.mean(err**2))), float(err.max()))
