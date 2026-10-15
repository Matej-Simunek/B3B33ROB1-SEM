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
from robotics_toolbox.maze.constants import CAMERA_HEIGHT, IMAGE_SIZE

TCP_GUESS = np.array([0.15, 0.0, 0.06])  # ring center in the flange frame [m]; refined by the calibration
F_PRIOR, F_SIGMA = 4500.0, 400.0  # from the 40 mm marker seen as ~158 px at ~1.14 m
TCP_BOUNDS = (np.array([-0.5, -0.5, -0.05]), np.array([0.5, 0.5, 0.35]))


@dataclass
class CalibrationResult:
    camera: CameraModel
    tcp: np.ndarray
    rms_px: float
    max_px: float


def _flange_poses(robot, qs) -> np.ndarray:
    return np.array([robot.fk(q) for q in qs])


def _ring_centers_from(poses: np.ndarray, tcp) -> np.ndarray:
    return poses[:, :3, :3] @ np.asarray(tcp, float) + poses[:, :3, 3]


def _ring_centers(robot, qs, tcp) -> np.ndarray:
    return _ring_centers_from(_flange_poses(robot, qs), tcp)


def _initial_guesses(robot, qs, pixels, tcp0, fast=False):
    """Camera/TCP starting points: PnP for a grid of TCP candidates (the TCP is the least known quantity), plus a
    top-down camera at several yaw angles for the nominal TCP. Yields parameter vectors [rvec, tvec, f, tcp]."""
    import cv2

    K = np.array([[F_PRIOR, 0, IMAGE_SIZE[0] / 2], [0, F_PRIOR, IMAGE_SIZE[1] / 2], [0, 0, 1.0]])
    poses = _flange_poses(robot, qs)
    scored = []
    step, tzs = (0.10, (0.03, 0.10)) if fast else (0.05, (0.03, 0.08, 0.14))
    for tx in np.arange(-0.30, 0.301, step):
        for ty in np.arange(-0.30, 0.301, step):
            for tz in tzs:
                tcp = np.array([tx, ty, tz])
                pts = _ring_centers_from(poses, tcp)
                ok, rvec, tvec = cv2.solvePnP(pts, pixels, K, None, flags=cv2.SOLVEPNP_EPNP)
                if not ok:
                    continue
                cam = CameraModel(F_PRIOR, Rotation.from_rotvec(rvec.ravel()).as_matrix(), tvec.ravel())
                if cam.center[2] < 0.5 or cam.center[2] > 2.0 or np.any(cam.R @ pts.T + cam.t[:, None] < 0.1):
                    continue
                err = np.linalg.norm(cam.project(pts) - pixels, axis=1)
                scored.append((float(np.mean(err**2)), np.concatenate([rvec.ravel(), tvec.ravel(), [F_PRIOR], tcp])))
    for _, x in sorted(scored, key=lambda s: s[0])[:2 if fast else 6]:
        yield x
    if fast:
        return
    pts0 = _ring_centers_from(poses, tcp0)
    center_xy = pts0[:, :2].mean(axis=0)
    for psi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        r = Rotation.from_euler("x", np.pi).as_matrix() @ Rotation.from_euler("z", psi).as_matrix()
        c = np.array([*center_xy, CAMERA_HEIGHT])
        yield np.concatenate([Rotation.from_matrix(r).as_rotvec(), -r @ c, [F_PRIOR], tcp0])


def fit_calibration(robot, qs, pixels, tcp0=TCP_GUESS, height_sigma=0.02, tcp_z_sigma=0.05, fast=False) -> CalibrationResult:
    qs, pixels = np.asarray(qs, float), np.asarray(pixels, float)
    assert len(qs) == len(pixels) >= 8, "need at least 8 samples"
    poses = _flange_poses(robot, qs)

    def residuals(x):
        cam, tcp = CameraModel.from_params(x[:7]), x[7:10]
        err = (cam.project(_ring_centers_from(poses, tcp)) - pixels).ravel()
        priors = [(cam.center[2] - CAMERA_HEIGHT) / height_sigma, (cam.f - F_PRIOR) / F_SIGMA,
                  (tcp[2] - tcp0[2]) / tcp_z_sigma]
        return np.concatenate([err, priors])

    lower = np.r_[-np.inf * np.ones(6), 2500.0, TCP_BOUNDS[0]]
    upper = np.r_[np.inf * np.ones(6), 9000.0, TCP_BOUNDS[1]]
    best = None
    for guess in _initial_guesses(robot, qs, pixels, tcp0, fast):
        guess = np.clip(guess, lower + 1e-9, upper - 1e-9)
        sol = least_squares(residuals, guess, bounds=(lower, upper), loss="soft_l1", f_scale=3.0,
                            x_scale=np.r_[np.ones(3), np.ones(3), 1000, np.ones(3)])
        if best is None or sol.cost < best.cost:
            best = sol
    cam, tcp = CameraModel.from_params(best.x[:7]), best.x[7:10]
    err = np.linalg.norm(cam.project(_ring_centers_from(poses, tcp)) - pixels, axis=1)
    return CalibrationResult(cam, tcp, float(np.sqrt(np.mean(err**2))), float(err.max()))


def fit_calibration_robust(robot, qs, pixels, **kwargs):
    """fit_calibration, then drop samples with a large residual (bad detections) and refit.

    Returns (result, kept_mask).
    """
    qs, pixels = np.asarray(qs, float), np.asarray(pixels, float)
    result = fit_calibration(robot, qs, pixels, **kwargs)
    err = np.linalg.norm(result.camera.project(_ring_centers(robot, qs, result.tcp)) - pixels, axis=1)
    keep = err <= max(3.0 * result.rms_px, 3.0)
    if keep.all() or keep.sum() < 8:
        return result, np.ones(len(qs), bool) if keep.all() else keep
    return fit_calibration(robot, qs[keep], pixels[keep], **kwargs), keep


def fit_calibration_ransac(robot, qs, pixels, trials=100, subset=8, inlier_px=30.0, seed=0, **kwargs):
    """Calibration that survives many gross outliers (wrong hoop detections): fit on random subsets, keep the model
    with the most inliers, then refit on its inliers. Returns (result, inlier_mask)."""
    qs, pixels = np.asarray(qs, float), np.asarray(pixels, float)
    n = len(qs)
    poses = _flange_poses(robot, qs)
    if n <= subset + 2:
        result = fit_calibration(robot, qs, pixels, **kwargs)
        return result, np.ones(n, bool)
    rng = np.random.default_rng(seed)
    best_mask, best_score = None, (-1, 0.0)
    for _ in range(trials):
        idx = rng.choice(n, subset, replace=False)
        try:
            res = fit_calibration(robot, qs[idx], pixels[idx], fast=True, **kwargs)
        except Exception:
            continue
        if res.rms_px > inlier_px:
            continue
        err = np.linalg.norm(res.camera.project(_ring_centers_from(poses, res.tcp)) - pixels, axis=1)
        mask = err < inlier_px
        score = (int(mask.sum()), -float(np.mean(err[mask])) if mask.any() else 0.0)
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None or best_mask.sum() < 8:
        raise RuntimeError("no consistent calibration found: too many wrong hoop detections")
    for _ in range(3):  # refit on the inliers, tighten the threshold to 3 x rms
        result = fit_calibration(robot, qs[best_mask], pixels[best_mask], **kwargs)
        err = np.linalg.norm(result.camera.project(_ring_centers_from(poses, result.tcp)) - pixels, axis=1)
        new_mask = err < max(3.0 * result.rms_px, 8.0)
        if new_mask.sum() < 8 or np.array_equal(new_mask, best_mask):
            break
        best_mask = new_mask
    return result, best_mask
