import numpy as np
from ctu_crs import CRS93
from scipy.spatial.transform import Rotation

from robotics_toolbox.maze.calibration import _ring_centers, fit_calibration
from robotics_toolbox.maze.camera_model import CameraModel
from robotics_toolbox.maze.reachability import flange_pose

TRUE_TCP = np.array([0.18, 0.02, 0.05])


def true_camera():
    r = Rotation.from_euler("x", np.pi).as_matrix() @ Rotation.from_euler("z", 0.6).as_matrix()
    return CameraModel(4400.0, r, -r @ np.array([0.45, 0.10, 1.17]))


def sample_poses(robot):
    qs = []
    for x in (0.35, 0.50, 0.65):
        for y in (-0.15, 0.0, 0.15):
            for z in (0.10, 0.20):
                for yaw in (0.0, 1.2, -1.2, 2.5):
                    for q in robot.ik(flange_pose(np.array([x, y, z]), yaw)):
                        if robot.in_limits(q):
                            qs.append(q)
                            break
    return qs


def test_backproject_inverts_project():
    cam = true_camera()
    pts = np.array([[0.4, 0.1, 0.033], [0.5, -0.1, 0.067]])
    for p in pts:
        assert np.allclose(cam.backproject(cam.project(p), p[2])[0], p)


def test_calibration_recovers_camera_and_tcp():
    robot = CRS93(tty_dev=None)
    cam, rng = true_camera(), np.random.default_rng(0)
    qs = sample_poses(robot)
    assert len(qs) > 30
    pixels = cam.project(_ring_centers(robot, qs, TRUE_TCP)) + rng.normal(0, 0.5, (len(qs), 2))
    result = fit_calibration(robot, qs, pixels)
    assert result.rms_px < 1.5
    for x, y in [(0.4, 0.0), (0.55, 0.1), (0.5, -0.12)]:
        truth = np.array([x, y, 0.033])
        est = result.camera.backproject(cam.project(truth), 0.033)[0]
        assert np.linalg.norm(est - truth) < 0.004
    assert np.linalg.norm(result.tcp[:2] - TRUE_TCP[:2]) < 0.01
