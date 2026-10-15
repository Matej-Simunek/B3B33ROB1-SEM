import numpy as np
from scipy.spatial.transform import Rotation

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
