"""Simulation of the whole scene (camera + rotating plate) for tests and for `run.py --sim`."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from robotics_toolbox.maze.aruco_base import ARUCO_DICT
from robotics_toolbox.maze.camera_model import CameraModel
from robotics_toolbox.maze.constants import IMAGE_SIZE, MARKER_EDGE, PLATFORM_HEIGHT
from robotics_toolbox.maze.table_motion import _rot

# Marker centers in the plate frame (x right, y up) [m]: ID 1 upper left, ID 2 lower right.
PLATE_MARKERS = {1: np.array([-0.037, 0.040]), 2: np.array([0.037, -0.040])}


def _plate_to_world(plate_xy, axis, omega, yaw0, t, plate0_xy):
    a = omega * t
    center = axis + _rot(a) @ (plate0_xy - axis)
    return center, yaw0 + a


def render_frame(camera: CameraModel, axis, omega, yaw0, plate0_xy, t) -> np.ndarray:
    """White image with ArUco markers rendered through the camera model (plate rotating about `axis`)."""
    img = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 235, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    s = MARKER_EDGE / 2
    local = np.array([[-s, s], [s, s], [s, -s], [-s, -s]])
    center, yaw = _plate_to_world(None, np.asarray(axis, float), omega, yaw0, t, np.asarray(plate0_xy, float))
    for marker_id, offset in PLATE_MARKERS.items():
        c = center + _rot(yaw) @ offset
        corners = c + local @ _rot(yaw).T
        dst = camera.project(np.column_stack([corners, np.full(4, PLATFORM_HEIGHT)])).astype(np.float32)
        tex = cv2.aruco.generateImageMarker(dictionary, marker_id, 200, borderBits=1)
        tex = cv2.copyMakeBorder(tex, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
        src = np.float32([[0, 0], [199, 0], [199, 199], [0, 199]])
        warped = cv2.warpPerspective(tex, cv2.getPerspectiveTransform(src, dst), IMAGE_SIZE, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(tex.shape, 255, np.uint8), cv2.getPerspectiveTransform(src, dst), IMAGE_SIZE)
        img[mask > 127] = warped[mask > 127, None]
    return img


def sim_frames(camera, axis, omega, yaw0, plate0_xy, duration=8.0, fps=2.0, t0=0.0):
    for t in np.arange(0.0, duration, 1.0 / fps):
        yield t0 + t, render_frame(camera, axis, omega, yaw0, plate0_xy, t0 + t)


SIM_TCP = np.array([0.18, 0.02, 0.05])


def sim_camera() -> CameraModel:
    """Ground-truth camera of the simulation: 1.17 m above the table, f ~ 4400 px, rotated by 0.6 rad."""
    r = Rotation.from_euler("x", np.pi).as_matrix() @ Rotation.from_euler("z", 0.6).as_matrix()
    return CameraModel(4400.0, r, -r @ np.array([0.45, 0.10, 1.17]))
