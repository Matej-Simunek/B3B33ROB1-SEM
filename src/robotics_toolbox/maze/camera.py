"""Camera access that does not require a connection to the robot (no motion)."""
from contextlib import contextmanager

import numpy as np
from ctu_crs.basler_camera import BaslerCamera

CAMERA_NAME = "camera-crs93"


@contextmanager
def open_camera(name: str = CAMERA_NAME, exposure_time: float = 0.0):
    """Open the Basler camera by name. Only one program may hold the camera at a time."""
    camera = BaslerCamera()
    camera.connect_by_name(name)
    camera.open()
    camera.set_parameters(exposure_time=exposure_time)
    try:
        yield camera
    finally:
        camera.close()


def grab_image(name: str = CAMERA_NAME, exposure_time: float = 0.0) -> np.ndarray:
    """Grab a single BGR image."""
    with open_camera(name, exposure_time) as camera:
        return camera.grab_image()
