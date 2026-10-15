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


def stream(duration: float, fps: float = 5.0, name: str = CAMERA_NAME, exposure_time: float = 0.0):
    """Yield (timestamp [s], BGR image) for `duration` seconds at about `fps`, camera kept open."""
    import time

    from pypylon import pylon

    with open_camera(name, exposure_time) as camera:
        cam = camera.camera
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        t0 = time.monotonic()
        period = 1.0 / fps
        next_t = 0.0
        try:
            while (t := time.monotonic() - t0) < duration:
                result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                try:
                    if result.GrabSucceeded() and t >= next_t:
                        yield t, camera.converter.Convert(result).GetArray().copy()
                        next_t = t + period
                finally:
                    result.Release()
        finally:
            cam.StopGrabbing()
