"""Print available Basler cameras and their resolution settings (robot does not move)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pypylon import pylon

from robotics_toolbox.maze.camera import CAMERA_NAME, open_camera

for d in pylon.TlFactory.GetInstance().EnumerateDevices():
    print("found:", d.GetUserDefinedName(), d.GetModelName(), d.GetIpAddress())

with open_camera(sys.argv[1] if len(sys.argv) > 1 else CAMERA_NAME) as cam:
    c = cam.camera
    print("model:", c.GetDeviceInfo().GetModelName())
    print("size:", c.Width.GetValue(), "x", c.Height.GetValue(), "(max", c.WidthMax.GetValue(), "x", c.HeightMax.GetValue(), ")")
    print("pixel format:", c.PixelFormat.GetValue())
    print("exposure [us]:", c.ExposureTime.GetValue(), "gain:", c.Gain.GetValue())
    img = cam.grab_image()
    print("grabbed array shape:", None if img is None else img.shape)
