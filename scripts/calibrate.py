"""Fit the camera -> robot calibration from data/calibration/samples.json (no robot needed).

    python scripts/calibrate.py [samples.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from ctu_crs import CRS93

from robotics_toolbox.maze.calibration import fit_calibration_robust
from robotics_toolbox.maze.constants import CAMERA_HEIGHT

root = Path(__file__).resolve().parents[1] / "data" / "calibration"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "samples.json"
samples = json.loads(path.read_text())
qs = np.array([s["q"] for s in samples])
pixels = np.array([s["pixel"] for s in samples])

result, keep = fit_calibration_robust(CRS93(tty_dev=None), qs, pixels)
cam = result.camera
print(f"samples: {keep.sum()} used of {len(keep)}")
print(f"reprojection error: rms {result.rms_px:.2f} px, max {result.max_px:.2f} px")
print(f"focal length {cam.f:.0f} px, camera at {cam.center.round(3)} m (measured height {CAMERA_HEIGHT} m)")
print(f"hoop center in the flange frame (tcp) = {result.tcp.round(3)} m")
if abs(cam.center[2] - CAMERA_HEIGHT) > 0.1:
    print("WARNING: calibrated camera height differs a lot from the measured one, check the samples")
if result.rms_px > 3:
    print("WARNING: large reprojection error; the hoop detection or the data may be bad")
out = root / "camera_robot.json"
cam.save(out, tcp=result.tcp.tolist(), rms_px=result.rms_px, n_samples=int(keep.sum()))
print(f"saved {out}")
