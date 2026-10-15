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
if not path.exists():
    sys.exit(f"{path} not found: first collect samples with scripts/collect_calibration.py (it moves the robot, "
             "see docs/ROBOT_CHECKLIST.md, step 1)")
samples = json.loads(path.read_text())
if len(samples) < 8:
    sys.exit(f"only {len(samples)} samples in {path}, at least 8 are needed (run collect_calibration.py again)")
qs = np.array([s["q"] for s in samples])
pixels = np.array([s["pixel"] for s in samples])

robot = CRS93(tty_dev=None)
flange = np.array([robot.fk(q)[:3, 3] for q in qs])
radius = np.array([s.get("radius", np.nan) for s in samples])
print(f"data: {len(samples)} samples")
print(f"  flange xyz range [m]: x {flange[:, 0].min():.2f}..{flange[:, 0].max():.2f}, "
      f"y {flange[:, 1].min():.2f}..{flange[:, 1].max():.2f}, z {flange[:, 2].min():.2f}..{flange[:, 2].max():.2f}")
print(f"  hoop pixel range: x {pixels[:, 0].min():.0f}..{pixels[:, 0].max():.0f}, y {pixels[:, 1].min():.0f}..{pixels[:, 1].max():.0f}")
print(f"  hoop radius [px]: {np.nanmin(radius):.0f}..{np.nanmax(radius):.0f} (should change smoothly with height)")
down = np.array([robot.fk(q)[:3, 2] @ [0, 0, -1] for q in qs])
if down.min() < 0.95:
    print(f"  WARNING: in {np.sum(down < 0.95)} samples the flange does not point down (min cos {down.min():.2f})")
result, keep = fit_calibration_robust(robot, qs, pixels)
cam = result.camera
print(f"samples: {keep.sum()} used of {len(keep)}")
print(f"reprojection error: rms {result.rms_px:.2f} px, max {result.max_px:.2f} px")
print(f"focal length {cam.f:.0f} px, camera at {cam.center.round(3)} m (measured height {CAMERA_HEIGHT} m)")
print(f"hoop center in the flange frame (tcp) = {result.tcp.round(3)} m")
if abs(cam.center[2] - CAMERA_HEIGHT) > 0.1:
    print("WARNING: calibrated camera height differs a lot from the measured one, check the samples")
if result.rms_px > 3:
    print("WARNING: large reprojection error; the hoop detection or the data may be bad")
err = np.linalg.norm(cam.project(np.array([(robot.fk(q) @ np.append(result.tcp, 1.0))[:3] for q in qs])) - pixels, axis=1)
worst = np.argsort(err)[::-1][:3]
print("worst samples (index: error px, pixel):", ", ".join(f"{i}: {err[i]:.1f} px {pixels[i].round(0)}" for i in worst))
if result.rms_px > 3 or abs(cam.center[2] - CAMERA_HEIGHT) > 0.1:
    sys.exit("calibration NOT saved (bad fit): send the output above and data/calibration/samples.json")
out = root / "camera_robot.json"
cam.save(out, tcp=result.tcp.tolist(), rms_px=result.rms_px, n_samples=int(keep.sum()))
print(f"saved {out}")
