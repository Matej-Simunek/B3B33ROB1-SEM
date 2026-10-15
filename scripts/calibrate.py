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

import argparse

import cv2

from robotics_toolbox.maze.detect import detect_hoop, hoop_usable

root = Path(__file__).resolve().parents[1] / "data" / "calibration"
ap = argparse.ArgumentParser()
ap.add_argument("samples", nargs="?", type=Path, default=root / "samples.json")
ap.add_argument("--min-coverage", type=float, default=0.5,
                help="fraction of the ring circumference that must be visible (0.5 = half circle)")
ap.add_argument("--radius-tolerance", type=float, default=0.2, help="allowed relative radius deviation from the median")
args = ap.parse_args()
path = args.samples
if not path.exists():
    sys.exit(f"{path} not found: first collect samples with scripts/collect_calibration.py (it moves the robot, "
             "see docs/ROBOT_CHECKLIST.md, step 1)")
samples = json.loads(path.read_text())

# Re-detect the hoop in the saved images with the current (robust) detector and drop wrong detections.
kept, dropped = [], []
thumbs = []
for k, smp in enumerate(samples):
    img_path = path.parent / smp["image"] if "image" in smp else None
    if img_path is not None and img_path.exists():
        image = cv2.imread(str(img_path))
        found = detect_hoop(image)
        status = "ok"
        if found is None:
            status = "no ring found"
        elif not hoop_usable(found, args.min_coverage):
            status = f"only {found.coverage * 100:.0f} % of the ring visible"
        line = f"  [{k:2d}] {status:34s}" + (f" center ({found.center[0]:.0f}, {found.center[1]:.0f}) r {found.radius:.0f} px, "
                                             f"coverage {found.coverage:.2f}" if found is not None else "")
        print(line)
        thumb = image.copy()
        if found is not None:
            cv2.circle(thumb, tuple(found.center.astype(int)), int(found.radius), (0, 0, 255) if status == "ok" else (0, 165, 255), 6)
        cv2.putText(thumb, f"{k} {status}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        thumbs.append(cv2.resize(thumb, (480, 300)))
        if status != "ok":
            dropped.append((k, status))
            continue
        smp = {**smp, "pixel": found.center.tolist(), "radius": found.radius}
    kept.append(smp)
if thumbs:
    cols = 4
    rows = [np.hstack(thumbs[r:r + cols] + [np.zeros_like(thumbs[0])] * (cols - len(thumbs[r:r + cols])))
            for r in range(0, len(thumbs), cols)]
    cv2.imwrite(str(path.parent / "detection_debug.png"), np.vstack(rows))
    print(f"overview of all detections saved to {path.parent / 'detection_debug.png'}")
if kept:
    median_r = float(np.median([smp["radius"] for smp in kept]))
    lo, hi = (1 - args.radius_tolerance) * median_r, (1 + args.radius_tolerance) * median_r
    good = [smp for smp in kept if lo <= smp["radius"] <= hi]
    dropped += [(None, f"radius {smp['radius']:.0f} px far from the median {median_r:.0f} px") for smp in kept if smp not in good]
    kept = good
print(f"detection: {len(kept)} usable samples, {len(dropped)} dropped {dropped[:8]}")
samples = kept
if len(samples) < 8:
    sys.exit("fewer than 8 usable samples: see the list above and data/calibration/detection_debug.png "
             "(try --min-coverage 0.5), or collect more (ring in view, no other blue objects)")
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
