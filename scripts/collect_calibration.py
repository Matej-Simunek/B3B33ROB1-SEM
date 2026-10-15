"""Collect camera-robot calibration samples: the robot moves the hoop to several poses and the camera finds it.

SAFETY: this MOVES THE ROBOT. Keep the emergency stop at hand, clear the workspace, low speed is used.
Procedure: bring the hoop (by hand-guided jogging, e.g. `move_crs93 q1 10`) to a pose where the whole ring is
visible in the camera and >= 15 cm above the table, then start this script. It moves around that pose
(+-8 cm in x/y, up to +12 cm up, +-30 deg yaw) and stores (joints, hoop center in pixels).

    python scripts/collect_calibration.py [--speed 0.2] [--no-home]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from ctu_crs import CRS93

from robotics_toolbox.maze.detect import detect_hoop
from robotics_toolbox.maze.reachability import R_DOWN  # noqa: F401  (flange pointing down convention)

OUT = Path(__file__).resolve().parents[1] / "data" / "calibration"

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=0.2, help="relative joint speed 0-1")
parser.add_argument("--no-home", action="store_true", help="skip homing (robot already homed)")
args = parser.parse_args()

OFFSETS = [(dx, dy, dz, yaw) for dz in (0.0, 0.06, 0.12) for dx in (-0.08, 0.0, 0.08) for dy in (-0.08, 0.0, 0.08)
           for yaw in (-0.5, 0.0, 0.5)]
rng = np.random.default_rng(1)
OFFSETS = [OFFSETS[i] for i in sorted(rng.choice(len(OFFSETS), 30, replace=False))]


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


robot = CRS93()
robot.initialize(home=not args.no_home)
robot.set_speed_relative(args.speed)
(OUT / "images").mkdir(parents=True, exist_ok=True)
base_q = robot.get_q()
base = robot.fk(base_q)
print(f"base pose: flange at {base[:3, 3].round(3)} m")
img = robot.grab_image()
hoop = detect_hoop(img)
if hoop is None:
    sys.exit("hoop not visible in the camera at the base pose: jog the robot so that the ring is in view")
print(f"hoop visible at pixel {hoop.center.round(0)}")
if input("Robot will move around this pose. Type 'go' to start: ") != "go":
    sys.exit("aborted")

samples, q = [], base_q
try:
    for k, (dx, dy, dz, yaw) in enumerate(OFFSETS):
        target = base.copy()
        target[:3, :3] = rotz(yaw) @ base[:3, :3]
        target[:3, 3] += [dx, dy, dz]
        sols = [s for s in robot.ik(target) if robot.in_limits(s) and robot.fk(s)[2, 3] > 0.10]
        if not sols:
            print(f"[{k}] no IK, skipped")
            continue
        q = min(sols, key=lambda s: np.linalg.norm(s - q))
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        cv2.waitKey(1)
        image = robot.grab_image()
        found = detect_hoop(image)
        if found is None:
            print(f"[{k}] hoop not found, skipped")
            continue
        name = f"images/{len(samples):03d}.png"
        cv2.imwrite(str(OUT / name), image)
        samples.append({"q": q.tolist(), "pixel": found.center.tolist(), "radius": found.radius, "image": name})
        print(f"[{k}] ok, pixel {found.center.round(0)}")
finally:
    robot.move_to_q(base_q)
    robot.wait_for_motion_stop()
    robot.close()
    (OUT / "samples.json").write_text(json.dumps(samples, indent=2))
    print(f"saved {len(samples)} samples to {OUT / 'samples.json'}")
