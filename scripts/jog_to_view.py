"""Interactively place the hoop in the camera view (flange pointing down) before collect_calibration.py.

SAFETY: this MOVES THE ROBOT in small steps you type. It knows NOTHING about objects on the table:
REMOVE the maze plate / puzzle and everything else from the workspace first. Keep the emergency stop at hand, clear the workspace.
Nothing moves without a typed command; the flange never goes below 12 cm above the table.

    python scripts/jog_to_view.py [--no-home] [--speed 0.2]

Commands (meters / degrees, relative to the current pose, flange stays pointing down):
    down [x y z]   go to a pose with the flange pointing down (default 0.45 0.0 0.35)
    x 0.02 | y -0.02 | z 0.03   move along the robot axes
    yaw 15         rotate the hoop around the flange axis
    look           grab an image and report whether the hoop ring is visible and where
    save           save the camera image with the detection to data/calibration/preview.png
    q              quit (the robot stays where it is; then run collect_calibration.py --no-home)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from ctu_crs import CRS93

from robotics_toolbox.maze.constants import IMAGE_SIZE
from robotics_toolbox.maze.detect import detect_hoop
from robotics_toolbox.maze.reachability import flange_pose

MIN_Z = 0.12  # lowest flange height [m]
MAX_STEP = 0.10  # largest single relative move [m]
MAX_JOINT_JUMP = 1.0  # rad, refuse larger moves (branch flips)

parser = argparse.ArgumentParser()
parser.add_argument("--no-home", action="store_true", help="robot is already homed")
parser.add_argument("--speed", type=float, default=0.2)
args = parser.parse_args()

robot = CRS93()
robot.initialize(home=not args.no_home)
robot.set_speed_relative(args.speed)
OUT = Path(__file__).resolve().parents[1] / "data" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)


def go(pose: np.ndarray) -> None:
    if pose[2, 3] < MIN_Z:
        print(f"refused: flange z would be {pose[2, 3]:.3f} m < {MIN_Z} m")
        return
    q_now = robot.get_q()
    sols = [q for q in robot.ik(pose) if robot.in_limits(q) and robot.fk(q)[2, 3] >= MIN_Z]
    if not sols:
        print("refused: pose is not reachable")
        return
    q = min(sols, key=lambda s: np.linalg.norm(s - q_now))
    if np.max(np.abs(q - q_now)) > MAX_JOINT_JUMP:
        print("refused: the move needs a big joint change; use smaller steps")
        return
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    print(f"flange now at {robot.fk(robot.get_q())[:3, 3].round(3)} m")


def go_down(x: float, y: float, z: float) -> None:
    """Move to a pose with the flange pointing down. The hoop yaw is free, so pick the one that needs the
    smallest joint change; the straight joint-space path must stay above MIN_Z. Asks before moving."""
    q_now = robot.get_q()
    best = None
    for yaw in np.linspace(-np.pi, np.pi, 36, endpoint=False):
        for q in robot.ik(flange_pose(np.array([x, y, z]), yaw)):
            if not robot.in_limits(q):
                continue
            path_z = min(robot.fk(q_now + s * (q - q_now))[2, 3] for s in np.linspace(0, 1, 40))
            if path_z < MIN_Z:
                continue
            change = float(np.max(np.abs(q - q_now)))
            if best is None or change < best[0]:
                best = (change, q, yaw)
    if best is None:
        print("refused: no reachable pose with a safe path")
        return
    change, q, yaw = best
    print(f"target flange ({x}, {y}, {z}) pointing down, hoop yaw {np.rad2deg(yaw):.0f} deg; "
          f"largest joint change {change:.2f} rad ({np.rad2deg(change):.0f} deg)")
    if input("Move there now? [y/N] ").strip().lower() != "y":
        print("not moved")
        return
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    print(f"flange now at {robot.fk(robot.get_q())[:3, 3].round(3)} m")


def look(save: bool = False) -> None:
    image = robot.grab_image()
    hoop = detect_hoop(image)
    if hoop is None:
        print("hoop ring NOT visible")
    else:
        cx, cy = hoop.center
        w, h = IMAGE_SIZE
        horiz = "left" if cx < w * 0.4 else "right" if cx > w * 0.6 else "center"
        vert = "top" if cy < h * 0.4 else "bottom" if cy > h * 0.6 else "middle"
        inside = hoop.radius < cx < w - hoop.radius and hoop.radius < cy < h - hoop.radius
        print(f"hoop visible at pixel ({cx:.0f}, {cy:.0f}), radius {hoop.radius:.0f} px: {vert} {horiz} of the image"
              + ("" if inside else "  -- touches the image border, move it towards the center"))
        cv2.circle(image, (int(cx), int(cy)), int(hoop.radius), (0, 0, 255), 3)
    if save:
        cv2.imwrite(str(OUT / "preview.png"), image)
        print(f"saved {OUT / 'preview.png'}")


print(__doc__)
try:
    while True:
        parts = input("jog> ").split()
        if not parts:
            continue
        cmd, vals = parts[0], [float(v) for v in parts[1:]]
        pose = robot.fk(robot.get_q())
        if cmd == "q":
            break
        elif cmd == "look":
            look()
        elif cmd == "save":
            look(save=True)
        elif cmd == "down":
            x, y, z = vals if len(vals) == 3 else (0.45, 0.0, 0.35)
            go_down(x, y, z)
            look()
        elif cmd in ("x", "y", "z") and len(vals) == 1:
            if abs(vals[0]) > MAX_STEP:
                print(f"refused: step larger than {MAX_STEP} m")
                continue
            if abs(pose[:3, 2] @ [0, 0, -1]) < 0.9:
                print("flange is not pointing down yet: use `down` first")
                continue
            target = pose.copy()
            target[:3, 3] += vals[0] * np.eye(3)["xyz".index(cmd)]
            go(target)
            look()
        elif cmd == "yaw" and len(vals) == 1:
            a = np.deg2rad(vals[0])
            rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            target = pose.copy()
            target[:3, :3] = rz @ pose[:3, :3]
            go(target)
            look()
        else:
            print("unknown command")
finally:
    robot.close()
