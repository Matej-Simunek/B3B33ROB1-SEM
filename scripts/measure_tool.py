"""Measure how far the hoop ring center hangs below the flange (robot does not move).

Needed because the camera calibration cannot determine this height well (it is only ~5 cm accurate) and a
wrong value makes the robot go too low or too high. Procedure:
  1. Put the robot in any pose where you can reach the hoop with a ruler (e.g. the pose from jog_to_view.py,
     flange pointing down); the ring should be roughly horizontal.
  2. Run this script. It reads the flange height from the robot and asks for the measured height of the
     CENTER OF THE RING (middle of its thickness) above the table surface.
  3. The difference is saved to data/calibration/tool.json and used by run.py.

    python scripts/measure_tool.py [--no-home]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from ctu_crs import CRS93

parser = argparse.ArgumentParser()
parser.add_argument("--no-home", action="store_true", default=True)
parser.parse_args()

OUT = Path(__file__).resolve().parents[1] / "data" / "calibration" / "tool.json"
robot = CRS93()
robot.initialize(home=False)  # does not move
try:
    T = robot.fk(robot.get_q())
    down = T[:3, 2] @ np.array([0, 0, -1])
    print(f"flange is at {T[:3, 3].round(3)} m (robot base frame, z = height above the table if the base is on the table)")
    if down < 0.95:
        print(f"WARNING: the flange does not point down (cos {down:.2f}); the result then only applies to this pose")
    text = input("Measured height of the ring center above the TABLE surface [cm]: ").strip().replace(",", ".")
    ring_z = float(text) / 100.0
    tcp_z = float(T[2, 3] - ring_z)
    print(f"-> the ring center is {tcp_z * 100:.1f} cm below the flange (this includes any offset of the robot base)")
    if not -0.05 < tcp_z < 0.35:
        sys.exit("value looks wrong (expected 0..30 cm); not saved")
    OUT.write_text(json.dumps({"tcp_z": tcp_z, "flange_z": float(T[2, 3]), "ring_z_measured": ring_z}, indent=2))
    print(f"saved {OUT}")
finally:
    robot.close()
