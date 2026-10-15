"""Run the whole runtime pipeline in one go.

Table rotation only (no calibration needed):  python scripts/run.py
Full pipeline, dry run (robot does not move):  python scripts/run.py --puzzle A
Full pipeline with robot motion:               python scripts/run.py --puzzle A --execute
Offline replay of a saved sequence:            python scripts/run.py --puzzle A --replay data/sequences/<folder>
Simulation of the whole scene:                 python scripts/run.py --puzzle D --sim
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotics_toolbox.maze.pipeline import CALIBRATION, run

parser = argparse.ArgumentParser()
parser.add_argument("--puzzle", choices=list("ABCD"), help="plan the hoop motion through this puzzle")
parser.add_argument("--execute", action="store_true", help="really move the robot (asks for confirmation)")
parser.add_argument("--replay", type=Path, help="folder with a saved sequence instead of the camera")
parser.add_argument("--sim", action="store_true", help="simulated camera, table and robot")
parser.add_argument("--direction", choices=["left", "right"], default="left", help="table direction in --sim")
parser.add_argument("--calibration", type=Path, default=CALIBRATION)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--fps", type=float, default=5.0)
parser.add_argument("--no-save", action="store_true")
args = parser.parse_args()

if args.sim:
    from robotics_toolbox.maze.sim_run import run_simulation

    run_simulation(args.puzzle or "A", args.direction)
else:
    run(args.replay, args.duration, args.fps, save=not args.no_save, puzzle=args.puzzle,
        calibration=args.calibration, do_execute=args.execute)
