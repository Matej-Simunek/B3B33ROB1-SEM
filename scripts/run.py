"""Run the whole runtime pipeline in one go (robot does not move yet).

Live:    python scripts/run.py [--duration 10] [--fps 5]
Replay:  python scripts/run.py --replay data/sequences/<folder>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robotics_toolbox.maze.pipeline import run

parser = argparse.ArgumentParser()
parser.add_argument("--replay", type=Path, help="folder with a saved sequence instead of the camera")
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--fps", type=float, default=5.0)
parser.add_argument("--no-save", action="store_true")
args = parser.parse_args()
run(args.replay, args.duration, args.fps, save=not args.no_save)
