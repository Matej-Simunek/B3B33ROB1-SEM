"""Map of the robot workspace: where can the hoop be threaded through each puzzle (for the report).

For every plate position on a grid and every plate yaw (the table rotates, so all yaws occur) the insertion
path is checked with the analytical IK of the CRS93 (no robot needed). Color = fraction of yaws that work.

    python scripts/reachability_map.py [--tcp 0.15 0.0 0.06] [--puzzles ABCD]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ctu_crs import CRS93

from robotics_toolbox.maze import planning
from robotics_toolbox.maze.constants import PLATFORM_HEIGHT
from robotics_toolbox.maze.plate_pose import puzzle_frame

parser = argparse.ArgumentParser()
parser.add_argument("--tcp", type=float, nargs=3, default=[0.15, 0.0, 0.06])
parser.add_argument("--puzzles", default="ABCD")
parser.add_argument("--step", type=float, default=0.05)
parser.add_argument("--yaws", type=int, default=12)
args = parser.parse_args()

robot, tool = CRS93(tty_dev=None), planning.Tool(np.array(args.tcp))
xs, ys = np.arange(0.20, 0.80 + 1e-9, args.step), np.arange(-0.40, 0.40 + 1e-9, args.step)
out_dir = Path(__file__).resolve().parents[1] / "report" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, len(args.puzzles), figsize=(4.2 * len(args.puzzles), 4.2), squeeze=False)
for ax, name in zip(axes[0], args.puzzles):
    hints = planning.hint_candidates(name, 6)
    frac = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ok = 0
            for yaw in np.linspace(0, 2 * np.pi, args.yaws, endpoint=False):
                T = np.eye(4)
                T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
                T[:2, 3], T[2, 3] = [x, y], PLATFORM_HEIGHT
                pose = puzzle_frame(T)
                ok += bool(planning.quick_feasible(robot, tool, name, lambda t, p=pose: p, 0.0, hints))
            frac[i, j] = ok / args.yaws
    im = ax.imshow(frac, origin="lower", extent=[xs[0] - args.step / 2, xs[-1] + args.step / 2, ys[0] - args.step / 2,
                                                 ys[-1] + args.step / 2], vmin=0, vmax=1, cmap="viridis")
    ax.set_title(f"puzzle {name}")
    ax.set_xlabel("x robot [m]")
    ax.set_ylabel("y robot [m]")
    print(f"puzzle {name}: {np.mean(frac > 0) * 100:.0f} % of positions workable for some yaw, "
          f"{np.mean(frac == 1) * 100:.0f} % for every yaw")
fig.colorbar(im, ax=axes.ravel().tolist(), label="fraction of plate yaws that work")
path = out_dir / "reachability.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"saved {path}")
