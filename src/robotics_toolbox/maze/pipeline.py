"""Runtime pipeline: observe the table, estimate its motion (more stages are added later)."""
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robotics_toolbox.maze.aruco_base import detect_markers
from robotics_toolbox.maze.table_motion import TableMotion, fit_table_motion, marker_observation

DATA = Path(__file__).resolve().parents[3] / "data" / "sequences"


def observe(frames) -> list[tuple[float, dict]]:
    observations = []
    for t, image in frames:
        markers = detect_markers(image)
        if markers:
            observations.append((t, {m.id: marker_observation(m) for m in markers}))
    return observations


def live_frames(duration: float, fps: float, save_dir: Path | None):
    from robotics_toolbox.maze.camera import stream

    times = []
    for k, (t, image) in enumerate(stream(duration, fps)):
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"{k:04d}.png"), image)
            times.append(t)
        yield t, image
    if save_dir is not None:
        (save_dir / "times.json").write_text(json.dumps(times))


def replay_frames(directory: Path):
    times = json.loads((directory / "times.json").read_text())
    for k, t in enumerate(times):
        yield t, cv2.imread(str(directory / f"{k:04d}.png"))


def estimate_table_motion(frames) -> TableMotion:
    observations = observe(frames)
    print(f"markers found in {len(observations)} frames")
    return fit_table_motion(observations)


def run(replay: Path | None = None, duration: float = 10.0, fps: float = 5.0, save: bool = True) -> TableMotion:
    if replay is not None:
        frames = replay_frames(replay)
    else:
        save_dir = None
        if save:
            save_dir = DATA / datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir.mkdir(parents=True)
            print(f"saving sequence to {save_dir}")
        frames = live_frames(duration, fps, save_dir)
    motion = estimate_table_motion(frames)
    print(f"table: {motion.direction}")
    print(f"  omega = {np.rad2deg(motion.omega):.2f} deg/s")
    print(f"  axis in image (x, y-up) = {motion.center.round(1)} px")
    print(f"  residuals: position {motion.rms_px:.2f} px, yaw {np.rad2deg(motion.yaw_residual_rad):.2f} deg")
    return motion
