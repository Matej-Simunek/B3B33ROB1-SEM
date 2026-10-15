"""Runtime pipeline: observe the table, find the plate, plan the hoop motion through the puzzle, execute."""
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robotics_toolbox.maze import planning
from robotics_toolbox.maze.aruco_base import detect_markers
from robotics_toolbox.maze.camera_model import CameraModel
from robotics_toolbox.maze.executor import ExecutionReport, TimedCRS, WallClock, execute
from robotics_toolbox.maze.plate_pose import PlateLayout, PlateTracker, marker_world_observation, puzzle_frame
from robotics_toolbox.maze.table_motion import TableMotion, fit_table_motion, marker_observation

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "sequences"
RUNS = ROOT / "data" / "runs"
CALIBRATION = ROOT / "data" / "calibration" / "camera_robot.json"
TOOL_Z = ROOT / "data" / "calibration" / "tool.json"  # measured ring height below the flange (scripts/measure_tool.py)


# ---------------------------------------------------------------- frames
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


# ---------------------------------------------------------------- table motion (pixel mode, no calibration)
def observe(frames) -> list[tuple[float, dict]]:
    observations = []
    for t, image in frames:
        markers = detect_markers(image)
        if markers:
            observations.append((t, {m.id: marker_observation(m) for m in markers}))
    return observations


def estimate_table_motion(frames) -> TableMotion:
    observations = observe(frames)
    print(f"markers found in {len(observations)} frames")
    return fit_table_motion(observations)


# ---------------------------------------------------------------- calibrated scene analysis
def load_calibration(path: Path = CALIBRATION) -> tuple[CameraModel, planning.Tool]:
    camera, extra = CameraModel.load(path)
    tcp = np.array(extra["tcp"], float)
    if TOOL_Z.exists():
        measured = json.loads(TOOL_Z.read_text())["tcp_z"]
        print(f"tool height below the flange: {measured:.3f} m (measured), calibration said {tcp[2]:.3f} m")
        tcp[2] = measured
    else:
        print(f"WARNING: tool height below the flange is only the calibration estimate ({tcp[2]:.3f} m); "
              "measure it with scripts/measure_tool.py before moving the robot")
    return camera, planning.Tool(tcp)


class SceneError(RuntimeError):
    """The scene cannot be analysed (e.g. the markers are not visible); the robot must not move."""


@dataclass
class Scene:
    motion: TableMotion
    layout: PlateLayout
    tracker: PlateTracker
    t_last: float


def analyze_scene(frames, camera: CameraModel) -> Scene:
    """Markers -> robot-frame observations -> table rotation + plate layout."""
    observations = []
    for t, image in frames:
        markers = detect_markers(image)
        if markers:
            observations.append((t, {m.id: marker_world_observation(m, camera) for m in markers}))
    if len(observations) < 3:
        raise SceneError(f"markers were found in only {len(observations)} frames; cannot track the plate")
    layout = PlateLayout.learn([o for _, o in observations])
    motion = fit_table_motion(observations)
    return Scene(motion, layout, PlateTracker(motion, layout), observations[-1][0])


def describe_scene(scene: Scene) -> None:
    m = scene.motion
    print(f"table: {m.direction}")
    print(f"  omega = {np.rad2deg(m.omega):.2f} deg/s, axis in robot xy = {m.center.round(3)} m")
    print(f"  fit residuals: position {m.rms_px * 1000:.2f} mm, yaw {np.rad2deg(m.yaw_residual_rad):.2f} deg")
    pose = scene.tracker.pose(scene.t_last)
    print(f"  plate at last frame: xy = {pose[:2, 3].round(3)} m, yaw = {np.rad2deg(np.arctan2(pose[1, 0], pose[0, 0])):.1f} deg")


def make_plan(scene: Scene, robot, tool, puzzle: str, q_now, t_start: float, speed_scale=0.5, hints=None) -> planning.Plan:
    pose_at = lambda t: puzzle_frame(scene.tracker.pose(t))  # noqa: E731
    omega = scene.motion.omega if scene.motion.rotating else 0.0
    return planning.plan_with_wait(robot, tool, puzzle, pose_at, q_now, t_start, omega, speed_scale=speed_scale, hints=hints)


def describe_plan(plan: planning.Plan, t_start: float) -> None:
    if not plan.success:
        print(f"NOT REACHABLE / no plan: {plan.reason}")
        print("The robot will not move.")
        return
    waits = [p for p in plan.phases if p == "wait"]
    print(f"plan OK: {len(plan.trajectory)} waypoints, duration {plan.duration:.1f} s"
          + (" (starts with waiting for a reachable table angle)" if waits else ""))
    print(f"  the robot starts moving {plan.idle_time:.0f} s after the start (waiting for a reachable table angle)")
    print(f"  handle direction in the puzzle frame: {np.round(plan.hint, 2)}")
    print(f"  lowest hoop center {plan.min_hoop_z * 100:.1f} cm and lowest flange {plan.min_flange_z * 100:.1f} cm above the table "
          "(check against the real heights of the puzzle foot / plate)")


@dataclass
class Result:
    scene: Scene | None
    plan: planning.Plan | None
    execution: ExecutionReport | None = None


def run_scene(frames, camera, tool, robot_model, puzzle, q_now, execute_with=None, clock=None, confirm=None,
              now=None, planning_lag=6.0, speed_scale=0.5) -> Result:
    """Everything after the images are in: analysis, plan, and optional execution."""
    try:
        scene = analyze_scene(frames, camera)
    except SceneError as e:
        print(f"{e}\nThe robot will not move.")
        return Result(None, planning.Plan(False, str(e)))
    describe_scene(scene)
    clock = clock or WallClock()
    t_start = (now() if now else clock.time()) + planning_lag
    wall0 = time.perf_counter()
    plan = make_plan(scene, robot_model, tool, puzzle, q_now, t_start, speed_scale)
    planning_lag = max(planning_lag, 2.0 * (time.perf_counter() - wall0))
    describe_plan(plan, t_start)
    if not plan.success or execute_with is None:
        return Result(scene, plan)
    if confirm is not None and not confirm(plan):
        print("aborted by the user, robot not moved")
        return Result(scene, plan)
    # re-plan from the current time and the current joints. Fast first try: only the handle direction found before;
    # if that fails, all directions. The start lag must be longer than the planning itself, else the plan is stale.
    q_now = execute_with.get_q()
    new_plan, took, lag, ready = None, 0.0, 4.0, False
    for hints in ([np.array(plan.hint)], None):
        lag = 4.0
        for _ in range(3):
            wall0 = time.perf_counter()
            new_plan = make_plan(scene, robot_model, tool, puzzle, q_now, clock.time() + lag, speed_scale, hints=hints)
            took = time.perf_counter() - wall0
            if not new_plan.success:
                break
            if took <= lag - 0.5:
                ready = True
                break
            lag = 1.5 * took + 1.0  # planning took longer than the lag: plan again with a bigger lag
        if ready:
            break
    plan = new_plan
    if not ready:
        print(f"re-planning failed: {plan.reason if not plan.success else 'planning is too slow'}")
        return Result(scene, plan)
    print(f"re-planned in {took:.1f} s; the robot starts moving in about {lag + plan.idle_time - took:.0f} s")
    report = execute(execute_with, plan.trajectory, clock)
    print("execution " + ("finished" if report.completed else f"ABORTED: {report.reason}") + f", max joint error {report.max_error:.3f} rad")
    try:  # record the run: planned vs measured joints, for offline analysis
        RUNS.mkdir(parents=True, exist_ok=True)
        path = RUNS / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{puzzle}.json"
        path.write_text(json.dumps({"puzzle": puzzle, "completed": report.completed, "reason": report.reason, "speed_scale": speed_scale,
                                    "plan": [(t - plan.trajectory[0][0], q.tolist()) for t, q in plan.trajectory],
                                    "phases": plan.phases, "log": report.log}))
        print(f"run log saved to {path}")
    except OSError as e:
        print(f"could not save the run log: {e}")
    return Result(scene, plan, report)


# ---------------------------------------------------------------- CLI orchestration
def run(replay: Path | None = None, duration: float = 10.0, fps: float = 5.0, save: bool = True,
        puzzle: str | None = None, calibration: Path = CALIBRATION, do_execute: bool = False,
        speed_scale: float = 0.5) -> Result | TableMotion:
    """Without a calibration: table rotation only (pixel mode). With one: the full pipeline for `puzzle`."""
    if replay is not None:
        frames = list(replay_frames(replay))
    else:
        save_dir = None
        if save:
            save_dir = DATA / datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir.mkdir(parents=True)
            print(f"saving sequence to {save_dir}")
        frames = list(live_frames(duration, fps, save_dir))

    if puzzle is None or not calibration.exists():
        if puzzle is not None:
            print(f"calibration {calibration} not found: only the table rotation is estimated (run scripts/calibrate.py)")
        motion = estimate_table_motion(frames)
        print(f"table: {motion.direction}")
        print(f"  omega = {np.rad2deg(motion.omega):.2f} deg/s, axis in image (x, y-up) = {motion.center.round(1)} px")
        print(f"  residuals: position {motion.rms_px:.2f} px, yaw {np.rad2deg(motion.yaw_residual_rad):.2f} deg")
        return motion

    from ctu_crs import CRS93

    camera, tool = load_calibration(calibration)
    assert 0 < speed_scale <= 1.0, "speed must be in (0, 1]"
    if do_execute:
        robot = CRS93()
        homed = input("Has the robot been homed since the control unit was powered on? [y/N] (N = it will home now and MOVE) ")
        robot.initialize(home=homed.strip().lower() != "y")
        q_now = robot.get_q()
        confirm = lambda plan: input(  # noqa: E731
            f"Execute {plan.duration:.0f} s trajectory; lowest hoop center {plan.min_hoop_z * 100:.1f} cm, lowest flange "
            f"{plan.min_flange_z * 100:.1f} cm above the table. Are these heights physically sensible? "
            "Keep the E-stop at hand. Type 'yes': ") == "yes"
        try:
            return run_scene(frames, camera, tool, robot, puzzle, q_now, execute_with=TimedCRS(robot), confirm=confirm,
                             speed_scale=speed_scale)
        finally:
            robot.close()
    model = CRS93(tty_dev=None)
    return run_scene(frames, camera, tool, model, puzzle, model.q_home, speed_scale=speed_scale)
