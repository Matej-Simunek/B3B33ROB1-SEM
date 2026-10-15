"""End-to-end simulation: rendered camera frames of a rotating plate, planning and simulated execution."""
import numpy as np
from ctu_crs import CRS93

from robotics_toolbox.maze import puzzles
from robotics_toolbox.maze.constants import PLATFORM_HEIGHT
from robotics_toolbox.maze.executor import SimRobot, VirtualClock
from robotics_toolbox.maze.pipeline import Result, run_scene
from robotics_toolbox.maze.planning import SUCCESS_HEIGHT, Tool, joint_speed_limits
from robotics_toolbox.maze.sim import SIM_TCP, sim_camera, sim_frames
from robotics_toolbox.maze.table_motion import _rot

AXIS = np.array([0.48, 0.02])  # true table axis in the robot frame
PLATE0 = np.array([0.40, 0.05])  # plate center at t=0
YAW0 = 0.2


def hoop_pose_after(result: Result, robot_model, tool: Tool) -> np.ndarray:
    return robot_model.fk(result.plan.trajectory[-1][1]) @ tool.T_flange_hoop


def run_simulation(puzzle: str = "A", direction: str = "left", omega_deg: float = 4.0, verbose: bool = True,
                   plate0=PLATE0, yaw0=YAW0) -> tuple[Result, dict]:
    omega = np.deg2rad(omega_deg) * (1 if direction == "left" else -1)
    camera, tool, robot_model = sim_camera(), Tool(SIM_TCP), CRS93(tty_dev=None)
    frames = list(sim_frames(camera, AXIS, omega, yaw0, plate0, duration=8.0, fps=2.0, t0=100.0))
    clock = VirtualClock(frames[-1][0] + 1.0)
    sim_robot = SimRobot(robot_model, clock, robot_model.q_home, joint_speed_limits(robot_model, 1.0))
    result = run_scene(frames, camera, tool, robot_model, puzzle, sim_robot.get_q(), execute_with=sim_robot, clock=clock,
                       now=clock.time, planning_lag=6.0)

    info = {"planned": result.plan is not None and result.plan.success, "success": False}
    if result.execution is not None and result.execution.completed:
        t_end = result.plan.trajectory[-1][0]
        # ground truth of the puzzle foot at the end time
        center = AXIS + _rot(omega * (t_end - 0.0)) @ (plate0 - AXIS)  # plate center rotates rigidly about the axis
        foot_truth = np.array([*center, PLATFORM_HEIGHT])
        hoop = sim_robot.fk(sim_robot.get_q()) @ tool.T_flange_hoop
        info["foot_error_xy"] = float(np.linalg.norm(hoop[:2, 3] - foot_truth[:2]))
        info["hoop_height_above_plate"] = float(hoop[2, 3] - PLATFORM_HEIGHT)
        info["success"] = info["foot_error_xy"] < 0.01 and info["hoop_height_above_plate"] < SUCCESS_HEIGHT
        if verbose:
            print(f"simulation: foot error {info['foot_error_xy'] * 1000:.1f} mm, hoop {info['hoop_height_above_plate'] * 100:.1f} cm above the plate "
                  f"-> {'SUCCESS' if info['success'] else 'FAIL'}")
    return result, info
