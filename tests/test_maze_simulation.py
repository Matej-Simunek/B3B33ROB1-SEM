import numpy as np
import pytest
from ctu_crs import CRS93

from robotics_toolbox.maze.executor import SimRobot, VirtualClock, execute
from robotics_toolbox.maze.planning import joint_speed_limits
from robotics_toolbox.maze.sim_run import run_simulation


@pytest.mark.parametrize("puzzle,direction", [("A", "left"), ("A", "right"), ("D", "left"), ("D", "right"),
                                              ("B", "left"), ("C", "left")])
def test_whole_pipeline_in_simulation(puzzle, direction):
    result, info = run_simulation(puzzle, direction, verbose=False)
    assert result.plan.success, result.plan.reason
    assert info["success"], info
    assert result.scene.motion.omega * (1 if direction == "left" else -1) > 0  # table direction recognised


def test_plate_outside_the_image_is_reported_and_robot_does_not_move():
    result, info = run_simulation("A", "left", plate0=np.array([1.4, 0.0]))
    assert not result.plan.success and "markers" in result.plan.reason
    assert result.execution is None and not info["success"]


def test_executor_aborts_when_robot_lags_behind():
    crs, clock = CRS93(tty_dev=None), VirtualClock(0.0)
    slow = SimRobot(crs, clock, crs.q_home, np.full(6, 0.01))
    target = crs.q_home + np.array([0.5, 0, 0, 0, 0, 0])
    trajectory = [(0.1 * k, crs.q_home + (target - crs.q_home) * k / 10) for k in range(11)]
    report = execute(slow, trajectory, clock, max_error=0.1)
    assert not report.completed and "lags" in report.reason


def test_executor_refuses_a_plan_in_the_past():
    crs, clock = CRS93(tty_dev=None), VirtualClock(50.0)
    robot = SimRobot(crs, clock, crs.q_home, joint_speed_limits(crs, 1.0))
    report = execute(robot, [(10.0, crs.q_home)], clock)
    assert not report.completed and not robot.commands


def test_executor_sends_each_waypoint_with_its_planned_duration():
    crs, clock = CRS93(tty_dev=None), VirtualClock(0.0)
    sim = SimRobot(crs, clock, crs.q_home, joint_speed_limits(crs, 1.0))
    sent = []
    sim.move_to_q_timed = lambda q, seconds: (sent.append(seconds), sim.move_to_q(q))
    trajectory = [(0.0, crs.q_home), (0.4, crs.q_home + 0.01), (1.0, crs.q_home + 0.02)]
    assert execute(sim, trajectory, clock).completed
    assert sent == pytest.approx([0.4, 0.6, 0.5])
