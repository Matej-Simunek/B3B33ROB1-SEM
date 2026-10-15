import numpy as np
import pytest
from ctu_crs import CRS93

from robotics_toolbox.maze.constants import PLATFORM_HEIGHT
from robotics_toolbox.maze.planning import Tool, flange_pose_for_hoop, joint_speed_limits, plan, plan_with_wait
from robotics_toolbox.maze.plate_pose import puzzle_frame
from robotics_toolbox.maze.table_motion import _rot


def plate_at(x, y, yaw, omega=0.0, axis=(0.45, 0.0)):
    def f(t):
        T = np.eye(4)
        a = omega * t
        p = np.array(axis) + _rot(a) @ (np.array([x, y]) - np.array(axis))
        T[:2, :2], T[:2, 3], T[2, 3] = _rot(yaw + a), p, PLATFORM_HEIGHT
        return puzzle_frame(T)

    return f


@pytest.fixture(scope="module")
def robot():
    return CRS93(tty_dev=None)


def test_tool_frame_is_consistent():
    tool = Tool(np.array([0.1, 0.05, 0.06]))
    T = tool.T_flange_hoop
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3))
    assert T[:3, 2] == pytest.approx([0, 0, 1])
    assert np.allclose(flange_pose_for_hoop(T, tool), np.eye(4))


def test_speed_limits_are_slow_and_positive(robot):
    v = joint_speed_limits(robot, 1.0)
    assert np.all(v > 0.05) and np.all(v < 3.0)


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_static_plan_reaches_the_foot_and_respects_limits(robot, name):
    result = plan(robot, Tool(), name, plate_at(0.40, 0.0, 0.0), robot.q_home)
    assert result.success, result.reason
    assert result.duration < 120
    T_at = plate_at(0.40, 0.0, 0.0)
    q_last = result.trajectory[-1][1]
    hoop = robot.fk(q_last) @ Tool().T_flange_hoop
    assert hoop[2, 3] < PLATFORM_HEIGHT + 0.04
    assert np.linalg.norm(hoop[:2, 3] - T_at(0)[:2, 3]) < 0.005
    qs = np.array([q for _, q in result.trajectory])
    assert all(robot.in_limits(q) for q in qs)
    v = np.abs(np.diff(qs, axis=0)) / np.diff([t for t, _ in result.trajectory])[:, None]
    assert np.all(v <= joint_speed_limits(robot, 0.5) * 1.05 + 1e-9)


def test_unreachable_puzzle_is_reported_not_planned(robot):
    result = plan(robot, Tool(), "A", plate_at(1.5, 0.0, 0.0), robot.q_home)
    assert not result.success and result.reason


def test_rotating_table_lets_the_robot_wait_for_a_reachable_angle(robot):
    still = plan_with_wait(robot, Tool(), "D", plate_at(0.45, 0.0, 0.0), robot.q_home, max_wait=5.0)
    assert not still.success
    omega = np.deg2rad(6.0)
    moving = plan_with_wait(robot, Tool(), "D", plate_at(0.45, 0.0, 0.0, omega), robot.q_home, omega=omega)
    assert moving.success, moving.reason


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_hoop_never_goes_below_the_floor(robot, name):
    from robotics_toolbox.maze.constants import HOOP_FLOOR

    result = plan(robot, Tool(), name, plate_at(0.40, 0.0, 0.0), robot.q_home)
    assert result.success, result.reason
    zs = [(robot.fk(q) @ Tool().T_flange_hoop)[2, 3] for _, q in result.trajectory]
    assert min(zs) >= HOOP_FLOOR - 1e-3  # 6.8 cm above the table = 3.5 cm above the ArUco plate
    assert zs[-1] == pytest.approx(HOOP_FLOOR, abs=2e-3)


def test_z_compensation_raises_the_commanded_heights_but_not_the_plan(robot, monkeypatch):
    from robotics_toolbox.maze import planning
    from robotics_toolbox.maze.constants import HOOP_FLOOR

    monkeypatch.setattr(planning, "Z_COMPENSATION", 0.02)
    result = plan(robot, Tool(), "A", plate_at(0.40, 0.0, 0.0), robot.q_home)
    assert result.success, result.reason
    zs = [(robot.fk(q) @ Tool().T_flange_hoop)[2, 3] for _, q in result.trajectory]
    assert min(zs) == pytest.approx(HOOP_FLOOR + 0.02, abs=2e-3)  # the robot is sent 2 cm higher
    assert result.min_hoop_z == pytest.approx(HOOP_FLOOR, abs=2e-3)  # the planned height is unchanged
