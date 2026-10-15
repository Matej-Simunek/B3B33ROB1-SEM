"""Execution of a planned joint trajectory on the robot (real or simulated).

The CRS controller executes coordinated moves asynchronously, so each waypoint is sent at its planned time and
the measured joint state is compared with the plan; a deviation aborts the run.
"""
import time
from dataclasses import dataclass

import numpy as np


class WallClock:
    def time(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class VirtualClock:
    def __init__(self, t: float = 0.0):
        self.t = t
        self.listeners = []

    def time(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)
        for f in self.listeners:
            f(self.t)


class SimRobot:
    """Offline CRS93 whose joints follow commanded targets at limited speed (asynchronous like the controller)."""

    def __init__(self, crs, clock: VirtualClock, q0, vmax):
        self.crs, self.clock, self.q, self.target, self.vmax = crs, clock, np.array(q0, float), np.array(q0, float), vmax
        self._t = clock.time()
        clock.listeners.append(self._advance)
        self.commands = []

    def _advance(self, t):
        dt, self._t = t - self._t, t
        delta = self.target - self.q
        step = np.clip(delta, -self.vmax * dt, self.vmax * dt)
        self.q = self.q + step

    def get_q(self):
        return self.q.copy()

    def move_to_q(self, q):
        assert self.crs.in_limits(q), "Joint limits violated."
        self.target = np.array(q, float)
        self.commands.append((self.clock.time(), self.target.copy()))

    def fk(self, q):
        return self.crs.fk(q)


class TimedCRS:
    """Real CRS robot whose waypoints are sent with a minimal duration (COORDMVT), so that every segment takes the
    time the plan assigned to it instead of "as fast as possible" (which makes the lag grow on fast joints)."""

    def __init__(self, crs):
        self.crs = crs

    def get_q(self):
        return self.crs.get_q()

    def move_to_q(self, q):
        self.crs.move_to_q(q)

    def move_to_q_timed(self, q, seconds: float):
        assert self.crs._initialized, "You need to initialize the robot before moving it."
        assert self.crs.in_limits(q), "Joint limits violated."
        self.crs._mars.coordmv(self.crs._joint_values_to_irc(q), min_time=max(float(seconds), 0.05))


def _send(robot, q, seconds: float) -> None:
    timed = getattr(robot, "move_to_q_timed", None)
    timed(q, seconds) if timed is not None else robot.move_to_q(q)


@dataclass
class ExecutionReport:
    completed: bool
    reason: str
    max_error: float
    samples: list


def execute(robot, trajectory, clock=None, max_error=0.15, on_step=None) -> ExecutionReport:
    """Send the trajectory [(t, q), ...] to `robot` (anything with get_q / move_to_q).

    Trajectory times are absolute on `clock` (the time base the table prediction was made in).
    """
    clock = clock or WallClock()
    if trajectory[0][0] < clock.time() - 0.2:
        return ExecutionReport(False, "plan is already in the past (planning took too long), robot not moved", 0.0, [])
    worst, samples, prev_target = 0.0, [], None
    for k, (t, q) in enumerate(trajectory):
        clock.sleep(t - clock.time())
        if prev_target is not None:
            err = float(np.max(np.abs(robot.get_q() - prev_target)))
            worst = max(worst, err)
            samples.append((t, err))
            if err > max_error:
                robot.move_to_q(robot.get_q())
                return ExecutionReport(False, f"robot lags {err:.3f} rad behind the plan at waypoint {k}", worst, samples)
        if on_step:
            on_step(k, t, q)
        _send(robot, q, trajectory[k + 1][0] - t if k + 1 < len(trajectory) else 0.5)
        prev_target = q
    clock.sleep(0.5)
    return ExecutionReport(True, "", worst, samples)
