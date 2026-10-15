"""Trajectory planning: hoop path in the puzzle frame -> time-parameterized joint trajectory of the CRS93.

The puzzle sits on a rotating table, so the puzzle frame moves: T_robot_puzzle(t). Waypoint k of the path is
reached at time t_k, which is chosen so that no joint exceeds its speed limit. The same engine is used for the
reachability check (simulation) and, later, for the execution.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from robotics_toolbox.maze import puzzles
from robotics_toolbox.maze.constants import PLATFORM_HEIGHT

SUCCESS_HEIGHT = 0.04  # hoop must end lower than this above the ArUco plate
MIN_FLANGE_HEIGHT = 0.06  # flange above the table [m]
MIN_HOOP_HEIGHT = 0.045  # hoop center above the table [m]
TIME_LIMIT = 120.0  # seconds per attempt
MAX_JOINT_STEP = 0.6  # rad between consecutive waypoints; larger means a branch flip


@dataclass
class Tool:
    """Hoop center in the flange frame. The ring normal is the flange z axis, the handle points back to the flange axis."""

    tcp: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.0, 0.06]))

    @property
    def T_flange_hoop(self) -> np.ndarray:
        x = -np.array([self.tcp[0], self.tcp[1], 0.0])
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        T = np.eye(4)
        T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
        T[:3, 3] = self.tcp
        return T


def flange_pose_for_hoop(T_robot_hoop: np.ndarray, tool: Tool) -> np.ndarray:
    return T_robot_hoop @ np.linalg.inv(tool.T_flange_hoop)


def joint_speed_limits(robot, scale: float = 0.5) -> np.ndarray:
    """Joint speed [rad/s] = scale * the default speeds of the controller."""
    irc_per_s = np.abs(robot._default_speed_irc256_per_ms) / 256 * 1000
    return scale * np.deg2rad(irc_per_s / np.abs(robot._deg_to_irc))


@dataclass
class Plan:
    success: bool
    reason: str = ""
    trajectory: list = field(default_factory=list)  # [(t, q), ...]
    phases: list = field(default_factory=list)  # phase label per trajectory sample
    hint: tuple = (0.0, 1.0, 0.0)
    min_hoop_z: float = float("nan")  # lowest hoop center above the table along the trajectory [m]
    min_flange_z: float = float("nan")

    @property
    def duration(self) -> float:
        return self.trajectory[-1][0] - self.trajectory[0][0] if self.trajectory else 0.0


def _ik_in_limits(robot, flange: np.ndarray, tool_height_check=True) -> list[np.ndarray]:
    out = []
    for q in robot.ik(flange):
        if robot.in_limits(q) and robot.fk(q)[2, 3] >= MIN_FLANGE_HEIGHT:
            out.append(q)
    return out


def _nearest(sols, q):
    return min(sols, key=lambda s: np.linalg.norm(s - q))


def hint_candidates(name: str, n_yaw: int = 12) -> list[np.ndarray]:
    """Handle directions (puzzle frame) to try: the hoop may roll freely about the rod."""
    first = puzzles.hoop_path(name)[0][:3, :3]
    z0 = first[:, 2]
    base = np.cross(z0, [0.0, 0.0, 1.0] if abs(z0[2]) < 0.9 else [1.0, 0.0, 0.0])
    base /= np.linalg.norm(base)
    return [Rotation.from_rotvec(z0 * a).apply(base) for a in np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)]


def follow_path(robot, tool, path, T_robot_puzzle_at, q0, t0, vmax, v_path=0.02):
    """Greedy time-parameterized IK along `path` (poses in the puzzle frame) starting from q0 at t0.

    Returns (trajectory, reason): trajectory [(t, q)] and reason "" on success.
    """
    traj = [(t0, q0)]
    q, t = q0, t0
    for k in range(1, len(path)):
        step = np.linalg.norm(path[k][:3, 3] - path[k - 1][:3, 3])
        dt = max(step / v_path, 0.05)
        for _ in range(6):
            flange = flange_pose_for_hoop(T_robot_puzzle_at(t + dt) @ path[k], tool)
            sols = _ik_in_limits(robot, flange)
            if not sols:
                return traj, f"no IK solution at path waypoint {k}/{len(path)} (t={t + dt - t0:.1f} s)"
            q_new = _nearest(sols, q)
            need = float(np.max(np.abs(q_new - q) / vmax))
            dt_new = max(need, step / v_path, 0.05)
            if abs(dt_new - dt) < 1e-3:
                break
            dt = dt_new
        if np.max(np.abs(q_new - q)) > MAX_JOINT_STEP:
            return traj, f"joint branch flip at path waypoint {k}/{len(path)}"
        t, q = t + dt, q_new
        traj.append((t, q))
    return traj, ""


ROD_CLEARANCE = 0.05  # hoop center stays this far from the rod centerline while approaching [m]
LIFT_HEIGHTS = (0.10, 0.18)  # via point above the entry pose, tried when the direct approach is not safe


def _segment(robot, tool, q0, q1, t0, t1, pose_at, rod):
    """Joint-space linear interpolation q0->q1 over [t0, t1]; returns (samples, reason)."""
    n = max(int(np.ceil((t1 - t0) / 0.25)), 2)
    samples = []
    for s in np.linspace(0.0, 1.0, n + 1):
        q, t = q0 + s * (q1 - q0), t0 + s * (t1 - t0)
        flange = robot.fk(q)
        hoop = (flange @ tool.T_flange_hoop)[:3, 3]
        if flange[2, 3] < MIN_FLANGE_HEIGHT or hoop[2] < MIN_HOOP_HEIGHT:
            return [], "approach would go too close to the table"
        if rod is not None:
            local = np.linalg.inv(pose_at(t)) @ np.append(hoop, 1.0)
            if np.min(np.linalg.norm(rod - local[:3], axis=1)) < ROD_CLEARANCE:
                return [], "approach would pass too close to the puzzle"
        samples.append((t, q))
    return samples, ""


def _entry_config(robot, tool, entry_pose, pose_at, t, q_ref):
    sols = _ik_in_limits(robot, flange_pose_for_hoop(pose_at(t) @ entry_pose, tool))
    return _nearest(sols, q_ref) if sols else None


def approach(robot, tool, entry_pose, T_robot_puzzle_at, q_now, t_now, vmax, margin=1.0, rod=None, horizon=45.0):
    """Move from q_now to the (moving) entry pose, directly or via a point above it.

    Scans candidate arrival times t_entry (the table rotates, so reachability depends on t_entry) and returns the
    earliest safe route; the robot stands still at q_now until it has to leave. Returns (trajectory, q_entry,
    t_entry, reason).
    """
    reason = "entry pose of the hoop is not reachable"

    def timed(q0, q1):
        return float(np.max(np.abs(q1 - q0) / vmax)) + margin

    def build(t_entry, q_entry, lift):
        if lift is None:
            dt = timed(q_now, q_entry)
            t_s = t_entry - dt
            if t_s < t_now - 1e-9:
                return None, ""
            seg, why = _segment(robot, tool, q_now, q_entry, t_s, t_entry, T_robot_puzzle_at, rod)
            return ([(t_now, q_now)] if t_s > t_now else []) + seg if not why else None, why
        lifted = entry_pose.copy()
        lifted[2, 3] += lift
        t_v = t_entry - 3.0
        for _ in range(3):
            q_via = _entry_config(robot, tool, lifted, T_robot_puzzle_at, t_v, q_entry)
            if q_via is None:
                return None, ""
            t_v = t_entry - timed(q_via, q_entry)
        dt1 = timed(q_now, q_via)
        t_s1 = t_v - dt1
        if t_s1 < t_now - 1e-9:
            return None, ""
        first, why1 = _segment(robot, tool, q_now, q_via, t_s1, t_v, T_robot_puzzle_at, rod)
        second, why2 = _segment(robot, tool, q_via, q_entry, t_v, t_entry, T_robot_puzzle_at, rod)
        if why1 or why2:
            return None, why1 or why2
        return ([(t_now, q_now)] if t_s1 > t_now else []) + first + second[1:], ""

    for t_entry in np.arange(t_now + 1.0, t_now + horizon, 0.5):
        q_entry = _entry_config(robot, tool, entry_pose, T_robot_puzzle_at, t_entry, q_now)
        if q_entry is None:
            continue
        for lift in (None,) + LIFT_HEIGHTS:
            traj, why = build(t_entry, q_entry, lift)
            if traj:
                return traj, q_entry, float(t_entry), ""
            reason = why or reason
    return [], None, t_now + horizon, reason


def plan(robot, tool: Tool, puzzle: str, T_robot_puzzle_at, q_now, t_now=0.0, speed_scale=0.5, v_path=0.02, hints=None) -> Plan:
    """Plan approach + insertion for one puzzle; tries several handle directions and returns the fastest success."""
    vmax = joint_speed_limits(robot, speed_scale)
    best, last_reason = None, "no candidate"
    rod = puzzles.centerline(puzzle, 0.004)
    for hint in hints if hints is not None else hint_candidates(puzzle):
        path = puzzles.hoop_path(puzzle, handle_hint=hint)
        a_traj, q_entry, t_entry, reason = approach(robot, tool, path[0], T_robot_puzzle_at, q_now, t_now, vmax,
                                                    rod=rod)
        if reason:
            last_reason = reason
            continue
        f_traj, reason = follow_path(robot, tool, path, T_robot_puzzle_at, q_entry, t_entry, vmax, v_path)
        if reason:
            last_reason = reason
            continue
        traj = a_traj + f_traj[1:]
        end_hoop_z = T_robot_puzzle_at(traj[-1][0])[2, 3] + path[-1][2, 3]
        if traj[-1][0] - t_now > TIME_LIMIT:
            last_reason = f"trajectory takes {traj[-1][0] - t_now:.0f} s (limit {TIME_LIMIT:.0f} s)"
            continue
        if end_hoop_z > PLATFORM_HEIGHT + SUCCESS_HEIGHT:
            last_reason = f"hoop ends {end_hoop_z - PLATFORM_HEIGHT:.3f} m above the plate"
            continue
        phases = ["approach"] * len(a_traj[:-1]) + ["insert"] * len(f_traj)
        cand = Plan(True, "", traj, phases, tuple(hint))
        poses = [robot.fk(q) for _, q in traj]
        cand.min_flange_z = float(min(T[2, 3] for T in poses))
        cand.min_hoop_z = float(min((T @ tool.T_flange_hoop)[2, 3] for T in poses))
        if best is None or cand.duration < best.duration:
            best = cand
    return best if best is not None else Plan(False, last_reason)


def entry_reachable(robot, tool, puzzle, T_robot_puzzle_at, t, hints) -> bool:
    for hint in hints:
        entry = puzzles.hoop_path(puzzle, handle_hint=hint, step=0.05)[0]
        if _ik_in_limits(robot, flange_pose_for_hoop(T_robot_puzzle_at(t) @ entry, tool)):
            return True
    return False


def quick_feasible(robot, tool, puzzle, T_robot_puzzle_at, t_entry, hints, v_path=0.02, spacing=0.015) -> list:
    """Cheap pre-check: the handle directions for which IK exists at sparse waypoints."""
    passing = []
    for hint in hints:
        path = puzzles.hoop_path(puzzle, handle_hint=hint, step=0.004)
        pos = np.array([p[:3, 3] for p in path])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
        picks = np.unique(np.searchsorted(s, np.arange(0.0, s[-1], spacing)).clip(0, len(path) - 1))
        if all(_ik_in_limits(robot, flange_pose_for_hoop(T_robot_puzzle_at(t_entry + s[k] / v_path) @ path[k], tool))
               for k in picks):
            passing.append(hint)
    return passing


def plan_with_wait(robot, tool, puzzle, T_robot_puzzle_at, q_now, t_now=0.0, omega=0.0, max_wait=60.0,
                   speed_scale=0.5, v_path=0.02, max_candidates=25) -> Plan:
    """Like plan(), but the robot may wait (standing still) for the table to rotate into a reachable angle.

    The returned trajectory starts with the waiting samples so its timing is absolute. Start times are screened
    by a cheap IK check (the insertion starts a few seconds after the start, so several entry delays are tried);
    if the screening finds nothing, a coarse grid of start times is planned directly.
    """
    hints = hint_candidates(puzzle)
    step = 1.0 if abs(omega) < 1e-6 else float(np.clip(np.deg2rad(5.0) / abs(omega), 0.25, 1.0))
    waits = np.arange(0.0, max_wait + 1e-9, step)

    def attempts():
        screened = []
        for wait in waits:
            passing = []
            for entry_delay in (3.0, 6.0, 10.0):
                passing = quick_feasible(robot, tool, puzzle, T_robot_puzzle_at, t_now + wait + entry_delay, hints, v_path)
                if passing:
                    break
            if passing:
                screened.append(wait)
                yield wait, passing, 0
        # fallback: all handle directions at screened start times, then an unscreened coarse grid
        for wait in screened[:6] + list(waits[:: max(int(5.0 / step), 1)]):
            yield wait, hints, 1

    reason, tried, seen = "no start time within the waiting window gives a feasible plan", [0, 0], set()
    for wait, candidates, phase in attempts():
        if tried[phase] >= max_candidates:
            continue
        key = (round(float(wait), 3), len(candidates))
        if key in seen:
            continue
        seen.add(key)
        result = plan(robot, tool, puzzle, T_robot_puzzle_at, q_now, t_now + wait, speed_scale, v_path, candidates)
        tried[phase] += 1
        if result.success and result.duration + wait <= TIME_LIMIT:
            if wait > 0:
                result.trajectory = [(t_now, q_now)] + result.trajectory
                result.phases = ["wait"] + result.phases
            return result
        reason = result.reason or reason
    return Plan(False, reason)
