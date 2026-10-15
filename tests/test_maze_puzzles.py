import numpy as np
import pytest

from robotics_toolbox.maze import puzzles


def test_lengths_match_drawings():
    assert puzzles.rod_length("A") == pytest.approx(0.200)
    assert puzzles.rod_length("B") == pytest.approx(0.080 + 0.070 * np.sqrt(2) + 0.050, abs=1e-3)
    d = 0.030 + 0.015 * np.sqrt(2) + 0.060 + 0.050 * np.pi / 2 + 0.050
    assert puzzles.rod_length("D") == pytest.approx(d, abs=2e-3)


@pytest.mark.parametrize("name", puzzles.PUZZLES)
def test_hoop_path_is_smooth_and_ends_at_foot(name):
    path = puzzles.hoop_path(name)
    pos = np.array([p[:3, 3] for p in path])
    assert np.linalg.norm(pos[-1][:2]) < 1e-6 and pos[-1][2] == pytest.approx(puzzles.END_Z)
    assert np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)) < 0.006
    for p in path:
        assert np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-9)
    # successive orientations change slowly (no sudden roll flips)
    for a, b in zip(path[:-1], path[1:]):
        cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
        assert np.arccos(np.clip(cos, -1, 1)) < 0.25


def test_hoop_path_starts_above_tip_for_a():
    path = puzzles.hoop_path("A")
    assert path[0][:3, 3] == pytest.approx([0, 0, 0.200 + puzzles.APPROACH], abs=1e-6)
    assert path[0][:3, 2] == pytest.approx([0, 0, -1])


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_horizontal_hoop_clears_the_rod_everywhere(name):
    """With the ring kept horizontal the 10x10 mm rod must stay well inside the 62 mm ring along the whole path."""
    line = puzzles.centerline(name, 0.001)
    offsets = np.array([[dx, dy, dz] for dx in (-0.005, 0.005) for dy in (-0.005, 0.005) for dz in (-0.005, 0.005)])
    rod = (line[:, None, :] + offsets[None]).reshape(-1, 3)
    for pose in puzzles.hoop_path(name):
        assert pose[:3, 2] == pytest.approx([0, 0, -1])
        c = pose[:3, 3]
        in_slab = rod[np.abs(rod[:, 2] - c[2]) <= 0.0033]  # ring thickness 6.6 mm
        if len(in_slab):
            assert np.max(np.linalg.norm(in_slab[:, :2] - c[:2], axis=1)) < 0.0312 - 0.006  # inner radius minus 6 mm


def test_d_still_turns_the_hoop():
    path = puzzles.hoop_path("D")
    tilts = [np.degrees(np.arccos(-p[2, 2])) for p in path]
    assert max(tilts) > 60  # at the horizontal arm the ring normal is nearly horizontal
