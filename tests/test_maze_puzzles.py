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
