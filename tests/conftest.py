import pytest


@pytest.fixture(autouse=True)
def no_z_compensation(monkeypatch):
    """The hand-tuned Z_COMPENSATION (maze/tuning.py) is for the real robot; tests assume a perfect one."""
    from robotics_toolbox.maze import planning

    monkeypatch.setattr(planning, "Z_COMPENSATION", 0.0)
