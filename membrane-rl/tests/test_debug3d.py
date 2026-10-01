"""The motion audit behind the 3D debug view: every change in the ball's
motion must have a logged cause."""

from __future__ import annotations

import numpy as np
import pytest

from membrane_rl.anomalies import KINDS, Anomaly
from membrane_rl.debug3d import audit, record_episode, video_camera
from membrane_rl.physics import Params
from membrane_rl.render import RenderConfig, Renderer

P = Params(frame_offset_max=0.3)


@pytest.mark.parametrize("seed", range(4))
def test_normal_episode_has_no_unexplained_frames(seed):
    rec = record_episode(375, seed=seed, params=P, grid=5)
    s = rec.summary()
    assert s["unexplained_frames"] == []
    assert s["max_free_fall_acc_residual"] < 1e-9
    assert "anomaly" not in s["status_counts"]


@pytest.mark.parametrize("kind", KINDS)
def test_anomalies_are_attributed_not_unexplained(kind):
    rec = record_episode(375, seed=3, params=P, anomaly=Anomaly(kind, start_frame=120), grid=5)
    s = rec.summary()
    assert s["unexplained_frames"] == []
    assert s["status_counts"].get("anomaly", 0) >= 1
    assert all(a.status == "anomaly" for a in rec.audit if a.frame == 120)


def test_unlogged_change_is_caught():
    rec = record_episode(375, seed=1, params=P, grid=5)
    free = [a.frame for a in rec.audit if a.status == "free fall"]
    i = free[len(free) // 2]
    rec.episode.states[i].vel[0] += 0.5          # a kick nobody logged
    flagged = {a.frame for a in audit(rec) if a.status == "UNEXPLAINED"}
    assert i in flagged


def test_video_camera_matches_renderer():
    """The debug view's default camera looks from where the mp4 camera does:
    screen right, screen up and depth agree with the renderer."""
    cfg = RenderConfig.for_size(320)
    r = Renderer(cfg)
    e = video_camera(cfg)["eye"]
    eye = np.array([e["x"], e["y"], e["z"]])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        (ax, ay), (bx, by) = r.project(*a), r.project(*b)
        d = b - a
        # screen y grows downward in the image
        assert (bx - ax) == pytest.approx(d @ right * cfg.scale, abs=1e-6)
        assert (ay - by) == pytest.approx(d @ up * cfg.scale, abs=1e-6)
        assert np.sign(r.depth(*b) - r.depth(*a)) == np.sign(d @ fwd)


def test_figure_builds():
    pytest.importorskip("plotly")
    from membrane_rl.debug3d import debug_figure
    rec = record_episode(40, seed=0, params=P, grid=5)
    fig = debug_figure(rec, stride=5)
    assert len(fig.frames) == 8
