"""Fast path: pixels -> 3D state -> physics twin (forecast + anomaly alarm)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.anomalies import Anomaly  # noqa: E402
from membrane_rl.episode import run_episode  # noqa: E402
from membrane_rl.perception import Camera, estimate_frame_center, observe  # noqa: E402
from membrane_rl.render import RenderConfig, Renderer  # noqa: E402
from membrane_rl.twin import PhysicsTwin, run_twin  # noqa: E402

SIZE = 480


def _frames(n, seed, anomaly=None):
    """Render in memory (no mp4 round trip) -- perception logic, not codecs."""
    r = Renderer(RenderConfig.for_size(SIZE))
    frames = []
    ep, _ = run_episode(n, seed=seed, anomaly=anomaly,
                        sim_hook=lambda sim, f: frames.append(np.asarray(r.render(sim))))
    return ep, frames


@pytest.fixture(scope="module")
def normal():
    return _frames(160, seed=11)


def test_camera_inverts_projection():
    cfg = RenderConfig.for_size(SIZE)
    r, cam = Renderer(cfg), Camera(cfg)
    for x, y, z in [(0.3, -0.2, 0.0), (-0.7, 0.4, 2.5), (0.0, 0.9, -0.2)]:
        sx, sy = r.project(x, y, 0.0)
        gx, gy = cam.ground(sx, sy)
        assert (gx, gy) == pytest.approx((x, y), abs=1e-9)
        assert cam.height(x, y, r.project(x, y, z)[1]) == pytest.approx(z, abs=1e-9)


def test_perception_recovers_3d_position(normal):
    ep, frames = normal
    cam = Camera(RenderConfig.for_size(SIZE))
    err = [np.abs(observe(f, cam).pos - s.pos) for f, s in zip(frames, ep.states)]
    err = np.array([e for e in err if np.isfinite(e).all()])
    assert len(err) > 0.9 * len(frames)
    assert np.median(err) < 0.02            # ~4 % of a ball radius
    fc = estimate_frame_center(frames, cam)
    assert np.linalg.norm(fc - ep.frame_center) < 0.03


def test_twin_tracks_normal_clip_without_alarm_and_forecasts(normal):
    ep, frames = normal
    twin = run_twin(frames, ep.params, SIZE)
    assert twin.locked and twin.alarms == []
    # forecast from mid-clip state lands near the truth 0.5 s later
    twin2 = PhysicsTwin(ep.params, estimate_frame_center(frames, Camera(RenderConfig.for_size(SIZE))))
    cam = Camera(RenderConfig.for_size(SIZE))
    for f in frames[:80]:
        twin2.update(observe(f, cam).pos)
    pred = twin2.forecast(31)[-1]
    assert np.linalg.norm(pred - ep.states[79 + 31].pos) < 0.25   # half a ball radius


@pytest.mark.parametrize("kind", ["teleport", "hover", "energy_kick"])
def test_twin_raises_alarm_at_anomaly(kind):
    mag = {"teleport": 2.0, "energy_kick": 0.8}.get(kind, 1.0)
    ep, frames = _frames(160, seed=11, anomaly=Anomaly(kind, 90, 40, mag))
    twin = run_twin(frames, ep.params, SIZE)
    t0 = 90 * ep.params.dt
    assert any(t0 - 0.05 <= a["t"] <= t0 + 0.3 for a in twin.alarms), twin.alarms
