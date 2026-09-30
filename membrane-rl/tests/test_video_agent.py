"""Tests for the video-agent layer: event log, anomalies, scoring, video I/O."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.anomalies import KINDS, Anomaly  # noqa: E402
from membrane_rl.episode import run_episode  # noqa: E402
from membrane_rl.scoring import score_clip, truth_events  # noqa: E402


def test_event_log_matches_counters():
    ep, sim = run_episode(400, seed=1)
    types = [e["type"] for e in ep.events]
    assert types.count("ring_bounce") == sim.n_ring_bounces
    assert types.count("wall_bounce") == sim.n_cyl_bounces
    assert "contact_start" in types
    frames = [e["frame"] for e in ep.events]
    assert frames == sorted(frames)
    # contact starts and ends alternate
    cs = [t for t in types if t.startswith("contact")]
    assert all(a != b for a, b in zip(cs, cs[1:]))


def test_event_time_is_inside_its_frame():
    ep, _ = run_episode(300, seed=2)
    dt = ep.params.dt
    for e in ep.events:
        assert (e["frame"] - 1) * dt - 1e-9 <= e["t"] <= e["frame"] * dt + 1e-9


def test_replay_is_deterministic():
    a, _ = run_episode(200, seed=5, anomaly=Anomaly("teleport", 80, magnitude=2.0))
    b, _ = run_episode(200, seed=5, anomaly=Anomaly("teleport", 80, magnitude=2.0))
    assert a.events == b.events
    assert np.array_equal(a.states[-1].pos, b.states[-1].pos)


# Which anomalies an energy monitor can see. The two it cannot are the point
# of the demo: only the picture gives them away.
_ENERGY_VISIBLE = {"gravity_flip": True, "energy_kick": True, "hover": True,
                   "teleport": False, "membrane_off": False}


@pytest.mark.parametrize("kind", KINDS)
def test_anomaly_logged_and_energy_signature(kind):
    start = 100
    mag = {"teleport": 2.0, "energy_kick": 0.8}.get(kind, 1.0)
    ep, _ = run_episode(200, seed=1, anomaly=Anomaly(kind, start, 40, mag))
    starts = [e for e in ep.events if e["type"] == "anomaly_start"]
    assert len(starts) == 1 and starts[0]["frame"] == start and starts[0]["kind"] == kind

    e = np.array(ep.energy)
    before = e[start - 1]
    jump = np.max(np.abs(e[start:start + 60] - before)) / before
    if _ENERGY_VISIBLE[kind]:
        assert jump > 0.05, f"{kind} should move the energy"
    else:
        assert jump < 0.02, f"{kind} should be invisible to an energy check"


def test_membrane_off_ball_falls_through():
    ep, _ = run_episode(300, seed=1, anomaly=Anomaly("membrane_off", 150))
    assert min(s.pos[2] for s in ep.states) < -1.0
    assert all(np.isfinite(s.pos).all() for s in ep.states)


def test_teleport_is_a_single_frame_jump():
    ep, _ = run_episode(200, seed=1, anomaly=Anomaly("teleport", 100, magnitude=2.0))
    jumps = [np.linalg.norm(ep.states[i].pos[:2] - ep.states[i - 1].pos[:2]) for i in range(1, 200)]
    assert jumps[99] > 0.9 and max(jumps[:99] + jumps[100:]) < 0.2


# ----------------------------------------------------------------------
def _key(**kw):
    ep, _ = run_episode(300, seed=3, **kw)
    return ep.answer_key()


def test_oracle_scores_perfectly():
    key = _key(anomaly=Anomaly("hover", 120))
    pred = [{"t": e["t"], "label": e["label"]} for e in truth_events(key)]
    s = score_clip(key, pred)
    assert s["summary"]["all"]["f1"] == 1.0
    assert s["anomaly"]["detected"] and s["anomaly"]["latency_s"] == 0.0


def test_empty_prediction_scores_zero_recall():
    key = _key()
    s = score_clip(key, [])
    assert s["summary"]["all"]["recall"] == 0.0
    assert s["anomaly"] == {"present": False, "false_alarm": False}


def test_false_alarm_and_synonyms():
    key = _key()
    s = score_clip(key, [{"t": 1.0, "label": "Physics Violation"}, {"t": 2.0, "label": "banana"}])
    assert s["anomaly"]["false_alarm"] is True
    assert s["n_unrecognised"] == 1


def test_timing_tolerance():
    key = _key()
    first = truth_events(key)[0]
    near = score_clip(key, [{"t": first["t"] + 0.2, "label": first["label"]}], tol_s=0.3)
    far = score_clip(key, [{"t": first["t"] + 0.5, "label": first["label"]}], tol_s=0.3)
    assert near["per_label"][first["label"]]["tp"] == 1
    assert far["per_label"][first["label"]]["tp"] == 0


# ----------------------------------------------------------------------
def test_video_roundtrip_and_tracker(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from membrane_rl.tracker import read_mp4, track_ball
    from membrane_rl.video import render_clip

    key = render_clip(tmp_path / "c.mp4", n_frames=40, seed=1, size=240)
    assert json.loads((tmp_path / "c.json").read_text())["n_frames"] == 40
    frames = list(read_mp4(tmp_path / "c.mp4"))
    assert len(frames) == 40 and frames[0].shape == (240, 240, 3)
    track = track_ball(frames)
    assert np.isfinite(track).all()      # ball visible and found in every frame
    assert key["fps"] == pytest.approx(62.5)


# ----------------------------------------------------------------------
# agent plumbing (no network)
from membrane_rl.agent import AgentConfig, build_messages, merge_events, parse_events, windows  # noqa: E402


def test_windows_cover_clip_with_overlap():
    w = windows(6.0, 2.0, 1.5)
    assert w[0] == (0.0, 2.0) and w[-1][1] == 6.0
    assert all(b[0] < a[1] for a, b in zip(w, w[1:]))       # consecutive windows overlap


@pytest.mark.parametrize("text,n", [
    ('{"events": [{"t": 0.4, "label": "wall_bounce"}]}', 1),
    ('Sure!\n```json\n{"events": [{"t": 0.4, "label": "anomaly", "why": "x"}]}\n```', 1),
    ('<think>{"events": [{"t": 9}]}</think>{"events": []}', 0),
    ('[{"t": 1.0, "type": "ring_bounce"}, {"t": "abc"}]', 1),
    ('{"events": [{"t": 7.5, "label": "wall_bounce"}]}', 0),   # outside 2 s window
    ('no json here', 0),
])
def test_parse_events_is_tolerant(text, n):
    assert len(parse_events(text, 2.0)) == n


def test_simulated_agent_scores_perfectly_through_windowing():
    """Feed each window its own ground truth, as a perfect model would answer;
    shifting + overlap de-duplication must reproduce a perfect score."""
    key = _key(anomaly=Anomaly("teleport", 150, magnitude=2.0))
    truth = truth_events(key)
    per_window = []
    for start, end in windows(key["duration_s"], 2.0, 1.5):
        evs = [{"t": e["t"] - start, "label": e["label"]} for e in truth if start <= e["t"] < end]
        per_window.append((start, evs))
    s = score_clip(key, merge_events(per_window, dedup_s=0.25))
    assert s["summary"]["all"]["f1"] == 1.0


def test_build_messages_frames_mode():
    frames = [np.zeros((64, 64, 3), np.uint8)] * 3
    msgs = build_messages(frames, [0.0, 0.1, 0.2], AgentConfig(image_size=32))
    parts = msgs[1]["content"]
    assert sum(p["type"] == "image_url" for p in parts) == 3
    assert msgs[0]["role"] == "system"


def test_resolve_provider_presets_and_overrides():
    from membrane_rl.agent import resolve_provider
    base, model, key = resolve_provider("wandb", env={"WANDB_API_KEY": "k"})
    assert base.startswith("https://api.inference.wandb.ai") and model and key == "k"
    assert resolve_provider("wandb", model="m", env={})[1] == "m"
    with pytest.raises(ValueError):
        resolve_provider("nvidia", env={})              # no default Cosmos id on purpose
    with pytest.raises(ValueError):
        resolve_provider("openai", model="m", env={})   # needs --base-url
    assert resolve_provider("local", model="m", env={})[2] == ""
