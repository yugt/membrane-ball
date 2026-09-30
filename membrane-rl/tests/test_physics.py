"""Correctness tests for the physics core.

The energy test is the load-bearing one: these labels are only trustworthy
because the dynamics conserve energy, so if this regresses, every dataset
generated afterwards is suspect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.physics import MembraneSim, Params, State  # noqa: E402
from membrane_rl.rewards import RewardConfig, compute_reward, parse_prediction  # noqa: E402


REFERENCE_IC = State(pos=np.array([0.2, 0.1, 4.0]), vel=np.array([1.5, 1.0, 0.0]))


def _leak(sim, steps=5000, move=None):
    e0 = sim.energy()["total"]
    worst = 0.0
    for i in range(steps):
        if move is not None:
            move(sim, i)
        sim.step()
        worst = max(worst, abs(sim.energy()["total"] - e0))
    return worst / e0 * 100.0


def test_matches_upstream_reference():
    """Bit-for-bit agreement with verify_energy.py in yugt/membrane-ball.

    Pinned so a refactor that silently changes the dynamics cannot slip through
    and quietly corrupt every label.
    """
    sim = MembraneSim(Params())
    sim.reset(REFERENCE_IC.copy())
    leak = _leak(sim)
    assert leak == pytest.approx(3.232784, abs=1e-4)
    assert sim.n_ring_bounces == 34
    assert sim.n_cyl_bounces == 117


@pytest.mark.parametrize("center", [(0.35, -0.20), (-0.40, 0.10), (0.25, 0.30)])
def test_offset_but_fixed_frame_conserves_energy(center):
    """A frame that is off-centre but FIXED is still an autonomous system."""
    sim = MembraneSim(Params(frame_offset_max=0.4))
    start = State(pos=np.array([center[0] + 0.2, center[1] + 0.1, 4.0]),
                  vel=np.array([1.5, 1.0, 0.0]))
    sim.reset(start, frame_center=center)
    assert _leak(sim) < 5.0


def test_moving_frame_breaks_conservation():
    """A moving frame does work on the ball, so energy is not conserved.

    This is why the frame is fixed within an episode: a time-dependent
    constraint invalidates the energy check that makes the labels trustworthy.
    """
    sim = MembraneSim(Params(frame_offset_max=0.4))
    sim.reset(REFERENCE_IC.copy(), frame_center=(0.0, 0.0))

    def wobble(s, i):
        s.frame_center = np.array([0.30 * np.sin(2 * np.pi * 0.5 * i * s.p.dt), 0.0])

    assert _leak(sim, move=wobble) > 5.0


def test_determinism():
    a = MembraneSim(Params(frame_offset_max=0.4), seed=7)
    b = MembraneSim(Params(frame_offset_max=0.4), seed=7)
    a.reset(); b.reset()
    a.rollout(200); b.rollout(200)
    assert np.allclose(a.state.pos, b.state.pos)
    assert np.allclose(a.frame_center, b.frame_center)


def test_frame_must_fit_inside_cylinder():
    with pytest.raises(ValueError):
        MembraneSim(Params(frame_offset_max=1.0))   # 1.0 + 1.0 > 1.45


def test_ball_stays_in_arena():
    sim = MembraneSim(Params(frame_offset_max=0.4), seed=3)
    for _ in range(15):
        sim.reset()
        for _ in range(400):
            sim.step()
            r = np.hypot(*sim.state.pos[:2])
            assert r <= sim.p.r_cyl + 1e-6


# ----------------------------------------------------------------------
def test_reward_is_monotone_in_error():
    cfg = RewardConfig()
    target = [0.0, 0.0, 1.0]
    rewards = [
        compute_reward(f"<answer>0, 0, {1.0 + d}</answer>", target, cfg)["reward"]
        for d in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    assert rewards == sorted(rewards, reverse=True)


def test_unparseable_is_not_scored_as_huge_error():
    out = compute_reward("I think it lands somewhere near the middle.", [0, 0, 1])
    assert out["parsed"] == 0.0
    assert out["reward"] == 0.0
    assert out["format"] == 0.0


@pytest.mark.parametrize("text,expected", [
    ("<answer>1, 2, 3</answer>", [1, 2, 3]),
    ("blah <answer> -0.5 0.25 1e-1 </answer> blah", [-0.5, 0.25, 0.1]),
    ("<ANSWER>1,2,3</ANSWER>", [1, 2, 3]),
    ("<answer>1, 2</answer>", None),
    ("no tags 1 2 3", None),
])
def test_parse_prediction(text, expected):
    got = parse_prediction(text)
    if expected is None:
        assert got is None
    else:
        assert np.allclose(got, expected)
