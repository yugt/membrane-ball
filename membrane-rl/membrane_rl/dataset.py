"""Trajectory-prediction dataset generation.

One sample = k rendered frames + the ball's true position N frames later.

The important object here is not the dataset, it is ``naive_ballistic``. It
answers the question that decides whether this whole project is worth running:
how much of the task is solved by "the ball keeps going the way it was going"?
If straight-line extrapolation already scores well at your chosen horizon, the
task has no headroom and RL will teach the model nothing. You calibrate N until
the naive baseline is clearly beaten by the true dynamics -- that is, until the
horizon reliably straddles a membrane contact.

This calibration costs zero GPU-hours and should be done before any training
code is written.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .physics import MembraneSim, Params, State
from .render import Renderer, RenderConfig

__all__ = ["SampleConfig", "naive_ballistic", "make_sample", "build_dataset", "PROMPT"]


PROMPT = """You are shown {k} frames from a physics simulation, in time order, {dt_ms} ms apart.

Setup: a ball (radius {R}) falls under gravity inside a vertical cylinder of radius {r_cyl}, \
centred on the origin. A circular elastic membrane of radius {r_frame} is clamped in the \
horizontal plane at z = 0 and is drawn as the green ring. The membrane does not move, but it is \
NOT always centred on the origin -- read its position off the green ring in the image. When the \
ball presses into the membrane it stretches and pushes back, and the membrane gets stiffer the \
further the contact point is from the ring's centre.

Axes: x and y are horizontal, z is vertical (up is positive), origin at the centre of the membrane ring.

Question: where is the centre of the ball {horizon_ms} ms after the LAST frame shown?

Think step by step about the ball's current velocity and whether it will contact the membrane \
during that interval. Then give your final answer as three numbers, x y z, inside answer tags:

<answer>x, y, z</answer>"""


@dataclass(frozen=True)
class SampleConfig:
    n_input_frames: int = 3      # k
    frame_stride: int = 4        # frames skipped between shown frames
    horizon: int = 25            # N frames ahead to predict
    warmup_min: int = 20
    warmup_max: int = 140
    require_contact_in_horizon: bool = False
    max_attempts: int = 40


def naive_ballistic(state: State, n_frames: int, p: Params) -> np.ndarray:
    """Gravity-only extrapolation, ignoring the membrane entirely.

    This is the "dumb" predictor. Its error is the yardstick every model result
    must be compared against; a model that does not beat it has learned nothing
    about the membrane.
    """
    t = n_frames * p.dt
    pos = np.asarray(state.pos, dtype=float).copy()
    vel = np.asarray(state.vel, dtype=float)
    pos[0] += vel[0] * t
    pos[1] += vel[1] * t
    pos[2] += vel[2] * t - 0.5 * p.gravity * t * t
    return pos


def make_sample(sim: MembraneSim, renderer: Renderer, cfg: SampleConfig):
    """Generate one sample. Returns ``(frames, record)`` or None if rejected."""
    p = sim.p
    for _ in range(cfg.max_attempts):
        sim.reset()
        warmup = int(sim.rng.integers(cfg.warmup_min, cfg.warmup_max + 1))
        sim.rollout(warmup)

        # ball fell out of the interesting region -- resample
        if sim.state.pos[2] < -2.0 or sim.state.pos[2] > 6.0:
            continue

        frames = []
        for i in range(cfg.n_input_frames):
            frames.append(renderer.render(sim))
            if i < cfg.n_input_frames - 1:
                sim.rollout(cfg.frame_stride)

        current = sim.state.copy()
        contact_before = sim.n_contact_frames
        cyl_before, ring_before = sim.n_cyl_bounces, sim.n_ring_bounces

        sim.rollout(cfg.horizon)
        target = sim.state.copy()

        contact_frames = sim.n_contact_frames - contact_before
        if cfg.require_contact_in_horizon and contact_frames == 0:
            continue

        naive = naive_ballistic(current, cfg.horizon, p)
        naive_err = float(np.linalg.norm(naive - target.pos))

        record = {
            "prompt": PROMPT.format(
                k=cfg.n_input_frames,
                dt_ms=int(round(cfg.frame_stride * p.dt * 1000)),
                R=p.ball_radius,
                r_cyl=p.r_cyl,
                r_frame=p.r_frame,
                horizon_ms=int(round(cfg.horizon * p.dt * 1000)),
            ),
            "answer": [round(float(v), 4) for v in target.pos],
            "current_state": {
                "pos": [round(float(v), 4) for v in current.pos],
                "vel": [round(float(v), 4) for v in current.vel],
            },
            "frame_center": [round(float(v), 4) for v in sim.frame_center],
            "frame_offset": round(float(np.linalg.norm(sim.frame_center)), 4),
            "naive_prediction": [round(float(v), 4) for v in naive],
            "naive_error": round(naive_err, 4),
            "naive_error_radii": round(naive_err / p.ball_radius, 4),
            "contact_frames_in_horizon": int(contact_frames),
            "cyl_bounces_in_horizon": int(sim.n_cyl_bounces - cyl_before),
            "ring_bounces_in_horizon": int(sim.n_ring_bounces - ring_before),
            "energy": {k: round(v, 4) for k, v in sim.energy().items()},
        }
        return frames, record
    return None


def calibrate_horizon(
    horizons,
    n_per: int = 400,
    params: Params | None = None,
    sample_cfg: SampleConfig | None = None,
    seed: int = 1234,
) -> list[dict]:
    """Sweep the prediction horizon N and report how hard the task is.

    Renders nothing -- this is physics only, so it runs in seconds and costs
    nothing. Run it BEFORE writing any training code.

    What to look for, per horizon:

    * ``frac_contact`` -- fraction of samples where the ball touches the
      membrane inside the horizon. These are the only samples that carry any
      information about the membrane; if this is near zero the task is pure
      ballistics and a VLM will not learn physics from it.
    * ``naive_err_contact`` -- median error of gravity-only extrapolation on
      exactly those samples, in ball radii. This is the floor a model must
      beat. If it is small, there is no headroom. If it is enormous, the target
      is effectively unpredictable and the reward will be noise.

    The sweet spot is a horizon where a healthy fraction of samples involve
    contact AND the naive error on them is several ball radii, while the
    no-contact samples stay easy -- that spread is what gives GRPO in-group
    variance to learn from.
    """
    params = params or Params()
    cfg = sample_cfg or SampleConfig()
    horizons = sorted({int(h) for h in horizons})
    h_max = horizons[-1]

    # One rollout per episode serves every horizon: step to h_max and read off
    # the state at each checkpoint. Simulating each horizon separately costs
    # len(horizons)x more for identical numbers.
    acc = {h: {"contact": [], "free": []} for h in horizons}
    sim = MembraneSim(params, seed=seed)

    for _ in range(n_per):
        sim.reset()
        warmup = int(sim.rng.integers(cfg.warmup_min, cfg.warmup_max + 1))
        sim.rollout(warmup)
        if sim.state.pos[2] < -2.0 or sim.state.pos[2] > 6.0:
            continue
        sim.rollout(cfg.frame_stride * (cfg.n_input_frames - 1))

        current = sim.state.copy()
        contact_before = sim.n_contact_frames

        nxt = 0
        for frame in range(1, h_max + 1):
            sim.step()
            while nxt < len(horizons) and horizons[nxt] == frame:
                h = horizons[nxt]
                err = float(np.linalg.norm(
                    naive_ballistic(current, h, params) - sim.state.pos
                )) / params.ball_radius
                key = "contact" if sim.n_contact_frames - contact_before > 0 else "free"
                acc[h][key].append(err)
                nxt += 1

    out = []
    for h in horizons:
        c, f = acc[h]["contact"], acc[h]["free"]
        total = len(c) + len(f)
        out.append({
            "horizon": h,
            "horizon_ms": int(round(h * params.dt * 1000)),
            "n": total,
            "frac_contact": round(len(c) / total, 3) if total else None,
            "naive_err_contact": round(float(np.median(c)), 3) if c else None,
            "naive_err_free": round(float(np.median(f)), 3) if f else None,
        })
    return out


def build_dataset(
    out_dir: str | Path,
    n_samples: int,
    seed: int = 0,
    sample_cfg: SampleConfig | None = None,
    params: Params | None = None,
    render_cfg: RenderConfig | None = None,
    split: str = "train",
) -> dict:
    """Write ``n_samples`` samples as PNGs + a JSONL manifest.

    Layout is deliberately plain (images on disk, one JSON object per line) so
    it can be adapted to whichever trainer you end up using without a rewrite.
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    sample_cfg = sample_cfg or SampleConfig()
    params = params or Params()
    sim = MembraneSim(params, seed=seed)
    renderer = Renderer(render_cfg)

    manifest = out_dir / split / "data.jsonl"
    naive_errs, contact_counts, rejected = [], [], 0

    with manifest.open("w") as fh:
        for idx in range(n_samples):
            got = make_sample(sim, renderer, sample_cfg)
            if got is None:
                rejected += 1
                continue
            frames, record = got
            paths = []
            for fi, frame in enumerate(frames):
                rel = f"images/{idx:06d}_{fi}.png"
                frame.save(out_dir / split / rel, optimize=True)
                paths.append(rel)
            record["images"] = paths
            record["id"] = idx
            fh.write(json.dumps(record) + "\n")
            naive_errs.append(record["naive_error_radii"])
            contact_counts.append(record["contact_frames_in_horizon"])

    stats = {
        "split": split,
        "n_written": len(naive_errs),
        "n_rejected": rejected,
        "sample_config": asdict(sample_cfg),
        "params": asdict(params),
        "naive_error_radii_mean": float(np.mean(naive_errs)) if naive_errs else None,
        "naive_error_radii_median": float(np.median(naive_errs)) if naive_errs else None,
        "frac_with_contact": float(np.mean([c > 0 for c in contact_counts])) if contact_counts else None,
    }
    (out_dir / split / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
