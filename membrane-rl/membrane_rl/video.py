"""Episode -> .mp4 (+ answer-key JSON).

Two flavours of the same clip:

* ``annotate=False`` -- what the video agent sees. No labels, no HUD: anything
  drawn here would leak the answer.
* ``annotate=True``  -- for humans (demo, debugging). Time, energy and the
  ground-truth event labels are burned in.

The encoder is ``imageio-ffmpeg``, whose wheel bundles an ffmpeg binary, so this
works on a fresh VM with ``pip install`` and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from .anomalies import Anomaly
from .episode import Episode, run_episode
from .physics import Params
from .render import RenderConfig, Renderer

__all__ = ["render_clip"]

_LABEL_HOLD_S = 0.6        # how long an event label stays on screen
_LABEL_COLOURS = {
    "contact_start": (80, 200, 255),
    "contact_end": (80, 200, 255),
    "ring_bounce": (50, 220, 120),
    "wall_bounce": (170, 180, 200),
    "apex": (255, 226, 78),
    "anomaly_start": (255, 70, 70),
    "anomaly_end": (255, 70, 70),
}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except TypeError:                                 # older Pillow
        return ImageFont.load_default()


def _annotate(img, frame: int, ep_events: list[dict], t: float, energy: float, e0: float):
    d = ImageDraw.Draw(img)
    s = img.size[0]
    f = _font(max(12, s // 34))
    d.text((s * 0.03, s * 0.03), f"t = {t:5.2f} s", fill=(220, 220, 230), font=f)
    d.text((s * 0.03, s * 0.03 + s // 26), f"E = {energy:6.2f}  ({100 * (energy - e0) / e0:+.1f}%)",
           fill=(220, 220, 230), font=f)
    y = s * 0.03
    for ev in ep_events:
        if 0.0 <= t - ev["t"] <= _LABEL_HOLD_S:
            label = ev["type"] + (f": {ev['kind']}" if "kind" in ev else "")
            w = d.textlength(label, font=f)
            d.text((s * 0.97 - w, y), label, fill=_LABEL_COLOURS.get(ev["type"], (255, 255, 255)), font=f)
            y += s // 26


def render_clip(
    out_path: str | Path,
    n_frames: int = 375,
    seed: int = 0,
    params: Params | None = None,
    anomaly: Anomaly | None = None,
    size: int = 720,
    annotate: bool = False,
    frame_center=None,
) -> dict:
    """Simulate, render and encode one clip; write ``<out>.json`` next to it.

    Returns the answer key. ``n_frames=375`` is 6 s at 62.5 fps.
    """
    import imageio_ffmpeg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(RenderConfig.for_size(size))

    # First pass: physics only, so the annotated flavour knows about events
    # before they happen on screen. Rendering happens on a replay below.
    ep, _ = run_episode(n_frames, seed=seed, params=params, anomaly=anomaly,
                        frame_center=frame_center)

    writer = imageio_ffmpeg.write_frames(
        str(out_path), (size, size), fps=ep.fps, codec="libx264",
        pix_fmt_out="yuv420p", quality=7, macro_block_size=8,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    e0 = ep.energy[0]

    def draw(sim, frame):
        img = renderer.render(sim)
        if annotate:
            _annotate(img, frame, ep.events, frame * ep.params.dt, sim.energy()["total"], e0)
        writer.send(np.asarray(img, dtype=np.uint8).tobytes())

    # Replay is deterministic (same seed, same anomaly), so it reproduces ep exactly.
    replay, _ = run_episode(n_frames, seed=seed, params=params, anomaly=anomaly,
                            frame_center=frame_center, sim_hook=draw)
    writer.close()
    assert replay.events == ep.events, "replay diverged -- simulation is not deterministic"

    key = ep.answer_key()
    key["video"] = out_path.name
    key["annotated"] = annotate
    key["size"] = size
    out_path.with_suffix(".json").write_text(json.dumps(key, indent=1))
    return key
