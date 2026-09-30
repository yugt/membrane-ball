"""Model-free baseline detector: pixels in, events out.

It reads only the .mp4 (never the answer key): finds the yellow ball by colour,
tracks its screen position, and flags moments where the motion stops looking
like free fall. It exists for two reasons:

* a floor -- a VLM agent that cannot beat colour thresholding is not adding
  anything, and the scoreboard should show that honestly;
* a fallback -- if the hosted VLM is slow or down on demo day, this still
  produces a live event stream.

Why free fall is easy to test in screen space: the camera projection is linear
in (x, y, z) and gravity only acts along z, so between collisions the ball's
screen x is linear in time and screen y is quadratic with a KNOWN constant
second derivative. Anything else is an interaction or an anomaly.

Limits (by design -- these are what the VLM should add): it cannot say WHAT
the ball hit (membrane vs ring vs wall), and it cannot see the membrane at
all, so a ball falling through a switched-off membrane looks like free fall.
"""

from __future__ import annotations

import numpy as np

from .render import RenderConfig

__all__ = ["track_ball", "detect_events"]

BALL_RGB = np.array(RenderConfig().c_ball, dtype=float)


def track_ball(frames, colour_tol: float = 40.0, min_pixels: int = 30, stride: int = 2):
    """Screen centroid of the ball per frame (full-res pixels); NaN when not visible.

    ``stride`` subsamples pixels -- the ball is tens of pixels wide, so every
    other pixel loses nothing and runs ~4x faster.
    """
    out = []
    tol2 = colour_tol * colour_tol
    ref = BALL_RGB.astype(np.int32)
    for f in frames:
        f = np.asarray(f)[::stride, ::stride].astype(np.int32)
        d2 = ((f - ref) ** 2).sum(axis=-1)
        mask = d2 < tol2
        if mask.sum() * stride * stride < min_pixels:
            out.append((np.nan, np.nan))
            continue
        ys, xs = np.nonzero(mask)
        out.append((xs.mean() * stride, ys.mean() * stride))
    return np.array(out)


def read_mp4(path):
    import imageio_ffmpeg

    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)
    w, h = meta["size"]
    for buf in gen:
        yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


def detect_events(
    track: np.ndarray,
    fps: float,
    size: int,
    gravity: float = 6.2,
    accel_k: float = 6.0,
    teleport_px: float = 0.06,
    hover_frames: int = 12,
    refractory_s: float = 0.35,
    edge: int = 3,
) -> list[dict]:
    """Turn a screen-space track into ``[{"t", "label"}]`` events.

    Thresholds are in units of the free-fall acceleration / image size, so
    they do not depend on resolution.
    """
    cfg = RenderConfig.for_size(size)
    dt = 1.0 / fps
    g_px = gravity * np.sin(np.deg2rad(cfg.tilt_deg)) * cfg.scale * dt * dt  # px/frame^2, +y is down

    x, y = track[:, 0], track[:, 1]
    vis = ~np.isnan(x)
    events: list[dict] = []

    # --- teleport: one-frame jump far larger than any real per-frame motion
    step = np.hypot(np.diff(x), np.diff(y))
    for i in np.nonzero(step > teleport_px * size)[0]:
        events.append({"t": round((i + 1) * dt, 3), "label": "anomaly", "why": "teleport"})

    # --- hover: ball stays put while visible (real free fall never does)
    still = np.r_[False, step < 0.05] & vis
    run = 0
    for i, s in enumerate(still):
        run = run + 1 if s else 0
        if run == hover_frames:
            events.append({"t": round((i - hover_frames + 1) * dt, 3), "label": "anomaly", "why": "hover"})

    # --- acceleration residual vs free fall, on a lightly smoothed track
    def smooth(v, k=3):
        return np.convolve(v, np.ones(k) / k, mode="same")

    ax = np.diff(smooth(x), 2)
    ay = np.diff(smooth(y), 2) - g_px
    resid = np.hypot(ax, ay)
    kicked = np.nan_to_num(resid, nan=0.0) > accel_k * g_px
    kicked[:edge] = False                  # smoothing artefacts at the clip edges
    kicked[len(kicked) - edge:] = False
    # One event per interaction: a membrane contact lasts ~0.3 s and kicks the
    # track both on the way in and on the way out, so detections inside a short
    # refractory window after the previous one are the same event.
    last_t = -1e9
    for i in np.nonzero(kicked)[0]:
        t = round((i + 1) * dt, 3)
        if t - last_t < refractory_s:
            continue
        last_t = t
        if not any(abs(e["t"] - t) < 0.1 for e in events):
            events.append({"t": t, "label": "bounce", "why": "accel"})

    # --- sustained upward acceleration in the air: gravity reversed
    w = 15
    for s0 in range(0, len(y) - w, w // 2):
        seg = slice(s0, s0 + w)
        if not vis[seg].all() or kicked[max(0, s0 - 1): s0 + w].any():
            continue
        a = np.polyfit(np.arange(w), y[seg], 2)[0] * 2
        if a < -0.5 * g_px:
            events.append({"t": round(s0 * dt, 3), "label": "anomaly", "why": "gravity"})
            break

    return sorted(events, key=lambda e: e["t"])
