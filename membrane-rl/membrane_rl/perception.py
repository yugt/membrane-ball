"""Pixels -> 3D ball position and membrane frame centre.

The camera is a known orthographic projection (``Renderer.project``), so a
single frame is enough to recover the full 3D position -- no depth guessing:

* the orange shadow ellipse sits at z = 0 under the ball, so its centre
  inverts to (x, y) exactly;
* screen x of a point does not depend on z, and screen y shifts linearly with
  z, so the ball's screen centre then gives z.

The green rim is the membrane frame at z = 0, so its centre inverts the same
way. Everything here is plain colour masking; it runs in ~1 ms per 720 px frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .render import RenderConfig

__all__ = ["Camera", "Observation", "observe", "estimate_frame_center"]

_BALL = np.array(RenderConfig().c_ball, np.int32)
_SHADOW = np.array((255, 150, 40), np.int32)
_RIM = np.array(RenderConfig().c_frame, np.int32)


class Camera:
    """Inverse of ``Renderer.project`` for a given ``RenderConfig``."""

    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        r, t = np.deg2rad(cfg.rotation_deg), np.deg2rad(cfg.tilt_deg)
        self.cr, self.sr, self.ct, self.st = np.cos(r), np.sin(r), np.cos(t), np.sin(t)
        self.half = cfg.size / 2.0

    def ground(self, sx: float, sy: float) -> tuple[float, float]:
        """Screen point known to lie at z = 0 -> world (x, y)."""
        u = (sx - self.half) / self.cfg.scale
        w = (sy - self.half - self.cfg.y_offset) / self.cfg.scale / self.ct
        return u * self.cr + w * self.sr, -u * self.sr + w * self.cr

    def height(self, x: float, y: float, sy: float) -> float:
        """World (x, y) plus the screen y of a point above it -> z."""
        ry = x * self.sr + y * self.cr
        return (ry * self.ct - (sy - self.half - self.cfg.y_offset) / self.cfg.scale) / self.st


@dataclass
class Observation:
    pos: np.ndarray          # (x, y, z); z is NaN if the ball is off-screen
    ball_px: tuple[float, float] | None
    ok: bool


def _mask(img: np.ndarray, rgb: np.ndarray, tol: int) -> np.ndarray:
    """Per-channel box test in int16 -- ~5x cheaper than a Euclidean distance
    and just as selective for these saturated, well-separated colours."""
    d = np.abs(img.astype(np.int16) - rgb.astype(np.int16))
    return (d[..., 0] < tol) & (d[..., 1] < tol) & (d[..., 2] < tol)


def _shadow_centre(img: np.ndarray) -> tuple[float, float] | None:
    # Coarse pass on every 4th pixel to find the ellipse, then a full-res pass
    # on just that region: the full-image mask is what made this slow.
    ys, xs = np.nonzero(_mask(img[::4, ::4], _SHADOW, 40))
    if xs.size < 4:
        return None
    pad = 8
    y0, y1 = max(0, ys.min() * 4 - pad), min(img.shape[0], ys.max() * 4 + pad)
    x0, x1 = max(0, xs.min() * 4 - pad), min(img.shape[1], xs.max() * 4 + pad)
    ys, xs = np.nonzero(_mask(img[y0:y1, x0:x1], _SHADOW, 40))
    if xs.size < 20:
        return None
    ys, xs = ys + y0, xs + x0
    x0, x1 = xs.min(), xs.max()
    # The dashed drop line runs exactly through the centre column, so the
    # left/right extremes of the ellipse are untouched by it; their y is the
    # ellipse centre's y.
    side = (xs <= x0 + 1) | (xs >= x1 - 1)
    return (x0 + x1) / 2.0, float(ys[side].mean())


def _ball_centre(img: np.ndarray, sx: float, r_px: float) -> float | None:
    """Screen y of the ball centre, from the vertical extent of the yellow disk
    in a column band around the (already known) screen x."""
    h = img.shape[0]
    lo, hi = int(max(0, sx - 0.5 * r_px)), int(min(img.shape[1], sx + 0.5 * r_px + 1))
    ys = np.nonzero(_mask(img[:, lo:hi], _BALL, 30).any(axis=1))[0]
    if ys.size < 3:
        return None
    top, bot = ys.min(), ys.max()
    if top <= 0 and bot >= h - 1:
        return None
    if top <= 0:                     # clipped at the top edge
        return bot - r_px
    if bot >= h - 1:                 # clipped at the bottom edge
        return top + r_px
    return (top + bot) / 2.0


def observe(img: np.ndarray, cam: Camera, ball_radius: float = 0.5) -> Observation:
    sh = _shadow_centre(img)
    if sh is None:
        return Observation(np.full(3, np.nan), None, False)
    x, y = cam.ground(*sh)
    r_px = ball_radius * cam.cfg.scale
    by = _ball_centre(img, sh[0], r_px)
    z = cam.height(x, y, by) if by is not None else np.nan
    return Observation(np.array([x, y, z]), (sh[0], by) if by is not None else None, by is not None)


def estimate_frame_center(frames, cam: Camera, n: int = 15) -> np.ndarray:
    """Centre of the green rim, median over the first ``n`` frames (the ball can
    hide part of the rim in any single frame)."""
    est = []
    for img in frames[:n]:
        ys, xs = np.nonzero(_mask(img, _RIM, 40))
        if xs.size < 50:
            continue
        est.append(cam.ground((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0))
    if not est:
        return np.zeros(2)
    return np.median(np.array(est), axis=0)
