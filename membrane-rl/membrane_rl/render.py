"""Headless renderer: simulation state -> small PNG.

Pure PIL, no display and no matplotlib, so this runs anywhere and is fast
enough to generate tens of thousands of frames on a laptop CPU.

Image size is the single biggest lever on training cost: VLM compute scales
with vision tokens, which scale with pixels. 320px is deliberately small.
Raise it only after you have measured that the task is bottlenecked by
perception rather than by reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .physics import MembraneSim

__all__ = ["RenderConfig", "Renderer"]


@dataclass(frozen=True)
class RenderConfig:
    size: int = 320
    scale: float = 62.0
    rotation_deg: float = 45.0
    tilt_deg: float = 35.0
    y_offset: float = 42.0
    grid: int = 13               # membrane mesh resolution
    bg: tuple[int, int, int] = (14, 14, 21)
    c_wall: tuple[int, int, int] = (52, 58, 76)
    c_frame: tuple[int, int, int] = (50, 220, 120)
    c_ball: tuple[int, int, int] = (255, 226, 78)
    c_mem_slack: tuple[int, int, int] = (0, 104, 158)
    c_mem_taut: tuple[int, int, int] = (208, 44, 86)

    @classmethod
    def for_size(cls, size: int, **overrides) -> "RenderConfig":
        """Same framing at a different resolution (e.g. 720 for demo video).

        320 px stays the default for training data -- vision tokens scale with
        pixels -- but a demo video for humans needs to be bigger.

        The camera is also pulled back a little so the whole cylinder
        (z = -1 .. 4.8) fits: in video the ball does leave the default crop,
        e.g. when it falls through a switched-off membrane.
        """
        k = size / 320.0
        return cls(size=size, scale=52.0 * k, y_offset=57.0 * k, **overrides)


class Renderer:
    def __init__(self, cfg: RenderConfig | None = None):
        self.cfg = cfg or RenderConfig()
        r = np.deg2rad(self.cfg.rotation_deg)
        t = np.deg2rad(self.cfg.tilt_deg)
        self._cos_r, self._sin_r = np.cos(r), np.sin(r)
        self._cos_t, self._sin_t = np.cos(t), np.sin(t)
        k = self.cfg.size / 320.0
        self._w1 = max(1, round(k))
        self._w2 = max(2, round(2 * k))

    def project(self, x, y, z):
        """Isometric projection matching the upstream game camera."""
        rx = x * self._cos_r - y * self._sin_r
        ry = x * self._sin_r + y * self._cos_r
        px = rx * self.cfg.scale
        py = (ry * self._cos_t - z * self._sin_t) * self.cfg.scale
        half = self.cfg.size / 2.0
        return half + px, half + py + self.cfg.y_offset

    def depth(self, x, y, z) -> float:
        """Distance along the view axis. Larger = further from the camera.

        Needed because without depth sorting the ball always paints over the
        membrane, and the viewer cannot tell whether it is above, below or
        pressed into it -- which is exactly the information the task depends on.
        """
        ry = x * self._sin_r + y * self._cos_r
        return ry * self._sin_t + z * self._cos_t

    def render(self, sim: MembraneSim) -> Image.Image:
        cfg = self.cfg
        img = Image.new("RGB", (cfg.size, cfg.size), cfg.bg)
        d = ImageDraw.Draw(img)

        self._draw_cylinder(d, sim)
        self._draw_height_axis(d, sim)
        behind, front = self._membrane_segments(sim)
        self._draw_segments(d, behind)
        self._draw_frame_ring(d, sim)
        self._draw_ball(d, sim)
        self._draw_segments(d, front)      # membrane that occludes the ball
        # Position annotations paint last so the mesh cannot bury them. They
        # are what make (x, y, z) readable at all: without them the three
        # frames of a sample look nearly identical.
        self._draw_shadow(d, sim)          # where the ball is in (x, y)
        self._draw_drop_line(d, sim)       # how high the ball is
        return img

    # ------------------------------------------------------------------
    def _ring_points(self, radius: float, z: float, n: int = 72, cx: float = 0.0, cy: float = 0.0):
        ang = np.linspace(0, 2 * np.pi, n, endpoint=True)
        return [
            self.project(cx + radius * np.cos(a), cy + radius * np.sin(a), z)
            for a in ang
        ]

    def _draw_cylinder(self, d: ImageDraw.ImageDraw, sim: MembraneSim) -> None:
        p = sim.p
        # a few horizontal rings give depth cues without costing pixels
        for z in (0.0, 1.6, 3.2, 4.8):
            d.line(self._ring_points(p.r_cyl, z), fill=self.cfg.c_wall, width=self._w1)
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x, y = p.r_cyl * np.cos(a), p.r_cyl * np.sin(a)
            d.line([self.project(x, y, 0.0), self.project(x, y, 4.8)],
                   fill=self.cfg.c_wall, width=self._w1)

    def _draw_height_axis(self, d: ImageDraw.ImageDraw, sim: MembraneSim) -> None:
        """Vertical ruler on the cylinder axis: gives z an absolute reference."""
        col = (74, 82, 104)
        d.line([self.project(0, 0, 0.0), self.project(0, 0, 5.0)], fill=col, width=self._w1)
        for z in range(0, 6):
            x0, y0 = self.project(0, 0, float(z))
            w = 7 if z % 2 == 0 else 4
            d.line([(x0 - w, y0), (x0 + w, y0)], fill=col, width=self._w1)

    def _draw_shadow(self, d: ImageDraw.ImageDraw, sim: MembraneSim) -> None:
        """Ellipse at z = 0 directly under the ball -- reads out (x, y)."""
        x, y, _ = sim.state.pos
        cx, cy = self.project(x, y, 0.0)
        rx = sim.p.ball_radius * self.cfg.scale * 0.9
        ry = rx * self._cos_t
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(255, 150, 40), width=self._w2)

    def _draw_drop_line(self, d: ImageDraw.ImageDraw, sim: MembraneSim) -> None:
        """Dashed vertical from the ball to its shadow -- reads out z."""
        x, y, z = sim.state.pos
        top = self.project(x, y, z - sim.p.ball_radius)
        bot = self.project(x, y, 0.0)
        n = max(2, int(abs(bot[1] - top[1]) / (7 * self._w1)))
        for i in range(n):
            if i % 2:
                continue
            t0, t1 = i / n, min(1.0, (i + 0.75) / n)
            d.line(
                [
                    (top[0] + (bot[0] - top[0]) * t0, top[1] + (bot[1] - top[1]) * t0),
                    (top[0] + (bot[0] - top[0]) * t1, top[1] + (bot[1] - top[1]) * t1),
                ],
                fill=(255, 150, 40),
                width=self._w1,
            )

    def _draw_segments(self, d: ImageDraw.ImageDraw, segs) -> None:
        for pts, col in segs:
            d.line(pts, fill=col, width=self._w1)

    def _membrane_segments(self, sim: MembraneSim):
        """Membrane mesh split into segments behind and in front of the ball."""
        nx, ny, u = sim.membrane_surface(grid=self.cfg.grid)
        g = self.cfg.grid

        # Colour encodes local stretch, matching the upstream visual language:
        # cyan when slack, hot pink where the membrane is taut.
        finite = u[np.isfinite(u)]
        dip = float(-finite.min()) if finite.size and finite.min() < 0 else 0.0
        t = float(np.clip(dip / max(1e-6, sim.p.ball_radius * 1.6), 0.0, 1.0))
        col = tuple(
            int(a + (b - a) * t)
            for a, b in zip(self.cfg.c_mem_slack, self.cfg.c_mem_taut)
        )

        ball_depth = self.depth(*sim.state.pos)
        behind, front = [], []

        def seg(i0, j0, i1, j1):
            if not (np.isfinite(u[i0, j0]) and np.isfinite(u[i1, j1])):
                return
            pts = [
                self.project(nx[i0, j0], ny[i0, j0], u[i0, j0]),
                self.project(nx[i1, j1], ny[i1, j1], u[i1, j1]),
            ]
            mid_depth = 0.5 * (
                self.depth(nx[i0, j0], ny[i0, j0], u[i0, j0])
                + self.depth(nx[i1, j1], ny[i1, j1], u[i1, j1])
            )
            (behind if mid_depth > ball_depth else front).append((pts, col))

        for i in range(g):
            for j in range(g):
                if i + 1 < g:
                    seg(i, j, i + 1, j)
                if j + 1 < g:
                    seg(i, j, i, j + 1)
        return behind, front

    def _draw_frame_ring(self, d: ImageDraw.ImageDraw, sim: MembraneSim) -> None:
        # The frame is fixed within an episode but its centre varies between
        # episodes, so the model has to actually locate the green ring rather
        # than memorise one scene layout.
        fx, fy = float(sim.frame_center[0]), float(sim.frame_center[1])
        d.line(
            self._ring_points(sim.p.r_frame, 0.0, cx=fx, cy=fy),
            fill=self.cfg.c_frame,
            width=self._w2,
        )

    def _draw_ball(self, d: ImageDraw.ImageDraw, sim: MembraneSim) -> None:
        x, y, z = sim.state.pos
        cx, cy = self.project(x, y, z)
        # approximate screen radius; exact perspective is unnecessary here
        rpx = sim.p.ball_radius * self.cfg.scale
        d.ellipse([cx - rpx, cy - rpx, cx + rpx, cy + rpx],
                  fill=self.cfg.c_ball, outline=(255, 255, 255))
