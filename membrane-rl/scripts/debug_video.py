#!/usr/bin/env python3
"""Play a clip's 3D debug view and record it as an mp4, with an orbiting camera.

    python scripts/debug_video.py clips/clip_002.mp4          # -> clips/clip_002_orbit.mp4

The flat clip cannot show depth: motion toward the camera looks like falling,
a far-wall bounce looks like a jump in mid-air. This records the same episode
from a camera that circles the cylinder (one loop per ``--period`` seconds of
clip time, always aimed at the cylinder axis), in real time, with the debug
view's arrows, events and per-frame verdicts, next to the flat clip itself
-- so a human can match each confusing moment in the clip to what actually
happened in 3D, without opening the interactive file.

The orbit starts from the clip camera's direction, so the first frame shows
the scene the way the mp4 does.

Needs ``membrane-rl[debug]`` plus a Chromium for Playwright
(``playwright install chromium``; or ``--chromium PATH`` for an existing one).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from membrane_rl.anomalies import Anomaly
from membrane_rl.debug3d import debug_figure, record_episode, video_camera
from membrane_rl.physics import Params
from membrane_rl.tracker import read_mp4
from membrane_rl.video import _font

# Moves the Plotly figure to frame ``i`` and the camera to ``cam`` in one redraw.
_STEP_JS = """async ([name, cam]) => {
  const gd = document.querySelector('.plotly-graph-div');
  const f = gd._transitionData._frameHash[name];
  await Plotly.animate(gd, {data: f.data, traces: f.traces,
                            layout: {title: f.layout.title, 'scene.camera': cam}},
                       {frame: {duration: 0, redraw: true}, transition: {duration: 0},
                        mode: 'immediate'});
}"""


def _orbit_camera(t: float, period: float, elevation_deg: float, distance: float,
                  phase: float) -> dict:
    a = phase + 2 * np.pi * t / period
    e = np.deg2rad(elevation_deg)
    return {"eye": {"x": distance * np.cos(e) * np.cos(a), "y": distance * np.cos(e) * np.sin(a),
                    "z": distance * np.sin(e)},
            "center": {"x": 0, "y": 0, "z": 0},          # the box is centred on the cylinder axis
            "up": {"x": 0, "y": 0, "z": 1},
            "projection": {"type": "perspective"}}


def _clip_panel(frame: np.ndarray | None, side: int, t: float) -> Image.Image:
    """The flat clip, scaled to ``side`` x ``side``, labelled as such."""
    img = Image.fromarray(frame).convert("RGB").resize((side, side)) if frame is not None \
        else Image.new("RGB", (side, side), (14, 16, 22))
    d = ImageDraw.Draw(img)
    d.text((14, side - 64), "flat clip (what the video agent sees)", fill=(230, 230, 235),
           font=_font(22))
    d.text((14, side - 36), f"t = {t:5.2f} s", fill=(200, 200, 210), font=_font(20))
    return img


async def _record(html: Path, out: Path, n_frames: int, fps: float, args,
                  clip: list[np.ndarray] | None = None) -> None:
    import imageio_ffmpeg
    from playwright.async_api import async_playwright

    eye = video_camera()["eye"]
    phase = float(np.arctan2(eye["y"], eye["x"]))
    W, H = args.width, args.height
    side = H if clip is not None else 0
    writer = imageio_ffmpeg.write_frames(str(out), (side + W, H), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=4,
                                         output_params=["-movflags", "+faststart"])
    writer.send(None)
    async with async_playwright() as p:
        launch = {"args": ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"]}
        if args.chromium:
            launch["executable_path"] = args.chromium
        browser = await p.chromium.launch(**launch)
        page = await browser.new_page(viewport={"width": W, "height": H})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(html.resolve().as_uri())
        await page.wait_for_function(
            "() => { const g = document.querySelector('.plotly-graph-div');"
            " return g && g._fullLayout && g._transitionData; }", timeout=120_000)
        # a recording has no use for the play button and slider
        await page.evaluate("() => Plotly.relayout(document.querySelector('.plotly-graph-div'),"
                            " {updatemenus: [], sliders: []})")
        t0 = time.time()
        for i in range(n_frames):
            cam = _orbit_camera(i / fps, args.period, args.elevation, args.distance, phase)
            await page.evaluate(_STEP_JS, [str(i), cam])
            shot = Image.open(io.BytesIO(await page.screenshot(type="png"))).convert("RGB")
            if clip is not None:
                both = Image.new("RGB", (side + W, H))
                both.paste(_clip_panel(clip[i] if i < len(clip) else None, side, i / fps), (0, 0))
                both.paste(shot, (side, 0))
                shot = both
            writer.send(np.asarray(shot, np.uint8).tobytes())
            if i % 50 == 0:
                print(f"  frame {i}/{n_frames} ({time.time() - t0:.0f}s)", file=sys.stderr)
        await browser.close()
    writer.close()
    if errors:
        raise RuntimeError(f"page errors while recording: {errors[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="clip .mp4 or answer-key .json")
    ap.add_argument("-o", "--out", help="output .mp4 (single clip only)")
    ap.add_argument("--period", type=float, default=6.0, help="seconds of clip time per orbit")
    ap.add_argument("--elevation", type=float, default=20.0, help="camera elevation, degrees")
    ap.add_argument("--distance", type=float, default=2.9, help="camera distance (scene units)")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=860)
    ap.add_argument("--chromium", help="path to a Chromium executable for Playwright")
    ap.add_argument("--frames", type=int, help="record only the first N frames (quick look)")
    ap.add_argument("--no-clip", action="store_true",
                    help="leave out the flat clip normally shown on the left")
    args = ap.parse_args()
    if args.out and len(args.clips) > 1:
        ap.error("--out needs a single clip")

    for c in args.clips:
        key_path = Path(c).with_suffix(".json")
        key = json.loads(key_path.read_text())
        params = Params(**{k: tuple(v) if isinstance(v, list) else v for k, v in key["params"].items()})
        anomaly = Anomaly(**key["anomaly"]) if key["anomaly"] else None
        rec = record_episode(key["n_frames"], seed=key["seed"], params=params, anomaly=anomaly)
        traj = np.array([s.pos for s in rec.episode.states])
        if np.abs(traj - np.array(key["trajectory"])).max() > 1e-4:
            sys.exit(f"{c}: re-simulation does NOT match the clip's answer key")
        kind = key["anomaly"]["kind"] if key["anomaly"] else "none"
        fig = debug_figure(rec, title=f"<b>{key_path.stem}</b> · seed {key['seed']} · anomaly: {kind}"
                                      f" · camera orbits the cylinder every {args.period:g} s")
        fig.update_layout(width=args.width, height=args.height)
        out = Path(args.out) if args.out else key_path.with_name(key_path.stem + "_orbit.mp4")
        with tempfile.TemporaryDirectory() as tmp:
            html = Path(tmp) / "debug.html"
            fig.write_html(html, include_plotlyjs=True, auto_play=False)
            print(f"{key_path.stem}: recording {rec.episode.n_frames} frames", file=sys.stderr)
            n = min(args.frames or rec.episode.n_frames, rec.episode.n_frames)
            mp4 = Path(c) if c.endswith(".mp4") else key_path.with_suffix(".mp4")
            clip = None if args.no_clip or not mp4.exists() else list(read_mp4(mp4))
            asyncio.run(_record(html, out, n, rec.episode.fps, args, clip))
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
