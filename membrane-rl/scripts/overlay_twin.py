#!/usr/bin/env python3
"""Render what the fast path sees on top of a clip (offline, for humans).

Draws, per frame: the ball position recovered from pixels, the twin's 1 s
forecast path, a mismatch meter, twin alarms and the twin's own event labels,
and -- in a separate corner, clearly marked -- the answer key for comparison.

    python scripts/overlay_twin.py clips/clip_003.mp4 -o clip_003_twin.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.render import RenderConfig, Renderer  # noqa: E402
from membrane_rl.tracker import read_mp4  # noqa: E402
from membrane_rl.twin import params_from_key, run_twin  # noqa: E402
from membrane_rl.video import _font  # noqa: E402

HOLD_S = 0.8
C_FORECAST = (120, 255, 255)
C_ALARM = (255, 70, 70)
C_TWIN = (200, 200, 255)
C_TRUTH = (150, 150, 160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--horizon", type=int, default=62)
    args = ap.parse_args()

    import imageio_ffmpeg

    clip = Path(args.clip)
    key = json.loads(clip.with_suffix(".json").read_text())
    size, fps = int(key.get("size") or 720), key["fps"]
    params = params_from_key(key)
    proj = Renderer(RenderConfig.for_size(size))
    frames = list(read_mp4(clip))
    font, small = _font(max(12, size // 34)), _font(max(10, size // 44))
    thr = None

    writer = imageio_ffmpeg.write_frames(args.out, (size, size), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=8,
                                         output_params=["-movflags", "+faststart"])
    writer.send(None)
    twin_events: list[dict] = []
    truth = [e for e in key["events"] if e["type"] in
             ("contact_start", "ring_bounce", "wall_bounce", "anomaly_start")]

    def hook(i, twin, obs, out):
        nonlocal thr
        thr = twin.cfg.threshold
        t = i / fps
        img = Image.fromarray(frames[i]).convert("RGB")
        d = ImageDraw.Draw(img)
        names = {"contact_start": "membrane", "ring_bounce": "rim", "wall_bounce": "wall"}
        for e in out.get("events", []) or []:
            if e["type"] in names:
                twin_events.append({"t": t, "label": names[e["type"]]})
        if out.get("alarm"):
            twin_events.append({"t": t, "label": "ANOMALY", "alarm": True})

        # 1 s forecast path
        if twin.locked:
            path = twin.forecast(args.horizon)
            pts = [proj.project(*p) for p in path[::3]]
            for k, (x, y) in enumerate(pts):
                r = max(2, size // 180)
                fade = 1.0 - k / max(1, len(pts))
                col = tuple(int(c * (0.35 + 0.65 * fade)) for c in C_FORECAST)
                d.ellipse([x - r, y - r, x + r, y + r], fill=col)
            ex, ey = proj.project(*path[-1])
            R = params.ball_radius * proj.cfg.scale
            d.ellipse([ex - R, ey - R, ex + R, ey + R], outline=C_FORECAST, width=2)
        # where perception thinks the ball is
        if np.isfinite(obs).all():
            bx, by = proj.project(*obs)
            d.line([bx - 8, by, bx + 8, by], fill=(255, 255, 255), width=2)
            d.line([bx, by - 8, bx, by + 8], fill=(255, 255, 255), width=2)

        # header
        s = size
        d.text((s * 0.03, s * 0.03), f"t = {t:5.2f} s", fill=(230, 230, 235), font=font)
        d.text((s * 0.03, s * 0.03 + s // 26), "fast path: pixels -> physics twin",
               fill=C_TWIN, font=small)
        d.text((s * 0.03, s * 0.03 + s // 26 + s // 40), "cyan = where the ball will be in 1 s",
               fill=C_FORECAST, font=small)
        # mismatch meter
        score = twin.score[-1] if twin.score else 0.0
        mx, my, mw, mh = s * 0.03, s * 0.93, s * 0.3, s * 0.018
        d.rectangle([mx, my, mx + mw, my + mh], outline=(90, 90, 110))
        frac = min(1.0, score / (3 * thr))
        d.rectangle([mx, my, mx + mw * frac, my + mh],
                    fill=C_ALARM if score > thr else (90, 200, 140))
        d.line([mx + mw / 3, my - 3, mx + mw / 3, my + mh + 3], fill=(230, 230, 235), width=1)
        d.text((mx, my - s // 32), "physics mismatch", fill=(200, 200, 210), font=small)

        # twin labels (right, top) and answer key (right, bottom)
        y = s * 0.03
        for e in twin_events:
            if 0 <= t - e["t"] <= HOLD_S:
                label = ("! " if e.get("alarm") else "twin: ") + e["label"]
                w = d.textlength(label, font=font)
                d.text((s * 0.97 - w, y), label, fill=C_ALARM if e.get("alarm") else C_TWIN, font=font)
                y += s // 26
        y = s * 0.80
        d.text((s * 0.97 - d.textlength("answer key", font=small), y), "answer key",
               fill=(110, 110, 120), font=small)
        y += s // 38
        for e in truth:
            te = e.get("visible_t", e["t"])
            if 0 <= t - te <= HOLD_S:
                label = e["type"].replace("contact_start", "membrane").replace("ring_bounce", "rim") \
                    .replace("wall_bounce", "wall").replace("anomaly_start", f"anomaly: {e.get('kind')}")
                w = d.textlength(label, font=small)
                d.text((s * 0.97 - w, y), label, fill=C_TRUTH, font=small)
                y += s // 38
        writer.send(np.asarray(img, dtype=np.uint8).tobytes())

    run_twin(frames, params, size, on_frame=hook)
    writer.close()
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
