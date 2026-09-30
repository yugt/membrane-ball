#!/usr/bin/env python3
"""Generate video clips with exact answer keys for the video-agent demo.

    python scripts/gen_clips.py --n 20 --out clips/

For every clip ``clip_XXX.mp4`` you get ``clip_XXX.json`` (events, energy,
trajectory, anomaly) and, with ``--annotated``, a ``clip_XXX_annotated.mp4``
with labels burned in for humans. Feed the plain .mp4 to the agent -- never
the annotated one, it contains the answers.

Clips are cheap to regenerate (~8 s each at 720 px), so they are not committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


from membrane_rl.anomalies import KINDS, random_anomaly  # noqa: E402
from membrane_rl.physics import Params  # noqa: E402
from membrane_rl.video import render_clip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="clips")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--anomaly-frac", type=float, default=0.5,
                    help="fraction of clips with one injected anomaly")
    ap.add_argument("--kinds", default=",".join(KINDS),
                    help=f"comma-separated subset of {KINDS}")
    ap.add_argument("--frame-offset-max", type=float, default=0.3)
    ap.add_argument("--annotated", action="store_true",
                    help="also write a human-facing copy with labels burned in")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    rng = np.random.default_rng(args.seed)
    params = Params(frame_offset_max=args.frame_offset_max)
    n_frames = int(round(args.seconds / params.dt))

    # Deterministic anomaly assignment: exactly round(n * frac) anomalous clips,
    # kinds cycled so every kind appears before any repeats.
    n_anom = int(round(args.n * args.anomaly_frac))
    anomalous = set(rng.choice(args.n, size=n_anom, replace=False).tolist())

    manifest = []
    t0 = time.time()
    k_i = 0
    for i in range(args.n):
        seed = args.seed * 100_000 + i
        anomaly = None
        if i in anomalous:
            anomaly = random_anomaly(rng, n_frames, kind=kinds[k_i % len(kinds)])
            k_i += 1
        name = f"clip_{i:03d}"
        key = render_clip(out / f"{name}.mp4", n_frames=n_frames, seed=seed,
                          params=params, anomaly=anomaly, size=args.size)
        if args.annotated:
            render_clip(out / f"{name}_annotated.mp4", n_frames=n_frames, seed=seed,
                        params=params, anomaly=anomaly, size=args.size, annotate=True)
        manifest.append({"clip": f"{name}.mp4", "answer_key": f"{name}.json",
                         "anomaly": anomaly.kind if anomaly else None,
                         "n_events": len(key["events"])})
        print(f"[{i + 1}/{args.n}] {name}  anomaly={manifest[-1]['anomaly']}  "
              f"({time.time() - t0:.0f}s)", file=sys.stderr)

    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    print(f"wrote {args.n} clips to {out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
