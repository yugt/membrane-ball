#!/usr/bin/env python3
"""Frame-by-frame 3D debug file (Plotly HTML) for a clip, plus its motion audit.

    python scripts/debug_plotly.py clips/clip_002.mp4                 # -> clips/clip_002_debug.html
    python scripts/debug_plotly.py clips/*.mp4 --audit-only           # audit table, no HTML

Re-simulates the clip's episode from its answer key, checks it reproduces the
clip exactly, then writes an animated 3D scene (rotate it; the "video camera"
button restores the mp4's view) with velocity / acceleration arrows, logged
events and a per-frame verdict on every change in motion. Exits non-zero if
any frame is UNEXPLAINED.

The HTML embeds plotly.js (~4.7 MB) so it opens offline; ``--cdn`` links it
instead. Needs ``membrane-rl[debug]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from membrane_rl.anomalies import Anomaly
from membrane_rl.debug3d import debug_figure, record_episode
from membrane_rl.physics import Params


def _params(key: dict) -> Params:
    return Params(**{k: tuple(v) if isinstance(v, list) else v for k, v in key["params"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="clip .mp4 or answer-key .json")
    ap.add_argument("-o", "--out", help="output .html (single clip only)")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th frame in the animation")
    ap.add_argument("--cdn", action="store_true", help="link plotly.js instead of embedding it")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    if args.out and len(args.clips) > 1:
        ap.error("--out needs a single clip")

    failed = False
    for c in args.clips:
        key_path = Path(c).with_suffix(".json")
        key = json.loads(key_path.read_text())
        anomaly = Anomaly(**key["anomaly"]) if key["anomaly"] else None
        rec = record_episode(key["n_frames"], seed=key["seed"], params=_params(key), anomaly=anomaly,
                             grid=5 if args.audit_only else 21)
        ep = rec.episode
        traj = np.array([s.pos for s in ep.states])
        if ep.frame_center != key["frame_center"] or \
                np.abs(traj - np.array(key["trajectory"])).max() > 1e-4:
            print(f"{c}: re-simulation does NOT match the clip's answer key", file=sys.stderr)
            failed = True
            continue
        s = rec.summary()
        bad = s["unexplained_frames"]
        failed |= bool(bad)
        kind = key["anomaly"]["kind"] if key["anomaly"] else "none"
        print(f"{key_path.stem}: anomaly={kind}  {s['status_counts']}  "
              f"unexplained={bad if bad else 'none'}  energy drift {s['energy_drift_pct']:.2f}%",
              file=sys.stderr)
        if args.audit_only:
            continue
        out = Path(args.out) if args.out else key_path.with_name(key_path.stem + "_debug.html")
        fig = debug_figure(rec, stride=args.stride,
                           title=f"<b>{key_path.stem}</b> · seed {key['seed']} · anomaly: {kind}")
        fig.write_html(out, include_plotlyjs="cdn" if args.cdn else True, auto_play=False)
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
