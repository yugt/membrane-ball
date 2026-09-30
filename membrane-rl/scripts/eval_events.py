#!/usr/bin/env python3
"""Score an event detector over a folder of clips.

Built-in detectors (no model, no network) -- run these first; they prove the
scoring path and set the bar:

    python scripts/eval_events.py --clips clips/ --detector oracle    # must be perfect
    python scripts/eval_events.py --clips clips/ --detector tracker   # pixel baseline

A real agent writes one ``<clip>.pred.json`` per clip -- a list of
``{"t": seconds, "label": ...}`` -- and is scored with:

    python scripts/eval_events.py --clips clips/ --detector preds

Labels: membrane_contact, ring_bounce, wall_bounce, anomaly (common synonyms
are accepted). Output reports both fine-grained and coarse (``interaction``)
scores; the coarse one is the fair comparison against the tracker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.scoring import aggregate, score_clip, truth_events  # noqa: E402


def detect(name: str, clip: Path, key: dict) -> list[dict]:
    if name == "oracle":
        return [{"t": e["t"], "label": e["label"]} for e in truth_events(key)]
    if name == "tracker":
        from membrane_rl.tracker import detect_events, read_mp4, track_ball
        track = track_ball(read_mp4(clip))
        size = int(key.get("size") or 720)
        return detect_events(track, key["fps"], size)
    if name == "preds":
        pred = clip.with_suffix(".pred.json")
        return json.loads(pred.read_text()) if pred.exists() else []
    raise SystemExit(f"unknown detector {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--detector", default="tracker", choices=["oracle", "tracker", "preds"])
    ap.add_argument("--tol", type=float, default=0.3, help="match window, seconds")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.clips)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    fine, coarse = [], []
    for i, row in enumerate(rows, 1):
        key = json.loads((root / row["answer_key"]).read_text())
        pred = detect(args.detector, root / row["clip"], key)
        fine.append(score_clip(key, pred, tol_s=args.tol))
        coarse.append(score_clip(key, pred, tol_s=args.tol, coarse=True))
        print(f"  [{i}/{len(rows)}] {row['clip']}", file=sys.stderr)

    print(json.dumps({
        "detector": args.detector,
        "fine": aggregate(fine),
        "coarse": aggregate(coarse)["events"],
    }, indent=2))


if __name__ == "__main__":
    main()
