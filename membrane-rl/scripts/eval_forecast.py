#!/usr/bin/env python3
"""How well does the physics twin predict where the ball will be 1 s ahead?

For every clip, every ``--every`` frames, the twin forecasts ``--horizon``
frames ahead from its current (pixel-derived) state. The forecast is compared
with the ball's true position at that time, and with a naive gravity-only
extrapolation from the same state. Windows that contain an injected anomaly
are skipped -- nothing should predict those.

    python scripts/eval_forecast.py --clips clips/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.dataset import naive_ballistic  # noqa: E402
from membrane_rl.tracker import read_mp4  # noqa: E402
from membrane_rl.twin import params_from_key, run_twin  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--horizon", type=int, default=62, help="frames ahead (62 = ~1 s)")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.clips)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    twin_err, naive_err, contact_twin, contact_naive, cost = [], [], [], [], []
    for row in rows[: args.limit or None]:
        key = json.loads((root / row["answer_key"]).read_text())
        params = params_from_key(key)
        truth = np.array(key["trajectory"])
        R, H = params.ball_radius, args.horizon
        anom = key.get("anomaly")
        a0 = anom["start_frame"] if anom else None
        contact_frames = {e["frame"] for e in key["events"] if e["type"] == "contact_start"}

        def hook(i, twin, obs, out):
            if not twin.locked or i % args.every or i + H >= len(truth):
                return
            if a0 is not None and i - 30 <= a0 <= i + H:
                return
            t0 = time.perf_counter()
            pred = twin.forecast(H)[-1]
            cost.append(time.perf_counter() - t0)
            naive = naive_ballistic(twin.sim.state, H, params)
            e_t = np.linalg.norm(pred - truth[i + H]) / R
            e_n = np.linalg.norm(naive - truth[i + H]) / R
            twin_err.append(e_t); naive_err.append(e_n)
            if any(i < f <= i + H for f in contact_frames):
                contact_twin.append(e_t); contact_naive.append(e_n)

        run_twin(read_mp4(root / row["clip"]), params, int(key.get("size") or 720), on_frame=hook)
        print(f"  {row['clip']}", file=sys.stderr)

    def med(v):
        return round(float(np.median(v)), 3) if v else None

    print(json.dumps({
        "horizon_s": round(args.horizon * params.dt, 3),
        "n_forecasts": len(twin_err),
        "median_error_radii": {"twin": med(twin_err), "naive_gravity_only": med(naive_err)},
        "median_error_radii_when_membrane_contact_in_window": {
            "twin": med(contact_twin), "naive_gravity_only": med(contact_naive), "n": len(contact_twin)},
        "within_1_radius": {"twin": round(float(np.mean(np.array(twin_err) <= 1)), 3),
                            "naive_gravity_only": round(float(np.mean(np.array(naive_err) <= 1)), 3)},
        "forecast_cost_ms": round(1000 * float(np.median(cost)), 1) if cost else None,
    }, indent=2))


if __name__ == "__main__":
    main()
