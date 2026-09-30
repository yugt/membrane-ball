#!/usr/bin/env python3
"""Generate train/test splits.

The two splits use DISJOINT bands of membrane-frame offset. Train sees frames
placed within `--train-offset-max` of the origin; test sees frames placed
strictly further out. A split that differs only by RNG seed measures nothing --
it draws from the same distribution the model was trained on. Holding out an
unseen band of geometry is a real generalisation probe, and it is the number
worth reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.dataset import SampleConfig, build_dataset  # noqa: E402
from membrane_rl.physics import Params  # noqa: E402
from membrane_rl.render import RenderConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/membrane")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=25,
                    help="frames ahead to predict; 25-30 is the calibrated band")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--train-offset-max", type=float, default=0.25)
    ap.add_argument("--test-offset-max", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sample_cfg = SampleConfig(
        n_input_frames=args.frames,
        frame_stride=args.stride,
        horizon=args.horizon,
    )
    render_cfg = RenderConfig(size=args.image_size)

    train_params = Params(frame_offset_min=0.0, frame_offset_max=args.train_offset_max)
    test_params = Params(frame_offset_min=args.train_offset_max,
                         frame_offset_max=args.test_offset_max)

    for split, n, params, seed in (
        ("train", args.train, train_params, args.seed),
        ("test", args.test, test_params, args.seed + 99991),
    ):
        stats = build_dataset(args.out, n, seed=seed, sample_cfg=sample_cfg,
                              params=params, render_cfg=render_cfg, split=split)
        print(json.dumps({k: v for k, v in stats.items()
                          if k not in ("params", "sample_config")}, indent=2))
        print(f"  frame offsets: [{params.frame_offset_min}, {params.frame_offset_max}]\n")


if __name__ == "__main__":
    main()
