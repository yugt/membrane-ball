#!/usr/bin/env python3
"""Run the VLM event agent over clips and write ``<clip>.pred.json`` for scoring.

    python scripts/run_agent.py --clips clips/ --limit 2 --dry-run   # inspect payload, no network

    export WANDB_API_KEY=...
    python scripts/run_agent.py --clips clips/ --limit 2 --provider wandb
    python scripts/run_agent.py --clips clips/ --limit 2 --provider nvidia --model <cosmos id>
    python scripts/run_agent.py --clips clips/ --limit 2 --provider local --model <served id>
    OPENAI_API_KEY=... python scripts/run_agent.py --clips clips/ --provider openai \
        --base-url https://.../v1 --model <id>
    python scripts/eval_events.py --clips clips/ --detector preds

``--dry-run`` builds every request and prints its size without sending, so the
pipeline can be checked with no key and no network.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.agent import (PROVIDERS, AgentConfig, build_messages,  # noqa: E402
                               call_model, merge_events, parse_events,
                               resolve_provider, windows)
from membrane_rl.tracker import read_mp4  # noqa: E402


def cut_video_b64(clip: Path, start: float, end: float) -> str:
    """Re-encode one window as a small mp4 (for mode='video')."""
    import base64

    import imageio_ffmpeg
    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-ss", f"{start}", "-t", f"{end - start}",
         "-i", str(clip), "-vf", "scale=448:448", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"],
        capture_output=True, check=True)
    return base64.b64encode(out.stdout).decode()


def run_clip(clip: Path, key: dict, cfg: AgentConfig, dry_run: bool, log) -> list[dict]:
    fps = key["fps"]
    frames = list(read_mp4(clip))
    duration = len(frames) / fps
    per_window = []
    for start, end in windows(duration, cfg.window_s, cfg.hop_s):
        step = max(1, int(round(fps / cfg.sample_fps)))
        idx = list(range(int(start * fps), min(len(frames), int(end * fps)), step))
        times = [round(i / fps - start, 3) for i in idx]
        video_b64 = cut_video_b64(clip, start, end) if cfg.mode == "video" else None
        msgs = build_messages([frames[i] for i in idx], times, cfg, video_b64=video_b64)
        size_kb = len(json.dumps(msgs)) / 1024
        if dry_run:
            log(f"    window {start:.2f}-{end:.2f}s: {len(idx)} frames, request {size_kb:.0f} KB")
            continue
        t0 = time.time()
        text = call_model(msgs, cfg)
        evs = parse_events(text, end - start)
        log(f"    window {start:.2f}-{end:.2f}s: {len(evs)} events, {time.time() - t0:.1f}s")
        per_window.append((start, evs))
    return merge_events(per_window, cfg.dedup_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--provider", default="wandb", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default="", help="overrides the provider's default")
    ap.add_argument("--base-url", default="", help="overrides the provider's default")
    ap.add_argument("--project", default="", help="W&B Inference: <entity>/<project>")
    ap.add_argument("--mode", default="frames", choices=["frames", "video"])
    ap.add_argument("--window", type=float, default=AgentConfig.window_s)
    ap.add_argument("--hop", type=float, default=AgentConfig.hop_s)
    ap.add_argument("--sample-fps", type=float, default=AgentConfig.sample_fps)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        base_url, model, api_key = resolve_provider(args.provider, args.model, args.base_url)
    except ValueError as exc:
        if not args.dry_run:
            raise SystemExit(str(exc))
        base_url, model, api_key = args.base_url or "(dry-run)", args.model or "(dry-run)", ""
    cfg = AgentConfig(model=model, base_url=base_url, api_key=api_key, project=args.project,
                      mode=args.mode, window_s=args.window, hop_s=args.hop,
                      sample_fps=args.sample_fps)
    key_env = PROVIDERS[args.provider]["key_env"]
    if key_env and not api_key and not args.dry_run:
        raise SystemExit(f"set {key_env} for provider {args.provider!r}, or use --dry-run")
    print(f"provider={args.provider} model={model} base_url={base_url} mode={args.mode}",
          file=sys.stderr)

    root = Path(args.clips)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    for i, row in enumerate(rows[: args.limit or None], 1):
        clip = root / row["clip"]
        key = json.loads((root / row["answer_key"]).read_text())
        print(f"[{i}] {clip.name}", file=sys.stderr)
        events = run_clip(clip, key, cfg, args.dry_run, lambda m: print(m, file=sys.stderr))
        if not args.dry_run:
            clip.with_suffix(".pred.json").write_text(json.dumps(events, indent=1))


if __name__ == "__main__":
    main()
