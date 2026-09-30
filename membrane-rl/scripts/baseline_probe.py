#!/usr/bin/env python3
"""Zero-shot baseline probe -- the project's go/no-go gate.

``calibrate_horizon`` bounds the HEADROOM (how wrong dumb extrapolation is).
This script measures the other half: how well the actual base model does before
any training. You need both numbers to know whether training is worth paying
for.

Three outcomes, and what each means:

* Base model already accurate  -> no headroom, change the task (raise N).
* Base model catastrophically wrong AND never partially right -> the reward
  will be flat and GRPO has nothing to bootstrap from. Make the task easier
  (lower N), or coarsen the answer, before spending a cent on GPUs.
* Base model in between, with visible spread across repeated samples of the
  same prompt -> this is the regime GRPO needs. Proceed.

The third column to watch is ``parse_rate``. If the base model cannot even
produce the answer format, your first problem is formatting, not physics, and a
short SFT warm-up will save you a lot of wasted RL steps.

Backends
--------
--backend openai   any OpenAI-compatible chat endpoint (works with vLLM's
                   server, so the same script probes a local model on a rented
                   box and a hosted API)
--backend hf       local transformers, for CPU smoke tests with a tiny VLM
--backend oracle   no model at all: answers with ground truth (upper bound)
--backend naive    no model at all: gravity-only extrapolation (lower bound)

Start with oracle and naive. They cost nothing and prove the whole scoring path
is wired correctly before a model is ever involved.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_rl.physics import Params, State  # noqa: E402
from membrane_rl.dataset import naive_ballistic  # noqa: E402
from membrane_rl.rewards import RewardConfig, compute_reward  # noqa: E402


def load_split(root: Path, split: str):
    path = root / split / "data.jsonl"
    if not path.exists():
        raise SystemExit(f"no dataset at {path} -- run scripts/gen_dataset.py first")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ----------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------
def backend_oracle(rec, root, split, **_):
    x, y, z = rec["answer"]
    return f"<answer>{x}, {y}, {z}</answer>"


def backend_naive(rec, root, split, params: Params, horizon: int, **_):
    st = State(
        pos=np.array(rec["current_state"]["pos"], dtype=float),
        vel=np.array(rec["current_state"]["vel"], dtype=float),
    )
    p = naive_ballistic(st, horizon, params)
    return f"<answer>{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}</answer>"


def backend_openai(rec, root, split, model, base_url, api_key, temperature, **_):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    content = [{"type": "text", "text": rec["prompt"]}]
    for rel in rec["images"]:
        b64 = _b64(root / split / rel)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=768,
    )
    return resp.choices[0].message.content or ""


def backend_hf(rec, root, split, model, temperature, _cache={}, **__):
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if "m" not in _cache:
        _cache["proc"] = AutoProcessor.from_pretrained(model)
        _cache["m"] = AutoModelForVision2Seq.from_pretrained(model)
        _cache["m"].eval()
    proc, mdl = _cache["proc"], _cache["m"]

    images = [Image.open(root / split / rel).convert("RGB") for rel in rec["images"]]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": rec["prompt"]}]
    text = proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )
    inputs = proc(text=text, images=images, return_tensors="pt")
    out = mdl.generate(**inputs, max_new_tokens=768,
                       do_sample=temperature > 0, temperature=max(temperature, 1e-5))
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)[0]


BACKENDS = {
    "oracle": backend_oracle,
    "naive": backend_naive,
    "openai": backend_openai,
    "hf": backend_hf,
}


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/membrane")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="oracle", choices=sorted(BACKENDS))
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--samples-per-item", type=int, default=1,
                    help="repeat each prompt to measure in-group spread, which "
                         "is exactly what GRPO learns from. Use >1 with "
                         "temperature > 0.")
    ap.add_argument("--horizon", type=int, default=25, help="only used by --backend naive")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.data)
    records = load_split(root, args.split)[: args.limit]
    fn = BACKENDS[args.backend]
    cfg = RewardConfig()
    params = Params()

    rows = []
    for i, rec in enumerate(records, 1):
        per_item = []
        for _ in range(args.samples_per_item):
            try:
                completion = fn(
                    rec, root, args.split,
                    model=args.model, base_url=args.base_url, api_key=args.api_key,
                    temperature=args.temperature, params=params, horizon=args.horizon,
                )
            except Exception as exc:                      # noqa: BLE001
                print(f"  [{i}] backend error: {exc}", file=sys.stderr)
                completion = ""
            per_item.append(compute_reward(completion, rec["answer"], cfg))
        rows.append({"id": rec.get("id", i), "scores": per_item,
                     "contact": rec["contact_frames_in_horizon"] > 0})
        if i % 20 == 0:
            print(f"  ...{i}/{len(records)}", file=sys.stderr)

    flat = [s for r in rows for s in r["scores"]]
    parsed = [s for s in flat if s["parsed"] > 0]
    errs = [s["error_radii"] for s in parsed]
    contact_errs = [s["error_radii"] for r in rows if r["contact"]
                    for s in r["scores"] if s["parsed"] > 0]
    free_errs = [s["error_radii"] for r in rows if not r["contact"]
                 for s in r["scores"] if s["parsed"] > 0]

    def med(v):
        return round(statistics.median(v), 3) if v else None

    # spread within a prompt: no spread means no learning signal for GRPO
    spreads = [
        max(s["reward"] for s in r["scores"]) - min(s["reward"] for s in r["scores"])
        for r in rows if len(r["scores"]) > 1
    ]

    summary = {
        "backend": args.backend,
        "model": args.model or None,
        "split": args.split,
        "n_items": len(rows),
        "samples_per_item": args.samples_per_item,
        "parse_rate": round(len(parsed) / len(flat), 3) if flat else 0.0,
        "mean_reward": round(float(np.mean([s["reward"] for s in flat])), 4) if flat else 0.0,
        "median_error_radii": med(errs),
        "median_error_contact": med(contact_errs),
        "median_error_free": med(free_errs),
        "acc_within_1_radius": round(float(np.mean([e <= 1.0 for e in errs])), 3) if errs else None,
        "acc_within_2_radii": round(float(np.mean([e <= 2.0 for e in errs])), 3) if errs else None,
        "mean_in_group_reward_spread": round(float(np.mean(spreads)), 4) if spreads else None,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)

    print(
        "\nRead it like this:\n"
        "  acc_within_2_radii near 1.0  -> no headroom, raise --horizon\n"
        "  acc_within_2_radii near 0.0  -> check median_error_radii; if it is huge\n"
        "                                  the reward is flat, lower the horizon\n"
        "  parse_rate < 0.9             -> fix formatting with a short SFT warm-up\n"
        "  mean_in_group_reward_spread near 0 -> GRPO has no signal; raise\n"
        "                                  temperature or make the task harder",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
