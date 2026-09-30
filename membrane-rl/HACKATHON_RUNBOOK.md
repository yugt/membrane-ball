# Hackathon runbook — Real-Time Video Agents Hack (SF, Fri Oct 2)

**Pitch (first sentence, every time):** video agents are hard to trust because
real footage has no answer key. We generate physics video *with* an exact,
timestamped answer key, score a video agent against it live, and show where it
beats — and loses to — a model-free baseline.

## What exists before the event (disclose this)

Everything in `membrane-rl/` on branch `claude/laughing-noether-tpl7lw` of
`yugt/membrane-ball`, committed before Oct 2:

| Piece | File | Status |
|---|---|---|
| Physics with ground-truth event log | `membrane_rl/physics.py` | done, parity-pinned to upstream |
| 5 injected anomalies | `membrane_rl/anomalies.py` | done |
| Clip + answer-key generator | `membrane_rl/video.py`, `scripts/gen_clips.py` | done, ~4 s/clip at 720 px |
| Scorer (P/R/F1, anomaly latency) | `membrane_rl/scoring.py`, `scripts/eval_events.py` | done |
| Pixel-tracker baseline | `membrane_rl/tracker.py` | done |
| VLM agent client (OpenAI-compatible) | `membrane_rl/agent.py`, `scripts/run_agent.py` | done offline; **never called a real model** |
| Tests | `tests/` | 39 passing |

**Build on the day** (new commits, dated Oct 2): real model integration and
prompt tuning, a live/streaming UI, the scoreboard (W&B), search over events,
the real-footage segment, the demo itself.

## Setup on the event VM (5 min)

```bash
git clone -b claude/laughing-noether-tpl7lw https://github.com/yugt/membrane-ball.git
cd membrane-ball/membrane-rl
python3 -m venv .venv && . .venv/bin/activate      # or: uv venv && uv pip install -r requirements.txt
pip install -r requirements.txt
python -m pytest -q tests/                         # expect 39 passed
python scripts/gen_clips.py --n 20 --out clips/ --annotated
python scripts/eval_events.py --clips clips/ --detector oracle    # must be all 1.0
python scripts/eval_events.py --clips clips/ --detector tracker   # baseline numbers below
```

If `pip` is missing on the VM: `python3 -m ensurepip` or use `uv`. If the VM has
no internet to PyPI, everything except the agent needs only
`numpy pillow imageio-ffmpeg`.

## Baseline to beat (20 clips, 6 s each, tol 0.3 s)

| | tracker |
|---|---|
| "something happened here" (coarse F1) | **0.90** |
| *what* was hit — membrane / ring / wall | **0.00** (it cannot tell) |
| anomaly: teleport, hover | 100 % |
| anomaly: gravity_flip | 50 % |
| anomaly: energy_kick, membrane_off | **0 %** |
| false alarms on normal clips | 0 % |

The story writes itself: the tracker times events well but cannot *name* them
and cannot see the membrane. The VLM's job is fine-grained labels plus the
anomalies the tracker misses. Two anomalies (`teleport`, `membrane_off`) are
also invisible to an energy check — only the picture gives them away.

## Day schedule (08:30–18:30)

| Time | Do | Done when |
|---|---|---|
| 08:30–09:30 | Setup above on the event VM; get API keys from organisers; confirm model names | tests pass, oracle = 1.0 |
| 09:30–10:30 | **Model smoke test** (below): one window, one call, look at raw output | a parseable JSON reply |
| 10:30–12:30 | `run_agent.py` on 5 clips → `eval_events.py --detector preds`; tune prompt/window/fps | a first real score table |
| 12:30–14:30 | Live view: play clip, stream agent events next to ground truth; W&B logging | one clip plays with both timelines |
| 14:30–16:00 | Event search ("show every ring bounce"), real-footage clip | query returns timestamps |
| 16:00–17:15 | Full eval on 20 clips, final table, record backup video of the demo | numbers frozen |
| 17:15–18:30 | **No new code.** Rehearse 3× | |

**Cut order if late:** real footage → search → W&B → live view (fall back to
the annotated mp4 + score table). **Never cut:** agent vs tracker vs ground
truth table.

## Model smoke test (the one thing not verified yet)

```bash
export NVIDIA_API_KEY=...   # from organisers / build.nvidia.com
python scripts/run_agent.py --clips clips/ --limit 1 --dry-run         # payload sizes, no network
python scripts/run_agent.py --clips clips/ --limit 1                    # frames mode (any VLM)
python scripts/run_agent.py --clips clips/ --limit 1 --mode video       # if endpoint takes video
python scripts/eval_events.py --clips clips/ --detector preds --limit 1
```

Check, in this order, and write the answers down:

1. **Model id** — default is `nvidia/cosmos-reason1-7b` at
   `https://integrate.api.nvidia.com/v1`. The organisers may give a different
   id, endpoint, or a VAST-hosted one. Pass `--model` / `--base-url`.
2. **Video input** — does the endpoint accept `video_url` content? If it errors,
   stay in `frames` mode (works with any vision chat model).
3. **Latency per window** — the log prints it. Real-time needs < window hop
   (1.5 s). If slower, present as "near real-time" (analyse the last 2 s every
   few seconds) or raise `--hop`.
4. **Output format** — Cosmos Reason may emit `<think>…</think>` before the
   answer; `parse_events` strips it. If replies are prose, lower temperature
   and put the JSON example last in the prompt.
5. **Request size** — ~300 KB/window in frames mode at 448 px, ~120 KB in
   video mode. If rejected, lower `--sample-fps` or `AgentConfig.image_size`.

## Known obstacles and fixes (from prep)

| Symptom | Cause | Fix |
|---|---|---|
| Browser game ring collision in wrong place | `game_web/game.js` centres ring at origin, paddle moves | don't use the game for ground truth; clips come from `membrane_rl` |
| Energy readout jumps +185 % with membrane off | elastic PE computed geometrically | fixed: PE uses `membrane_scale` |
| Ball flies out of frame on gravity flip | long flips | fixed: 15–30 frames |
| Ball cut off at bottom / top | training camera crop | fixed: `RenderConfig.for_size` pulls camera back |
| Tracker double-counts membrane contacts | kick on entry and exit | fixed: 0.35 s refractory |
| mp4 won't encode on fresh VM | no system ffmpeg | `imageio-ffmpeg` wheel bundles one |
| Headless box, no display | — | nothing here needs a display |
| Agent timestamps outside the window | hallucination | `parse_events` drops them |
| Same event reported by two overlapping windows | overlap by design | `merge_events` de-dups within 0.25 s |

## Questions judges will ask

- **"Is this just a toy?"** — It's a test bench. The answer key is what makes
  the numbers trustworthy; the same agent then runs on real footage (show it).
- **"Why not a real physics engine / real video?"** — Real video has no exact
  labels; engines are black boxes. Here every event is logged at 0.8 ms
  resolution and anomalies are injected with known onset.
- **"What did you build today?"** — Point at the Oct 2 commits.
