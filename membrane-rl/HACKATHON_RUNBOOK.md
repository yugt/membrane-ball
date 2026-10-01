# Hackathon runbook — Real-Time Video Agents Hack (SF, Fri Oct 2)

**Pitch (first sentence, every time):** video agents are hard to trust because
real footage has no answer key. We generate physics video *with* an exact,
timestamped answer key, score a video agent against it live, and show where it
beats — and loses to — a model-free baseline.

**Architecture — two speeds:**

```
pixels ──► perception (1 ms) ──► physics twin (0.2 ms) ──► 1 s forecast (7 ms)
  │           3D ball + rim           │  innovation ─► anomaly alarm (≤ 0.12 s)
  │                                   └─ twin's own events ─► membrane / rim / wall
  └──► Cosmos over 2 s windows (seconds) ──► what happened, and WHY, in words
```

A language model is far too slow (seconds per call) to say where the ball will
be in 1 s, and poor at numeric dynamics. The fast path predicts and alarms; the
slow path (Cosmos) explains and answers questions; the answer key scores both.

## What exists before the event (disclose this)

The new repo depends on the published `membrane-rl` package from
`yugt/membrane-ball` (everything in `membrane-rl/`, committed before Oct 2 and
merged into `main` in PR #1; developed on branch `claude/laughing-noether-tpl7lw`):

| Piece | File | Status |
|---|---|---|
| Physics with ground-truth event log | `membrane_rl/physics.py` | done, parity-pinned to the repo's original `verify_energy.py` |
| 5 injected anomalies | `membrane_rl/anomalies.py` | done |
| Clip + answer-key generator | `membrane_rl/video.py`, `scripts/gen_clips.py` | done, ~4 s/clip at 720 px |
| Scorer (P/R/F1, anomaly latency) | `membrane_rl/scoring.py`, `scripts/eval_events.py` | done |
| Pixel-tracker baseline | `membrane_rl/tracker.py` | done |
| Pixels → 3D ball + rim centre | `membrane_rl/perception.py` | done, ~1 ms/frame, error ≈ 1.4 % of ball radius |
| Physics twin: forecast + anomaly alarm | `membrane_rl/twin.py`, `scripts/eval_forecast.py` | done, numbers below |
| Offline overlay of the fast path (for humans) | `scripts/overlay_twin.py` | done; the live version is day-of work |
| VLM agent client (W&B / NVIDIA / local / any OpenAI-compatible) | `membrane_rl/agent.py`, `scripts/run_agent.py` | done offline; **never called a real model** |
| 3D debug view + per-frame motion audit | `membrane_rl/debug3d.py`, `scripts/debug_plotly.py` | done; every frame of 30 held-out clips explained |
| Tests | `tests/` | 59 passing |

**Build on the day** (new commits, dated Oct 2): real model integration and
prompt tuning, the live view (video + forecast path + twin alarms + Cosmos
explanations), the scoreboard (W&B), search over events, the real-footage
segment, the demo itself.

## Setup on the event VM (5 min)

In the fresh event repo, install the package (add `[agent]` for the VLM client):

```bash
python3 -m venv .venv && . .venv/bin/activate      # or: uv venv (then `uv pip install ...`)
pip install "git+https://github.com/yugt/membrane-ball#subdirectory=membrane-rl"
pip install "membrane-rl[agent,debug] @ git+https://github.com/yugt/membrane-ball#subdirectory=membrane-rl"   # + openai, plotly
python -c "import membrane_rl"                     # sanity check
```

The scripts are not part of the installed package. Get them from the source repo
and run them from `membrane-rl/scripts/` (or copy the ones you need into the new
repo):

```bash
git clone https://github.com/yugt/membrane-ball.git
cd membrane-ball/membrane-rl
pip install -e ".[dev,agent,debug]"                # or, inside the clone: `uv sync` and prefix commands with `uv run`
python -m pytest -q tests/                         # expect 59 passed
python scripts/gen_clips.py --n 20 --out clips/ --annotated
python scripts/eval_events.py --clips clips/ --detector oracle    # must be all 1.0
python scripts/eval_events.py --clips clips/ --detector tracker   # baseline numbers below
python scripts/eval_events.py --clips clips/ --detector twin      # fast path
python scripts/eval_forecast.py --clips clips/                    # 1 s forecast accuracy
python scripts/debug_plotly.py clips/clip_*.mp4 --audit-only     # every frame explained? (exit 1 if not)
python scripts/debug_plotly.py clips/clip_002.mp4                 # 3D debug view of one clip
```

If `pip` is missing on the VM: `python3 -m ensurepip` or use `uv`. The
`git+https` install needs GitHub access as well as PyPI; with no internet at all,
everything except the agent needs only `numpy pillow imageio-ffmpeg` (copy the
repo over and install it with `pip install -e membrane-rl`, or put `membrane-rl/`
on `PYTHONPATH`).

## Numbers to beat (30 held-out clips, seed 7, 6 s each, tol 0.3 s)

Thresholds were set on a different 20-clip set; these clips were not used for tuning.

| | tracker (pixels only) | **physics twin** |
|---|---|---|
| names what was hit — F1 over membrane / rim / wall / anomaly | 0.04 | **0.97** |
| "something happened here" (coarse F1) | **0.88** | 0.85 |
| anomaly caught: teleport · hover | 100 % · 100 % | 100 % · 100 % |
| anomaly caught: gravity_flip | 100 % (0.21 s) | **100 %** (0.11 s) |
| anomaly caught: energy_kick · membrane_off | **0 % · 0 %** | **100 % · 100 %** (≤ 0.02 s) |
| false alarms on normal clips | 0 % | 0 % |
| where is the ball in 1 s — median error | — | **0.11 ball radii** (gravity-only: 9.2); 95 % within 1 radius |
| … with a membrane bounce inside that second | — | **0.12** (gravity-only: 11.3) |
| cost per frame | ~4 ms | ~1.3 ms + 7 ms per forecast |

Honest caveats for Q&A: the twin knows the physical constants (same
simulator); everything about the episode — rim position, ball state — comes
from pixels. `membrane_off` is timed from the first frame the ball is deeper
than a live membrane ever allows (−0.3; live minimum over 200 episodes −0.18).

Where Cosmos must add value: explaining *why* in words, answering questions,
and anything outside the twin's model — e.g. **real footage**, where no twin
exists. That is the natural end of the demo.

## Day schedule (08:30–18:30)

| Time | Do | Done when |
|---|---|---|
| 08:30–09:30 | Setup above on the event VM; get API keys from organisers; confirm model names | tests pass, oracle = 1.0 |
| 09:30–10:30 | **Model smoke test** (below): one window, one call, look at raw output | a parseable JSON reply |
| 10:30–12:30 | `run_agent.py` on 5 clips → `eval_events.py --detector preds`; tune prompt/window/fps | a first real score table |
| 12:30–14:30 | Live view: clip + twin forecast path + twin alarms + Cosmos text, ground truth beside it; W&B logging | one clip plays with all three |
| 14:30–16:00 | Event search ("show every ring bounce"), real-footage clip | query returns timestamps |
| 16:00–17:15 | Full eval on 20 clips, final table, record backup video of the demo | numbers frozen |
| 17:15–18:30 | **No new code.** Rehearse 3× | |

**Cut order if late:** real footage → search → W&B → live view (fall back to
the annotated mp4 + score table). **Never cut:** agent vs tracker vs ground
truth table.

## Model smoke test (the one thing not verified yet)

The agent speaks the OpenAI chat protocol, so the model is a flag:

| `--provider` | Endpoint | Key env | Default model | Notes |
|---|---|---|---|---|
| `wandb` | `api.inference.wandb.ai/v1` | `WANDB_API_KEY` | `Qwen/Qwen3.8-27B` | event sponsor; multimodal Qwen takes images. `--project entity/project` for usage tracking |
| `nvidia` | `integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` | none — pass `--model` | hosted `cosmos-reason1-7b` was **deprecated Mar 2026**; use the Cosmos Reason id the organisers give |
| `local` | `localhost:8000/v1` | — | none | vLLM/SGLang you run on the event GPU |
| `openai` | `--base-url` | `OPENAI_API_KEY` | none | any other OpenAI-compatible service |

Plan: **Cosmos** (organiser endpoint) is the video-understanding model the
judges expect to see; **W&B Qwen** is the fallback and the text LLM for the
search/Q&A layer. Try both on 5 clips and put both rows in the results table —
"which model catches which anomaly" is itself a finding.

```bash
python scripts/run_agent.py --clips clips/ --limit 1 --dry-run                 # payload sizes, no network
python scripts/run_agent.py --clips clips/ --limit 1 --provider wandb          # frames mode
python scripts/run_agent.py --clips clips/ --limit 1 --provider nvidia --model <id> --mode video
python scripts/eval_events.py --clips clips/ --detector preds --limit 1
```

Check, in this order, and write the answers down:

1. **Model id / endpoint** — the organisers may serve Cosmos on CoreWeave behind
   their own URL. Pass `--provider openai --base-url <url> --model <id>`.
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
| Twin false alarms at rim hits | rim bounce is discontinuous; sub-pixel error flips it | detector threshold ×3 near the rim, held 0.3 s after leaving it |
| Ball airborne above the membrane looks pressed into it | renderer depth sort had its sign flipped (fixed Oct 1) | regenerate any clips rendered before the fix |
| Video left-right mirrored vs a real camera; 3D view never matched it | projection used +rx for screen right (fixed Oct 1) | same; `debug_plotly.py`'s "video camera" now reproduces the mp4 view |
| A normal clip looks anomalous (ball "falls faster", jumps in mid-air) | projection: motion toward the camera looks like falling; far-wall bounces look causeless | open the clip's debug file: the 3D view and the per-frame causes show what happened |
| `membrane_off` "detected 1 s late" | injected long before the ball reaches the membrane | score from `visible_t` (first physically impossible frame) |
| Twin reports imaginary membrane hits after ball falls through | re-locks far below a membrane its model still has | ball below −0.3 inside the rim ⇒ one alarm, stop simulating |
| Same event reported by two overlapping windows | overlap by design | `merge_events` de-dups within 0.25 s |

## Questions judges will ask

- **"Is this just a toy?"** — It's a test bench. The answer key is what makes
  the numbers trustworthy; the same agent then runs on real footage (show it).
- **"Why not a real physics engine / real video?"** — Real video has no exact
  labels; engines are black boxes. Here every event is logged at 0.8 ms
  resolution and anomalies are injected with known onset.
- **"Why not just ask the VLM where the ball goes?"** — seconds of latency to
  predict one second ahead, and numeric dynamics is its weak spot. The twin
  does it in 7 ms at 0.12 ball radii; the VLM explains.
- **"Doesn't the twin cheat by knowing the physics?"** — it knows the laws, not
  the episode. That's also why real footage needs the VLM.
- **"What did you build today?"** — Point at the Oct 2 commits.
