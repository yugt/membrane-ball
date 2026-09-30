# membrane-rl

A verifiable-reward multimodal RL environment built on the physics from
[`yugt/membrane-ball`](https://github.com/yugt/membrane-ball).

A VLM is shown rendered frames of a ball interacting with a clamped elastic
membrane and must predict where the ball will be N frames later. Ground truth
comes from the simulator, so the reward is exact, free, and unlimited — the
property that makes this a usable RLVR environment rather than another
dataset-collection problem.

**Status: Phase 0–1 complete and verified on CPU. No GPU has been used or is
needed yet.**

---

## Why the frame does not move

The membrane frame is **fixed for the whole of an episode**. A moving frame is a
time-dependent constraint: it does work on the ball, mechanical energy is no
longer conserved, and the energy check that makes these labels trustworthy stops
meaning anything. Measured:

| configuration | energy leak over 5000 steps |
|---|---|
| frame at origin, fixed | 3.23 % (bit-for-bit vs upstream `verify_energy.py`) |
| frame at (0.35, −0.20), fixed | 1.98 % |
| frame at (−0.40, 0.10), fixed | 3.43 % |
| frame at (0.25, 0.30), fixed | 2.78 % |
| frame wobbling at 0.5 Hz | **6.30 %** — past the 5 % threshold |

The frame *centre* is still resampled per episode. Each trajectory stays an
autonomous, energy-conserving system, and the dataset gains a geometric axis
that forces the model to actually locate the green ring instead of memorising
one scene layout. `tests/test_physics.py` pins all of this.

## Single source of truth

Upstream carries three copies of the physics (`game_web/game.js`,
`game_python/physics.py`, `verify_energy.py`). That was a tidiness problem
before; here it is a correctness problem, because labels must come from exactly
one implementation. `membrane_rl/physics.py` is a faithful port of the
energy-conserving reference loop and is the only implementation in this project.
The port is pinned to upstream to 6 significant figures (leak 3.232784 %, 34 ring
bounces, 117 cylinder bounces) so a refactor cannot silently corrupt every
dataset generated afterwards.

---

## What the calibration says

Run before writing any training code, costs nothing:

```bash
python3 -c "
from membrane_rl.physics import Params
from membrane_rl.dataset import calibrate_horizon
import json; print(json.dumps(calibrate_horizon(
    [10,20,25,30,50,80], n_per=250,
    params=Params(frame_offset_max=0.4)), indent=2))"
```

Median error of gravity-only extrapolation, in ball radii:

| N (frames) | horizon | frac. with contact | naive err (contact) | naive err (free) |
|---|---|---|---|---|
| 10 | 160 ms | 0.23 | 1.01 | 0.00 |
| 20 | 320 ms | 0.35 | 2.99 | 0.00 |
| **25** | **400 ms** | **0.40** | **4.41** | **0.83** |
| **30** | **480 ms** | **0.43** | **5.62** | **1.64** |
| 50 | 800 ms | 0.55 | 10.55 | 4.65 |
| 80 | 1280 ms | 0.71 | 19.16 | 7.47 |

At short horizons `naive err (free)` is exactly 0, because with no membrane
contact gravity-only extrapolation *is* the true dynamics — those samples carry
no information about the membrane at all. It only becomes non-zero at N ≥ 25,
when cylinder-wall bounces enter the window.

**N = 25–30 is the calibrated band**: ~40 % of samples involve contact, those
samples leave 4–6 ball radii of headroom, and the free-flight samples are no
longer free points. That spread is what gives GRPO in-group variance to learn
from.

## What is still unknown

The sweep bounds the *headroom*. It says nothing about what a VLM can actually
do, and that is the other half of the go/no-go decision. Two risks remain open:

1. **Perception, not physics.** Reading (x, y, z) off an isometric wireframe is
   hard. The renderer draws a shadow ellipse at z = 0 and a dashed drop line
   specifically so position is recoverable; without them the three frames of a
   sample look nearly identical. Whether that is enough for a 3B model is
   unmeasured. If `median_error_free` is large in the probe, the model cannot
   read positions and the first thing to fix is rendering, not the training
   recipe.
2. **The task may need quantitative simulation.** Beating naive extrapolation on
   contact samples requires something close to integrating the dynamics. A 3B
   VLM may plateau well short of that. The shaped reward still ranks "closer"
   above "further", so RL is not hopeless, but the achievable ceiling is
   genuinely unknown. If the probe shows a flat reward, drop to N = 15–20 or
   coarsen the answer (predict z only, or a discretised cell) before spending
   money.

Run the probe next. It is the last free step.

---

## Video-agent layer (hackathon prep)

The same simulator also produces **video clips with an exact, timestamped
answer key** — membrane contacts, ring and wall bounces, and five injected
physically-impossible anomalies — so a video agent's output can be scored
instead of eyeballed. See [`HACKATHON_RUNBOOK.md`](HACKATHON_RUNBOOK.md).

```bash
pip install -r requirements.txt
python scripts/gen_clips.py --n 20 --out clips/ --annotated
python scripts/eval_events.py --clips clips/ --detector tracker     # model-free baseline
python scripts/eval_events.py --clips clips/ --detector twin        # physics twin (fast path)
python scripts/eval_forecast.py --clips clips/                      # 1 s forecast accuracy
python scripts/run_agent.py  --clips clips/ --limit 1 --dry-run     # VLM agent, no network
```

## Layout

```
membrane_rl/
  physics.py     single source of truth; deterministic, energy-conserving
  render.py      headless PIL renderer with depth sorting + position annotations
  dataset.py     sample generation, naive baseline, horizon calibration
  rewards.py     parsing + shaped verifiable reward
  anomalies.py   injected physically-impossible segments with logged onset
  episode.py     one clip = frames + answer key (events, energy, trajectory)
  video.py       episode -> .mp4 + .json (plain for agents, annotated for humans)
  scoring.py     event P/R/F1 and anomaly latency against the answer key
  tracker.py     model-free pixel baseline
  agent.py       VLM event agent over any OpenAI-compatible endpoint
  perception.py  pixels -> 3D ball position and rim centre
  twin.py        physics twin: 1 s forecast + anomaly alarm from innovation
scripts/
  gen_dataset.py     train/test splits with disjoint frame-offset bands
  baseline_probe.py  zero-shot probe; oracle / naive / openai / hf backends
  gen_clips.py       video clips + answer keys
  eval_events.py     score oracle / tracker / agent predictions
  run_agent.py       run the VLM agent, write <clip>.pred.json
  eval_forecast.py   1 s forecast accuracy of the twin vs gravity-only
tests/
  test_physics.py      energy, determinism, geometry, reward monotonicity
  test_video_agent.py  event log, anomalies, scoring, video I/O, agent parsing
  test_twin.py         camera inversion, perception accuracy, twin forecast + alarms
```

## Quick start

```bash
pip install -r requirements.txt
python3 -m pytest -q tests/                       # 46 tests, ~10 s
python3 scripts/gen_dataset.py --train 2000 --test 300 --horizon 25
python3 scripts/baseline_probe.py --backend oracle --split test   # upper bound
python3 scripts/baseline_probe.py --backend naive  --split test --horizon 25
```

Verified on this scaffold:

| backend | mean reward | median err (radii) | contact | free |
|---|---|---|---|---|
| oracle (upper bound) | 1.100 | 0.00 | 0.00 | 0.00 |
| naive ballistic (lower bound) | 0.474 | 1.70 | 4.68 | 0.92 |

Any model result must land inside that band to be believable. A model that does
not beat 0.474 has learned nothing about the membrane.

## The held-out split is structural

`gen_dataset.py` gives train and test **disjoint bands of frame offset** (train
≤ 0.25, test 0.25–0.40). A split that differs only by RNG seed measures nothing —
it draws from the distribution the model was trained on. Generalising to unseen
frame placements is a real probe, and it is the number worth reporting.

Note the data supply is effectively unbounded: initial position, frame position
and horizon are all continuous. The binding constraint is therefore **not sample
count** but coverage of the regime that carries information (contact events) and
keeping train and test genuinely distinct.

## Probing a real model

Any OpenAI-compatible endpoint works, including vLLM's server, so the same
command probes a hosted API and a local model on a rented box:

```bash
python3 scripts/baseline_probe.py --backend openai \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --split test --limit 200 --samples-per-item 4 --temperature 0.8
```

`--samples-per-item > 1` measures `mean_in_group_reward_spread`. GRPO learns
from variation *within* a group of samples for the same prompt; if that spread
is ~0 there is no gradient no matter how wrong the model is. Check it before
renting anything.

## Next steps

1. Run the probe against Qwen2.5-VL-3B. Free on CPU if slow, or ~$2 on a rented
   4090/5090. **This is the go/no-go.**
2. If the band is healthy, wire GRPO with SmolVLM-256M on CPU to prove the
   plumbing before touching a real model.
3. Only then rent a GPU. On a 5090 (Blackwell, sm_120), check `vllm` imports and
   loads a VLM in the first 20 minutes — flash-attn on sm_120 is still rough and
   `VLLM_FLASH_ATTN_VERSION=2` is the documented fallback. If it needs a source
   build, kill the pod and take an A6000 instead; the time is worth more than
   the price difference.
