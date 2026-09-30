# Technical Overview

Membrane Breakout 3D is a small physics/gameplay project that demonstrates a ball interacting with a circular elastic membrane frame. The repository contains a browser implementation, a Python/Pygame prototype, and numerical verification scripts.

## Architecture

- `game_web/` contains the primary interactive browser experience using Three.js.
- `game_python/` contains a local Pygame prototype with the same core simulation ideas.
- `verify_energy.py` and `verify_physics_stage5.py` are command-line checks for energy behavior and simulation stability. `verify_energy.py` runs the shared implementation in `membrane-rl/membrane_rl/physics.py`; `verify_physics_stage5.py` is a separate experiment with its own loop.
- `membrane-rl/` is a pip-installable package (`membrane_rl`) with evaluation tooling built on the same physics (see below).
- `physics_presentation.html` is a presentation artifact explaining the model and visuals.

## Membrane Model

The current implementation uses a massless circular membrane approximation. Contact is represented by solving for a contact radius between the ball and the membrane, then evaluating a logarithmic/conformal membrane shape for visualization.

The runtime simulation combines:

- Ball state integration using substepped symplectic Euler updates.
- Contact-force calculation from the membrane potential.
- Off-center energy scaling for impacts away from the membrane center.
- Rigid boundary responses for the circular frame, cylinder wall, and bricks.

This is intentionally lightweight enough for interactive rendering while still exposing physics quantities such as kinetic energy, gravitational potential energy, elastic energy, velocity, and acceleration.

## Browser Version

The browser game is in `game_web/` and is intended as the main demo. It renders:

- A 3D membrane grid and circular paddle frame.
- A bouncing ball with velocity and acceleration arrows.
- Brick collision and respawn behavior.
- A live energy plot overlay.

Run it with:

```bash
python3 -m http.server 8000 -d game_web
```

Then open `http://localhost:8000/`.

## Python Version

The Python prototype mirrors the core gameplay loop in Pygame and is useful for local debugging and comparison.

```bash
uv run python game_python/game.py
```

## Evaluation tooling (membrane-rl)

`membrane-rl/` reuses the physics to build evaluation tools for AI models. `membrane_rl/physics.py` is the repo's reference implementation and the only one used for labels:

- A verifiable-reward environment for vision-language models: rendered frames with the exact future ball position as ground truth.
- A video test bench: clips with an exact, timestamped answer key (membrane, rim and wall contacts plus five injected impossible-physics anomalies), a scorer, and a pixel-tracker baseline.
- A physics twin that recovers the 3D ball position from pixels and keeps a simulator in lock-step. On 30 held-out clips its 1 s forecast has a median error of 0.11 ball radii (gravity-only: 9.2), and it catches all 5 anomaly kinds with 0 false alarms.
- A VLM agent client for OpenAI-compatible endpoints.

Install with `uv sync` inside this repo, or `pip install "git+https://github.com/yugt/membrane-ball#subdirectory=membrane-rl"` elsewhere (extra `[agent]` adds the client). The command-line scripts are not part of the installed package; they live in `membrane-rl/scripts/` and run as `uv run python membrane-rl/scripts/<script>.py`. See `membrane-rl/README.md`.

## Verification

The repository includes a smoke-test script:

```bash
./scripts/verify.sh
```

It runs four steps: a Python syntax check, `verify_energy.py` (which uses the shared physics; expected output 3.232784 % drift, 117 wall / 34 rim bounces), `verify_physics_stage5.py` (a separate experiment; it now fails above 5 % drift), and the 46 `membrane-rl` tests. The current verification checks are not formal proofs, but they provide quick reproducibility signals for an evaluator.

## Known Limitations

- The browser game and the Pygame prototype keep their own copies of the physics logic; `membrane-rl/membrane_rl/physics.py` is the reference implementation (used by `verify_energy.py` and all evaluation tooling).
- The visual membrane model is designed for real-time interaction and is not a full finite-element solver.
- Generated HTML artifacts are kept for presentation/reference and are not part of the core runtime.
- The browser version depends on Three.js from a CDN, so internet access is required unless the dependency is vendored.
