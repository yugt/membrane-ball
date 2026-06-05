# Technical Overview

Membrane Breakout 3D is a small physics/gameplay project that demonstrates a ball interacting with a circular elastic membrane frame. The repository contains a browser implementation, a Python/Pygame prototype, and numerical verification scripts.

## Architecture

- `game_web/` contains the primary interactive browser experience using Three.js.
- `game_python/` contains a local Pygame prototype with the same core simulation ideas.
- `verify_energy.py` and `verify_physics_stage5.py` are command-line checks for energy behavior and simulation stability.
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

## Verification

The repository includes a smoke-test script:

```bash
./scripts/verify.sh
```

It compiles the Python files and runs the two main numerical verification scripts. The current verification checks are not formal proofs, but they provide quick reproducibility signals for an evaluator.

## Known Limitations

- The Python and web implementations duplicate similar physics logic instead of sharing one source of truth.
- The visual membrane model is designed for real-time interaction and is not a full finite-element solver.
- Generated HTML artifacts are kept for presentation/reference and are not part of the core runtime.
- The browser version depends on Three.js from a CDN, so internet access is required unless the dependency is vendored.
