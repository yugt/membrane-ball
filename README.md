# Membrane Breakout 3D

A physics-based 3D Breakout prototype where the player controls an elastic circular membrane frame to bounce a ball and break glowing bricks.

The project is built to show three things clearly:

- A playable browser demo using Three.js.
- A Python/Pygame prototype of the same idea.
- Numerical verification scripts that track energy behavior in the membrane-ball system.

## Review Links

<a href="https://www.youtube.com/watch?v=-JdWYVzKI3c&t=1s" title="Watch the Membrane Breakout 3D demo video">
  <img src="https://img.youtube.com/vi/-JdWYVzKI3c/maxresdefault.jpg" alt="Membrane Breakout 3D demo video" width="100%">
</a>

- Browser game: [`game_web/index.html`](game_web/index.html)
- Demo video: [YouTube walkthrough](https://www.youtube.com/watch?v=-JdWYVzKI3c&t=1s)
- Physics presentation: [`physics_presentation.html`](physics_presentation.html)
- Technical overview: [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md)
- Smoke verification: `./scripts/verify.sh`

## Quick Start

### Browser Demo

```bash
python3 -m http.server 8000 -d game_web
```

Open:

```text
http://localhost:8000/
```

Controls:

- Move the mouse or touch-drag to move the membrane frame.
- Drag the scene to rotate the camera.
- Refresh the page to reset the simulation.

### Python Prototype

This repo uses `uv` for reproducible local execution.

```bash
uv run python game_python/game.py
```

### Verification

Run the smoke test:

```bash
./scripts/verify.sh
```

Or run the verification scripts individually:

```bash
uv run python verify_energy.py
uv run python verify_physics_stage5.py
```

## Repository Structure

```text
Membrane/
├── game_web/                 # Main Three.js browser implementation
│   ├── index.html
│   ├── style.css
│   ├── game.js
│   └── MOBILE_PACKAGING.md
├── game_python/              # Pygame prototype and reusable Python physics module
│   ├── game.py
│   ├── physics.py
│   └── verify_physics.py
├── docs/
│   └── TECHNICAL_OVERVIEW.md # Evaluator-focused implementation overview
├── scripts/
│   └── verify.sh             # Syntax and physics smoke checks
├── verify_energy.py          # Energy conservation verification
├── verify_physics_stage5.py  # Additional physics verification scenario
├── record_and_analyze.py     # Plotly analysis/presentation generator
└── physics_presentation.html # Presentation artifact
```

## Implementation Notes

The current membrane model is a real-time massless circular membrane approximation. It solves a contact radius for ball/membrane interaction, applies a lightweight elastic force model, and visualizes the resulting membrane shape as a logarithmic/conformal surface.

The simulation uses substepped symplectic Euler integration to keep the browser and Python versions stable enough for interactive playback. The live overlay tracks kinetic, gravitational, elastic, and total energy.

## What To Evaluate

- `game_web/index.html`: primary interactive demo.
- `physics_presentation.html`: mathematical presentation of the membrane model and constraints.
- `game_web/game.js`: Three.js rendering, gameplay loop, contact forces, and live energy plot.
- `game_python/physics.py`: compact Python membrane/ball model.
- `verify_energy.py`: reproducible energy-conservation smoke check.
- `docs/TECHNICAL_OVERVIEW.md`: technical summary and known limitations.

## Requirements

- Python 3.12 or newer for `uv` execution.
- `uv` for dependency-managed Python commands.
- A modern browser with WebGL support.
- Internet access for the browser demo's Three.js CDN dependency.

## Known Limitations

- The Python and web versions duplicate similar physics logic instead of sharing a single source of truth.
- The membrane is an interactive approximation, not a full finite-element simulation.
- Some checked-in HTML files are generated/presentation artifacts rather than source code.
