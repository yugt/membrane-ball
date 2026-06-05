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

## Physics Model Overview

The game models a ball interacting with a clamped circular membrane. For interview review, the key idea is that the ball, gravity, and membrane are treated as one coupled mechanical system.

When the paddle frame is stationary, the total mechanical energy is tracked as:

$$
E_{\text{total}} = K + U_g + U_e
$$

with:

$$
K = \frac{1}{2}m\lVert \mathbf{v} \rVert^2
$$

$$
U_g = mgz_b
$$

and an off-center membrane energy approximation:

$$
U_e(x_b, y_b, z_b) = \frac{U_{e,\text{centered}}(z_b)}{1 - (x_b^2 + y_b^2)}
$$

The denominator increases the effective stiffness as the ball approaches the clamped boundary of the circular membrane.

The membrane shape $u(x, y)$ is based on a massless circular membrane approximation. The elastic energy objective is:

$$
E_{\text{elastic}} = \frac{1}{2}T\iint_{\Omega}\lVert \nabla u \rVert^2\,dx\,dy
$$

Inside the contact patch, the membrane conforms to the sphere:

$$
u(x, y) = z_b - \sqrt{R^2 - r_{\text{ball}}^2}
$$

Outside the contact patch, the membrane solves the Laplace equation and becomes a logarithmic funnel:

$$
\nabla^2 u = 0 \quad \Rightarrow \quad u(d) = A\ln(d)
$$

The contact radius $r_c$ and coefficient $A$ are solved by matching height and slope at the contact boundary:

$$
\frac{\partial u_{\text{sphere}}}{\partial d}\Big|_{d=r_c}
=
\frac{\partial u_{\text{funnel}}}{\partial d}\Big|_{d=r_c}
\quad \Rightarrow \quad
A = \frac{r_c^2}{\sqrt{R^2-r_c^2}}
$$

The simulation then applies the contact force through the work-energy relationship:

$$
dU_e = -\mathbf{F}_{\text{contact}} \cdot d\mathbf{r}_b
$$

Rigid constraints such as the cylinder wall and circular frame use a standard velocity reflection rule:

$$
\mathbf{v}_{\text{new}} = \mathbf{v} - (1 + e)(\mathbf{v}\cdot\mathbf{n})\mathbf{n}
$$

The browser game uses restitution $e = 0.95$ for game feel, while verification scripts use idealized settings to check energy behavior.

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
