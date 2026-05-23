# Membrane Breakout 3D

A playable physics-based Breakout game where the player controls an **elastic membrane frame (trampoline paddle)** to bounce a ball and demolish glowing bricks floating above. 

This repository contains a clean, scratch-built game structure. The legacy high-fidelity scientific simulation code has been preserved as reference and is `.gitignore`d.

---

## Repository Structure

```
Membrane/
├── game_python/             # Phase 1: Local Python Prototype (Pygame)
│   ├── game.py              # Pygame application & isometric wireframe rendering
│   └── physics.py           # Real-time Finite Difference Wave equation solver
├── game_web/                # Phase 2: Beautiful WebGL Browser Game (Three.js)
│   ├── index.html           # HTML5 structure with dashboard HUD overlay
│   ├── style.css            # Dark cyber-neon CSS layout
│   └── game.js              # Three.js 3D WebGL renderer and dynamic wave solver
├── .gitignore               # Ignores standard virtual envs and caches
└── README.md                # Project documentation & instructions
```

---

## Core Game Mechanics

### 1. Real-time Elastic Membrane Waves
Instead of using slow optimization loops (L-BFGS), the game integrates a **2D Finite Difference Method (FDM) wave equation** at 60 FPS in real time:
$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - d \frac{\partial u}{\partial t}$$
This integrates in `<0.5ms` per frame, causing the membrane to flex, ripple, and oscillate beautifully upon ball impact.

### 2. Physical Coupling & Slice Mechanics
When the ball hits the membrane, it deforms it based on depth penetration. The membrane applies an upward spring restoring force onto the ball. 
* **The "Slice" Effect (Horizontal Momentum Transfer):** Moving the paddle frame quickly in the plane just as the ball hits transmits horizontal kinetic energy, letting you "slice" the ball to direct it towards specific bricks.

---

## Quick Start Instructions

### Option A: Web-based 3D Game (WebGL + Three.js) — *Recommended*
Since the WebGL version runs natively in any browser without installing Python packages:
1. Open the [game_web/index.html](file:///Users/yugt/Documents/Membrane/game_web/index.html) file directly in your browser, or start a local HTTP server inside the `game_web/` directory.
2. Slide your mouse to move the trampoline paddle in 3D perspective space.
3. Bounce the ball and demolish the blocks!

### Option B: Local Python Game (Pygame + NumPy)
To run the Pygame prototype locally:
1. Initialize virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install pygame numpy
   ```
2. Run the game:
   ```bash
   python game_python/game.py
   ```
3. Control the paddle frame using your mouse. Press `SPACEBAR` to start/retry!
