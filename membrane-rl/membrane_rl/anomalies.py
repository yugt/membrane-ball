"""Physically impossible segments, injected on purpose.

Each anomaly breaks the dynamics in a way a careful viewer can see, and its
exact onset is logged, so a video agent's "something is wrong here" can be
scored against ground truth instead of eyeballed.

The kinds are chosen to span what different detectors can catch:

* ``gravity_flip``   ball accelerates upward in free flight   (energy check: yes)
* ``energy_kick``    ball suddenly speeds up                  (energy check: yes)
* ``membrane_off``   ball falls straight through the membrane (energy check: no --
                     elastic energy just stays at 0; only the picture shows it)
* ``teleport``       ball jumps sideways between two frames   (energy check: no --
                     a horizontal jump leaves every energy term unchanged)
* ``hover``          ball freezes in mid-air                  (energy check: yes)

The two "no" rows matter for the pitch: an energy monitor alone misses them, a
video agent should not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .physics import MembraneSim

__all__ = ["KINDS", "Anomaly", "random_anomaly"]

KINDS = ("gravity_flip", "energy_kick", "membrane_off", "teleport", "hover")


@dataclass(frozen=True)
class Anomaly:
    kind: str
    start_frame: int
    duration: int = 40           # frames; ignored by instantaneous kinds
    magnitude: float = 1.0       # kind-specific, see apply()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown anomaly kind {self.kind!r}; choose from {KINDS}")

    @property
    def instantaneous(self) -> bool:
        return self.kind in ("energy_kick", "teleport")

    @property
    def persistent(self) -> bool:
        # Switching the membrane back on while the ball is below it would fire
        # a huge spurious force, so once it is off it stays off.
        return self.kind == "membrane_off"

    @property
    def end_frame(self) -> int | None:
        if self.persistent:
            return None
        return self.start_frame + (1 if self.instantaneous else self.duration)

    def to_dict(self) -> dict:
        return asdict(self)

    def apply(self, sim: MembraneSim, frame: int) -> None:
        """Call once before stepping into ``frame`` (i.e. before ``sim.step()``)."""
        if frame == self.start_frame:
            sim._emit("anomaly_start", sim.state.pos, at_frame=frame, kind=self.kind)
            if self.kind == "gravity_flip":
                sim.gravity_scale = -abs(self.magnitude)
            elif self.kind == "membrane_off":
                sim.membrane_scale = 0.0
            elif self.kind == "hover":
                sim.gravity_scale = 0.0
                sim.state.vel[:] = 0.0
            elif self.kind == "energy_kick":
                sim.state.vel *= 1.0 + abs(self.magnitude)
            elif self.kind == "teleport":
                self._teleport(sim)

        if not self.instantaneous and self.end_frame is not None and frame == self.end_frame:
            sim.gravity_scale = 1.0
            sim.membrane_scale = 1.0
            sim._emit("anomaly_end", sim.state.pos, at_frame=frame, kind=self.kind)

    def _teleport(self, sim: MembraneSim) -> None:
        """Jump sideways by ``magnitude`` ball radii, staying inside the arena."""
        p = sim.p
        pos = sim.state.pos
        jump = abs(self.magnitude) * p.ball_radius
        limit = p.r_cyl - p.ball_radius - 1e-3
        for ang in np.linspace(0.0, 2 * np.pi, 16, endpoint=False):
            cand = pos[:2] + jump * np.array([np.cos(ang), np.sin(ang)])
            if np.hypot(*cand) <= limit:
                pos[0], pos[1] = cand
                return


def random_anomaly(rng: np.random.Generator, n_frames: int, kind: str | None = None) -> Anomaly:
    """An anomaly that starts in the middle half of the clip, so it is always on screen."""
    kind = kind or str(rng.choice(KINDS))
    start = int(rng.integers(n_frames // 4, n_frames // 2))
    magnitude = {
        "gravity_flip": 1.0,
        "energy_kick": float(rng.uniform(0.6, 1.0)),
        "membrane_off": 1.0,
        "teleport": float(rng.uniform(1.5, 2.5)),
        "hover": 1.0,
    }[kind]
    # A long gravity flip launches the ball out of the top of the frame; 15-30
    # frames is plenty to see and keeps it in view.
    duration = int(rng.integers(15, 30)) if kind == "gravity_flip" else int(rng.integers(30, 60))
    return Anomaly(kind=kind, start_frame=start, duration=duration, magnitude=magnitude)
