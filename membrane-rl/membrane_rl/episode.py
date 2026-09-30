"""One clip = one episode: frames + an exact, timestamped answer key.

This is the unit the video-agent demo works with. The answer key (``events``,
``energy``) comes from the simulator, never from a model, so any detector's
output can be scored with ``membrane_rl.scoring``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .anomalies import Anomaly
from .physics import MembraneSim, Params, State

__all__ = ["Episode", "run_episode"]


@dataclass
class Episode:
    params: Params
    seed: int
    frame_center: list[float]
    anomaly: Anomaly | None
    states: list[State] = field(default_factory=list)       # frame i -> state
    events: list[dict] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)       # total energy per frame

    @property
    def n_frames(self) -> int:
        return len(self.states)

    @property
    def fps(self) -> float:
        # one sim frame per video frame, so video time == simulation time
        return 1.0 / self.params.dt

    def answer_key(self) -> dict:
        """Everything a scorer needs, as plain JSON."""
        return {
            "seed": self.seed,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "duration_s": round(self.n_frames / self.fps, 3),
            "frame_center": self.frame_center,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
            "events": self.events,
            "energy_total": [round(e, 4) for e in self.energy],
            "trajectory": [[round(float(v), 4) for v in s.pos] for s in self.states],
            "params": {k: v for k, v in asdict(self.params).items()},
        }


def run_episode(
    n_frames: int,
    seed: int = 0,
    params: Params | None = None,
    anomaly: Anomaly | None = None,
    frame_center=None,
    sim_hook=None,
) -> tuple[Episode, MembraneSim]:
    """Simulate ``n_frames`` frames. ``sim_hook(sim, frame)`` runs after each frame
    (the renderer uses it so frames are drawn from live simulator state)."""
    params = params or Params(frame_offset_max=0.3)
    sim = MembraneSim(params, seed=seed)
    sim.reset(frame_center=frame_center)
    ep = Episode(params=params, seed=seed,
                 frame_center=[round(float(v), 4) for v in sim.frame_center],
                 anomaly=anomaly)

    def record():
        ep.states.append(sim.state.copy())
        ep.energy.append(sim.energy()["total"])
        if sim_hook is not None:
            sim_hook(sim, sim.frame)

    record()                                   # frame 0 = initial condition
    for frame in range(1, n_frames):
        if sim.state.pos[2] < -3.0 and sim.membrane_scale > 0.0:
            # Far below a LIVE membrane the contact solve is meaningless and
            # would fire a huge force. Only reachable if an anomaly switched the
            # membrane back on under the ball; freeze rather than explode.
            sim.frame += 1
            record()
            continue
        if anomaly is not None:
            anomaly.apply(sim, frame)
        sim.step()
        record()
    ep.events = list(sim.events)
    return ep, sim
