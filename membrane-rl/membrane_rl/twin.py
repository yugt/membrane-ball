"""Physics twin: the fast path of the video agent.

A copy of the simulator is kept in lock-step with what the camera sees. Every
frame it

1. **predicts** the next ball position by stepping the simulator (membrane,
   rim and wall included -- this is what a language model cannot do);
2. **compares** with the position ``perception`` measured -- the difference is
   the *innovation*;
3. **corrects** its state a little toward the measurement (an alpha-beta
   filter whose process model is the real physics).

Consequences:

* ``forecast(n)`` rolls the corrected state forward -- a 1 s prediction costs
  ~15 ms, so it can be refreshed several times a second;
* while the physics holds, the innovation is just measurement noise. When the
  physics breaks -- the ball speeds up by itself, falls through the membrane,
  hangs in mid-air -- the innovation grows a persistent bias within a few
  frames. That is the anomaly detector, and it needs no model and no labels;
* the twin's own simulator emits contact / rim / wall events, so the fast path
  also names what the ball hit.

The slow path (Cosmos over clip windows) then explains *what* went wrong in
words. See HACKATHON_RUNBOOK.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import MembraneSim, Params, State

__all__ = ["TwinConfig", "PhysicsTwin", "params_from_key", "run_twin"]


@dataclass(frozen=True)
class TwinConfig:
    alpha: float = 0.3            # position correction gain
    beta: float = 0.05            # velocity correction gain (small: trust the physics)
    ema: float = 0.25             # smoothing of the innovation for the detector
    threshold: float = 0.02       # detector threshold on the smoothed innovation (world units)
    min_run: int = 3              # frames above threshold before raising an anomaly
    refractory_s: float = 1.0     # one alarm per anomaly
    relock: float = 0.25          # innovation this large means we lost the ball: re-initialise
    # Rim hits are discontinuous: a sub-pixel position error decides whether a
    # bounce happens this frame or the next, which looks like a physics
    # mismatch. Near the rim the detector threshold is multiplied by this.
    rim_margin: float = 0.3       # extra distance (world units) that counts as "at the rim"
    rim_threshold_mult: float = 3.0
    # Deeper than a live membrane ever lets the ball go (-0.18 over 200 random
    # episodes), inside the rim: the model no longer describes the scene.
    impossible_depth: float = -0.3
    warmup: int = 4               # frames used to initialise velocity


@dataclass
class PhysicsTwin:
    params: Params
    frame_center: np.ndarray
    cfg: TwinConfig = field(default_factory=TwinConfig)

    def __post_init__(self):
        self.sim = MembraneSim(self.params)
        self.sim.frame_center = np.asarray(self.frame_center, float)
        self.dt = self.params.dt
        self.frame = -1
        self._buf: list[tuple[int, np.ndarray]] = []
        self.locked = False
        self.bias = np.zeros(3)
        self.innovation: list[float] = []      # |innovation| per frame (NaN if unobserved)
        self.score: list[float] = []           # |smoothed innovation| per frame
        self.alarms: list[dict] = []
        self._run = 0
        self._last_alarm_t = -1e9
        self.invalid = False                   # model known not to describe the scene

    # ------------------------------------------------------------------
    def _init_from_buffer(self) -> None:
        """Fit position and velocity to the first few observations, using the
        known gravity (free flight is by far the common case at start-up)."""
        (f0, _), (f1, p1) = self._buf[0], self._buf[-1]
        ts = np.array([(f - f1) * self.dt for f, _ in self._buf])
        P = np.array([p for _, p in self._buf])
        g = np.array([0.0, 0.0, -self.params.gravity])
        Y = P - 0.5 * np.outer(ts ** 2, g)
        A = np.column_stack([np.ones_like(ts), ts])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)      # rows: p(t1), v(t1)
        self.sim.state = State(pos=coef[0].copy(), vel=coef[1].copy())
        self.sim._was_contact = False
        self.locked = True
        self.bias[:] = 0.0

    def update(self, obs: np.ndarray | None) -> dict:
        """Feed one frame's observation (``None``/NaN if not visible)."""
        self.frame += 1
        t = self.frame * self.dt
        have = obs is not None and np.isfinite(obs).all()

        if have and self._impossible(obs):
            # Below a membrane that should have stopped it. One alarm, then stop
            # simulating: re-locking here would run the membrane solver far
            # outside its domain and report imaginary contacts.
            if not self.invalid and t - self._last_alarm_t > self.cfg.refractory_s:
                alarm = {"t": round(t, 3), "label": "anomaly", "why": "below the membrane"}
                self.alarms.append(alarm)
                self._last_alarm_t = t
            else:
                alarm = None
            self.invalid, self.locked = True, False
            self._buf.clear()
            self.innovation.append(np.nan)
            self.score.append(float(np.linalg.norm(self.bias)))
            return {"t": t, "locked": False, "alarm": alarm, "events": []}

        if not self.locked:
            if have:
                self._buf.append((self.frame, np.asarray(obs, float)))
                if len(self._buf) >= self.cfg.warmup:
                    self._init_from_buffer()
                    self._buf.clear()
            self.innovation.append(np.nan)
            self.score.append(0.0)
            return {"t": t, "locked": self.locked, "alarm": None}

        n_events = len(self.sim.events)
        self.sim.step()                                   # 1. predict
        pred = self.sim.state.pos
        alarm = None

        if have:
            r = np.asarray(obs, float) - pred             # 2. innovation
            rn = float(np.linalg.norm(r))
            self.innovation.append(rn)
            self.bias = (1 - self.cfg.ema) * self.bias + self.cfg.ema * r
            s = float(np.linalg.norm(self.bias))
            self.score.append(s)

            thr = self.cfg.threshold * (self.cfg.rim_threshold_mult if self._near_rim(obs) else 1.0)
            self._run = self._run + 1 if s > thr else 0
            if (self._run == self.cfg.min_run or rn > self.cfg.relock) and \
                    t - self._last_alarm_t > self.cfg.refractory_s:
                # time the alarm at the onset of the run, not when it was confirmed
                onset = t - (self._run - 1) * self.dt if rn <= self.cfg.relock else t
                alarm = {"t": round(onset, 3), "label": "anomaly",
                         "why": "jump" if rn > self.cfg.relock else "physics mismatch",
                         "innovation": round(rn, 4)}
                self.alarms.append(alarm)
                self._last_alarm_t = t

            if rn > self.cfg.relock:                      # lost it: re-initialise
                self.locked = False
                self._buf = [(self.frame, np.asarray(obs, float))]
                self._run = 0
            else:                                         # 3. correct
                self.sim.state.pos += self.cfg.alpha * r
                self.sim.state.vel += (self.cfg.beta / self.dt) * r
        else:
            # Coasting on the model alone: whatever the twin "sees" happen now is
            # its own imagination, not an observation -- don't report it.
            del self.sim.events[n_events:]
            self.innovation.append(np.nan)
            self.score.append(float(np.linalg.norm(self.bias)))

        new = self.sim.events[n_events:]
        return {"t": t, "locked": self.locked, "alarm": alarm, "events": new,
                "pos": self.sim.state.pos.copy()}

    def _impossible(self, pos) -> bool:
        dx, dy = pos[0] - self.sim.frame_center[0], pos[1] - self.sim.frame_center[1]
        return pos[2] < self.cfg.impossible_depth and np.hypot(dx, dy) <= self.params.r_frame

    def _near_rim(self, pos) -> bool:
        """Is the ball within reach of the rigid rim (a torus at z = 0)?"""
        dx, dy = pos[0] - self.sim.frame_center[0], pos[1] - self.sim.frame_center[1]
        d_rim = np.hypot(np.hypot(dx, dy) - self.params.r_frame, pos[2])
        return d_rim < self.params.ball_radius + self.cfg.rim_margin

    # ------------------------------------------------------------------
    def forecast(self, n_frames: int) -> np.ndarray:
        """Positions for the next ``n_frames`` frames from the current estimate."""
        sim = MembraneSim(self.params)
        sim.reset(self.sim.state.copy(), frame_center=self.sim.frame_center)
        return np.array([sim.step().pos for _ in range(n_frames)])

    def labelled_events(self) -> list[dict]:
        """Twin simulator events mapped to scoreable labels, plus alarms."""
        names = {"contact_start": "membrane_contact", "ring_bounce": "ring_bounce",
                 "wall_bounce": "wall_bounce"}
        out = [{"t": e["t"], "label": names[e["type"]]}
               for e in self.sim.events if e["type"] in names]
        return sorted(out + list(self.alarms), key=lambda e: e["t"])


def params_from_key(answer_key: dict) -> Params:
    """The physical constants are assumed known (it is the same simulator);
    everything about the *episode* -- frame centre, ball state -- comes from pixels."""
    return Params(**{k: (tuple(v) if isinstance(v, list) else v)
                     for k, v in answer_key["params"].items()})


def run_twin(frames, params: Params, size: int, cfg: TwinConfig | None = None, on_frame=None):
    """Perception + twin over a clip. ``on_frame(i, twin, obs, out)`` lets a
    caller (live view, forecast eval) hook every frame. Returns the twin."""
    from .perception import Camera, estimate_frame_center, observe
    from .render import RenderConfig

    cam = Camera(RenderConfig.for_size(size))
    frames = list(frames)
    twin = PhysicsTwin(params, estimate_frame_center(frames, cam), cfg or TwinConfig())
    for i, img in enumerate(frames):
        obs = observe(img, cam, params.ball_radius).pos
        out = twin.update(obs)
        if on_frame is not None:
            on_frame(i, twin, obs, out)
    return twin
