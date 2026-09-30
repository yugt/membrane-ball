"""Single-source physics core for the membrane-ball system.

This is the ONLY implementation of the dynamics in this project. Training
labels, rendering and evaluation all read from here, so ground truth cannot
drift between components.

Ported from ``verify_energy.py`` in yugt/membrane-ball -- the energy-conserving
reference loop. The other two copies in that repo (``game_web/game.js`` and
``game_python/physics.py``) are display-only and must NEVER be used to generate
labels: they carry game-feel restitution (e=0.95) and a coarser membrane
evaluation, so their trajectories differ from the reference.

Model summary
-------------
A ball of radius R interacts with a clamped circular membrane of unit radius.
Inside the contact patch the membrane conforms to the sphere; outside it solves
Laplace's equation and becomes a logarithmic funnel.  The contact radius r_c is
found by Newton's method on

    sqrt(R^2 - r_c^2) + r_c^2 ln(r_c) / sqrt(R^2 - r_c^2) = z_b

Off-centre impacts are handled by the coordinate scaling factor 1/(1 - r_b^2),
which stiffens the membrane as the ball approaches the clamped boundary.

Why the frame does not move
---------------------------
The membrane frame is fixed for the whole of an episode. A frame that moves is
a time-dependent constraint: it does work on the ball, so mechanical energy is
no longer conserved and the energy check stops being a meaningful correctness
test on the labels. The frame CENTRE is still sampled per episode -- that keeps
each trajectory an autonomous, energy-conserving system while giving the
dataset a second geometric axis of variation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Params", "State", "MembraneSim"]


@dataclass(frozen=True)
class Params:
    """Physical and integration parameters.

    Defaults match the settings under which the upstream repo verifies energy
    conservation. Changing them invalidates that test, so the dataset generator
    records the params it used alongside every sample.
    """

    tension: float = 30.0
    gravity: float = 6.2
    ball_mass: float = 1.0
    ball_radius: float = 0.5

    r_cyl: float = 1.45          # rigid outer cylinder, always centred on the origin
    r_frame: float = 1.0         # clamped membrane boundary / paddle ring

    dt: float = 0.016            # one rendered frame
    substeps: int = 20           # physics substeps per frame

    # Per-episode frame placement. The frame is FIXED within an episode; these
    # only control where it may be placed at reset time.
    # max = 0.0 reproduces the upstream single-scene configuration exactly.
    #
    # The min/max pair exists so train and test can occupy DISJOINT annuli of
    # frame placement. A held-out split that differs only by RNG seed measures
    # nothing -- the samples come from the same distribution. Holding out an
    # unseen band of frame offsets is a real generalisation probe, and it is
    # the number worth putting in a writeup.
    frame_offset_min: float = 0.0
    frame_offset_max: float = 0.0

    # Sampling bounds for random initial conditions. Horizontal position is
    # sampled relative to the frame centre, so the ball starts over the
    # membrane wherever the frame happens to be.
    z0_range: tuple[float, float] = (2.5, 4.5)
    xy0_radius: float = 0.55
    v0_xy: float = 1.8
    v0_z: float = 1.0

    def sub_dt(self) -> float:
        return self.dt / self.substeps

    def validate(self) -> None:
        """The frame must stay inside the cylinder or the geometry is nonsense."""
        if self.frame_offset_max + self.r_frame > self.r_cyl:
            raise ValueError(
                f"frame_offset_max ({self.frame_offset_max}) + r_frame ({self.r_frame}) "
                f"exceeds r_cyl ({self.r_cyl}): the membrane ring would clip through "
                f"the cylinder wall. Enlarge r_cyl or reduce the offset."
            )


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray

    def copy(self) -> "State":
        return State(pos=self.pos.copy(), vel=self.vel.copy())


class MembraneSim:
    """Deterministic, headless simulator.

    Given the same ``seed`` the same trajectory is produced on any machine.
    That property is what makes labels reproducible, so any training sample can
    be traced back to the exact trajectory that generated it.
    """

    def __init__(self, params: Params | None = None, seed: int | None = None):
        self.p = params or Params()
        self.p.validate()
        self.rng = np.random.default_rng(seed)
        self.state = State(pos=np.zeros(3), vel=np.zeros(3))
        self.frame_center = np.zeros(2)      # fixed for the whole episode
        self.n_cyl_bounces = 0
        self.n_ring_bounces = 0
        self.n_contact_frames = 0

        # Runtime multipliers used ONLY to inject physically impossible
        # segments (see anomalies.py). At 1.0 they are exact no-ops in IEEE
        # arithmetic, so the upstream parity test still pins the dynamics.
        self.gravity_scale = 1.0
        self.membrane_scale = 1.0

        # Timestamped ground-truth event log. ``frame`` is the index of the
        # first rendered frame that shows the event (frame 0 = state at reset).
        self.frame = 0
        self.events: list[dict] = []
        self._was_contact = False

    # ------------------------------------------------------------------
    # membrane geometry
    # ------------------------------------------------------------------
    def solve_contact_radius(self, z_b: float) -> float:
        """Newton solve for the contact radius. Returns 0.0 when not in contact."""
        R = self.p.ball_radius
        if z_b >= R:
            return 0.0
        r_c = R * 0.5
        for _ in range(12):
            r_c = float(np.clip(r_c, 1e-7, R * 0.9999))
            S = np.sqrt(R**2 - r_c**2)
            f_val = S + (r_c**2 * np.log(r_c)) / S
            f_prime = (r_c * np.log(r_c) / S) * (2.0 + (r_c**2) / (S**2))
            diff = f_val - z_b
            if abs(diff) < 1e-10:
                break
            r_c = r_c - diff / f_prime
        return float(np.clip(r_c, 0.0, R * 0.9999))

    def elastic_energy_centered(self, z_b: float) -> float:
        return self._elastic_from_rc(self.solve_contact_radius(z_b))

    def _elastic_from_rc(self, r_c: float) -> float:
        """Elastic energy given an already-solved contact radius.

        The Newton solve dominates the inner loop, so callers that already have
        r_c must not pay for it twice.
        """
        if r_c <= 0.0:
            return 0.0
        R = self.p.ball_radius
        S = np.sqrt(max(1e-15, R**2 - r_c**2))
        term1 = -0.5 * r_c**2
        term2 = -(R**2) * np.log(S / R)
        term3 = -(r_c**4 * np.log(r_c)) / (S**2)
        return float(max(0.0, np.pi * self.p.tension * (term1 + term2 + term3)))

    def _rel_xy(self, pos) -> tuple[float, float]:
        """Ball position relative to the (fixed) frame centre."""
        return float(pos[0] - self.frame_center[0]), float(pos[1] - self.frame_center[1])

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------
    def energy(self, state: State | None = None) -> dict[str, float]:
        s = state or self.state
        p = self.p
        ke = 0.5 * p.ball_mass * float(np.sum(s.vel**2))
        pe_grav = p.ball_mass * p.gravity * float(s.pos[2])

        dx, dy = self._rel_xy(s.pos)
        r_b2 = dx * dx + dy * dy
        if np.sqrt(r_b2) <= 1.0 and self.membrane_scale > 0.0:
            pe_elastic = self.elastic_energy_centered(float(s.pos[2])) / (1.0 - r_b2)
        else:
            pe_elastic = 0.0

        return {
            "ke": ke,
            "pe_grav": pe_grav,
            "pe_elastic": pe_elastic,
            "total": ke + pe_grav + pe_elastic,
        }

    # ------------------------------------------------------------------
    # initial conditions
    # ------------------------------------------------------------------
    def reset(
        self,
        state: State | None = None,
        frame_center: np.ndarray | tuple[float, float] | None = None,
    ) -> State:
        p = self.p

        if frame_center is not None:
            self.frame_center = np.asarray(frame_center, dtype=float)[:2].copy()
        elif p.frame_offset_max > 0.0:
            theta = self.rng.uniform(0.0, 2 * np.pi)
            # area-uniform within the annulus [min, max]
            lo2, hi2 = p.frame_offset_min**2, p.frame_offset_max**2
            r = np.sqrt(self.rng.uniform(lo2, hi2))
            self.frame_center = np.array([r * np.cos(theta), r * np.sin(theta)])
        else:
            self.frame_center = np.zeros(2)

        if state is not None:
            self.state = state.copy()
        else:
            theta = self.rng.uniform(0.0, 2 * np.pi)
            r = p.xy0_radius * np.sqrt(self.rng.uniform(0.0, 1.0))
            pos = np.array(
                [
                    self.frame_center[0] + r * np.cos(theta),
                    self.frame_center[1] + r * np.sin(theta),
                    self.rng.uniform(*p.z0_range),
                ]
            )
            vel = np.array(
                [
                    self.rng.uniform(-p.v0_xy, p.v0_xy),
                    self.rng.uniform(-p.v0_xy, p.v0_xy),
                    self.rng.uniform(-p.v0_z, 0.0),
                ]
            )
            self.state = State(pos=pos, vel=vel)

        self.n_cyl_bounces = 0
        self.n_ring_bounces = 0
        self.n_contact_frames = 0
        self.gravity_scale = 1.0
        self.membrane_scale = 1.0
        self.frame = 0
        self.events = []
        self._was_contact = False
        return self.state.copy()

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def step(self) -> State:
        """Advance one rendered frame (``substeps`` symplectic Euler substeps)."""
        p = self.p
        pos = self.state.pos
        vel = self.state.vel
        sub_dt = p.sub_dt()
        in_contact = False
        self.frame += 1

        for k in range(p.substeps):
            self._t = (self.frame - 1 + (k + 1) / p.substeps) * p.dt
            dx, dy = self._rel_xy(pos)
            r_b2 = dx * dx + dy * dy
            r_b = np.sqrt(r_b2)
            r_c = self.solve_contact_radius(float(pos[2]))
            ue_centered = self._elastic_from_rc(r_c)

            fz_centered = 0.0
            contact_now = r_b <= 1.0 and r_c > 0.0 and self.membrane_scale > 0.0
            if contact_now != self._was_contact:
                self._emit("contact_start" if contact_now else "contact_end", pos)
                self._was_contact = contact_now
            if r_b <= 1.0 and r_c > 0.0:
                in_contact = in_contact or contact_now
                S = np.sqrt(max(1e-15, p.ball_radius**2 - r_c**2))
                fz_centered = 2.0 * np.pi * p.tension * (r_c**2) / S

            factor = 1.0 / (1.0 - r_b2) if r_b < 1.0 else 1.0
            force_z = fz_centered * factor * self.membrane_scale

            # Lateral restoring force points toward the frame centre; it is the
            # gradient of the off-centre stiffening term, so it must be taken
            # in frame-relative coordinates.
            force_x = force_y = 0.0
            if r_b < 1.0 and ue_centered > 0.0:
                denom = (1.0 - r_b2) ** 2
                force_x = -(2.0 * dx / denom) * ue_centered * self.membrane_scale
                force_y = -(2.0 * dy / denom) * ue_centered * self.membrane_scale

            # symplectic Euler: velocity first, then position with the new velocity
            vel[0] += (force_x / p.ball_mass) * sub_dt
            vel[1] += (force_y / p.ball_mass) * sub_dt
            vz_before = vel[2]
            vel[2] += (-p.gravity * self.gravity_scale + force_z / p.ball_mass) * sub_dt
            pos += vel * sub_dt
            if vz_before > 0.0 >= vel[2] and not contact_now:
                self._emit("apex", pos)

            self._bounce_cylinder(pos, vel)
            self._bounce_ring(pos, vel)

        if in_contact:
            self.n_contact_frames += 1
        return self.state.copy()

    def _bounce_cylinder(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Outer arena wall -- always centred on the origin, independent of the frame."""
        p = self.p
        ball_r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        limit = p.r_cyl - p.ball_radius
        if ball_r >= limit and ball_r > 1e-9:
            nx, ny = pos[0] / ball_r, pos[1] / ball_r
            v_dot_n = vel[0] * nx + vel[1] * ny
            if v_dot_n > 0:
                vel[0] -= 2.0 * v_dot_n * nx
                vel[1] -= 2.0 * v_dot_n * ny
                self.n_cyl_bounces += 1
                self._emit("wall_bounce", pos)
                pos[0] = limit * nx
                pos[1] = limit * ny

    def _bounce_ring(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Rigid rim of the membrane frame, centred on ``frame_center`` at z = 0."""
        p = self.p
        fx, fy = float(self.frame_center[0]), float(self.frame_center[1])
        dx, dy = self._rel_xy(pos)
        ball_r = np.sqrt(dx * dx + dy * dy)
        if ball_r > 1e-6:
            cx = fx + p.r_frame * (dx / ball_r)
            cy = fy + p.r_frame * (dy / ball_r)
        else:
            cx, cy = fx + p.r_frame, fy

        ex, ey, ez = pos[0] - cx, pos[1] - cy, pos[2]
        d_ring = np.sqrt(ex**2 + ey**2 + ez**2)
        if d_ring <= p.ball_radius and d_ring > 1e-9:
            nx, ny, nz = ex / d_ring, ey / d_ring, ez / d_ring
            v_dot = vel[0] * nx + vel[1] * ny + vel[2] * nz
            if v_dot < 0:
                vel[0] -= 2.0 * v_dot * nx
                vel[1] -= 2.0 * v_dot * ny
                vel[2] -= 2.0 * v_dot * nz
                self.n_ring_bounces += 1
                self._emit("ring_bounce", pos)
            pos[0] = cx + p.ball_radius * nx
            pos[1] = cy + p.ball_radius * ny
            pos[2] = p.ball_radius * nz

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _emit(self, etype: str, pos, at_frame: int | None = None, **extra) -> None:
        """Append a ground-truth event at the current substep time.

        ``at_frame`` is for events injected between steps (anomalies), which
        take effect exactly at a frame boundary.
        """
        if at_frame is None:
            frame, t = self.frame, getattr(self, "_t", self.frame * self.p.dt)
        else:
            frame, t = at_frame, at_frame * self.p.dt
        self.events.append({
            "frame": frame,
            "t": round(float(t), 5),
            "type": etype,
            "pos": [round(float(v), 4) for v in pos],
            **extra,
        })

    def rollout(self, n_frames: int) -> list[State]:
        """Advance ``n_frames`` and return the state after each one."""
        return [self.step() for _ in range(n_frames)]

    def membrane_surface(self, grid: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Membrane height field on a ``grid x grid`` patch around the frame centre.

        Returns absolute x, y (world coordinates) and the height u. Points
        outside the clamped unit circle are NaN so the renderer can mask them.
        Uses the exact Moebius conformal mapping, which is what makes off-centre
        contact look right rather than merely plausible.
        """
        p = self.p
        fx, fy = float(self.frame_center[0]), float(self.frame_center[1])
        bx, by = self._rel_xy(self.state.pos)
        z_b = float(self.state.pos[2])
        r_b2 = bx * bx + by * by
        r_b = np.sqrt(r_b2)

        lin = np.linspace(-1.0, 1.0, grid)
        nx, ny = np.meshgrid(lin, lin, indexing="ij")   # frame-relative
        u = np.zeros_like(nx)
        outside = (nx**2 + ny**2) >= 1.0

        wx, wy = nx + fx, ny + fy                        # world coordinates

        r_c = self.solve_contact_radius(z_b) if (r_b <= 1.0 and self.membrane_scale > 0.0) else 0.0
        if r_c <= 0.0:
            u[outside] = np.nan
            return wx, wy, u

        S = np.sqrt(max(1e-15, p.ball_radius**2 - r_c**2))
        A = (r_c**2) / S

        num_x = nx - bx
        num_y = ny - by
        den_x = 1.0 - (nx * bx + ny * by)
        den_y = nx * by - ny * bx
        den2 = den_x**2 + den_y**2
        den2 = np.where(den2 < 1e-15, 1e-15, den2)
        d = np.sqrt((num_x**2 + num_y**2) / den2)

        r_ball_plane = np.sqrt(num_x**2 + num_y**2)
        inside_patch = d <= r_c

        u_sphere = z_b - np.sqrt(np.maximum(0.0, p.ball_radius**2 - r_ball_plane**2))

        with np.errstate(divide="ignore", invalid="ignore"):
            u_log = A * np.log(np.where(d > 1e-12, d, 1e-12))

        dist_ball = np.where(r_ball_plane > 1e-6, r_ball_plane, 1.0)
        ux, uy = num_x / dist_ball, num_y / dist_ball
        proj_a = bx * ux + by * uy
        r_boundary = r_c * (1.0 - r_b2) / (1.0 + r_c * proj_a)
        u_sphere_b = z_b - np.sqrt(np.maximum(0.0, p.ball_radius**2 - r_boundary**2))
        local_jump = u_sphere_b - A * np.log(r_c)

        s = np.clip((1.0 - d) / max(1e-9, (1.0 - r_c)), 0.0, 1.0)
        u_funnel = u_log + s * local_jump

        u = np.where(inside_patch, u_sphere, u_funnel)
        u[outside] = np.nan
        return wx, wy, u
