"""Frame-by-frame 3D debug view of an episode, plus an audit of its motion.

A rendered clip is a 2D projection: motion toward the camera and falling look
the same, and a bounce off the far wall looks like a jump in mid-air. Before
trusting a clip (or blaming the physics) open its debug file: the same
episode in a rotatable 3D scene, with the ball's velocity and acceleration
drawn every frame, the logged events, and the cause of every change in motion.

The audit is the part that makes the scene trustworthy. Between two frames
the only things allowed to change the ball's motion are gravity, the
membrane, a wall or rim bounce, and an injected anomaly -- each of which the
simulator logs. Every frame is checked against free fall; a deviation with
no logged cause, or a logged bounce/contact that changes the energy by more
than any real one does, is reported as UNEXPLAINED. A healthy episode has none.

    from membrane_rl.debug3d import record_episode, debug_figure
    rec = record_episode(n_frames=375, seed=7, anomaly=None)
    print(rec.summary())
    debug_figure(rec).write_html("debug.html")

The audit needs only numpy; the figure needs ``plotly`` (``membrane-rl[debug]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .anomalies import Anomaly
from .episode import Episode, run_episode
from .physics import Params
from .render import RenderConfig

__all__ = ["FrameAudit", "Recording", "record_episode", "audit", "video_camera", "debug_figure"]

# Free fall under symplectic Euler is exact up to float rounding; anything
# larger than this is a real change in the motion.
_ACC_TOL = 1e-6      # world units / s^2
_POS_TOL = 1e-9      # world units
# Bounces and the membrane are conservative. Over 40 random episodes the
# largest one-frame energy change they cause is 0.35 % (integration error at
# a rim hit during contact); more than this in one frame is not physics.
_ENERGY_STEP_TOL_PCT = 1.0


@dataclass
class FrameAudit:
    frame: int
    t: float
    acc: np.ndarray            # (v_i - v_{i-1}) / dt, world units / s^2
    expected: np.ndarray       # gravity alone, as scaled during this frame
    acc_residual: float        # |acc - expected|
    pos_residual: float        # |x_i - free-fall prediction from frame i-1|
    energy_step_pct: float     # |E_i - E_{i-1}| as % of the initial energy
    causes: list[str]          # logged reasons the motion may deviate
    status: str                # "free fall" | "explained" | "anomaly" | "UNEXPLAINED"


@dataclass
class Recording:
    """An episode re-simulated with everything the debug view needs."""
    episode: Episode
    membrane: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    contact: list[bool] = field(default_factory=list)          # membrane pushed during step into frame i
    gravity_scale: list[float] = field(default_factory=list)   # in force during step into frame i
    membrane_scale: list[float] = field(default_factory=list)
    frozen: list[bool] = field(default_factory=list)           # frame i was held, not simulated
    audit: list[FrameAudit] = field(default_factory=list)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for a in self.audit:
            counts[a.status] = counts.get(a.status, 0) + 1
        bad = [a for a in self.audit if a.status == "UNEXPLAINED"]
        return {
            "n_frames": self.episode.n_frames,
            "status_counts": counts,
            "unexplained_frames": [a.frame for a in bad],
            "max_free_fall_acc_residual": max(
                (a.acc_residual for a in self.audit if a.status == "free fall"), default=0.0),
            "energy_drift_pct": _energy_drift_pct(self.episode),
        }


def _energy_drift_pct(ep: Episode) -> float:
    E = np.asarray(ep.energy)
    if ep.anomaly is not None:
        E = E[: ep.anomaly.start_frame]
    if E.size < 2 or E[0] == 0:
        return 0.0
    return float((E.max() - E.min()) / abs(E[0]) * 100)


def record_episode(
    n_frames: int,
    seed: int = 0,
    params: Params | None = None,
    anomaly: Anomaly | None = None,
    grid: int = 21,
) -> Recording:
    """Re-run ``run_episode`` exactly as ``video.render_clip`` does, capturing the
    membrane surface and per-step contact for every frame, then audit it."""
    membrane, contact, gscale, mscale, frozen = [], [], [], [], []
    last = {"contact_frames": 0, "pos": None}

    def hook(sim, frame):
        membrane.append(tuple(np.asarray(a, np.float32) for a in sim.membrane_surface(grid=grid, polar=True)))
        contact.append(sim.n_contact_frames > last["contact_frames"])
        gscale.append(float(sim.gravity_scale))
        mscale.append(float(sim.membrane_scale))
        # run_episode holds the ball, rather than stepping, far below a live membrane
        prev = last["pos"]
        frozen.append(prev is not None and prev[2] < -3.0 and sim.membrane_scale > 0.0)
        last["contact_frames"] = sim.n_contact_frames
        last["pos"] = sim.state.pos.copy()

    ep, _ = run_episode(n_frames, seed=seed, params=params, anomaly=anomaly, sim_hook=hook)
    rec = Recording(episode=ep, membrane=membrane, contact=contact, gravity_scale=gscale,
                    membrane_scale=mscale, frozen=frozen)
    rec.audit = audit(rec)
    return rec


def audit(rec: Recording) -> list[FrameAudit]:
    """Explain every change in the ball's motion, frame by frame."""
    ep = rec.episode
    p = ep.params
    dt, n, h = p.dt, p.substeps, p.sub_dt()
    by_frame: dict[int, list[str]] = {}
    for e in ep.events:
        if e["type"] in ("wall_bounce", "ring_bounce"):
            by_frame.setdefault(e["frame"], []).append(e["type"].replace("_", " "))
    anomaly_frames: dict[int, str] = {}
    if ep.anomaly is not None:
        a = ep.anomaly
        anomaly_frames[a.start_frame] = f"anomaly: {a.kind} starts"
        if a.end_frame is not None:
            anomaly_frames[a.end_frame] = f"anomaly: {a.kind} ends"

    E = np.asarray(ep.energy)
    e_ref = abs(E[0]) or 1.0
    out: list[FrameAudit] = []
    for i in range(1, ep.n_frames):
        s0, s1 = ep.states[i - 1], ep.states[i]
        gs = rec.gravity_scale[i]
        g = np.array([0.0, 0.0, -p.gravity * gs])
        acc = (s1.vel - s0.vel) / dt
        # symplectic Euler, n substeps of free fall from s0
        x_pred = s0.pos + n * h * s0.vel + h * h * g * n * (n + 1) / 2
        acc_res = float(np.linalg.norm(acc - g))
        pos_res = float(np.linalg.norm(s1.pos - x_pred))
        de = float(abs(E[i] - E[i - 1]) / e_ref * 100)

        causes = list(by_frame.get(i, []))
        if rec.contact[i]:
            causes.append("membrane")
        if gs != 1.0:
            causes.append(f"anomaly: gravity x{gs:g}")
        if i in anomaly_frames:
            causes.append(anomaly_frames[i])
        if rec.membrane_scale[i] == 0.0 and s1.pos[2] < p.ball_radius and \
                np.hypot(*(s1.pos[:2] - np.asarray(ep.frame_center))) < p.r_frame:
            causes.append("anomaly: membrane off, ball passing through")
        if rec.frozen[i]:
            causes.append("anomaly: held far below the membrane")
        if any(c.startswith("anomaly") for c in causes):
            status = "anomaly"
        elif acc_res <= _ACC_TOL and pos_res <= _POS_TOL:
            status = "free fall"
        elif causes and de <= _ENERGY_STEP_TOL_PCT:
            status = "explained"
        else:
            status = "UNEXPLAINED"
        out.append(FrameAudit(i, i * dt, acc, g, acc_res, pos_res, de, causes, status))
    return out


# ----------------------------------------------------------------------------
# 3D figure
# ----------------------------------------------------------------------------

def video_camera(cfg: RenderConfig | None = None, distance: float = 2.4) -> dict:
    """Plotly ``scene.camera`` that reproduces the clip's view exactly.

    The renderer looks along -(0, sin t, cos t) in its rotated frame, which in
    world coordinates is the direction below; with an orthographic projection
    and equal axis scaling the 3D view and the mp4 frame coincide.
    """
    cfg = cfg or RenderConfig()
    r, t = np.deg2rad(cfg.rotation_deg), np.deg2rad(cfg.tilt_deg)
    d = np.array([np.sin(t) * np.sin(r), np.sin(t) * np.cos(r), np.cos(t)]) * distance
    return {"eye": {"x": float(d[0]), "y": float(d[1]), "z": float(d[2])},
            "up": {"x": 0, "y": 0, "z": 1},
            "projection": {"type": "orthographic"}}


def _sphere(c, r, n=18):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n))
    return tuple(np.asarray(a, np.float32) for a in (
        c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v), c[2] + r * np.cos(v)))


_STATUS_COLOUR = {"free fall": "#9aa0a6", "explained": "#2e7d32", "anomaly": "#c62828",
                  "UNEXPLAINED": "#ff00ff"}
_EVENT_STYLE = {"contact_start": ("#1e88e5", "circle"), "ring_bounce": ("#43a047", "diamond"),
                "wall_bounce": ("#8e24aa", "square"), "anomaly_start": ("#e53935", "x"),
                "anomaly_end": ("#e53935", "cross")}


def debug_figure(rec: Recording, stride: int = 1, trail: int = 40,
                 vel_scale: float = 0.15, acc_scale: float = 0.02, title: str = ""):
    """Animated figure: 3D scene (left) and per-frame traces (right)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ep = rec.episode
    p = ep.params
    dt = p.dt
    S = np.array([s.pos for s in ep.states])
    V = np.array([s.vel for s in ep.states])
    T = np.arange(ep.n_frames) * dt
    A = np.vstack([np.full(3, np.nan), [a.acc for a in rec.audit]])
    status = ["start"] + [a.status for a in rec.audit]
    causes = [[]] + [a.causes for a in rec.audit]
    fc = ep.frame_center
    z_top = max(4.8, float(np.nanmax(S[:, 2])) + p.ball_radius + 0.2)
    z_bot = min(-1.0, float(np.nanmin(S[:, 2])) - p.ball_radius - 0.2)

    fig = make_subplots(
        rows=3, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.06, vertical_spacing=0.08,
        specs=[[{"type": "scene", "rowspan": 3}, {"type": "xy"}], [None, {"type": "xy"}],
               [None, {"type": "xy"}]],
        subplot_titles=("", "acceleration (world units/s²)", "speed and height", "total energy"))

    # ---- dynamic traces first (indices 0..7), so frames can address them ----
    def dyn(i, first=False):
        bx, by, bz = _sphere(S[i], p.ball_radius)
        mx, my, mu = rec.membrane[i]
        if not first:
            mx = my = None      # the frame is fixed: frames only update the heights
        tail = S[max(0, i - trail): i + 1]
        v_end = S[i] + vel_scale * V[i]
        a = A[i] if np.isfinite(A[i]).all() else np.zeros(3)
        a_end = S[i] + acc_scale * a
        return [
            go.Surface(x=bx, y=by, z=bz, colorscale=[[0, "#f9d71c"], [1, "#f9d71c"]],
                       showscale=False, opacity=0.9, name="ball", hoverinfo="skip"),
            go.Surface(x=mx, y=my, z=mu, colorscale=[[0, "#1565c0"], [1, "#90caf9"]],
                       cmin=-0.6, cmax=0.0, showscale=False, opacity=0.75, name="membrane",
                       hoverinfo="skip"),
            go.Scatter3d(x=tail[:, 0], y=tail[:, 1], z=tail[:, 2], mode="lines",
                         line=dict(color="#ff9800", width=4), name="trail (last %d frames)" % trail),
            go.Scatter3d(x=[S[i, 0], v_end[0]], y=[S[i, 1], v_end[1]], z=[S[i, 2], v_end[2]],
                         mode="lines+markers", line=dict(color="#00897b", width=7),
                         marker=dict(size=[0, 4]), name=f"velocity ×{vel_scale:g}"),
            go.Scatter3d(x=[S[i, 0], a_end[0]], y=[S[i, 1], a_end[1]], z=[S[i, 2], a_end[2]],
                         mode="lines+markers", line=dict(color="#d81b60", width=7),
                         marker=dict(size=[0, 4]), name=f"acceleration ×{acc_scale:g}"),
            go.Scatter3d(x=[S[i, 0]], y=[S[i, 1]], z=[0.0], mode="markers",
                         marker=dict(size=5, color="#ff9800", symbol="circle-open"),
                         name="ground point (shadow)"),
        ]

    def cursors(i):
        t = T[i]
        return [go.Scatter(x=[t, t], y=[-1e3, 1e3], mode="lines", line=dict(color="black", width=1),
                           showlegend=False, hoverinfo="skip", xaxis=f"x{k}", yaxis=f"y{k}")
                for k in ("", "2", "3")]

    for tr in dyn(0, first=True):
        fig.add_trace(tr, row=1, col=1)
    for k, tr in enumerate(cursors(0)):
        fig.add_trace(tr, row=k + 1, col=2)
    n_dyn = len(fig.data)

    # ---- static 3D context ----
    th = np.linspace(0, 2 * np.pi, 49)
    zz = np.linspace(z_bot, z_top, 2)
    TH, ZZ = np.meshgrid(th, zz)
    fig.add_trace(go.Surface(x=p.r_cyl * np.cos(TH), y=p.r_cyl * np.sin(TH), z=ZZ,
                             colorscale=[[0, "#b0bec5"], [1, "#b0bec5"]], showscale=False,
                             opacity=0.15, name="cylinder wall", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=fc[0] + p.r_frame * np.cos(th), y=fc[1] + p.r_frame * np.sin(th),
                               z=np.zeros_like(th), mode="lines", line=dict(color="#2ecc71", width=6),
                               name="rim (membrane frame)"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="lines",
                               line=dict(color="rgba(120,120,120,0.35)", width=2),
                               name="full path", visible="legendonly"), row=1, col=1)
    for et, (col, sym) in _EVENT_STYLE.items():
        es = [e for e in ep.events if e["type"] == et]
        if not es:
            continue
        P = np.array([e["pos"] for e in es])
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", marker=dict(size=4, color=col, symbol=sym),
            name=et.replace("_", " "), text=[f"{et} t={e['t']:.3f}s frame {e['frame']}" for e in es],
            hoverinfo="text"), row=1, col=1)

    # ---- 2D panels ----
    fig.add_trace(go.Scatter(x=T, y=A[:, 2], name="a_z", line=dict(color="#d81b60")), row=1, col=2)
    fig.add_trace(go.Scatter(x=T, y=np.hypot(A[:, 0], A[:, 1]), name="|a_xy|",
                             line=dict(color="#6d4c41")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[T[0], T[-1]], y=[-p.gravity] * 2, name="-g",
                             line=dict(color="black", dash="dash")), row=1, col=2)
    bad = [i for i, s in enumerate(status) if s in ("anomaly", "UNEXPLAINED")]
    if bad:
        fig.add_trace(go.Scatter(x=T[bad], y=np.nan_to_num(A[bad, 2]), mode="markers",
                                 marker=dict(color=[_STATUS_COLOUR[status[i]] for i in bad], size=7,
                                             symbol="x"),
                                 name="anomaly / UNEXPLAINED frames"), row=1, col=2)
    fig.add_trace(go.Scatter(x=T, y=np.linalg.norm(V, axis=1), name="|v|",
                             line=dict(color="#00897b")), row=2, col=2)
    fig.add_trace(go.Scatter(x=T, y=S[:, 2], name="z", line=dict(color="#ff9800")), row=2, col=2)
    fig.add_trace(go.Scatter(x=T, y=ep.energy, name="energy", line=dict(color="#3949ab")), row=3, col=2)
    # contact and bounce spikes reach hundreds; clip them so free fall at -g is legible
    fig.update_yaxes(range=[-3 * p.gravity, 4 * p.gravity], row=1, col=2)
    nf = [i for i, st in enumerate(status) if st not in ("free fall", "start")]
    if nf:
        fig.add_trace(go.Scatter(
            x=T[nf], y=[3.4 * p.gravity] * len(nf), mode="markers",
            marker=dict(color=[_STATUS_COLOUR[status[i]] for i in nf], size=5, symbol="line-ns-open"),
            text=[f"frame {i}: {status[i]} ({', '.join(causes[i])})" for i in nf], hoverinfo="text",
            name="frame verdict (grey = free fall, not drawn)"), row=1, col=2)
    fig.update_yaxes(range=[min(0.0, float(S[:, 2].min())) - 0.3,
                            max(float(np.linalg.norm(V, axis=1).max()), float(S[:, 2].max())) + 0.5],
                     row=2, col=2)
    e_arr = np.asarray(ep.energy)
    pad = 0.05 * (e_arr.max() - e_arr.min() + 1e-6) + 0.1
    fig.update_yaxes(range=[e_arr.min() - pad, e_arr.max() + pad], row=3, col=2)
    for k in (1, 2, 3):
        fig.update_xaxes(range=[0, T[-1]], row=k, col=2)
    fig.update_xaxes(title_text="t (s)", row=3, col=2)

    # ---- frames ----
    def label(i):
        a = A[i]
        a_txt = "—" if not np.isfinite(a).all() else f"({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f})"
        c = ", ".join(causes[i]) or "none"
        col = _STATUS_COLOUR.get(status[i], "#555")
        return (f"{title}<br>t = {T[i]:.3f} s  ·  frame {i}  ·  "
                f"<span style='color:{col}'><b>{status[i]}</b></span>  ·  causes: {c}<br>"
                f"pos ({S[i,0]:+.3f}, {S[i,1]:+.3f}, {S[i,2]:+.3f})  ·  "
                f"v ({V[i,0]:+.2f}, {V[i,1]:+.2f}, {V[i,2]:+.2f})  ·  a {a_txt}  ·  "
                f"E = {ep.energy[i]:.3f}")

    idx = list(range(0, ep.n_frames, stride))
    fig.frames = [go.Frame(name=f"{i}", data=dyn(i) + cursors(i), traces=list(range(n_dyn)),
                           layout=go.Layout(title_text=label(i))) for i in idx]
    step_args = lambda name: [[name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",
                                       "transition": {"duration": 0}}]
    fig.update_layout(
        title=dict(text=label(0), x=0.01, y=0.985, yanchor="top", font=dict(size=13)),
        height=860, margin=dict(t=110), template="plotly_white", legend=dict(x=0.0, y=0.0, bgcolor="rgba(255,255,255,0.6)"),
        scene=dict(
            xaxis=dict(range=[-1.6, 1.6], title="x"), yaxis=dict(range=[-1.6, 1.6], title="y"),
            zaxis=dict(range=[z_bot, z_top], title="z"),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=(z_top - z_bot) / 3.2),
            camera=video_camera()),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left",
                          yanchor="top",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, {"frame": {"duration": int(dt * 1000 * stride),
                                                          "redraw": True}, "fromcurrent": True,
                                                "transition": {"duration": 0}}]),
                              dict(label="⏸ Pause", method="animate", args=step_args(None)),
                              dict(label="video camera", method="relayout",
                                   args=["scene.camera", video_camera()])])],
        sliders=[dict(currentvalue=dict(prefix="t = "), len=0.75, x=0.22, y=-0.02,
                      steps=[dict(label=f"{T[i]:.2f}s", method="animate", args=step_args(f"{i}"))
                             for i in idx])],
    )
    return fig
