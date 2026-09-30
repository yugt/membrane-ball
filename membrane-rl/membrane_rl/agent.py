"""VLM event agent: clip -> timestamped events, via any OpenAI-compatible API.

The clip is cut into short overlapping windows; each window is sent as a
handful of timestamped frames (``mode="frames"``, works with any vision chat
model) or as a video clip (``mode="video"``, for endpoints that accept video
such as NVIDIA's hosted Cosmos Reason -- verify on the day). The model answers
with JSON events in window-relative seconds; they are shifted to clip time and
de-duplicated across window overlaps.

Everything except ``call_model`` is pure and unit-tested offline, so on the day
only the network part can surprise you.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = ["AgentConfig", "PROVIDERS", "resolve_provider", "SYSTEM_PROMPT", "windows", "build_messages", "parse_events", "merge_events"]

SYSTEM_PROMPT = """You watch short clips from a physics simulation and report events.

Scene: a yellow ball moves inside a transparent vertical cylinder (grey lines). \
A circular elastic membrane (blue/pink mesh, green rim) is stretched horizontally \
near the bottom. Gravity pulls the ball down. Normally the ball falls, presses into \
the membrane (the mesh dents and turns pink), and is thrown back up; it can also bounce \
off the green rim or the cylinder wall. The orange ellipse is the ball's shadow on the \
membrane plane and the dashed line shows its height.

Report every event you see, with its time in seconds from the start of THIS clip:
- "membrane_contact": the ball starts pressing into the membrane
- "ring_bounce": the ball bounces off the green rim
- "wall_bounce": the ball bounces off the cylinder wall
- "anomaly": something physically impossible starts, e.g. the ball speeds up by \
itself, accelerates upward in mid-air, stops in mid-air, jumps to another place, \
or passes through the membrane. Add a short "why".

Answer with JSON only, no prose:
{"events": [{"t": 0.42, "label": "membrane_contact"}, {"t": 1.10, "label": "anomaly", "why": "..."}]}
If nothing happens, answer {"events": []}."""


# Every provider below speaks the OpenAI chat-completions protocol, so switching
# is a flag, not a code change. Model ids are best guesses as of 2026-09-30 --
# the event will hand out its own endpoints (Cosmos on CoreWeave via VAST);
# confirm ids there and pass --model / --base-url.
PROVIDERS: dict[str, dict] = {
    # W&B Inference (CoreWeave). Multimodal Qwen models accept image input.
    "wandb": {"base_url": "https://api.inference.wandb.ai/v1", "key_env": "WANDB_API_KEY",
              "model": "Qwen/Qwen3.8-27B"},
    # NVIDIA hosted API. The old cosmos-reason1-7b endpoint was deprecated in
    # March 2026; use whatever Cosmos Reason id the organisers give you.
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY",
               "model": ""},
    # A vLLM / SGLang server you started yourself (e.g. Cosmos Reason on the event GPU).
    "local": {"base_url": "http://localhost:8000/v1", "key_env": "", "model": ""},
    # Anything else OpenAI-compatible (DashScope, OpenRouter, a proxy): set --base-url.
    "openai": {"base_url": "", "key_env": "OPENAI_API_KEY", "model": ""},
}


def resolve_provider(name: str, model: str = "", base_url: str = "", env=None) -> tuple[str, str, str]:
    """Return ``(base_url, model, api_key)``; explicit arguments win over presets."""
    import os
    env = os.environ if env is None else env
    preset = PROVIDERS[name]
    base = base_url or preset["base_url"]
    mdl = model or preset["model"]
    key = env.get(preset["key_env"], "") if preset["key_env"] else ""
    if not base:
        raise ValueError(f"provider {name!r} needs --base-url")
    if not mdl:
        raise ValueError(f"provider {name!r} has no default model; pass --model")
    return base, mdl, key


@dataclass(frozen=True)
class AgentConfig:
    model: str = PROVIDERS["wandb"]["model"]
    base_url: str = PROVIDERS["wandb"]["base_url"]
    api_key: str = ""
    project: str = ""             # W&B Inference: "<entity>/<project>" for usage tracking
    mode: str = "frames"          # "frames" | "video"
    window_s: float = 2.0
    hop_s: float = 1.5            # < window_s, so events at a boundary are seen twice
    sample_fps: float = 6.0       # frames per second sent in "frames" mode
    image_size: int = 448         # downscale before sending: tokens ~ pixels
    temperature: float = 0.2
    max_tokens: int = 512
    dedup_s: float = 0.25         # same-label events closer than this are one event


def windows(duration_s: float, window_s: float, hop_s: float) -> list[tuple[float, float]]:
    out, t = [], 0.0
    while t < duration_s - 1e-9:
        out.append((round(t, 4), round(min(t + window_s, duration_s), 4)))
        if t + window_s >= duration_s:
            break
        t += hop_s
    return out


def _jpeg_b64(frame: np.ndarray, size: int) -> str:
    img = Image.fromarray(frame)
    if img.size[0] != size:
        img = img.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(frames: list[np.ndarray], times: list[float], cfg: AgentConfig,
                   video_b64: str | None = None) -> list[dict]:
    """Chat messages for one window. ``times`` are window-relative seconds."""
    content: list[dict] = []
    if cfg.mode == "video":
        if video_b64 is None:
            raise ValueError("mode='video' needs video_b64")
        content.append({"type": "text", "text": f"Clip length: {times[-1]:.2f} s."})
        content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})
    else:
        content.append({"type": "text", "text": f"{len(frames)} frames in time order. "
                        "Each image is preceded by its time in seconds."})
        for f, t in zip(frames, times):
            content.append({"type": "text", "text": f"t = {t:.2f} s"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(f, cfg.image_size)}"}})
    content.append({"type": "text", "text": "List the events as JSON."})
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def parse_events(text: str, window_len: float) -> list[dict]:
    """Tolerant JSON extraction: code fences, prose around it, bare lists, <think> blocks."""
    if not text:
        return []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _JSON_RE.search(text)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = obj.get("events", []) if isinstance(obj, dict) else obj
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or "t" not in it:
            continue
        try:
            t = float(it["t"])
        except (TypeError, ValueError):
            continue
        if not (-0.1 <= t <= window_len + 0.1):
            continue                      # hallucinated timestamp outside the window
        ev = {"t": max(0.0, t), "label": str(it.get("label", it.get("type", "")))}
        if "why" in it:
            ev["why"] = str(it["why"])
        out.append(ev)
    return out


def merge_events(per_window: list[tuple[float, list[dict]]], dedup_s: float) -> list[dict]:
    """Shift window-relative events to clip time and drop overlap duplicates."""
    flat = sorted(
        ({**e, "t": round(start + e["t"], 3)} for start, evs in per_window for e in evs),
        key=lambda e: e["t"],
    )
    out: list[dict] = []
    for e in flat:
        if any(o["label"] == e["label"] and abs(o["t"] - e["t"]) < dedup_s for o in out[-6:]):
            continue
        out.append(e)
    return out


def call_model(messages: list[dict], cfg: AgentConfig) -> str:
    from openai import OpenAI

    kw = {"project": cfg.project} if cfg.project else {}
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY", **kw)
    resp = client.chat.completions.create(
        model=cfg.model, messages=messages,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    return resp.choices[0].message.content or ""
