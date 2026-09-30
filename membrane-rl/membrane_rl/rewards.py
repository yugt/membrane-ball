"""Verifiable rewards for the trajectory-prediction task.

The whole point of this environment is that the reward is computed against
simulator ground truth, not against a learned judge. Nothing here calls a
model.

Design notes
------------
* The accuracy reward is SHAPED (continuous in the error), not 0/1. With GRPO
  you compare samples inside a group; a binary reward gives zero gradient
  whenever every sample in the group is wrong, which is exactly the regime you
  start in. A continuous reward still ranks "close" above "far".
* Errors are measured in ball radii, so the scale is physical and stays
  meaningful if you change the geometry.
* Format reward is kept small. Make it too large and the model learns to emit
  perfectly-formatted garbage -- a failure mode you will actually hit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

__all__ = ["RewardConfig", "parse_prediction", "accuracy_reward", "compute_reward"]

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class RewardConfig:
    tau_radii: float = 1.0        # error scale of the exponential, in ball radii
    format_weight: float = 0.1
    accuracy_weight: float = 1.0
    ball_radius: float = 0.5
    # Errors beyond this are all equally bad; prevents one wild outlier from
    # dominating the group-relative advantage.
    clip_radii: float = 6.0


def parse_prediction(text: str) -> np.ndarray | None:
    """Extract a 3-vector from ``<answer>x, y, z</answer>``.

    Returns None when the model did not produce a parseable answer, which the
    caller must treat as a format failure rather than as a huge error -- those
    are different signals and conflating them makes reward curves unreadable.
    """
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    body = m.group(1) if m else None
    if body is None:
        return None
    nums = _NUM_RE.findall(body)
    if len(nums) < 3:
        return None
    try:
        vals = [float(n) for n in nums[:3]]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return np.asarray(vals, dtype=float)


def accuracy_reward(pred: np.ndarray, target: np.ndarray, cfg: RewardConfig) -> tuple[float, float]:
    """Return ``(reward, error_in_radii)``."""
    err = float(np.linalg.norm(np.asarray(pred, float) - np.asarray(target, float)))
    err_radii = err / cfg.ball_radius
    clipped = min(err_radii, cfg.clip_radii)
    return math.exp(-clipped / cfg.tau_radii), err_radii


def compute_reward(
    completion: str,
    target,
    cfg: RewardConfig | None = None,
) -> dict[str, float]:
    """Score one completion against the ground-truth position.

    Returned dict is intentionally rich: log every component during training.
    When a run goes wrong you almost always need to know whether format or
    accuracy moved, and reconstructing that afterwards is impossible.
    """
    cfg = cfg or RewardConfig()
    target = np.asarray(target, dtype=float)

    pred = parse_prediction(completion)
    if pred is None:
        return {
            "reward": 0.0,
            "format": 0.0,
            "accuracy": 0.0,
            "error_radii": float("nan"),
            "parsed": 0.0,
        }

    acc, err_radii = accuracy_reward(pred, target, cfg)
    reward = cfg.format_weight * 1.0 + cfg.accuracy_weight * acc
    return {
        "reward": reward,
        "format": 1.0,
        "accuracy": acc,
        "error_radii": err_radii,
        "parsed": 1.0,
    }


def batch_reward(completions, targets, cfg: RewardConfig | None = None) -> list[dict]:
    return [compute_reward(c, t, cfg) for c, t in zip(completions, targets)]
