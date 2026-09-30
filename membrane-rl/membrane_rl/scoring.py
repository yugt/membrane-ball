"""Score a video agent's event list against the simulator's answer key.

Agents are asked for coarse, human-level events -- the kind a person watching
the clip could name -- so the fine-grained simulator log is collapsed first:

    contact_start  -> membrane_contact
    ring_bounce    -> ring_bounce
    wall_bounce    -> wall_bounce
    anomaly_start  -> anomaly          (carries ``kind``)

``contact_end``, ``apex`` and ``anomaly_end`` are not scored by default: they
are real, but asking an agent for them mostly measures prompt wording.

Matching is one-to-one within ``tol_s`` seconds, same label, greedy by time
difference. Report precision/recall/F1 per label plus anomaly latency; a live
scoreboard can call ``score_clip`` after every agent update.
"""

from __future__ import annotations

from collections import defaultdict

__all__ = ["LABELS", "truth_events", "score_clip", "aggregate"]

LABELS = ("membrane_contact", "ring_bounce", "wall_bounce", "anomaly")

_COLLAPSE = {
    "contact_start": "membrane_contact",
    "ring_bounce": "ring_bounce",
    "wall_bounce": "wall_bounce",
    "anomaly_start": "anomaly",
}

# Accept the obvious synonyms an LLM will produce instead of our exact labels.
_ALIASES = {
    "contact": "membrane_contact", "membrane": "membrane_contact",
    "membrane_hit": "membrane_contact", "bounce_membrane": "membrane_contact",
    "frame_bounce": "ring_bounce", "rim_bounce": "ring_bounce", "ring": "ring_bounce",
    "wall": "wall_bounce", "cylinder_bounce": "wall_bounce",
    "bounce": "interaction", "collision": "interaction", "interaction": "interaction",
    "anomalous": "anomaly", "impossible": "anomaly", "physics_violation": "anomaly",
}


def normalise_label(label: str) -> str | None:
    key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    if key in LABELS or key == "interaction":
        return key
    return _ALIASES.get(key)


def truth_events(answer_key: dict, merge_s: float = 0.1) -> list[dict]:
    """Collapse the simulator log into scoreable events.

    Several raw events can fire within a few substeps (e.g. a glancing ring hit
    registers twice); a person sees one event, so same-label events closer
    than ``merge_s`` are merged.
    """
    out: list[dict] = []
    for ev in answer_key["events"]:
        label = _COLLAPSE.get(ev["type"])
        if label is None:
            continue
        if out and out[-1]["label"] == label and ev["t"] - out[-1]["t"] < merge_s:
            continue
        item = {"t": ev["t"], "label": label}
        if "kind" in ev:
            item["kind"] = ev["kind"]
        out.append(item)
    return out


def _match(truth: list[float], pred: list[float], tol: float) -> list[tuple[int, int]]:
    pairs = sorted(
        (abs(tt - pp), i, j)
        for i, tt in enumerate(truth)
        for j, pp in enumerate(pred)
        if abs(tt - pp) <= tol
    )
    used_t, used_p, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i); used_p.add(j); out.append((i, j))
    return out


def score_clip(answer_key: dict, predicted: list[dict], tol_s: float = 0.3,
               coarse: bool = False) -> dict:
    """``predicted`` = ``[{"t": seconds, "label": str}, ...]`` from the agent.

    Unknown labels are counted (``n_unrecognised``) and ignored, so a chatty
    agent is not silently rewarded or punished for vocabulary.

    ``coarse=True`` collapses every non-anomaly label into ``interaction``:
    "something hit something here", without saying what. That is the level a
    pixel tracker can reach, so it is the fair comparison against one.
    """
    truth = truth_events(answer_key)
    if coarse:
        truth = _coarsen(truth)
    pred_by, n_unrec = defaultdict(list), 0
    for p in predicted:
        label = normalise_label(p.get("label", p.get("type", "")))
        if label is None:
            n_unrec += 1
            continue
        if coarse and label != "anomaly":
            label = "interaction"
        pred_by[label].append(float(p["t"]))

    per_label = {}
    for label in (("interaction", "anomaly") if coarse else LABELS):
        tt = [e["t"] for e in truth if e["label"] == label]
        pp = sorted(pred_by.get(label, []))
        m = _match(tt, pp, tol_s)
        per_label[label] = {"tp": len(m), "fp": len(pp) - len(m), "fn": len(tt) - len(m)}

    anomaly = answer_key.get("anomaly")
    pred_anom = sorted(pred_by.get("anomaly", []))
    if anomaly:
        t0 = anomaly["start_frame"] / answer_key["fps"]
        hits = [p for p in pred_anom if p >= t0 - tol_s]
        anomaly_result = {
            "present": True, "kind": anomaly["kind"], "t_true": round(t0, 3),
            "detected": bool(hits),
            "latency_s": round(hits[0] - t0, 3) if hits else None,
        }
    else:
        anomaly_result = {"present": False, "false_alarm": bool(pred_anom)}

    return {
        "per_label": per_label,
        "summary": _prf(per_label),
        "anomaly": anomaly_result,
        "n_unrecognised": n_unrec,
    }


def _coarsen(truth: list[dict], merge_s: float = 0.15) -> list[dict]:
    out: list[dict] = []
    for e in truth:
        if e["label"] == "anomaly":
            out.append(e)
            continue
        if out and out[-1]["label"] == "interaction" and e["t"] - out[-1]["t"] < merge_s:
            continue       # e.g. a ring hit and a membrane contact in the same instant
        out.append({"t": e["t"], "label": "interaction"})
    return out


def _prf(per_label: dict) -> dict:
    out = {}
    for label, c in list(per_label.items()) + [("all", {
        k: sum(v[k] for v in per_label.values()) for k in ("tp", "fp", "fn")
    })]:
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if (tp + fp) and (tp + fn) else None)
        out[label] = {"precision": _r(prec), "recall": _r(rec), "f1": _r(f1)}
    return out


def _r(x):
    return None if x is None else round(x, 3)


def aggregate(results: list[dict]) -> dict:
    """Micro-average over clips, plus anomaly detection rate by kind."""
    per_label: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        for l, c in r["per_label"].items():
            for k in c:
                per_label[l][k] += c[k]
    per_label = dict(per_label)
    by_kind = defaultdict(lambda: {"n": 0, "detected": 0, "latencies": []})
    false_alarms = normals = 0
    for r in results:
        a = r["anomaly"]
        if a["present"]:
            b = by_kind[a["kind"]]
            b["n"] += 1
            b["detected"] += a["detected"]
            if a["latency_s"] is not None:
                b["latencies"].append(a["latency_s"])
        else:
            normals += 1
            false_alarms += a["false_alarm"]
    return {
        "n_clips": len(results),
        "events": _prf(per_label),
        "anomaly_by_kind": {
            k: {"n": v["n"], "detection_rate": round(v["detected"] / v["n"], 3),
                "median_latency_s": (sorted(v["latencies"])[len(v["latencies"]) // 2]
                                     if v["latencies"] else None)}
            for k, v in sorted(by_kind.items())
        },
        "false_alarm_rate": round(false_alarms / normals, 3) if normals else None,
    }
