"""Route selection + lifecycle (A5, minimal). Contract section 8.

A task with a promoted model is served by it; everything else passes through to the
model the caller asked for. Promotion is gated on a held-out eval; demote is unconditional.
"""
from __future__ import annotations

import json

from traceforge.db import DB

PROMOTION_MARGIN = 0.05     # contract section 8(a): acc(candidate) >= acc(teacher) - 0.05


class GateError(Exception):
    """Promotion refused; the message is the reason (served as HTTP 409)."""


def pick_upstream(db: DB, task_id: str, requested: str) -> str:
    for m in reversed(db.models()):
        if m["task_id"] == task_id and m["state"] == "promoted":
            return m["name"]
    return requested


def shadow_model(db: DB, task_id: str) -> str | None:
    for m in reversed(db.models()):
        if m["task_id"] == task_id and m["state"] == "shadow":
            return m["name"]
    return None


def to_shadow(db: DB, model_id: str) -> None:
    if not any(m["id"] == model_id for m in db.models()):
        raise GateError(f"no model {model_id}")
    db.record_model_state(model_id, "shadow")


MIN_SHADOW_CALLS = 50       # contract section 8(b)


def shadow_gate(db: DB, model: dict, teacher: str) -> None:
    """Contract 8(b): over >= 50 graded shadow calls, the candidate's accuracy must be
    within the margin of the teacher's accuracy on the same task's proxied traffic."""
    rows = [t for t in db.traces(task_id=model["task_id"], limit=100_000)
            if t["grade"] in ("correct", "incorrect")]
    cand = [t for t in rows if t["source"] == "shadow" and t["model"] == model["name"]]
    teach = [t for t in rows if t["source"] == "proxy" and t["model"] == teacher]
    if len(cand) < MIN_SHADOW_CALLS:
        raise GateError(f"shadow: {len(cand)} graded calls, need {MIN_SHADOW_CALLS}")
    if not teach:
        raise GateError("shadow: no graded teacher traffic on this task to compare against")
    acc = lambda ts: sum(t["grade"] == "correct" for t in ts) / len(ts)
    if acc(cand) < acc(teach) - PROMOTION_MARGIN:
        raise GateError(f"shadow acc {acc(cand):.3f} < teacher {acc(teach):.3f} - {PROMOTION_MARGIN}")


def promote(db: DB, model_id: str, teacher: str) -> None:
    """Both gates (contract section 8): held-out eval AND shadow traffic."""
    model = next((m for m in db.models() if m["id"] == model_id), None)
    if model is None:
        raise GateError(f"no model {model_id}")
    _eval_gate(db, model, teacher)
    shadow_gate(db, model, teacher)
    db.record_model_state(model_id, "promoted")


def _eval_gate(db: DB, model: dict, teacher: str) -> None:
    for run in db.eval_runs():
        # B's eval.persist: results_json = score_table()["accuracy"] (a list),
        # stats_json = score_table()["paired"].
        accs = {a["model"]: a["acc"] for a in json.loads(run["results_json"] or "[]")}
        if model["name"] in accs and teacher in accs:
            cand, teach = accs[model["name"]], accs[teacher]
            if cand is None or teach is None:
                continue
            if cand < teach - PROMOTION_MARGIN:
                raise GateError(f"acc {cand:.3f} < teacher {teach:.3f} - {PROMOTION_MARGIN}")
            return
    raise GateError(f"no held-out eval containing both {model['name']} and {teacher}")


def demote(db: DB, model_id: str) -> None:
    db.record_model_state(model_id, "demoted")
