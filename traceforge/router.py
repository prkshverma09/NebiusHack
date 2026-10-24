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


def promote(db: DB, model_id: str, teacher: str) -> None:
    model = next((m for m in db.models() if m["id"] == model_id), None)
    if model is None:
        raise GateError(f"no model {model_id}")
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
            db.record_model_state(model_id, "promoted")
            return
    raise GateError(f"no held-out eval containing both {model['name']} and {teacher}")


def demote(db: DB, model_id: str) -> None:
    db.record_model_state(model_id, "demoted")
