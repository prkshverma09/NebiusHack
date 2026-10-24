"""datasets.build against fixture traces written through A's DB API (contract B3 acceptance)."""
from __future__ import annotations

import json

import pytest

# traceforge/db.py is A-owned. Until it is committed, skip loudly rather than fail
# collection — a collection error aborts the WHOLE pytest session, not just this file.
pytest.importorskip("traceforge.db", reason="needs Hacker A's traceforge/db.py (not yet committed)")

import traceforge.datasets as ds  # noqa: E402
from traceforge.db import DB  # noqa: E402

M = "teacher"


def prompt(k, pad=""):
    return [{"role": "system", "content": "tools: f"}, {"role": "user", "content": f"q{k} {pad}"}]


def add(db, k, grade, pad="", answer='[f(x=1)]'):
    tid = db.insert_trace(task_id="t", model=M, request={"messages": prompt(k, pad)},
                          response={"choices": [{"message": {"role": "assistant", "content": answer}}]},
                          prompt_tokens=1, completion_tokens=1, latency_ms=1, cost_usd=None,
                          source="runner")
    db.update_grade(tid, grader="toolace", correct={"correct": True, "incorrect": False}.get(grade),
                    detail="fixture")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "OUT_ROOT", tmp_path / "out")
    hold = tmp_path / "holdout.jsonl"
    hold.write_text(json.dumps({"messages": prompt(999) + [{"role": "assistant", "content": "[f()]"}]}) + "\n",
                    encoding="utf-8")
    return DB(tmp_path / "t.db"), tmp_path / "t.db", hold


def test_only_correct_traces_become_rows(env):
    db, path, hold = env
    for k in range(20):
        add(db, k, "correct")
    for k in range(20, 25):
        add(db, k, "incorrect")
    add(db, 25, "error")
    add(db, 3, "correct")                                  # duplicate prompt
    m = ds.build(model=M, name="v2", holdout=hold, db_path=path)
    assert (m["n_train"], m["n_val"]) == (18, 2)
    assert m["dropped"] == {"duplicate_prompt": 1}
    assert m["verdicts"] == {"correct": 21, "incorrect": 5, "error": 1}
    assert m["teacher_acceptance"] == pytest.approx(21 / 26)
    rows = [json.loads(l) for l in (ds.OUT_ROOT / "v2/train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["messages"][-1] == {"role": "assistant", "content": "[f(x=1)]"} for r in rows)
    assert len(db.traces(limit=1000)) == 27                # build never mutates traces


def test_9k_row_is_dropped_not_truncated(env):
    db, path, hold = env
    add(db, 1, "correct")
    add(db, 2, "correct", pad="word " * 9000)              # ~9k tokens
    m = ds.build(model=M, name="cap", holdout=hold, db_path=path)
    assert m["dropped"] == {"over_token_cap": 1}
    assert m["token_audit"]["max"] <= 7500


def test_holdout_leakage_refuses_to_build(env):
    db, path, hold = env
    add(db, 1, "correct")
    add(db, 999, "correct")                                # same prompt as the holdout item
    with pytest.raises(ds.DatasetError, match="LEAKAGE"):
        ds.build(model=M, name="leak", holdout=hold, db_path=path)
    assert not (ds.OUT_ROOT / "leak").exists()


def test_no_correct_traces_is_an_error_not_an_empty_file(env):
    db, path, hold = env
    add(db, 1, "incorrect")
    with pytest.raises(ds.DatasetError, match="no correct traces"):
        ds.build(model=M, name="empty", holdout=hold, db_path=path)
