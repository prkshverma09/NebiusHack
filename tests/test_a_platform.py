"""Hacker A acceptance tests (plans/hacker-a/IMPLEMENTATION_PLAN.md, A1-A8)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fakes import PROMPT, FakeGrader, FakeUpstream, make_gateway, make_settings
from traceforge.costs import PriceTable
from traceforge.db import DB
from traceforge.fingerprint import fingerprint
from traceforge.orchestrator import Orchestrator, OrchestratorError
from traceforge.router import GateError, demote, pick_upstream, promote, to_shadow


def body(model="teacher/T", messages=PROMPT) -> bytes:
    return json.dumps({"model": model, "messages": messages}).encode()


async def settle(db: DB, n: int, timeout: float = 10) -> list[dict]:
    """Background writes land asynchronously; wait until n traces are graded or exist."""
    end = time.time() + timeout
    while time.time() < end:
        rows = db.traces(limit=1000)
        if len(rows) >= n and all(r["grade"] is not None for r in rows[:n]):
            return rows
        await asyncio.sleep(0.05)
    return db.traces(limit=1000)


# ---- A2: fingerprint + costs -------------------------------------------------------
def test_fingerprint_groups_by_system_and_tools_not_user_turn():
    a = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q1"}]}
    b = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q2"}]}
    c = {"messages": [{"role": "system", "content": "S2"}, {"role": "user", "content": "q1"}]}
    assert fingerprint(a) == fingerprint(b) != fingerprint(c)
    assert len(fingerprint(a)) == 12


def test_fingerprint_matches_b_gen_traces_fallback():
    """B passes the task label in-payload as `_task`; ids must equal B's spec fallback."""
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    canon = json.dumps({"system": "S", "tools": [], "response_format": None, "task": "toolace"},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert fingerprint({"messages": msgs, "_task": "toolace"}) == \
        hashlib.sha256(canon.encode()).hexdigest()[:12]


def test_unknown_model_costs_none_not_zero():
    pt = PriceTable({"m": (1e-7, 3e-7)})
    assert pt.cost_usd("m", {"prompt_tokens": 1000, "completion_tokens": 100}) == pytest.approx(1.3e-4)
    assert pt.cost_usd("other", {"prompt_tokens": 1}) is None


# ---- A1: db -------------------------------------------------------------------------
def test_regrading_does_not_double_count(tmp_path):
    db = DB(tmp_path / "t.db")
    tid = db.insert_trace(task_id="t", model="m", request={}, response={}, prompt_tokens=1,
                          completion_tokens=1, latency_ms=1, cost_usd=None)
    db.update_grade(tid, grader="g", correct=True)
    db.update_grade(tid, grader="g", correct=True)
    assert db.tasks()[0]["n_correct"] == 1
    db.update_grade(tid, grader="g", correct=False)
    assert db.tasks()[0]["n_correct"] == 0
    db.update_grade(tid, grader="g", correct=None)
    assert db.trace(tid)["grade"] == "error"      # grader failure is not 'incorrect'


def test_failed_write_does_not_poison_connection(tmp_path):
    """Regression (found live 15:25): a locked write left the implicit transaction open;
    the next read pinned a stale snapshot and every later write failed forever."""
    db = DB(tmp_path / "t.db")
    db._con.execute("PRAGMA busy_timeout=100")
    holder = sqlite3.connect(tmp_path / "t.db", timeout=1)
    holder.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.OperationalError):
        db.insert_trace(task_id="t", model="m", request={}, response=None, prompt_tokens=None,
                        completion_tokens=None, latency_ms=0, cost_usd=None)
    # A read on the *write* connection (update_grade reads before it writes). Without the
    # rollback this pins a snapshot older than the holder's commit and the retry fails.
    db._con.execute("SELECT COUNT(*) FROM traces").fetchone()
    holder.execute("INSERT INTO tasks VALUES ('x','x',0,0,0)")
    holder.commit()
    db.insert_trace(task_id="t", model="m", request={}, response=None, prompt_tokens=None,
                    completion_tokens=None, latency_ms=0, cost_usd=None)
    assert len(db.traces()) == 1


# ---- A3: gateway ------------------------------------------------------------------
def test_proxy_returns_upstream_bytes_verbatim_and_writes_trace(tmp_path):
    up = FakeUpstream(answer="[f(x=1)]")
    gw = make_gateway(tmp_path, up)

    async def go():
        st, raw, _ = await gw.proxy(body())
        return st, raw, await settle(gw.db, 1)
    st, raw, rows = asyncio.run(go())
    assert st == 200 and json.loads(raw)["choices"][0]["message"]["content"] == "[f(x=1)]"
    assert len(rows) == 1 and rows[0]["source"] == "proxy"
    assert rows[0]["cost_usd"] == pytest.approx(100 * 2e-7 + 10 * 6e-7)
    assert rows[0]["grade"] is None or rows[0]["grade"] == "error"   # no gold -> never 'correct'


def test_known_prompt_is_graded_live(tmp_path):
    gold = {json.dumps(PROMPT): ("fake", "[f(x=1)]")}
    for answer, want in (("[f(x=1)]", "correct"), ("[f(x=2)]", "incorrect")):
        gw = make_gateway(tmp_path / want, FakeUpstream(answer=answer), gold=gold)

        async def go():
            await gw.proxy(body())
            return await settle(gw.db, 1)
        assert asyncio.run(go())[0]["grade"] == want


def test_unreachable_upstream_is_a_recorded_error_not_a_missing_trace(tmp_path):
    gw = make_gateway(tmp_path, FakeUpstream(fail=httpx.ConnectError("getaddrinfo failed")))

    async def go():
        st, raw, _ = await gw.proxy(body())
        return st, raw, await settle(gw.db, 1)
    st, raw, rows = asyncio.run(go())
    assert st == 502 and json.loads(raw)["error"]["type"] == "upstream_unreachable"
    assert rows[0]["grade"] == "error" and rows[0]["grade_detail"] == "upstream_unreachable"


def test_locked_db_does_not_delay_caller_and_trace_still_lands(tmp_path):
    gw = make_gateway(tmp_path, FakeUpstream())
    gw.db._con.execute("PRAGMA busy_timeout=200")
    holder = sqlite3.connect(gw.db.path, timeout=1, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    threading.Timer(1.5, holder.commit).start()

    async def go():
        t0 = time.time()
        st, _, _ = await gw.proxy(body())
        resp_ms = (time.time() - t0) * 1000
        rows = []
        for _ in range(200):
            rows = gw.db.traces()
            if rows:
                break
            await asyncio.sleep(0.05)
        return st, resp_ms, rows
    st, resp_ms, rows = asyncio.run(go())
    assert st == 200 and resp_ms < 500 and len(rows) == 1


def test_shadow_answers_same_prompt_without_touching_the_response(tmp_path):
    up = FakeUpstream(answer="[f(x=1)]")
    gold = {json.dumps(PROMPT): ("fake", "[f(x=1)]")}
    gw = make_gateway(tmp_path, up, gold=gold)
    mid = gw.db.register_model(name="pupil/P", task_id=fingerprint({"messages": PROMPT}),
                               base_model="student/S")
    to_shadow(gw.db, mid)

    async def go():
        st, raw, _ = await gw.proxy(body())
        return raw, await settle(gw.db, 2)
    raw, rows = asyncio.run(go())
    assert json.loads(raw)["model"] == "teacher/T"           # caller got the teacher
    by_source = {r["source"]: r for r in rows}
    assert by_source["shadow"]["model"] == "pupil/P" and by_source["shadow"]["grade"] == "correct"
    assert [r["model"] for r in up.requests] == ["teacher/T", "pupil/P"]


# ---- A5: router -------------------------------------------------------------------
def _eval_run(db: DB, accs: dict) -> None:
    db._w("INSERT INTO eval_runs VALUES (?,?,?,?,?,?,?)",
          ("ev-1", "b", json.dumps(list(accs)), 10,
           json.dumps([{"model": m, "acc": a} for m, a in accs.items()]), "[]", time.time()))


def _graded(db: DB, task: str, model: str, source: str, n: int, n_correct: int) -> None:
    for i in range(n):
        tid = db.insert_trace(task_id=task, model=model, request={}, response={},
                              prompt_tokens=1, completion_tokens=1, latency_ms=1,
                              cost_usd=None, source=source)
        db.update_grade(tid, grader="g", correct=i < n_correct)


def test_promotion_requires_eval_and_shadow_gates(tmp_path):
    db = DB(tmp_path / "t.db")
    mid = db.register_model(name="pupil/P", task_id="t", base_model="s")
    with pytest.raises(GateError, match="no held-out eval"):
        promote(db, mid, teacher="teacher/T")
    _eval_run(db, {"pupil/P": 0.74, "teacher/T": 0.776})
    with pytest.raises(GateError, match="shadow: 0 graded"):
        promote(db, mid, teacher="teacher/T")
    _graded(db, "t", "teacher/T", "proxy", 50, 39)
    _graded(db, "t", "pupil/P", "shadow", 50, 37)
    promote(db, mid, teacher="teacher/T")
    assert pick_upstream(db, "t", "teacher/T") == "pupil/P"
    demote(db, mid)
    assert pick_upstream(db, "t", "teacher/T") == "teacher/T"


def test_promotion_refused_below_margin(tmp_path):
    db = DB(tmp_path / "t.db")
    mid = db.register_model(name="pupil/P", task_id="t", base_model="s")
    _eval_run(db, {"pupil/P": 0.60, "teacher/T": 0.776})
    with pytest.raises(GateError, match="< teacher"):
        promote(db, mid, teacher="teacher/T")


# ---- A4: orchestrator -------------------------------------------------------------
def test_orchestrator_submit_sequence_and_batch_minimum(tmp_path):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if req.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": f"file-{len(seen)}"})
        return httpx.Response(200, json={"id": "ftjob-1", "status": "running"})
    (tmp_path / "train.jsonl").write_text("{}\n")
    (tmp_path / "val.jsonl").write_text("{}\n")
    o = Orchestrator(make_settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = o.submit(train=tmp_path / "train.jsonl", val=tmp_path / "val.jsonl",
                   model="Qwen/Qwen3-30B-A3B-Instruct-2507", suffix="pupil-x")
    assert [p for _, p in seen] == ["/v1/files", "/v1/files", "/v1/fine_tuning/jobs"]
    hp = out["request"]["hyperparameters"]
    assert hp["lora"] and hp["batch_size"] * hp["context_length"] >= 131072


def test_deploy_rejects_ids_the_api_would_reject(tmp_path):
    o = Orchestrator(make_settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(500))))
    with pytest.raises(OrchestratorError, match="model-artifact_"):
        o.deploy(name="p", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507", custom_weights_id="ftckpt-1")
    with pytest.raises(OrchestratorError, match="no custom-weights deploy config"):
        o.deploy(name="p", base_model="meta-llama/Llama-3.1-8B-Instruct",
                 custom_weights_id="model-artifact_x")


# ---- A6: server -------------------------------------------------------------------
@pytest.fixture
def client(tmp_path):
    from traceforge.server import Services, create_app
    gold = {json.dumps(PROMPT): ("fake", "[f(x=1)]")}
    gw = make_gateway(tmp_path, FakeUpstream(answer="[f(x=1)]"), gold=gold)
    jobs = lambda r: httpx.Response(200, json={"id": "ftjob-1", "status": "running",
                                               "model": "m", "trained_tokens": 0, "data": []})
    svc = Services(settings=make_settings(tmp_path), db=gw.db, gateway=gw,
                   orchestrator=Orchestrator(make_settings(tmp_path),
                                             client=httpx.Client(transport=httpx.MockTransport(jobs))))
    with TestClient(create_app(svc)) as c:
        yield c, gw


def test_dashboard_renders_with_empty_db(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "TraceForge" in r.text
    assert c.get("/traces").json() == {"traces": []}
    assert c.get("/tasks").json() == {"tasks": []}
    assert c.get("/stats/cost").json() == {"by_model": []}
    assert c.get("/models").json() == []


def test_endpoint_table_end_to_end(client):
    c, gw = client
    r = c.post("/v1/chat/completions", content=body())
    assert r.status_code == 200
    for _ in range(100):
        tr = c.get("/traces").json()["traces"]
        if tr and tr[0]["grade"]:
            break
        time.sleep(0.05)
    assert tr[0]["grade"] == "correct" and tr[0]["answer"] == "[f(x=1)]"
    assert c.get("/tasks").json()["tasks"][0]["n_correct"] == 1
    rp = c.post("/replay", json={"trace_id": tr[0]["id"], "model": "pupil/P"}).json()
    assert rp["verdict"] == "correct" and rp["original"]["verdict"] == "correct"
    assert c.post("/replay", json={"trace_id": "nope", "model": "x"}).status_code == 404
    assert c.post("/grades/run", json={}).json()["graded"]["correct"] == 0   # nothing left ungraded
    mid = gw.db.register_model(name="pupil/P", task_id="t", base_model="s")
    assert c.post(f"/models/{mid}/promote").status_code == 409
    assert c.post(f"/models/{mid}/shadow").status_code == 200
    assert c.post(f"/models/{mid}/demote").status_code == 200
    assert c.get("/jobs/ftjob-1").json()["status"] == "running"
    assert c.post("/distill", json={"dataset_id": "missing"}).status_code == 404
