"""SQLite store. Contract section 4 -- DDL verbatim.

Every CREATE is IF NOT EXISTS: B's scripts/gen_traces.py and eval.persist() create or
expect the same tables, so whichever side touches the db first wins and both keep working.
One connection, guarded by an RLock, WAL so B's scripts can write while the server reads.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS traces(id TEXT PRIMARY KEY, created_at REAL, task_id TEXT, session_id TEXT,
       model TEXT, request_json TEXT, response_json TEXT,
       prompt_tokens INT, completion_tokens INT, latency_ms INT,
       cost_usd REAL, grade TEXT,
       grade_detail TEXT, source TEXT DEFAULT 'proxy');
CREATE TABLE IF NOT EXISTS grades(id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, grader TEXT,
       verdict TEXT, detail TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY, name TEXT, first_seen REAL, n_traces INT, n_correct INT);
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, task_id TEXT, name TEXT, train_path TEXT, val_path TEXT,
         n_train INT, n_val INT, n_tokens INT, train_file_id TEXT, val_file_id TEXT,
         created_at REAL);
CREATE TABLE IF NOT EXISTS finetune_jobs(id TEXT PRIMARY KEY, dataset_id TEXT, nebius_job_id TEXT, base_model TEXT,
              hyperparams_json TEXT, status TEXT, trained_tokens INT,
              checkpoint_id TEXT, created_at REAL, finished_at REAL, error TEXT);
CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, name TEXT, task_id TEXT, source TEXT, base_model TEXT,
       state TEXT,
       created_at REAL);
CREATE TABLE IF NOT EXISTS eval_runs(id TEXT PRIMARY KEY, benchmark TEXT, models_json TEXT, n_items INT,
          results_json TEXT, stats_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS eval_items(id INTEGER PRIMARY KEY AUTOINCREMENT, eval_run_id TEXT, item_id TEXT,
           model TEXT, verdict TEXT, detail TEXT, latency_ms INT, tokens INT);
CREATE INDEX IF NOT EXISTS ix_traces_task ON traces(task_id, created_at);
"""

MODEL_STATES = ("candidate", "shadow", "promoted", "demoted")


class DB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._con.row_factory = sqlite3.Row
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.executescript(DDL)
            self._con.commit()
        # Reads get their own connection + lock. WAL lets them run beside a writer, so
        # the request path (router lookup) and the dashboard never queue behind a write
        # that is itself waiting out B's write lock.
        self._rlock = threading.Lock()
        self._rcon = sqlite3.connect(self.path, check_same_thread=False, timeout=30,
                                     isolation_level=None)   # autocommit: no stale snapshot
        self._rcon.row_factory = sqlite3.Row

    def _q(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._rlock:
            return [dict(r) for r in self._rcon.execute(sql, args).fetchall()]

    def _w(self, sql: str, args: tuple = ()) -> None:
        with self._lock, self._txn():
            self._con.execute(sql, args)

    def _txn(self):
        """Commit on success; ROLL BACK on failure. A failed write otherwise leaves the
        implicit transaction open, the next read on this shared connection pins a stale
        WAL snapshot, and every retry fails 'database is locked' forever -- one lock
        from B's concurrent writer would silently end all trace logging."""
        con = self._con

        class _T:
            def __enter__(self):
                return con

            def __exit__(self, exc_type, *_):
                if exc_type is None:
                    con.commit()
                elif con.in_transaction:
                    con.rollback()
                return False
        return _T()

    # ---- traces ---------------------------------------------------------------------
    def insert_trace(self, *, task_id: str, model: str, request: dict, response: dict | None,
                     prompt_tokens: int | None, completion_tokens: int | None,
                     latency_ms: int, cost_usd: float | None, session_id: str | None = None,
                     source: str = "proxy", task_name: str | None = None) -> str:
        tid = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock, self._txn():
            self._con.execute(
                "INSERT INTO traces(id,created_at,task_id,session_id,model,request_json,response_json,"
                "prompt_tokens,completion_tokens,latency_ms,cost_usd,grade,grade_detail,source)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?)",
                (tid, now, task_id, session_id, model, json.dumps(request),
                 json.dumps(response) if response is not None else None,
                 prompt_tokens, completion_tokens, latency_ms, cost_usd, source))
            self._con.execute(
                "INSERT INTO tasks(task_id,name,first_seen,n_traces,n_correct) VALUES (?,?,?,1,0)"
                " ON CONFLICT(task_id) DO UPDATE SET n_traces = n_traces + 1",
                (task_id, task_name or task_id, now))
        return tid

    def update_grade(self, trace_id: str, *, grader: str, correct: bool | None,
                     detail: str = "") -> None:
        """correct=None records a grader error -- distinct from 'incorrect'."""
        verdict = "error" if correct is None else ("correct" if correct else "incorrect")
        with self._lock, self._txn():
            prev = self._con.execute("SELECT task_id, grade FROM traces WHERE id=?",
                                     (trace_id,)).fetchone()
            if prev is None:
                raise KeyError(f"no trace {trace_id}")
            self._con.execute("UPDATE traces SET grade=?, grade_detail=? WHERE id=?",
                              (verdict, detail, trace_id))
            self._con.execute(
                "INSERT INTO grades(trace_id,grader,verdict,detail,created_at) VALUES (?,?,?,?,?)",
                (trace_id, grader, verdict, detail, time.time()))
            delta = (verdict == "correct") - (prev["grade"] == "correct")
            if delta:
                self._con.execute("UPDATE tasks SET n_correct = n_correct + ? WHERE task_id=?",
                                  (delta, prev["task_id"]))

    def traces(self, task_id: str | None = None, grade: str | None = None,
               limit: int = 200) -> list[dict]:
        sql, args = "SELECT * FROM traces WHERE 1=1", []
        if task_id:
            sql += " AND task_id=?"
            args.append(task_id)
        if grade == "ungraded":
            sql += " AND grade IS NULL"
        elif grade:
            sql += " AND grade=?"
            args.append(grade)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return self._q(sql, tuple(args))

    def trace(self, trace_id: str) -> dict | None:
        rows = self._q("SELECT * FROM traces WHERE id=?", (trace_id,))
        return rows[0] if rows else None

    # ---- tasks / models -------------------------------------------------------------
    def tasks(self) -> list[dict]:
        return self._q(
            "SELECT t.*, (SELECT m.state FROM models m WHERE m.task_id=t.task_id"
            " ORDER BY m.created_at DESC LIMIT 1) AS model_state"
            " FROM tasks t ORDER BY t.first_seen")

    def register_model(self, *, name: str, task_id: str | None, base_model: str,
                       source: str = "finetune", state: str = "candidate") -> str:
        mid = uuid.uuid4().hex[:12]
        self._w("INSERT INTO models VALUES (?,?,?,?,?,?,?)",
                (mid, name, task_id, source, base_model, state, time.time()))
        return mid

    def record_model_state(self, model_id: str, state: str) -> None:
        if state not in MODEL_STATES:
            raise ValueError(f"state must be one of {MODEL_STATES}")
        self._w("UPDATE models SET state=? WHERE id=?", (state, model_id))

    def models(self) -> list[dict]:
        return self._q("SELECT * FROM models ORDER BY created_at")

    # ---- jobs / evals ---------------------------------------------------------------
    def upsert_job(self, *, nebius_job_id: str, base_model: str, status: str,
                   trained_tokens: int | None = None, hyperparams: dict | None = None,
                   dataset_id: str | None = None, error: str | None = None) -> None:
        self._w(
            "INSERT INTO finetune_jobs(id,dataset_id,nebius_job_id,base_model,hyperparams_json,"
            "status,trained_tokens,created_at,error) VALUES (?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET status=excluded.status,"
            " trained_tokens=excluded.trained_tokens, error=excluded.error",
            (nebius_job_id, dataset_id, nebius_job_id, base_model,
             json.dumps(hyperparams or {}), status, trained_tokens, time.time(), error))

    def jobs(self) -> list[dict]:
        return self._q("SELECT * FROM finetune_jobs ORDER BY created_at")

    def eval_runs(self) -> list[dict]:
        return self._q("SELECT * FROM eval_runs ORDER BY created_at DESC")

    # ---- aggregates -----------------------------------------------------------------
    def cost_by_model(self) -> list[dict]:
        return self._q(
            "SELECT model, COUNT(*) AS n, SUM(cost_usd) AS usd,"
            " SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens,"
            " SUM(grade='correct') AS n_correct, SUM(grade IS NOT NULL) AS n_graded"
            " FROM traces GROUP BY model ORDER BY n DESC")
