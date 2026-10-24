"""Drive dataset prompts through a model, grade every answer, write trace rows (B2).

Standalone path (contract §4): writes `traces` + `grades` rows directly with
`source='runner'`, so the data path runs without A's server.

  python scripts/gen_traces.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
      --items data/datasets/toolace-fc-v1/train.jsonl --task toolace

Each trace gets the verdict column the platform logs lack: correct | incorrect
| error. Transport errors are recorded as `error`, never `incorrect`.
Calls go through `traceforge.eval.call_model`, so responses are cached per
(model, item) and a re-run after a 503 storm only re-calls what failed.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from traceforge.config import load_settings
from traceforge.eval import EvalItem, call_model, grade_call, load_holdout
from traceforge.fingerprint import fingerprint  # A-owned (contract §3); B only calls it
from traceforge.graders import get_grader

# Contract §4, verbatim. IF NOT EXISTS so A's db.py owns the canonical DDL
# and this never conflicts with it.
DDL = """
CREATE TABLE IF NOT EXISTS traces(id TEXT PRIMARY KEY, created_at REAL, task_id TEXT, session_id TEXT,
       model TEXT, request_json TEXT, response_json TEXT,
       prompt_tokens INT, completion_tokens INT, latency_ms INT,
       cost_usd REAL, grade TEXT, grade_detail TEXT, source TEXT DEFAULT 'proxy');
CREATE TABLE IF NOT EXISTS grades(id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, grader TEXT,
       verdict TEXT, detail TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY, name TEXT, first_seen REAL, n_traces INT, n_correct INT);
"""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True, help="JSONL of {messages:[..., gold assistant]}")
    ap.add_argument("--task", default="toolace", help="grader key; also the task label")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    s = load_settings()
    items = load_holdout(args.items, args.limit)
    grader = get_grader(args.task)
    client = OpenAI(api_key=s.nebius_api_key, base_url=s.upstream_base_url, timeout=120, max_retries=0)
    print(f"[traces] {len(items)} items -> {args.model}, grader={args.task}, db={s.db_path}", flush=True)

    con = sqlite3.connect(s.db_path, timeout=60)
    con.executescript(DDL)
    counts: collections.Counter = collections.Counter()
    t0 = time.time()

    def one(it: EvalItem):
        out = call_model(client, args.model, it)
        return (it, out, *grade_call(grader, it, out))

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, fut in enumerate(cf.as_completed([ex.submit(one, it) for it in items]), 1):
            it, out, verdict, detail = fut.result()
            counts[verdict] += 1
            tid = f"tr-{hashlib.sha256((args.model + it.item_id).encode()).hexdigest()[:16]}"
            usage = (out["response"] or {}).get("usage") or {}
            task_id = fingerprint({"messages": it.messages}, task_header=args.task)
            now = time.time()
            # INSERT OR REPLACE: re-runs update a trace's verdict instead of duplicating it.
            con.execute(
                "INSERT OR REPLACE INTO traces(id,created_at,task_id,session_id,model,request_json,"
                "response_json,prompt_tokens,completion_tokens,latency_ms,cost_usd,grade,grade_detail,source)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tid, now, task_id, it.item_id, args.model,
                 json.dumps({"model": args.model, "messages": it.messages, "_gold": it.ground_truth}),
                 json.dumps(out["response"]) if out["response"] else None,
                 usage.get("prompt_tokens"), usage.get("completion_tokens"), out["latency_ms"],
                 None, verdict, detail, "runner"))
            con.execute("INSERT INTO grades(trace_id,grader,verdict,detail,created_at) VALUES (?,?,?,?,?)",
                        (tid, grader.name, verdict, detail, now))
            con.execute(
                "INSERT INTO tasks(task_id,name,first_seen,n_traces,n_correct) VALUES (?,?,?,0,0)"
                " ON CONFLICT(task_id) DO NOTHING", (task_id, args.task, now))
            # Commit per row: a long-held write lock blocks A's server from logging
            # traces into the same sqlite file (seen as "database is locked").
            con.commit()
            if n % 50 == 0 or n == len(items):
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  [{n}/{len(items)}] {dict(counts)}  {rate:.1f}/s", flush=True)

    # Recompute task counters from the rows themselves rather than incrementing.
    con.execute("""UPDATE tasks SET
        n_traces = (SELECT COUNT(*) FROM traces t WHERE t.task_id = tasks.task_id),
        n_correct = (SELECT COUNT(*) FROM traces t WHERE t.task_id = tasks.task_id AND t.grade='correct')""")
    con.commit()
    print("\n[traces] grade counts in db for this model:")
    for g, c in con.execute("SELECT grade, COUNT(*) FROM traces WHERE model=? GROUP BY grade", (args.model,)):
        print(f"  {g:10} {c}")
    con.close()
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
