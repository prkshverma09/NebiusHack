# Integration Plan — merging A (platform) + B (data/eval) into TraceForge

> Both hackers worked in separate directories implementing `plans/CONTRACT.md`.
> This plan merges, wires, verifies end-to-end, and produces the demo. Budget ~45 min for integration proper; the E2E run rides on the already-submitted LoRA job's timeline.

## 0. Pre-flight (5 min)

- [ ] Both trees pass their own unit tests: `pytest -q` in each dir
- [ ] `plans/CONTRACT.md` unchanged in both dirs (`diff` them — divergence = stop and reconcile before merging)
- [ ] `NEBIUS_API_KEY` valid; `data/STATUS.md` records the deploy-path finding (serverless LoRA vs dedicated endpoint)

## 1. Merge (10 min)

Files are disjoint by contract — a clean overlay:

```bash
# into integration dir
cp -r hacker-a-work/traceforge ./        # A's package root
cp -r hacker-a-work/dashboard hacker-a-work/scripts ./
cp -r hacker-b-work/traceforge/graders traceforge/   # B's subtree
cp -r hacker-b-work/traceforge/data    traceforge/
cp     hacker-b-work/traceforge/datasets.py traceforge/
cp     hacker-b-work/traceforge/eval.py traceforge/
cp -r hacker-b-work/scripts/* scripts/               # gen_traces, run_eval (watch: no filename collisions vs A's smoke/submit)
rsync -a --ignore-existing hacker-a-work/tests/ tests/
rsync -a --ignore-existing hacker-b-work/tests/ tests/
```

Conflict rules: identical basenames in `tests/` → rename with `a_`/`b_` prefix; any same-path source file → contract violation, reconcile by hand (should not happen).

## 2. Wiring checklist (10 min)

- [ ] `server.py` grader registry: replace empty default with `from traceforge.graders import get_grader`; inject via `Services.build()`
- [ ] `POST /grades/run` → iterates ungraded `traces` for task, calls `grader.grade(...)`, writes `grades` + updates `traces.grade`
- [ ] `POST /datasets` → calls `datasets.build(task_id, ...)`; returns `{dataset_id, n_train, n_val, n_tokens, dropped_rows}`
- [ ] `POST /distill` → orchestrator consumes `datasets.train_path/val_path`
- [ ] `POST /evals` → calls `eval.run_eval(...)`; `GET /evals/{id}` serves `score_table()`
- [ ] `POST /replay` → looks up trace, re-calls `model`, re-grades with task's grader
- [ ] Router `pick_upstream` maps `task_id → promoted model name` from `models` table
- [ ] `import traceforge` + `create_app()` importable with zero env (test doubles injected)

## 3. E2E runbook — the real pipeline (in order)

```bash
# 0) day-0 (should already be done — A0.2)
python scripts/smoke_nebius.py

# 1) traces: teacher answers BFCL train items, writes + grades rows directly
python scripts/gen_traces.py --teacher $TEACHER_MODEL --items data/bfcl_train.jsonl --db $TF_DB_PATH

# 2) dataset: verified traces → JSONL (+ xlam floor merged), token audit asserts ≤7500
python -c "from traceforge.datasets import build; build(task_id='<fp>', out_dir='data/datasets/bfcl-v1', also='data/xlam_floor.jsonl')"

# 3) distill — HARD GATE: submit no later than T+1:45 regardless of dashboard state
python scripts/submit_job.py --train data/datasets/bfcl-v1/train.jsonl --val data/datasets/bfcl-v1/val.jsonl --suffix pupil-fc-v1 --watch

# 4) deploy + smoke
python scripts/submit_job.py --deploy --job-id <id> --name pupil-fc-v1
python -c "...chat.completions.create(model='pupil-fc-v1', ...)"

# 5) eval: the money shot — base vs pupil vs teacher on held-out items
python scripts/run_eval.py --benchmark bfcl --models meta-llama/Llama-3.1-8B-Instruct,pupil-fc-v1,$TEACHER_MODEL --items data/bfcl_test.jsonl

# 6) serve + promote (if gate passes)
uvicorn traceforge.server:create_app --factory --port 8000
curl -X POST localhost:8000/models/<id>/shadow   # optional if clock tight
curl -X POST localhost:8000/models/<id>/promote
```

## 4. Fallback matrix — decide by gate, not vibes

| If broken | Then |
|---|---|
| Serverless LoRA deploy dead (TF-07 finding) | Dedicated endpoint for 2–3h (~$5–15); if quota denied → run eval via batch inference on the checkpoint path, demo shows score table + architecture |
| Teacher trace yield too low by T+1:00 | Submit job on **xlam floor alone**; merge verified traces into a v2 job later |
| 8k context rows dropped >20% (TF-03) | Rebuild dataset trimming system prompt/tools; prefer dropping `parallel_multiple` items |
| Job still queued at T+2:30 | Demo pivots to: traces → dataset → *job running* → pre-computed base-vs-teacher eval; promotion shown as pending state |
| Eval harness unfinished | Fall back to grader-only accuracy script over held-out items (B's `run_eval` is priority #1 for B) |
| Venue network dies | Labelled replay: `data/replay/` recorded responses; dashboard renders from DB only — never re-grade |

## 5. Demo finalization (per review TF-05/TF-12)

- [ ] Pitch line: *"Token Factory logs every call and Data Lab turns logs into datasets — but nothing records whether the answer was right. We added that verdict column and closed the loop."* No Mubit, no Banyan-echo names.
- [ ] Score table shows **delta vs base** as headline + McNemar p-value (TF-04)
- [ ] Pre-picked failure-replay items (B's DOD list) verified against deployed `pupil-fc-v1`
- [ ] Dashboard: trace feed + cost counter + score bars rendering real values; screenshot the full page as backup
- [ ] Cost tally from `GET /stats/cost` — know the real "under $N" number before saying it
- [ ] Backup artifacts in `data/demo/`: score table markdown, 3 graded trace JSONs, 30s screen recording of live eval

## 6. Definition of done (project)

1. Teacher traces → verified → JSONL → real LoRA job → deployed `pupil-*` endpoint — all real, all logged
2. Score table: base / pupil / teacher on held-out items, paired stats
3. One live call through the gateway served by a promoted model (or honest pending state)
4. Demo rehearsed end-to-end ≤3 min with fallbacks staged
5. README: architecture diagram, runbook, real spend number, the verdict-column pitch
