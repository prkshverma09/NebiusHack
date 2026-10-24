# Hacker B — Data & Eval (benchmarks, graders, traces, datasets, evals)

> You build the **numbers**: real datasets, deterministic graders, teacher traces, SFT JSONL, and the eval that produces the score table. Hacker A builds the proxy/server in parallel — **you do not need their server**: `gen_traces.py` writes trace rows to the same SQLite schema directly (`source='runner'`), so the whole data path runs standalone.
>
> **Read first:** `plans/CONTRACT.md` + `spec/REVIEW-traceforge-findings.md` (esp. TF-03/04/06/09/13).
> Per TF-13: **BFCL is the primary (maybe only) task.** Spider/SQL is stretch, built only after the BFCL score table exists. Per TF-06: the xlam-60k floor comes FIRST in parallel — it de-risks the 14:15-ish job gate.

## Your files (nobody else touches)

`traceforge/graders/{__init__,base,bfcl_ast,sql_exec}.py`, `traceforge/data/{bfcl,spider,xlam}.py`,
`traceforge/datasets.py`, `traceforge/eval.py`, `scripts/{gen_traces,run_eval}.py`, your tests.

## Work packets — parallelize with subagents

### Wave 0 — parallel from minute zero (both are day-0 risk killers)

| Pkt | File | Spec | Why first |
|---|---|---|---|
| **B0** | `data/xlam.py` | Accept HF licence for `Salesforce/xlam-function-calling-60k` **now** (gated — do it interactively), download, convert N=~800 rows to SFT JSONL already in OpenAI message/tool_call format. **This file alone can satisfy the job-submission gate.** | TF-06: guaranteed floor; zero grader dependency |
| **B1** | `graders/bfcl_ast.py` | BFCL AST matcher: parse model's function call(s) vs ground-truth — compare function name, param names/values (order-insensitive dict compare, tolerate enum case). Handle: single, multiple (list), parallel, irrelevance (model should *not* call). Output `Verdict`. | TF-06: the grader gates everything; hardest piece, start immediately. Use their own repo's `eval_checker` logic as reference but keep it self-contained (~150 lines) |

### Wave 1 — two parallel subagents (after B1 exists to grade with; B0 runs alongside)

| Pkt | File | Spec | Acceptance |
|---|---|---|---|
| **B2** | `data/bfcl.py` + `scripts/gen_traces.py` | Load BFCL slice (categories: `simple`, `multiple`, `parallel`, `irrelevance`; ~350 train items + ~120 held-out test, per TF-09). `gen_traces.py --teacher $TEACHER_MODEL --items train --db $TF_DB_PATH`: sends items to teacher via plain `openai` client (NOT via A's server — standalone), writes `traces` rows with full request/response/tools, `source='runner'`. Then runs `BFCLGrader` on each and updates `grade`/`grade_detail`. | Running it produces graded trace rows; `SELECT grade, COUNT(*) FROM traces GROUP BY grade` shows real numbers |
| **B3** | `datasets.py` | `build(task_id, out_dir)` → filter `grade='correct'` + dedupe by prompt hash → write `{train,val}.jsonl` (90/10). **Token audit (TF-03)**: count tokens per row (tiktoken or `len//3` estimate), print histogram, **hard-assert max ≤ 7500**, drop violators and report count. Insert `datasets` row. Merge mode: `--also xlam.jsonl` concatenates the floor dataset. | On fixture traces → valid JSONL; assert fires on an injected 9k row |

### Wave 2 — two parallel subagents

| Pkt | File | Spec | Acceptance |
|---|---|---|---|
| **B4** | `eval.py` + `scripts/run_eval.py` | `run_eval(models: list[str], items, grader, client)` — concurrent per-model runner over held-out items, per-item `EvalResult` persisted (`eval_runs`/`eval_items`). Stats (TF-04): **McNemar** on pupil-vs-teacher disagreements + bootstrap CI on the base-vs-pupil delta. `score_table()` → dict + markdown + ASCII bar chart. CLI: `run_eval.py --models base,pupil-fc-v1,$TEACHER --items test`. | Fixture run over 20 items × 2 fake models → correct table incl. p-value |
| **B5** | `graders/base.py` + `graders/__init__.py` | `Verdict`, `Grader` protocol (contract §5), `get_grader(task_name)` registry mapping `bfcl`→BFCLGrader, `sql`→SQLExecGrader | Import test; A wires it into `/grades/run` |

### Wave 3 — stretch only (after the BFCL score table exists!)

| Pkt | File | Spec |
|---|---|---|
| **B6** | `data/spider.py` + `graders/sql_exec.py` | Spider slice: ~15 small schemas w/ sqlite DBs, ~400 train + ~150 test. `SQLExecGrader`: run generated SQL (30s timeout, read-only) vs gold SQL on same DB → result-set equality → Verdict. Extract SQL from markdown fences/semicolons robustly. |
| **B7** | v2 staircase | Harvest promoted-model failures on new items → teacher-corrected rows → second `datasets.py` build (this is the "learning staircase" stretch demo) |

## Parallelization cheat-sheet

```
min 0:    you start B1 (hardest, gates all) + bg subagent on B0 (xlam licence/download)
wave 1:   2 bg subagents → B2 traces-gen, B3 dataset builder   (B0 may still run)
wave 2:   2 bg subagents → B4 eval+stats, B5 registry          (you: glue + fixtures)
wave 3:   stretch → B6 Spider, B7 staircase — ONLY if score table done
```

## What you hand A (integration surface — contract §5/§6/§7)

- `traceforge/graders/` package: `get_grader()` registry — A injects it into `/grades/run` and shadow grading
- `traceforge/eval.py::run_eval` — A's `/evals` endpoint calls it; `score_table()` renders for dashboard + demo
- `data/datasets/<name>/{train,val}.jsonl` + `datasets` DB row — A's `/datasets` + `/distill` consume
- Graded `traces` rows — A's dashboard + dataset endpoints read them

## Definition of done (your half)

- [ ] xlam floor JSONL exists (insurance, before anything else finishes)
- [ ] ≥300 teacher traces graded on BFCL train split (300–400 is enough — TF-09)
- [ ] `datasets.py` emits JSONL with max-row assert passing
- [ ] `run_eval.py` produces the 3-model table with McNemar p + CI — **this is the demo's money shot, own it**
- [ ] Failure-replay shortlist: ≥3 item_ids where base fails and pupil is expected to pass (verify post-eval)
- [ ] No LLM-as-judge anywhere in the headline path
