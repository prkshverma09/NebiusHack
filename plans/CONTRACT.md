# TraceForge — Integration Contract (single source of truth)

Both hackers build to this document. **If it's not here, it doesn't exist.**
Copy this file into your working directory. Changes require both hackers to agree — edit via a commit, not chat.

Review findings this contract bakes in (`spec/REVIEW-traceforge-findings.md` — required reading):

- **TF-01**: student = `meta-llama/Llama-3.1-8B-Instruct` (only cheap serverless LoRA deploy path).
- **TF-03**: fine-tune context cap is **8k**; with `packing: true` over-length rows truncate *silently*. Dataset builder must assert serialised rows ≤ **7,500 tokens** and drop/diagnose violators.
- **TF-04**: stats use a **paired** test (McNemar) over identical items; headline = accuracy delta vs **base student**, not "within N pts of teacher".
- **TF-05**: no Mubit branding anywhere in code/UI. Deployed models are named `pupil-<task>-v<n>`.
- **TF-06**: `xlam-function-calling-60k` is a guaranteed dataset floor, not an optional extra.
- **TF-07**: `POST /v0/models` deploy call is smoke-tested before anything depends on it.
- **TF-12**: Nebius already has capture (Inference Logs) + datasets (Data Lab). Our gap: the **verdict column**. Demo pitch: *"they record what was said, never whether it was right — we added that column and closed the loop."*

## 1. Final repo layout (post-integration)

```
traceforge/                 # python package
  __init__.py
  config.py                 # A — env loading, Settings
  schemas.py                # A — dataclasses below (contract-owned; A writes first)
  db.py                     # A — sqlite layer + DDL below
  fingerprint.py            # A — task fingerprinting
  gateway.py                # A — proxy endpoint logic
  costs.py                  # A — price table + cost computation
  router.py                 # A — route selection, shadow fan-out, promote/demote
  orchestrator.py           # A — Nebius files/jobs/deploy calls
  server.py                 # A — FastAPI app factory + endpoints
  graders/
    __init__.py             # B — registry {task_name: Grader}
    base.py                 # B — Grader protocol + Verdict
    bfcl_ast.py             # B — AST match grader (PRIMARY)
    sql_exec.py             # B — exec-match grader (STRETCH, task B only)
  data/
    bfcl.py                 # B — BFCL slice loading/item prep
    spider.py               # B — Spider slice (STRETCH)
    xlam.py                 # B — xlam-60k floor dataset
  datasets.py               # B — verified traces → JSONL + token audit
  eval.py                   # B — eval runner + paired stats + score table
scripts/
  gen_traces.py             # B — drives items through a chat endpoint, writes trace rows
  run_eval.py               # B — CLI eval entrypoint
  smoke_nebius.py           # A — day-0 checks (key, teacher call, job API, deploy API)
  submit_job.py             # A — one-shot upload→job CLI (emergency path)
dashboard/
  index.html                # A — single-file dashboard
tests/                      # each hacker tests own modules; shared fakes in tests/fakes.py (A)
data/                       # runtime artifacts: db, jsonl, cache (gitignored)
```

Ownership: **A owns everything not marked B.** Files are disjoint by construction — no file has two owners.

## 2. Env / config

```
NEBIUS_API_KEY              # required
TF_DB_PATH=./data/traceforge.db
UPSTREAM_BASE_URL=https://api.tokenfactory.nebius.com/v1/
TEACHER_MODEL=deepseek-ai/DeepSeek-V3-0324        # or Qwen/Qwen3-235B-A22B-Instruct-2507
STUDENT_BASE=meta-llama/Llama-3.1-8B-Instruct
TF_PORT=8000
```

## 3. Task fingerprint

```python
fingerprint(payload) -> str  # sha256(canonical_json({
                             #   "system": first system message content or "",
                             #   "tools": request.tools or [],
                             #   "response_format": request.response_format or None,
                             #   "task": header "X-TF-Task" or ""
                             # }))[:12]
```

Canonical JSON = sorted keys, `separators=(",",":")`, UTF-8. `X-TF-Task` is an optional human label (`bfcl`, `sql`); two calls with same system+tools and no header → same fingerprint.

## 4. SQLite schema (A implements verbatim; B reads/writes via these columns only)

```sql
traces(id TEXT PK, created_at REAL, task_id TEXT, session_id TEXT,
       model TEXT, request_json TEXT, response_json TEXT,
       prompt_tokens INT, completion_tokens INT, latency_ms INT,
       cost_usd REAL, grade TEXT,                -- NULL|correct|incorrect|error
       grade_detail TEXT, source TEXT DEFAULT 'proxy')  -- proxy|runner|shadow
grades(id INTEGER PK AUTOINCREMENT, trace_id TEXT, grader TEXT,
       verdict TEXT, detail TEXT, created_at REAL)
tasks(task_id TEXT PK, name TEXT, first_seen REAL, n_traces INT, n_correct INT)
datasets(id TEXT PK, task_id TEXT, name TEXT, train_path TEXT, val_path TEXT,
         n_train INT, n_val INT, n_tokens INT, train_file_id TEXT, val_file_id TEXT,
         created_at REAL)
finetune_jobs(id TEXT PK, dataset_id TEXT, nebius_job_id TEXT, base_model TEXT,
              hyperparams_json TEXT, status TEXT, trained_tokens INT,
              checkpoint_id TEXT, created_at REAL, finished_at REAL, error TEXT)
models(id TEXT PK, name TEXT, task_id TEXT, source TEXT, base_model TEXT,
       state TEXT,                     -- candidate|shadow|promoted|demoted
       created_at REAL)
eval_runs(id TEXT PK, benchmark TEXT, models_json TEXT, n_items INT,
          results_json TEXT, stats_json TEXT, created_at REAL)
eval_items(id INTEGER PK AUTOINCREMENT, eval_run_id TEXT, item_id TEXT,
           model TEXT, verdict TEXT, detail TEXT, latency_ms INT, tokens INT)
```

`source='runner'` means the row came from `scripts/gen_traces.py` writing directly (standalone path — lets B generate traces before A's server exists). Same columns, same semantics.

## 5. Shared dataclasses (`schemas.py`, A owns)

```python
@dataclass(frozen=True)
class Verdict:           # returned by every grader
    correct: bool
    detail: str = ""     # machine-readable reason, e.g. "exec_match:false"

class Grader(Protocol):  # B implements one per task type
    name: str
    def grade(self, *, prompt_messages: list[dict], response: dict,
              ground_truth: dict) -> Verdict: ...

@dataclass(frozen=True)
class EvalResult:
    item_id: str; model: str; verdict: bool; detail: str
    latency_ms: int; tokens: int
```

## 6. API surface (A implements; B's scripts may call these OR write rows directly)

| Endpoint | Body → Response |
|---|---|
| `POST /v1/chat/completions` | OpenAI payload → proxied response (verbatim bytes); trace logged |
| `GET /traces?task_id=&grade=` | → `{traces: [...]}` |
| `GET /tasks` | → `{tasks: [{task_id,name,n_traces,n_correct,model_state}]}` |
| `POST /grades/run` | `{task_id?, limit?}` → `{graded: n}` (looks up grader by task name in registry) |
| `POST /datasets` | `{task_id, name}` → `{dataset_id, n_train, n_val, n_tokens, dropped_rows}` |
| `POST /distill` | `{dataset_id, suffix?}` → `{job_id}` (starts async: upload→create→poll→deploy→register model as `candidate`) |
| `GET /jobs/{id}` | → `{status, trained_tokens, checkpoint_id, deployed_model?}` |
| `GET /models` | → `[{name, task_id, state}]` |
| `POST /models/{id}/shadow` / `/promote` / `/demote` | lifecycle transitions |
| `POST /evals` | `{benchmark, models: [...], n_items?}` → `{eval_run_id}` (async) |
| `GET /evals/{id}` | → `{scores: {model: acc}, stats: {mcnemar_p, ci}, items: [...]}` |
| `GET /stats/cost` | → `{teacher_usd, pupil_usd, saved_usd}` |
| `POST /replay` | `{trace_id, model}` → `{response, verdict}` |
| `GET /` | dashboard `index.html` |

## 7. SFT JSONL format (B emits; Nebius consumes)

One row per verified trace: `{"messages":[...]}` — messages copied verbatim from the trace request, plus the assistant turn from the response. Preserve `tool_calls` on the assistant row **if** Nebius accepts them (day-0 check on 1 row). Fallback contract: serialize the tool call as JSON in `assistant.content` and keep `tools` in the system prompt — graders score both forms identically.

Every builder run prints: row count, token-length histogram, max row tokens. **Hard assert: max ≤ 7500** (TF-03).

## 8. Promotion gate

`promote` allowed only if BOTH: (a) held-out eval exists for the candidate with `acc(candidate) ≥ acc(teacher) − 0.05` on its task; (b) shadow win-rate over ≥50 calls ≥ teacher − margin. Otherwise 409 with reason. `demote` is unconditional (rollback).

## 9. Hard gates — ANCHORED TO WALL-CLOCK (TF-02, v2)

> ⚠️ **The previous version of this table was relative (`T+0:15` … `T+3:45`) with `T` never anchored.
> Read against the real clock it reproduced TF-02, the exact finding it was written to fix:**
> at T = 14:09 it put deployment at **16:54** and the score table at **17:24**, against judging at
> **16:30** — an 84-minute overrun. Relative gates are estimates wearing a deadline's clothes.
> **These times are absolute. Read the clock; do not re-derive them.**

Back-planned from judging at 16:30 (judges walk the room; be demo-ready, not mid-run):

| By (wall clock) | Must be true |
|---|---|
| **14:25** | `NEBIUS_API_KEY` works; `smoke_nebius.py` green **incl. the `POST /v0/models` deploy probe** (TF-07) |
| **14:40** | xlam licence accepted (`gated: auto` — instant), rows downloaded, `{train,val}.jsonl` written, token audit passed (TF-03) |
| **14:50** | **LoRA job SUBMITTED on `Llama-3.1-8B-Instruct`. HARD DEADLINE — the single irrecoverable gate.** |
| 15:45 | job `succeeded`; checkpoint deployed; `pupil-*` answers one real call |
| 16:05 | 3-model eval complete over held-out items; McNemar + CI computed |
| 16:15 | score table + cost ratio rendered in the dashboard |
| 16:25 | demo rehearsed once, end to end |

**Why 14:50 and not later.** Job wall-clock including queue is unmeasured (TF-11 Q2 — still
unanswered). Deployment must be done by 15:45, so a 55-minute job needs submission by 14:50. If a
mentor gives you the real figure, re-derive this one number and nothing else.

**Everything else is built while the job trains.** The server, router, dashboard, eval harness, BFCL
grader and shadow mode all have ~55 minutes of free wall-clock after submission. Nothing that can be
built during training belongs before the gate.

Order of truth if anything slips: **job submitted > eval numbers > dashboard > second task > shadow mode**.

### 9a. Critical-path change: xlam is the trace source, not the fallback

Verified: each `Salesforce/xlam-function-calling-60k` row carries **`query` + `tools` + `answers`**,
where `answers` holds the ground-truth call(s) as `{name, arguments}`. **The dataset supplies the prompt
AND the ground truth.** Consequences:

1. **Generate teacher traces over xlam prompts and grade against xlam's own `answers`.** The thesis —
   distillation from *verified teacher traces* — is fully preserved. A job trained on raw xlam rows is
   fine-tuning on a public dataset and is a visibly weaker story; do not settle for it unless the clock
   forces it.
2. **The grader for the critical path is a dict comparison**, not a BFCL AST matcher: compare `name`,
   then `arguments` key-by-key, order-insensitive. ~25 lines. B1's full AST matcher is no longer a
   blocker — it moves to the eval path, where it has until 16:05.
3. **BFCL becomes the held-out eval set.** This is *better* than the original plan: train on xlam,
   evaluate on BFCL, and the score table demonstrates **generalisation across distributions** rather
   than memorisation of one. Say that on stage — it pre-empts the obvious judge question.

Revised pre-gate ordering — only these three things exist before 14:50:

```
A0.2 smoke_nebius.py   (deploy probe decides serverless vs dedicated NOW)
B0   xlam -> teacher traces -> graded -> JSONL + token audit
A4   orchestrator / submit_job.py   (the emergency CLI path, no server needed)
```

If traces are not graded by 14:45, submit on raw xlam rows and regenerate v2 during training. **A
submitted job beats a better dataset.**
