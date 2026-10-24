# Hacker A — Platform (gateway, store, router, orchestrator, API, dashboard)

> You build the **product machinery**. Hacker B builds benchmarks/graders/evals in parallel — you never wait on them: your only coupling is `plans/CONTRACT.md` (schema, fingerprint, endpoints) and the grader `Protocol` they implement to.
>
> **Read first:** `plans/CONTRACT.md` + `spec/REVIEW-traceforge-findings.md` (esp. TF-01/03/05/07/12/13).
> Ship order per review: job submitted > eval numbers > dashboard > second task > shadow mode.

## Your files (nobody else touches)

`traceforge/{__init__,config,schemas,db,fingerprint,costs,gateway,router,orchestrator,server}.py`,
`dashboard/index.html`, `scripts/{smoke_nebius,submit_job}.py`, `tests/fakes.py` + your unit tests.

## Work packets — parallelize with subagents

Packets in the same wave are file-disjoint and designed to run as **parallel background subagents**.
Rule: never let two agents edit the same file; each packet below names exactly one owner file.

### Wave 0 — sequential, do yourself first (~15 min, unblocks everything)

- **A0.1 Scaffold + env**: `uv venv`, `pip install fastapi uvicorn openai httpx pytest`; package skeleton; `config.py` loading `NEBIUS_API_KEY` etc.; `ConfigError` on missing keys.
- **A0.2 `scripts/smoke_nebius.py`** — *day-0 verification (TF-07/08), highest-value 15 min of the day:*
  1. `GET /models` on Token Factory with the key.
  2. One teacher chat call → confirm model id works.
  3. `client.files.create` with a 3-row dummy JSONL → confirm upload.
  4. **Deploy-path probe**: check docs/console for `POST /v0/models` shape; if a finished job exists anywhere, attempt deploy. Record result — decides serverless vs dedicated-endpoint fallback **now**, not at 15:00.
  5. Print which of `Llama-3.1-8B-Instruct` appears in the deployable-LoRA list.
- **A0.3 `schemas.py` + `db.py` skeleton** — contract §4/§5 verbatim. Push early if sharing a repo; B codes against it.

### Wave 1 — three parallel subagents

| Pkt | File | Spec | Acceptance |
|---|---|---|---|
| **A1** | `db.py` | Full DDL from contract §4; one connection + RLock, WAL; helpers: `insert_trace`, `update_grade`, `traces(task_id,grade)`, `tasks()`, `record_model_state`, cost aggregates | `pytest tests/test_db.py` green against tmp sqlite |
| **A2** | `fingerprint.py` + `costs.py` | Contract §3 fingerprint (canonical JSON, sha256[:12]); price table per-model ($/M in/out), `cost_usd(model, usage)` | Snapshot test: same request → same 12-hex id; price for unknown model → None |
| **A3** | `gateway.py` | `async def proxy(request) -> Response`: forwards to `{UPSTREAM_BASE_URL}/chat/completions` via httpx (pass body bytes through), extracts usage + `x-request-id` if present, writes trace row (`source='proxy'`), returns response bytes verbatim. Fail-open on log errors. | Unit test with `httpx.MockTransport`: response identical, row written |

### Wave 2 — three parallel subagents

| Pkt | File | Spec | Acceptance |
|---|---|---|---|
| **A4** | `orchestrator.py` | `submit(dataset_id, paths)` → `files.create`×2 → `fine_tuning.jobs.create(model=STUDENT_BASE, hyperparameters={lora:true, lora_r:16, lora_alpha:16, lora_dropout:0.05, n_epochs:3, learning_rate:1e-5, batch_size:8, packing:true}, suffix=...)`; `poll(job_id)` loop; `deploy(checkpoint)` → `POST /v0/models`; `register_model(task_id)` → row state=`candidate`. Also standalone `scripts/submit_job.py` CLI wrapping it (emergency path if server isn't up by 14:15 gate) | Dry-run unit test with mocked client asserts call sequence + hyperparams; CLI `--help` works |
| **A5** | `router.py` | `pick_upstream(task_id)` → teacher | pupil-name | shadow pair. Lifecycle fns `to_shadow/to_promote/to_demote` enforcing contract §8 gate (reads eval stats + shadow verdicts). Shadow fan-out: fire-and-forget `asyncio` task answering same prompt on candidate, grading via injected grader registry, writing `grades` row with `source='shadow'` | Gate unit tests: no eval → 409; acc within margin → allowed |
| **A6** | `server.py` + `dashboard/index.html` | `create_app(services)` factory (importable without env — inject clients); all endpoints from contract §6; grader registry injected (defaults to empty → `grades/run` returns 501 until B lands); dashboard = one static file: trace feed, task groups, model states, cost counters, score bars from `GET /evals/{id}` | `TestClient(create_app(...))` covers endpoint table; page renders with empty DB |

### Wave 3 — you, on the critical path

- **A7 Deploy smoke**: when B's first JSONL lands (or dummy file), run the real `submit→poll→deploy` path on a tiny dataset (3 rows) **if clock permits** — this is the TF-07 verification. If serverless LoRA deploy fails → pivot: dedicated endpoint call or score via batch before teardown; record decision in `data/STATUS.md`.
- **A8 `tests/fakes.py`**: `FakeNebius` (files/jobs/chat), `FakeGrader` — shared test doubles both hackers may import.

## Parallelization cheat-sheet

```
Wave 0: yourself          → scaffold, smoke_nebius, schemas+db skeleton
Wave 1: 3 bg subagents    → A1 db, A2 fingerprint+costs, A3 gateway
Wave 2: 3 bg subagents    → A4 orchestrator, A5 router, A6 server+dashboard
Wave 3: yourself          → A7 deploy smoke, A8 fakes, wiring, integration
```

## Interfaces you consume from B (do NOT build)

- `traceforge/graders/registry` — `get_grader(task_name) -> Grader` (inject into server; absent → grading disabled)
- `traceforge/eval.py::run_eval(models, items, grader) -> stats` — called by your `/evals` endpoint if present; until then `/evals` returns 501
- Trace JSONL at `data/datasets/<name>/{train,val}.jsonl` (paths recorded in `datasets` table)

## Definition of done (your half)

- [ ] `smoke_nebius.py` green, deploy path known & documented
- [ ] Proxy returns byte-identical responses and writes complete trace rows
- [ ] `submit_job.py` submitted a real job before the hard gate
- [ ] Deployed `pupil-*` model answers a real call
- [ ] All contract §6 endpoints implemented (501s allowed only for B-owned bits)
- [ ] Dashboard loads from `GET /` and renders real numbers
- [ ] No Mubit strings anywhere; models named `pupil-*` (TF-05)
