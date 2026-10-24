# Day-0 findings against the real Token Factory API (14:20–14:46, 24 Oct)

Measured, not assumed. Each item overrides `plans/CONTRACT.md` where they conflict.
Evidence: `scripts/smoke_nebius.py`, live OpenAPI spec (`/openapi.json`, v20261022).

## Contract changes

| Contract said | Reality | Now |
|---|---|---|
| Student `Llama-3.1-8B-Instruct` (TF-01) | Trainable, but **no Llama-8B endpoint flavor accepts custom weights** — unservable | **Student = `Qwen/Qwen3-30B-A3B-Instruct-2507`** (LoRA) |
| Deploy via `POST /v0/models` (TF-07) | **410** "LoRA adapter deployment is no longer available"; route not in spec | `POST /v0/dedicated_endpoints` + `custom_weights_id` (`model-artifact_...`) |
| Teacher `DeepSeek-V3-0324` | `model_not_found` | **Teacher = `Qwen/Qwen3-235B-A22B-Instruct-2507`** (same family as student) |
| Trace source xlam-60k (9a) | Gated; account not yet authorised (403) | **ToolACE** (`Team-ACE/ToolACE`, ungated) |
| Assistant turn = JSON calls | ToolACE uses its own call DSL | e.g. `[Market Trends API(trend_type="MARKET_INDEXES", country="us")]` — **B's grader must parse this** |

## Jobs running

| Job | Base | Hyperparams | Submitted |
|---|---|---|---|
| `ftjob-70b3094d7e4f4d9093bfc5b82245f8bd` | Qwen3-30B-A3B-Instruct-2507 | LoRA r16/α16, lr 2e-4, 5 ep, batch 16, packing | 14:45:35 — **primary** |
| `ftjob-dc1fa2a7a68842e7914e6579c14f859c` | Llama-3.1-8B-Instruct | LoRA r16/α16, lr 1e-4, 3 ep, batch 8, packing | 14:33:38 — hedge, unservable on Nebius |

Qwen requires `batch_size × context_length ≥ 131072` (hence batch 16).

## Dataset — `data/datasets/toolace-fc-v1/` (build: `scripts/build_xlam_dataset.py --source toolace`)

- train 1200 / val 150 / **holdout 250 — never trained on; the eval set**
- single-turn rows only (user → one assistant call); refusals/clarifications dropped
- token audit (TF-03): max 2582, p99 1574 — far under 7500

## Still open

1. **How a fine-tuning checkpoint becomes a `model-artifact_` id.** No route in the spec. Docs call custom weights "beta, on request" — but the API already answers "ID does not exist" rather than a permission error, so it may be enabled. Ask a Nebius mentor now.
2. Serverless inference is **intermittently 503** for all models (worked 14:37, down 14:41). Trace generation and evals need retry/backoff.
3. Dedicated-endpoint $/hour for H200 unmeasured — read billing docs before quoting cost.
