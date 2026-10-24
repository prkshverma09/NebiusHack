# TraceForge PRD — Technical Review & Findings

**Reviewed:** 2026-10-24, 13:05 BST (hackathon day, mid-build)
**Against:** `spec/prd-traceforge.html` @ commit `392a149` (v1); TF-12/TF-13 added 13:25 against `c8436f6` (v2)
**Method:** every platform claim re-checked against Nebius Token Factory docs; competitor and
prior-art sweep via web research; mubit.ai landing page + `llms.txt` read directly.

**Label key**

| Label | Meaning |
|---|---|
| `BLOCKER` | Will break the demo or invalidate the headline claim if unaddressed |
| `HIGH` | Likely to cost significant time or credibility |
| `MEDIUM` | Should fix, has a cheap workaround |
| `LOW` | Tuning / polish |
| `VERIFIED-OK` | Checked against primary source, PRD is correct |
| `UNVERIFIED` | Nobody has checked this; needs a day-0 test or a Nebius mentor |

---

## Summary table

| ID | Label | Finding |
|---|---|---|
| TF-01 | `BLOCKER` | §09 is factually wrong: Llama-3.2-1B/3B have **no serverless LoRA deploy path**. The "1B matches frontier" headline is the one config that cannot be cheaply served. |
| TF-02 | `BLOCKER` | Day plan is calibrated to a 7.5h day. The slot it allocates for LoRA job submission (12:30) **has already passed** with nothing built. |
| TF-03 | `HIGH` | The **8k fine-tuning context cap** appears nowhere in the PRD, yet §11 requires verbatim `tools` arrays. With `packing: True` over-length rows truncate **silently**. |
| TF-04 | `HIGH` | Headline metric "within 2 pts of teacher" is **inside the noise band** at n≈400. Not defensible as stated. |
| TF-05 | `HIGH` | Mubit framing is a positioning liability. Use the **idea**, not the brand. Rename `Bonsai` (echoes their `Banyan`). |
| TF-06 | `MEDIUM` | BFCL AST grader is the critical-path time sink and **gates** dataset → job → eval. `xlam-function-calling-60k` should be a primary floor, not optional enrichment. |
| TF-07 | `MEDIUM` | `POST /v0/models` (deploy checkpoint) is the one unverified API call in the PRD, and the whole demo sits behind it. |
| TF-08 | `MEDIUM` | No `NEBIUS_API_KEY` provisioned at time of review. Nothing can start until it exists. |
| TF-09 | `LOW` | 1,500 teacher traces is more than the remaining window supports. 300–400 is sufficient. |
| TF-10 | `INFO` | Nebius **already markets distillation** and ships a Custom Speculator API. Position accordingly — novelty is not the pitch. **Partially corrected — see TF-12.** |
| TF-11 | `UNVERIFIED` | Fine-tune queue wall-clock, fine-tune pricing, and whether Nebius has an internal router/eval product. |
| TF-12 | `CORRECTION` | **Token Factory already has trace capture.** Inference Logs are collected automatically, and Data Lab builds datasets from them via SQL. The real gap is far narrower and much sharper: **there is no verdict field.** |
| TF-13 | `HIGH` | PRD v2 (`c8436f6`) roughly doubles scope — two benchmarks, two LoRA jobs, shadow mode, promotion gate — while remaining time fell to ~3h. |

---

## TF-01 `BLOCKER` — The 1B headline has no serving path

**PRD says** (§09, Student A): `meta-llama/Llama-3.2-1B-Instruct` — *"'1B matches frontier' is the
stronger headline; LoRA + serverless deploy supported"*.

**Docs say** (`docs.tokenfactory.nebius.com/fine-tuning/models`, section *"Base LoRA adapter models
available for deployment"*) — serverless LoRA deployment is supported for **exactly two** base models:

- `meta-llama/Llama-3.1-8B-Instruct` (LoRA and full fine-tuning)
- `meta-llama/Llama-3.3-70B-Instruct` (LoRA)

Llama-3.2-1B and 3.2-3B appear in the **fine-tunable** list but are **absent from the deployable**
list. They can be trained and then served only via a **Dedicated endpoint** (provisioned GPU, hourly
billing, quota required).

§15 half-catches this as a generic risk; §09 asserts the opposite. The two sections contradict.

**Fix:** make **Llama-3.1-8B-Instruct the headline student**. "8B matches a 235B teacher" is still a
strong claim, and it is the only one with a cheap, demonstrable serving path. Keep 1B as a stretch goal
explicitly contingent on a dedicated endpoint.

---

## TF-02 `BLOCKER` — The plan is built for a day that no longer exists

PRD §14 schedules 09:00–16:30 across two parallel tracks and submits the LoRA job at **12:30**.

Actual: build opened 11:30, this review written 13:05, judging 16:30. **~3h25m remains**, not 7.5h,
and the job-submission slot has already elapsed with no code and no API key.

Everything downstream — deploy, eval, score table, the demo's money shot — sits behind that one job.

**Fix — revised critical path (times are hard gates, not estimates):**

| By | Must be true |
|---|---|
| 13:20 | `NEBIUS_API_KEY` works; teacher call returns; `POST /v0/models` smoke-tested (TF-07) |
| 14:00 | Grader runs on a handful of items; train JSONL emitted and length-checked (TF-03) |
| **14:15** | **LoRA job submitted on 8B. Hard deadline.** Nothing after this point can recover a missed job. |
| 15:15 | Checkpoint deployed, endpoint answers one smoke call |
| 15:45 | 3-model eval complete, score table rendered |
| 16:15 | Demo rehearsed once, end to end |

The dashboard is the most cuttable item in the plan and the most tempting to build first. Build it last.

---

## TF-03 `HIGH` — 8k context cap vs verbatim tool schemas

**Docs:** *"The context length for all fine-tuning models is 8k."* This limit is absent from the PRD.

**PRD §11 requires:** *"Persist full traces incl. `tools`/`tool_calls` verbatim"*, and §15 mitigates a
different risk by *"preserve tools array + assistant tool_calls verbatim in JSONL"*.

BFCL `parallel_multiple` items carry several complete function schemas plus a system prompt. Serialised
with the assistant turn, these rows can approach or exceed 8k tokens. Because `packing: True` is set,
**over-length rows do not error — they truncate**, and the student learns to begin a tool call and stop
mid-structure. The failure surfaces only as unexplained low eval accuracy after a 40-minute job.

**Fix:** before submitting, print the token-length distribution of the serialised JSONL and assert the
max is under ~7.5k. Drop or trim offending rows. Two minutes of work; prevents an unexplainable
training run.

---

## TF-04 `HIGH` — The headline metric cannot be measured at this sample size

**PRD §12:** *"≥ −2pts — Bonsai accuracy vs teacher on held-out test split. Target: parity or better;
floor: within 2 points."*

At n≈400 and p≈0.85 the binomial standard error is ~1.8pp, giving a 95% interval of roughly ±3.5pp. A
2-point gap therefore sits **inside the noise**. The number cannot distinguish genuine parity from a
real deficit, and a sharp judge can dismantle it.

**Fix — cheap and strictly better:** you grade *identical items* across models, so use a **paired**
test rather than two independent proportions — McNemar on the per-item agreement table, or a bootstrap
CI over the same items. Paired intervals are substantially tighter and are the statistically correct
comparison for this design.

Also promote **"≥ +15pts over base student"** to the headline. That gap is comfortably outside noise,
and it is the claim that actually proves distillation did something.

---

## TF-05 `HIGH` — Use the idea, not the brand

**We do not use Mubit technically.** No API, no SDK, no dependency; the architecture is FastAPI +
SQLite + Nebius, and §04 states *"the only external dependency is Token Factory."* That is correct and
should stay true.

**But the PRD's framing leans on them heavily:**

- Subtitle: *"A working clone of mubit.ai's core loop"*
- §03 is structured as *"What Mubit does → what TraceForge builds"*
- §17 naming note: Mubit's family is **Banyan**; ours is **Bonsai** — both cultivated trees, a
  transparent echo
- Suggested judge line: *"Mubit sells this as a managed service; we rebuilt the provable core of it"*

**Why this costs us:**

1. It frames original work as a knockoff of a waitlist-stage startup instead of as a **missing Nebius
   capability** — a strictly weaker pitch to Nebius judges.
2. Judges almost certainly do not know Mubit. Explaining who they are burns demo seconds and shifts the
   frame onto someone else's product.
3. Mubit is a real company with a brand. Publicly presenting "a clone of mubit.ai" is needless exposure
   for zero gain.

**Fix:**

- Strip the Mubit frame from the demo script, the README and the dashboard entirely.
- **Rename `Bonsai`** to something with no lineage to `Banyan`.
- Keep Mubit **internal only**, as competitive validation: a funded company whose entire product is the
  layer Token Factory lacks. That is evidence the gap is real — it is not a thing to say on stage.
- Pitch in Nebius's own terms: *"Token Factory can train a student from traces but gives you nothing to
  discover what to train on. We built that half."*

---

## TF-06 `MEDIUM` — The grader is on the critical path and gates everything

Nothing downstream can run until grading works: no grades → no verified traces → no dataset → no job.
A correct BFCL AST comparator is fiddly, and the PRD treats `xlam-function-calling-60k` as *"optional
enrichment … if teacher yield is low"*.

In a 3h window that dependency ordering is backwards. A thin teacher yield discovered at 14:00 leaves
no time to react.

**Fix:** treat `xlam-function-calling-60k` as a **guaranteed dataset floor**, obtained in parallel from
the start (it is already in tool-call format — one download, no grader needed), and let graded teacher
traces be the *quality* layer added on top. The job can then be submitted on time regardless of how
grading goes. Note the HF dataset is gated — accept the licence now, not at 14:10.

---

## TF-07 `MEDIUM` — The deploy call is unverified and everything sits behind it

PRD §07 specifies:

```
POST https://api.tokenfactory.nebius.com/v0/models
{ "source": "{job_id}:{checkpoint_id}",
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "name": "bonsai-bfcl-v1" }
```

This is the only call in the PRD not corroborated against a docs page during this review, and the
entire demo depends on it. Note also that the `base_model` string here
(`Meta-Llama-3.1-8B-Instruct`) differs from the id used in the fine-tune call
(`Llama-3.1-8B-Instruct`) — one of the two is likely wrong.

Compounding risk: a **March 2026 deprecation notice** states *"We are deprecating LoRA per-token
deployments as part of a shift toward … Dedicated Endpoints."* The cheap path may be mid-retirement.

**Fix:** smoke-test the deploy endpoint against *any* existing job/checkpoint before building anything
on it. If it is gone, the dedicated-endpoint fallback needs budgeting at 13:20, not discovering at 15:30.

---

## TF-08 `MEDIUM` — No API key at review time

`NEBIUS_API_KEY` was not set in the environment and no `.env` existed in the project directory. This is
the hard blocker ahead of all others.

---

## TF-09 `LOW` — Reduce teacher trace volume

1,500 train items (§10) is more than the remaining window supports and more than the claim needs.
**300–400 graded traces** is enough to move an 8B student materially on a narrow task, costs less, and
generates faster. Scale up only if time remains after the job is submitted.

---

## TF-10 `INFO` — Nebius already markets distillation; adjust the pitch

Checked, because "is this new to the platform?" determines the pitch:

- Nebius's **post-training factory** page already claims *"built-in distillation and speculative
  decoding pipelines compress powerful teachers into compact students that run 3–5× faster."*
- A **Custom Speculator API** exists today: `method.type="spec-draft"`, `loss: {"type": "kl"}`,
  `context_length: 8192` — real KL-divergence distillation from teacher behaviour, exposed as an API.
- Competitors ship the full idea: **AWS Bedrock Model Distillation** (GA) and **Bedrock Intelligent
  Prompt Routing**; **OpenPipe** (*"swap the SDK"* trace capture → cheap fine-tune, YC, now W&B);
  Fireworks FireOptimizer; Predibase. Routers: Martian, NotDiamond, RouteLLM, OpenRouter Auto.
- Prior art in the literature: NeurIPS **"Gatekeeper"** (confidence tuning for cascades), Google
  **speculative cascades**, DeepSeek-R1-Distill.

**Therefore novelty is not available as a pitch, and should not be attempted.** The defensible claim is
the *gap*.

> ⚠️ **This section's description of the gap was wrong and is superseded by TF-12.** It claimed Nebius
> has "no capture, no eval harness, no router." Nebius *does* have capture — automatic Inference Logs
> plus Data Lab. Read TF-12 for the corrected, narrower gap. The keyword searches behind this paragraph
> ("observability", "trace capture", "evals") missed it because Nebius files it under *Datasets*; a
> depth-2 site map found it immediately. Searching with my own vocabulary rather than the platform's is
> what produced the error.

A credible platform ask to raise with Nebius staff on the day: **expose teacher logprobs on serverless
inference.** Their speculator API trains on KL against teacher distributions, but the chat API returns
only truncated top-k logprobs — so customers cannot do the distillation Nebius itself does internally.

---

## TF-11 `UNVERIFIED` — Ask a Nebius mentor, in person, now

These cannot be resolved by research, and each can reshape the plan. Their clock runs independently of
ours, so the asks go out early.

1. Does this hackathon account have **fine-tuning quota**, and **dedicated-endpoint quota**?
2. **How long does a LoRA job actually take** end to end, including queue? (The only public datapoint
   found — ~10 min for 160 rows — was a self-managed GPU on Nebius AI Cloud, a *different system* with
   no managed queue. It is not evidence about Token Factory.)
3. Is **serverless LoRA deployment still live** for Llama-3.1-8B, given the March 2026 deprecation?
4. What is **fine-tuning priced at** per trained token? No public price card exists.
5. Does Token Factory have an **internal or unreleased router / eval product**? If yes, TF-10's
   positioning needs rework. Research found no public route, which is **not** the same as "it does not
   exist."

---

## TF-12 `CORRECTION` — Nebius already has capture. The gap is the verdict.

**This supersedes part of TF-10.** Checked at 13:20 against
`docs.tokenfactory.nebius.com/data-lab/*` and `/ai-models-inference/observability`.

**What Token Factory already has — more than TF-10 credited:**

- **Inference Logs**: *"automatically collected chat completions generated via API or Playground
  (unless Zero Data Retention is enabled)."* Capture is **on by default**. No gateway required.
- **Data Lab**: a unified workspace over inference logs and datasets, with *"a unified interface for
  preparing datasets via SQL queries"* and *"create datasets from inference logs … select a time range
  and model."*
- **Inference Observability**: latency p50/p90/p99, TTFT, output TPS, KV-cache hit rate, error rate by
  status code, GPU utilisation, active replicas. **Caveat: *"Observability Metrics are only accessible
  for Dedicated Endpoints inference"*** — serverless users get none of it.

So "we built the missing front half of their funnel" is **not true** and must not be said on stage —
Nebius will know, and it is the one claim a Nebius judge is best placed to demolish.

**The actual gap, and it is a single column.** The fields Data Lab imports from inference logs are:

```
id, choice_id, prompt, completion, dialogue, started, completed,
raw_prompt, raw_completion, completion_tokens, prompt_tokens,
ai_project_id, model_flavor_id, tenant_user_account_id, error_string
```

There is **no verdict, grade or outcome field.** `error_string` is transport-level only. Token Factory
records *what was said* and *whether the call succeeded* — never *whether the answer was right*. You can
filter logs by model and time range; you cannot write `WHERE verdict = 'correct'`, because that column
does not exist.

That single absence breaks the loop, because grade-filtering is what separates distillation from
imitation — train on all teacher traces and you copy its mistakes too.

**The platform ask, in one sentence:** *add a writable, SQL-filterable `verdict` field to inference
logs.* Then Data Lab's existing SQL dataset builder becomes a distillation dataset builder, and the
whole loop closes inside Token Factory with no new product surface.

**Ranked novelty for the platform** (strongest first):

| # | Gap | Status |
|---|---|---|
| 1 | **Verdict/outcome on inference logs** — graded, SQL-filterable | Genuinely absent; one column; highest-value ask |
| 2 | **Task fingerprinting** — group logs by task shape (system + tools + output format) | Data Lab filters by *model* and *time*, never by task. PRD v2's fingerprint is a real addition |
| 3 | **Shadow mode / promotion gate** — prove a candidate on live traffic, then swap, with rollback | No safe-swap primitive found. Bedrock has built-in eval; Nebius does not appear to |
| 4 | **Teacher logprobs on serverless inference** | Their speculator trains on `loss: {"type": "kl"}` against teacher distributions, but the chat API returns only truncated top-k — customers cannot replicate the recipe Nebius uses internally |
| 5 | **Router** | No public route found |
| — | Trace capture, dataset-from-logs, distillation, SFT/LoRA, observability metrics | **Already shipped. Claim none of these.** |

**Revised pitch:** *"Token Factory already logs every chat completion and Data Lab already turns logs
into a dataset with SQL. The one column it doesn't have is whether the answer was right — so you can
filter by model and by time, but never by correctness. We added that column and closed the loop."*

That is specific, respectful of what they built, obviously implementable, and credits them with 90% of
the pipeline — a far stronger position than claiming to have filled a hole they do not have.

---

## TF-13 `HIGH` — PRD v2 raised scope while the clock fell

PRD v2 (`c8436f6`) adds a second benchmark (Spider text-to-SQL with exec-match grading), a second LoRA
job, shadow mode, a promotion gate, and task fingerprinting. Scope roughly doubled. Remaining time at
the moment v2 landed: **~3h09m**.

Credit where due — v2 is a better *product* than v1, and it already cites the Data Lab thesis (*"your
best dataset is already in your logs"*), which is the right instinct and precedes TF-12.

But the additions multiply the exposure to TF-02's single unmeasured variable. Two LoRA jobs do not
halve queue risk, they double the number of things that must clear it. And **exec-match SQL grading
needs a live SQLite database per Spider item** — a second grader, with its own failure modes, on the
critical path that already gates everything.

**Recommendation:** keep v2 as the *narrative* and ship v1's scope. One task (BFCL), one LoRA job on
8B. Add the second task only once the first adapter is deployed and evaluated. Shadow mode is a
dashboard ticker over machinery that must already work — it is the last thing to build, not a parallel
track.

If the fleet story matters for the pitch, say it as roadmap: *"the fingerprint makes this per-task
automatically; we promoted one specialist today."* One promoted specialist with real numbers beats two
half-trained ones with a missing score table.

---

## Verified as correct in the PRD

Credit where due — these were checked against primary sources and hold up:

| `VERIFIED-OK` | Evidence |
|---|---|
| SFT API shape | `files.create` → `fine_tuning.jobs.create` with `hyperparameters: {lora, lora_r, lora_alpha, lora_dropout, n_epochs, learning_rate, batch_size, packing}` matches the docs |
| OpenAI-compatible base URL | `https://api.tokenfactory.nebius.com/v1/` — correct; a plain `openai` client works |
| Fine-tunable model list | Llama-3.2-1B/3B, Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Qwen3-14B/32B, gpt-oss-20b/120b, DeepSeek-V3-0324 — all confirmed |
| Qwen3/gpt-oss/DeepSeek deploy via dedicated endpoints only | Confirmed |
| March 2026 LoRA per-token deprecation notice | Confirmed, exists |
| "No LLM-as-judge for the headline metric" | Correct and important — keep this rule |
| Grading before dataset build | Correct, and it is *why* a student can exceed its teacher: filtering to `grade=correct` distils the teacher's successes rather than its full distribution, mistakes included |
| BFCL precedent for small-model parity | xLAM / Hammer sub-4B fine-tunes do beat GPT-4-class models on the public leaderboard — the claim is defensible, not magical |
| Budget headroom | $15–35 against $150 is sound; the contingency line is well judged |
| Scope cut vs Mubit | Implementing capture → verify → distill → evaluate and stubbing memory/refresh/rollback is the right call for one day |

---

## One design question worth having an answer ready for

A judge can reasonably ask: *"your grade comes from a benchmark's ground truth — where does it come
from in production?"*

The answer is the pluggable `Grader` interface (§6.3), which is genuinely the extensible part of the
design. Mubit's own FAQ makes the same point from the other direction, and it is worth internalising:
*"a successful API response alone does not prove the answer was correct."*
