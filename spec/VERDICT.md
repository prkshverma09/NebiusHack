# TraceForge — Go / No-Go Verdict

**Written:** 2026-10-24 13:42 BST · **Judging:** 16:30 · **Remaining: 2h47m**
**Against:** `spec/prd-traceforge.html` @ `c8436f6` (v2)
**Companion:** `spec/REVIEW-traceforge-findings.md` (TF-01…TF-13, the technical layer)

---

## Verdict in one line

**The idea is strong and unusually well-matched to this specific hackathon. The PRD as written is
not shippable in the time remaining — not by a margin that can be closed by working faster.**

Ship a hard cut (Part 4). The cut keeps the only genuinely novel part and takes the single
unmeasured variable off the critical path.

---

## Part 1 — Is it a good idea for *this* hackathon?

**Yes, on fit. This is close to the ideal project for this room.**

| Event brief says | TraceForge |
|---|---|
| *"Put open models to work"* | Entire thesis: replace a frontier model with an open student |
| *"A working example with a clear purpose"* | A working API + measurable before/after, nothing simulated |
| *"Show where the models struggle"* | Failure taxonomy + base-student misses are the core visual |
| *"a mini-benchmark comparing how different open models perform"* | Literally the score table: base / distilled / teacher |
| Judges include **Nebius** staff | Demo is a Token Factory showcase and ends in a concrete platform ask |

Two further points in its favour:

1. **The result is likely, not hopeful.** xLAM-1B and Hammer-3B fine-tunes genuinely beat
   GPT-4-class models on the public BFCL leaderboard. The headline is precedented, not magical.
2. **No LLM-as-judge anywhere in the headline metric.** Deterministic grading is the single
   biggest credibility advantage over the median hackathon demo, and the PRD gets this right.

**The honest weaknesses, in order of severity:**

1. **Scope vs clock (decisive).** See Part 3.
2. **Novelty is not available as a pitch.** OpenPipe (YC, now W&B) is trace-capture → cheap
   fine-tune, shipped. AWS Bedrock Model Distillation is GA. Fireworks FireOptimizer and Predibase
   likewise. The market answer to "has this been done" is yes, thoroughly.
3. **The headline metric sits inside its own noise band.** "Within 2 pts of teacher" at n≈400 and
   p≈0.85 carries a ±3.5pp interval. Use a paired comparison and lead with **"+15 pts over base
   student"** instead — that gap is real, and it is the claim that proves distillation did
   something. (TF-04.)
4. **The strongest headline has no cheap serving path.** Llama-3.2-1B is fine-tunable but is *not*
   in Token Factory's serverless-LoRA-deployable list; only Llama-3.1-8B and 3.3-70B are. "1B
   matches 235B" is the one config that cannot be served without a dedicated endpoint. (TF-01.)

---

## Part 2 — Are the core features already present?

Two different questions, two different answers. Both matter.

### 2a. In the repo: **no. Nothing is built.**

```
README.md                           14 bytes
spec/prd-traceforge.html            PRD v2
spec/REVIEW-traceforge-findings.md  the review
```

Zero `.py`, zero `.jsonl`, no `.env`, and `NEBIUS_API_KEY` is still not set. Measured at 13:42, not
inferred. All nine `Must` functional requirements are unstarted.

### 2b. On the platform: **yes — roughly half of it is already Nebius product.**

This is the finding that should reshape the pitch, and the one a Nebius judge is best placed to
test. Checked against `docs.tokenfactory.nebius.com`:

| TraceForge "core feature" | Already on Token Factory? |
|---|---|
| Trace capture | **Shipped.** Inference Logs are collected *automatically* for every API/Playground completion unless ZDR is enabled. No gateway needed. |
| Traces → training dataset | **Shipped.** Data Lab: *"create datasets from inference logs"*, prepared via SQL. |
| Distillation | **Marketed.** The post-training factory page claims built-in distillation + speculative decoding. |
| KL-distillation from a teacher | **Shipped as API.** Custom Speculator: `method.type="spec-draft"`, `loss: {"type": "kl"}`. |
| LoRA / SFT, deploy, call your own weights | **Shipped.** |
| Latency / throughput observability | **Shipped** — but Dedicated Endpoints only. |
| **Verdict / grade / outcome on a log** | **ABSENT.** The log schema has `error_string` (transport-level only). No correctness field. |
| **Task fingerprinting** (group logs by task shape) | **Absent.** Data Lab filters by model and by time, never by task. |
| **Shadow mode + promotion gate + rollback** | **Absent.** No safe-swap primitive found. |
| Router | No public route found. |

**So the "we built the missing front half of their funnel" framing in the PRD is false, and must not
be said on stage.** The defensible version is sharper and considerably better:

> *"Token Factory already logs every completion, and Data Lab already turns those logs into a
> dataset with SQL. The one column it doesn't have is whether the answer was **right** — so you can
> filter by model and by time, but never by correctness. We added that column and closed the loop."*

That credits Nebius with 90% of the pipeline, is obviously implementable as one field, and leaves
TraceForge owning exactly the three things that *are* missing: **verdict, fingerprint, promotion
gate.**

Corollary for the build: **the promotion gate is the differentiator, and it is roughly 50 lines of
state plus a threshold.** It should not be the thing that gets cut.

---

## Part 3 — Why the PRD cannot ship today, specifically

The binding constraint is not feature count. It is **LoRA fine-tune queue wall-clock on Token
Factory, which nobody has measured.** Every item in the money shot — deploy, eval, score table,
shadow ticker, promotion flip — sits downstream of one job finishing.

PRD v2 schedules job submission at **12:30**. That has passed. v2 also *doubles* v1's scope: two
benchmarks, two graders (one needing a live SQLite DB per item), two LoRA jobs, shadow mode,
promotion gate, Data Lab push, Pareto chart. Two jobs do not halve queue risk — they double the
number of things that must clear it.

With 2h47m left and no key, the realistic submit-by gate is **14:20**: about 35 minutes to get a
key, produce a valid JSONL, and submit. Nothing else belongs on the critical path until that job
is in.

---

## Part 4 — The cut I recommend

**Principle: build the machinery so that *either* outcome is a demo.** The gateway, grader, eval
harness and dashboard are identical whether or not a trained adapter exists. That makes the
fine-tune **upside** rather than a single point of failure.

### Critical path (hard gates, not estimates)

| By | Must be true |
|---|---|
| **14:00** | Key works. Teacher call returns. `POST /v0/models` smoke-tested against *any* existing checkpoint — this call is uncorroborated and everything sits behind it (TF-07). |
| **14:20** | **ONE LoRA job submitted on `Llama-3.1-8B-Instruct`. Hard deadline.** |
| 15:10 | Gateway + trace store + grader + eval harness running against teacher and base student |
| 15:35 | Checkpoint deployed; one smoke call answers |
| 15:55 | 3-model eval complete; score table rendered; promotion gate evaluated on real numbers |
| 16:15 | Rehearsed once, end to end |

### To make 14:20 reachable, take the grader off the critical path

Use **`xlam-function-calling-60k`** — already in tool-call format, so the *training* JSONL needs no
grader at all. Accept the HF licence now; it is gated. The grader is then needed only for *eval*,
where you need it regardless, and it gets built while the job trains. This single change is what
makes the 14:20 gate achievable.

Before submitting: **print the token-length distribution and assert max < ~7.5k.** The fine-tuning
context cap is 8k, and with `packing: True` over-length rows **truncate silently** rather than
erroring — the student learns to begin a tool call and stop mid-structure, and you discover it as
unexplained low accuracy 40 minutes later. (TF-03.)

### Keep

- One task (tool calls), one LoRA job, **Llama-3.1-8B** as the student
- Gateway + fingerprint + trace store + cost ledger
- One deterministic grader, for eval
- 3-bar score table on held-out items, **paired** comparison (McNemar, or bootstrap over the same items)
- **Shadow mode + promotion gate + rollback** — the differentiator, and cheap
- The verdict-column insight as the closing platform ask

### Cut

Spider / second task · second LoRA job · SQL exec-grader · Llama-3.2-1B student · Data Lab push ·
Pareto chart · failure taxonomy · `/replay` · live judge input · v2 staircase retrain

Keep the multi-task story as **narrative**: *"the fingerprint makes this per-task automatically; we
promoted one specialist today."* One promoted specialist with real numbers beats two half-trained
ones and a missing score table.

### Insurance (build it anyway — it is free)

If the queue overruns, the same machinery still demos an **escalation gate**: route by fingerprint,
grade both models, show where the small model fails and when to escalate to the teacher. No
fine-tuning required, every number still real. A weaker story, but a *working demo* — and it costs
nothing extra, because it reuses the gateway, grader and eval harness you were building anyway.

---

## Part 5 — Framing fixes before anyone presents this

1. **Strip the Mubit frame entirely** from the demo, README and dashboard. It positions original
   work as a clone of a waitlist-stage startup rather than as a missing Nebius capability —
   strictly weaker in front of these judges, and it burns demo seconds explaining a company nobody
   in the room knows. Rename `Bonsai`; it transparently echoes Mubit's `Banyan`. Keep Mubit
   internal, as evidence the gap is real. (TF-05.)
2. **Do not claim novelty. Claim the gap.** See Part 2b.
3. **Lead with "+15 pts over base student"**, not "within 2 of teacher".

---

## Still unresolved — ask a Nebius mentor in person, now

Their clock runs independently of ours, so these go out before anything else.

1. Does this account have **fine-tuning quota**? Dedicated-endpoint quota?
2. **How long does a LoRA job actually take end to end, including queue?** This is the variable the
   whole plan hangs on, and there is no public datapoint for Token Factory specifically.
3. Is **serverless LoRA deployment still live** for Llama-3.1-8B, given the March 2026 per-token
   LoRA deprecation notice?
4. Does Token Factory have an internal or unreleased **router / eval product**? If yes, Part 2b's
   positioning needs rework. Research found no public route — which is not the same as "none
   exists".
