"""Eval runner + paired statistics. Contract §5/§6, TF-04.

Every model sees the SAME items with the SAME messages, so comparisons are
paired. Headline stat is McNemar (exact binomial over discordant pairs) plus a
paired bootstrap CI on the accuracy delta — not "within N pts of teacher".

Transport failures are verdict "error", never "incorrect": an outage must not
read as a model being wrong. Items that errored for EITHER model of a pair are
excluded from that pair's stats, and the exclusion count is reported.

Responses are cached per (model, item) under data/evals/cache/, so a run
interrupted by a 503 storm resumes and only re-calls what failed.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import math
import random
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traceforge.config import REPO_ROOT
from traceforge.schemas import EvalResult, Grader

CACHE_DIR = REPO_ROOT / "data" / "evals" / "cache"
RUNS_DIR = REPO_ROOT / "data" / "evals"


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    messages: list[dict]       # prompt only: everything before the gold assistant turn
    ground_truth: Any          # gold, in whatever form the grader accepts


def load_holdout(path: str | Path, limit: int | None = None) -> list[EvalItem]:
    """Holdout JSONL rows are {"messages":[system,user,assistant]}; the final
    assistant turn is the gold answer and is stripped from the prompt."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            raise ValueError(f"holdout row without a final assistant turn: {line[:80]}")
        prompt = msgs[:-1]
        iid = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()[:12]
        items.append(EvalItem(iid, prompt, msgs[-1]["content"]))
        if limit and len(items) >= limit:
            break
    return items


# --------------------------------------------------------------------------- calling
def _cache_path(model: str, item_id: str) -> Path:
    return CACHE_DIR / model.replace("/", "__") / f"{item_id}.json"


def call_model(client, model: str, item: EvalItem, *, max_tokens: int = 1024,
               retries: int = 4) -> dict:
    """Returns {"response": dict|None, "latency_ms", "tokens", "error": str|None}.

    Only successful responses are cached; errors are retried on the next run.
    """
    cp = _cache_path(model, item.item_id)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    last = "unknown"
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=model, messages=item.messages,
                                               temperature=0, max_tokens=max_tokens)
            out = {"response": r.model_dump(), "latency_ms": int((time.time() - t0) * 1000),
                   "tokens": (r.usage.total_tokens if r.usage else 0), "error": None}
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(out), encoding="utf-8")
            return out
        except Exception as e:  # noqa: BLE001 — classified below, never swallowed
            status = getattr(e, "status_code", None)
            last = f"{type(e).__name__}:{status}"
            if status is not None and status < 500 and status != 429:
                break                       # 4xx (bad model id, auth) won't heal on retry
            time.sleep(min(2 ** attempt, 8) + random.random())
    return {"response": None, "latency_ms": 0, "tokens": 0, "error": last}


def grade_call(grader: Grader, item: EvalItem, out: dict) -> tuple[str, str]:
    """The ONE place a model call becomes a verdict: correct | incorrect | error.

    Shared by run_eval and scripts/gen_traces.py so both paths get the same
    guards. Two things are `error`, never `incorrect`, because neither says
    anything about the model: a transport failure, and a gold label the grader
    cannot parse (3% of ToolACE train is malformed, e.g. `name=>(k="v"; ...)`).
    """
    if out["error"]:
        return "error", out["error"]
    try:
        v = grader.grade(prompt_messages=item.messages, response=out["response"],
                         ground_truth=item.ground_truth)
    except ValueError as e:                 # DSLParseError subclasses ValueError
        return "error", f"gold_unparseable:{str(e)[:80]}"
    return ("correct" if v.correct else "incorrect"), v.detail


def run_eval(models: list[str], items: list[EvalItem], grader: Grader, client, *,
             workers: int = 16, progress: bool = True) -> list[EvalResult]:
    """Run every (model, item) pair concurrently and grade it.

    `EvalResult.verdict` is a bool per contract §5; errors are flagged via a
    `detail` prefix of "error:" and excluded from stats by `paired_stats`.
    """
    jobs = [(m, it) for m in models for it in items]
    results: list[EvalResult] = []
    done = 0

    def one(m: str, it: EvalItem) -> EvalResult:
        out = call_model(client, m, it)
        verdict, detail = grade_call(grader, it, out)
        if verdict == "error":
            return EvalResult(it.item_id, m, False, f"error:{detail}", 0, 0)
        return EvalResult(it.item_id, m, verdict == "correct", detail,
                          out["latency_ms"], out["tokens"])

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in cf.as_completed([ex.submit(one, m, it) for m, it in jobs]):
            results.append(fut.result())
            done += 1
            if progress and (done % 50 == 0 or done == len(jobs)):
                errs = sum(is_error(r) for r in results)
                print(f"  [{done}/{len(jobs)}] errors so far: {errs}", flush=True)
    return results


# --------------------------------------------------------------------------- stats
def is_error(r: EvalResult) -> bool:
    return r.detail.startswith("error:")


def accuracy(results: list[EvalResult], model: str) -> dict:
    rs = [r for r in results if r.model == model]
    ok = [r for r in rs if not is_error(r)]
    n_ok = sum(r.verdict for r in ok)
    lat = sorted(r.latency_ms for r in ok)
    return {"model": model, "n": len(ok), "correct": n_ok, "errors": len(rs) - len(ok),
            "acc": (n_ok / len(ok)) if ok else None,
            "median_latency_ms": (lat[len(lat) // 2] if lat else None)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: binomial(b+c, 0.5) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_stats(results: list[EvalResult], model_a: str, model_b: str, *,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Compare B against A over items where NEITHER errored."""
    by = {(r.model, r.item_id): r for r in results}
    ids = sorted({r.item_id for r in results})
    pairs = [(by[(model_a, i)], by[(model_b, i)]) for i in ids
             if (model_a, i) in by and (model_b, i) in by
             and not is_error(by[(model_a, i)]) and not is_error(by[(model_b, i)])]
    n = len(pairs)
    excluded = len(ids) - n
    if n == 0:
        return {"a": model_a, "b": model_b, "n_paired": 0, "excluded": excluded,
                "delta": None, "ci95": None, "mcnemar_p": None, "b_only": 0, "a_only": 0}
    a_only = sum(a.verdict and not b.verdict for a, b in pairs)    # A right, B wrong
    b_only = sum(b.verdict and not a.verdict for a, b in pairs)    # B right, A wrong
    diffs = [int(b.verdict) - int(a.verdict) for a, b in pairs]
    delta = sum(diffs) / n
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    ci = (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1])
    return {"a": model_a, "b": model_b, "n_paired": n, "excluded": excluded,
            "delta": delta, "ci95": ci, "mcnemar_p": mcnemar_exact(a_only, b_only),
            "b_only": b_only, "a_only": a_only}


FAILURE_KINDS = {
    "wrong_values": "right function, wrong argument value",
    "missing_args": "right function, required argument missing",
    "extra_args": "invented a parameter not in the schema",
    "no_match_for": "called the wrong function",
    "call_count": "wrong number of calls (extra or missing)",
    "no_call_emitted": "answered in prose / empty list instead of calling",
    "irrelevance": "called a tool when it should have abstained",
    "unparseable_call": "emitted a malformed call",
}


def failure_kind(detail: str) -> str:
    """Map a grader detail string to one taxonomy bucket.

    Details look like "Fn Name:wrong_values:arg" or "call_count:5!=2"; the kind
    is the first known token, since function names may themselves contain ':'.
    """
    for kind in FAILURE_KINDS:
        if detail.startswith(kind) or f":{kind}" in detail:
            return kind
    return "other"


def failure_taxonomy(results: list[EvalResult], models: list[str]) -> dict:
    """{model: {kind: count}} over graded failures (errors excluded)."""
    out = {}
    for m in models:
        c = collections.Counter(failure_kind(r.detail) for r in results
                                if r.model == m and not is_error(r) and not r.verdict)
        out[m] = dict(c.most_common())
    return out


_CAT_ID = re.compile(r"^([a-z_]+?)_\d+$")


def by_category(results: list[EvalResult], models: list[str]) -> dict:
    """{category: {model: acc}} for benchmarks whose ids are "<category>_<n>" (BFCL).
    Returns {} when ids carry no category (ToolACE prompt hashes)."""
    cats: dict[str, dict[str, list[bool]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        m = _CAT_ID.match(r.item_id)
        if not m:
            return {}
        if not is_error(r):
            cats[m.group(1)][r.model].append(r.verdict)
    return {c: {mo: (sum(v) / len(v) if v else None) for mo, v in per.items()} for c, per in cats.items()}


def failure_shortlist(results: list[EvalResult], weak: str, strong: str, k: int = 5) -> list[dict]:
    """Items the strong model gets right and the weak one gets wrong — demo replay candidates."""
    by = {(r.model, r.item_id): r for r in results}
    out = []
    for (m, i), r in sorted(by.items()):
        if m != weak or is_error(r) or r.verdict:
            continue
        s = by.get((strong, i))
        if s and not is_error(s) and s.verdict:
            out.append({"item_id": i, "weak_detail": r.detail})
    return out[:k]


# --------------------------------------------------------------------------- report
def score_table(results: list[EvalResult], models: list[str], baseline: str) -> dict:
    accs = [accuracy(results, m) for m in models]
    comps = [paired_stats(results, baseline, m) for m in models if m != baseline]
    lines = ["| model | acc | correct/n | errors | p50 latency |", "|---|---|---|---|---|"]
    for a in accs:
        acc = f"{a['acc']:.1%}" if a["acc"] is not None else "n/a"
        lat = f"{a['median_latency_ms']} ms" if a["median_latency_ms"] is not None else "n/a"
        lines.append(f"| `{a['model']}` | **{acc}** | {a['correct']}/{a['n']} | {a['errors']} | {lat} |")
    lines += ["", f"Paired vs `{baseline}` (items where neither errored):", "",
              "| model | Δ acc | 95% CI | McNemar p | wins/losses | n paired | excluded |",
              "|---|---|---|---|---|---|---|"]
    for c in comps:
        if c["delta"] is None:
            lines.append(f"| `{c['b']}` | n/a | n/a | n/a | – | 0 | {c['excluded']} |")
            continue
        lo, hi = c["ci95"]
        lines.append(f"| `{c['b']}` | {c['delta']:+.1%} | [{lo:+.1%}, {hi:+.1%}] "
                     f"| {c['mcnemar_p']:.3g} | {c['b_only']}/{c['a_only']} | {c['n_paired']} | {c['excluded']} |")
    cats = by_category(results, models)
    if cats:
        lines += ["", "By category:", "",
                  "| category | " + " | ".join(f"`{m.split('/')[-1]}`" for m in models) + " |",
                  "|---|" + "---|" * len(models)]
        for c in sorted(cats):
            cells = [f"{cats[c][m]:.0%}" if cats[c].get(m) is not None else "n/a" for m in models]
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    tax = failure_taxonomy(results, models)
    kinds = [k for k in FAILURE_KINDS if any(k in t for t in tax.values())] + \
            (["other"] if any("other" in t for t in tax.values()) else [])
    if kinds:
        lines += ["", "Where they fail (graded failures, errors excluded):", "",
                  "| failure | " + " | ".join(f"`{m.split('/')[-1]}`" for m in models) + " |",
                  "|---|" + "---|" * len(models)]
        for k in kinds:
            label = FAILURE_KINDS.get(k, "unclassified")
            lines.append(f"| {label} | " + " | ".join(str(tax[m].get(k, 0)) for m in models) + " |")
    bars = []
    for a in accs:
        if a["acc"] is not None:
            name = a["model"][-34:]
            bars.append(f"{name:>34} {'#' * round(a['acc'] * 40):<40} {a['acc']:.1%}")
    return {"accuracy": accs, "paired": comps, "taxonomy": tax, "by_category": cats,
            "markdown": "\n".join(lines), "ascii": "\n".join(bars)}


def persist(results: list[EvalResult], table: dict, *, benchmark: str, models: list[str],
            db_path: Path | None = None) -> str:
    """Write the run to data/evals/<id>.json always, and to sqlite eval_runs /
    eval_items when A's tables exist (contract §4). Returns eval_run_id."""
    run_id = f"ev-{uuid.uuid4().hex[:8]}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"eval_run_id": run_id, "benchmark": benchmark, "models": models,
               "created_at": time.time(), "table": table,
               "items": [asdict(r) for r in results]}
    # Atomic: readers (tests, A's dashboard) must never see a half-written run file.
    tmp = RUNS_DIR / f".{run_id}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(RUNS_DIR / f"{run_id}.json")
    if db_path and Path(db_path).exists():
        con = sqlite3.connect(db_path, timeout=60)    # shared with A's server + gen_traces
        try:
            have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {"eval_runs", "eval_items"} <= have:
                con.execute("INSERT INTO eval_runs VALUES (?,?,?,?,?,?,?)",
                            (run_id, benchmark, json.dumps(models), len({r.item_id for r in results}),
                             json.dumps(table["accuracy"]), json.dumps(table["paired"]), time.time()))
                con.executemany(
                    "INSERT INTO eval_items(eval_run_id,item_id,model,verdict,detail,latency_ms,tokens)"
                    " VALUES (?,?,?,?,?,?,?)",
                    [(run_id, r.item_id, r.model,
                      "error" if is_error(r) else ("correct" if r.verdict else "incorrect"),
                      r.detail, r.latency_ms, r.tokens) for r in results])
                con.commit()
        finally:
            con.close()
    return run_id
