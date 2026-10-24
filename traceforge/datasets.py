"""Verified traces -> SFT JSONL (B3). Contract §7, TF-03.

Only traces whose verdict is `correct` become training rows: this is the
"verdict column" doing its job. The model's own answer is the assistant turn
(distillation), not the dataset's gold label.

Guards, each a hard failure rather than a warning:
  * leakage   — no training prompt may appear in the eval holdout
  * TF-03     — every row <= max_tokens; `packing: true` truncates silently
  * empty     — refuses to write a dataset with zero rows

  python -m traceforge.datasets --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
      --name toolace-verified-v2 --holdout data/datasets/toolace-fc-v1/holdout.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

from traceforge.config import REPO_ROOT, load_settings
from traceforge.db import DB  # A-owned; B only reads through it
from traceforge.eval import load_holdout

OUT_ROOT = REPO_ROOT / "data" / "datasets"


class DatasetError(RuntimeError):
    """A guard failed. The dataset is NOT written."""


def _prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:12]


def count_tokens(row: dict, enc) -> int:
    """Same accounting as A's build_xlam_dataset.count_tokens (content + 4/msg + 2),
    so v1 and v2 audits are comparable."""
    return sum(len(enc.encode(m.get("content") or "")) + 4 for m in row["messages"]) + 2


def build(*, model: str, name: str, holdout: str | Path, db_path: str | Path,
          max_tokens: int = 7500, val_frac: float = 0.1, task_id: str | None = None,
          source: str | None = None) -> dict:
    """Build {train,val}.jsonl from `model`'s correct traces. Returns the manifest.

    `source` restricts to one writer ('runner' = gen_traces, 'proxy' = A's gateway),
    so ad-hoc gateway test calls do not leak into a curated training set.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    rows = [r for r in DB(db_path).traces(task_id=task_id, limit=1_000_000)
            if r["model"] == model and (source is None or r["source"] == source)]
    verdicts = collections.Counter(r["grade"] for r in rows)
    holdout_hashes = {it.item_id for it in load_holdout(holdout)}   # item_id == prompt hash

    kept, seen, dropped, lengths = [], set(), collections.Counter(), []
    leaked = []
    for r in rows:
        if r["grade"] != "correct":
            continue
        req = json.loads(r["request_json"])
        prompt = req["messages"]
        h = _prompt_hash(prompt)
        if h in holdout_hashes:
            leaked.append(h)
            continue
        if h in seen:
            dropped["duplicate_prompt"] += 1
            continue
        msg = (json.loads(r["response_json"] or "{}").get("choices") or [{}])[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        if not content:
            dropped["no_text_answer"] += 1       # e.g. native tool_calls with empty content
            continue
        row = {"messages": prompt + [{"role": "assistant", "content": content}]}
        n = count_tokens(row, enc)
        if n > max_tokens:
            dropped["over_token_cap"] += 1
            continue
        seen.add(h)
        lengths.append(n)
        kept.append((h, row))

    if leaked:
        raise DatasetError(f"LEAKAGE: {len(leaked)} training prompts are in the eval holdout, "
                           f"e.g. {leaked[:3]} — refusing to build")
    if not kept:
        raise DatasetError(f"no correct traces for {model!r} (verdicts: {dict(verdicts)})")
    if max(lengths) > max_tokens:                # unreachable by construction; asserted anyway
        raise DatasetError(f"TF-03: max row {max(lengths)} > {max_tokens}")

    kept.sort(key=lambda x: x[0])                # deterministic split, independent of DB order
    n_val = max(1, round(len(kept) * val_frac))
    val, train = [r for _, r in kept[:n_val]], [r for _, r in kept[n_val:]]

    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, data in (("train.jsonl", train), ("val.jsonl", val)):
        (out / fname).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n",
                                 encoding="utf-8")

    graded = verdicts["correct"] + verdicts["incorrect"]
    ls = sorted(lengths)
    manifest = {
        "name": name, "source_model": model, "built_at": time.time(),
        "assistant_format": "model_answer_in_content (distilled, verdict-filtered)",
        "verdicts": dict(verdicts),
        "teacher_acceptance": (verdicts["correct"] / graded) if graded else None,
        "n_train": len(train), "n_val": len(val), "n_tokens": sum(lengths),
        "token_audit": {"max": ls[-1], "p50": ls[len(ls) // 2], "p99": ls[min(len(ls) - 1, int(len(ls) * .99))],
                        "cap": max_tokens},
        "dropped": dict(dropped), "leakage_checked_against": str(holdout), "source": source,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Contract §4 `datasets` row. A's DB has no writer for it; contract §6 makes this B's.
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.execute("INSERT OR REPLACE INTO datasets(id,task_id,name,train_path,val_path,n_train,n_val,"
                    "n_tokens,train_file_id,val_file_id,created_at) VALUES (?,?,?,?,?,?,?,?,NULL,NULL,?)",
                    (f"ds-{name}", task_id, name, str(out / "train.jsonl"), str(out / "val.jsonl"),
                     len(train), len(val), sum(lengths), time.time()))
        con.commit()
    finally:
        con.close()
    return manifest


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="whose correct traces become training rows")
    ap.add_argument("--name", required=True)
    ap.add_argument("--holdout", required=True, help="eval set to check for leakage")
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--max-tokens", type=int, default=7500)
    ap.add_argument("--source", default=None, help="only traces from this writer, e.g. runner")
    args = ap.parse_args()
    s = load_settings(require_key=False)
    m = build(model=args.model, name=args.name, holdout=args.holdout, db_path=s.db_path,
              max_tokens=args.max_tokens, task_id=args.task_id, source=args.source)
    print(json.dumps(m, indent=2))
    print(f"\n[PASS] leakage=0, max row {m['token_audit']['max']} <= {m['token_audit']['cap']} (TF-03)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
