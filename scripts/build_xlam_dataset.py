"""Build SFT JSONL from Salesforce/xlam-function-calling-60k (contract 9a).

Per contract 9a, xlam is the *trace source*, not a fallback: each row carries
`query` + `tools` + `answers`, so the dataset supplies both the prompt and the
ground truth. This script produces the dataset-floor job that secures the 14:50
gate; teacher-trace generation + grading produces v2 during training.

Assistant turn format: JSON-in-content (contract section 7 fallback form), NOT native
`tool_calls`. Upload validates only that the file is JSONL -- schema validation
happens at job creation. The content form cannot fail at that later step.

Token audit is a HARD assert: max row <= 7500 tokens (TF-03). With `packing: true`
over-length rows truncate *silently*, so a violator is a wrong training signal
that no error would ever surface.

Usage:
  python scripts/build_xlam_dataset.py --n-train 1200 --n-val 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traceforge.config import load_settings

REPO = Path(__file__).resolve().parent.parent

SYSTEM_TMPL = (
    "You are a function-calling AI model. You are given a set of tools in JSON.\n"
    "Choose the tool call(s) that answer the user's request.\n"
    "Respond with ONLY a JSON array of calls, each as "
    '{{"name": <tool name>, "arguments": {{<arg>: <value>}}}}.\n'
    "Do not add explanation.\n\n"
    "Available tools:\n{tools}"
)


def fetch_xlam(token: str | None) -> list[dict]:
    """Download the xlam rows. Tries the single-file hub path, then `datasets`."""
    from huggingface_hub import hf_hub_download

    repo = "Salesforce/xlam-function-calling-60k"
    cache = REPO / "data" / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for fname in ("xlam_function_calling_60k.json",
                  "data/xlam_function_calling_60k.json"):
        try:
            p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset",
                                token=token, cache_dir=str(cache))
            print(f"[info] downloaded {fname} -> {p}")
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - we try the next candidate path
            last_err = e
            print(f"[info] {fname} unavailable: {type(e).__name__}: {str(e)[:150]}")

    print("[info] falling back to datasets.load_dataset")
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, split="train", token=token)
        return [dict(r) for r in ds]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"could not obtain xlam rows; last hub error: {last_err}; "
            f"datasets error: {e}"
        ) from e


def _parse_maybe_json(v):
    """xlam stores tools/answers as JSON strings in some revisions, lists in others."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return v


def to_messages(row: dict) -> dict | None:
    """One xlam row -> one SFT row. Returns None if the row is unusable."""
    query = (row.get("query") or "").strip()
    tools = _parse_maybe_json(row.get("tools"))
    answers = _parse_maybe_json(row.get("answers"))
    if not query or not tools or not answers:
        return None
    if not isinstance(answers, list) or not answers:
        return None
    # Ground truth normalised to the shape the grader compares on (contract 9a:
    # name, then arguments key-by-key, order-insensitive).
    gt = []
    for a in answers:
        if not isinstance(a, dict) or "name" not in a:
            return None
        gt.append({"name": a["name"], "arguments": a.get("arguments", {})})
    return {
        "messages": [
            {"role": "system",
             "content": SYSTEM_TMPL.format(tools=json.dumps(tools, indent=None))},
            {"role": "user", "content": query},
            {"role": "assistant",
             "content": json.dumps(gt, separators=(",", ":"))},
        ]
    }


def toolace_to_messages(row: dict) -> dict | None:
    """One ToolACE row -> one SFT row. Single-turn only: user then assistant call.

    ToolACE embeds the tool list in `system` and writes calls in its own DSL,
    e.g. `[Market Trends API(trend_type="MARKET_INDEXES", country="us")]`.
    Both are kept verbatim: the system prompt already specifies that format,
    and the grader parses the same string it trained on.
    """
    conv = row.get("conversations") or []
    if len(conv) != 2:
        return None
    u, a = conv
    if u.get("from") != "user" or a.get("from") != "assistant":
        return None
    call = (a.get("value") or "").strip()
    if not (call.startswith("[") and call.endswith("]") and "(" in call):
        return None  # refusals / clarifications: not a verifiable call
    return {"messages": [
        {"role": "system", "content": row.get("system", "")},
        {"role": "user", "content": u.get("value", "")},
        {"role": "assistant", "content": call},
    ]}


def count_tokens(sft_row: dict, enc) -> int:
    """Token length of the serialised row, with per-message overhead."""
    total = 0
    for m in sft_row["messages"]:
        total += len(enc.encode(m.get("content") or "")) + 4
    return total + 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-holdout", type=int, default=250,
                    help="held-out items reserved for eval (never trained on)")
    ap.add_argument("--max-tokens", type=int, default=7500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", choices=["xlam", "toolace"], default="xlam")
    args = ap.parse_args()

    load_settings(require_key=False)  # loads .env so HF_TOKEN is visible
    token = os.environ.get("HF_TOKEN") or None
    print(f"[info] HF_TOKEN present: {bool(token)}")

    if args.source == "toolace":
        rows = json.loads((REPO / "data" / "toolace.json").read_text(encoding="utf-8"))
        convert = toolace_to_messages
    else:
        rows = fetch_xlam(token)
        convert = to_messages
    args.out = args.out or f"data/datasets/{args.source}-fc-v1"
    print(f"[info] {args.source} rows available: {len(rows)}")

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    need = args.n_train + args.n_val + args.n_holdout
    kept: list[dict] = []
    dropped = Counter()
    lengths: list[int] = []

    for row in rows:
        if len(kept) >= need:
            break
        sft = convert(row)
        if sft is None:
            dropped["unusable_row"] += 1
            continue
        n = count_tokens(sft, enc)
        if n > args.max_tokens:
            dropped["over_token_cap"] += 1
            continue
        lengths.append(n)
        kept.append(sft)

    if len(kept) < need:
        print(f"[WARN] only {len(kept)} usable rows for a target of {need}")

    train = kept[: args.n_train]
    val = kept[args.n_train: args.n_train + args.n_val]
    holdout = kept[args.n_train + args.n_val:]

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, data: list[dict]) -> Path:
        p = out / name
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n",
                     encoding="utf-8")
        return p

    tp, vp = write("train.jsonl", train), write("val.jsonl", val)
    hp = write("holdout.jsonl", holdout)

    # ---- token audit (TF-03) -----------------------------------------------------
    print("\n=== TOKEN AUDIT (TF-03) ===")
    if lengths:
        lengths_sorted = sorted(lengths)
        def pct(p: float) -> int:
            return lengths_sorted[min(len(lengths_sorted) - 1,
                                      int(len(lengths_sorted) * p))]
        print(f"rows kept   : {len(kept)}")
        print(f"min/med/max : {lengths_sorted[0]} / {pct(0.5)} / {lengths_sorted[-1]}")
        print(f"p90 / p99   : {pct(0.90)} / {pct(0.99)}")
        buckets = Counter(min(n // 500 * 500, 8000) for n in lengths)
        for b in sorted(buckets):
            print(f"  {b:>5}-{b + 499:<5} {'#' * min(60, buckets[b])} {buckets[b]}")
    print(f"dropped     : {dict(dropped)}")
    print(f"\ntrain={len(train)} -> {tp}")
    print(f"val  ={len(val)} -> {vp}")
    print(f"hold ={len(holdout)} -> {hp}  (NEVER trained on; eval uses this)")

    assert lengths, "no rows survived the audit"
    assert max(lengths) <= args.max_tokens, (
        f"HARD ASSERT FAILED (TF-03): max row {max(lengths)} > {args.max_tokens}"
    )
    print(f"\n[PASS] token audit: max {max(lengths)} <= {args.max_tokens}")
    (out / "manifest.json").write_text(json.dumps({
        "source": args.source,
        "assistant_format": "json_in_content" if args.source == "xlam"
                            else "toolace_call_dsl_in_content",
        "n_train": len(train), "n_val": len(val), "n_holdout": len(holdout),
        "max_tokens": max(lengths), "dropped": dict(dropped),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
