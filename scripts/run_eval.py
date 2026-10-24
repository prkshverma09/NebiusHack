"""CLI eval entrypoint (contract §1, B-owned).

  python scripts/run_eval.py \
      --models Qwen/Qwen3-30B-A3B-Instruct-2507,Qwen/Qwen3-235B-A22B-Instruct-2507 \
      --baseline Qwen/Qwen3-30B-A3B-Instruct-2507 --limit 60

Re-running is cheap: successful responses are cached per (model, item).
Exit code 2 means some calls failed in transport and the numbers are incomplete.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from traceforge.config import REPO_ROOT, load_settings
from traceforge.eval import failure_shortlist, load_holdout, persist, run_eval, score_table
from traceforge.graders import get_grader

DEFAULT_HOLDOUT = REPO_ROOT / "data/datasets/toolace-fc-v1/holdout.jsonl"


def main() -> int:
    # Windows consoles default to cp1252, which cannot print the table's "Δ".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--baseline", help="model the others are paired against (default: first)")
    ap.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    ap.add_argument("--bench", choices=["holdout", "bfcl"], default="holdout",
                    help="holdout = ToolACE held-out JSONL; bfcl = BFCL v3 cross-distribution slice")
    ap.add_argument("--per-category", type=int, default=40, help="bfcl only")
    ap.add_argument("--task", default=None, help="grader key (default: toolace, or bfcl with --bench bfcl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    s = load_settings()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    baseline = args.baseline or models[0]
    if baseline not in models:
        ap.error("--baseline must be one of --models")

    if args.bench == "bfcl":
        from traceforge.data.bfcl import load_bfcl
        items = load_bfcl(per_category=args.per_category)[: args.limit]
        task, benchmark = args.task or "bfcl", f"bfcl-v3-x{args.per_category}"
    else:
        items = load_holdout(args.holdout, args.limit)
        task, benchmark = args.task or "toolace", Path(args.holdout).parent.name
    print(f"[eval] {len(items)} items x {len(models)} models, bench={benchmark}, grader={task}, baseline={baseline}")
    client = OpenAI(api_key=s.nebius_api_key, base_url=s.upstream_base_url, timeout=90, max_retries=0)
    results = run_eval(models, items, get_grader(task), client, workers=args.workers)

    table = score_table(results, models, baseline)
    # Print before persisting: a locked DB must not swallow the numbers.
    print("\n" + table["markdown"] + "\n\n" + table["ascii"], flush=True)
    run_id = persist(results, table, benchmark=benchmark, models=models, db_path=s.db_path)
    scored = [a for a in table["accuracy"] if a["acc"] is not None]
    strong = max(scored, key=lambda a: a["acc"]) if scored else None
    if strong and strong["model"] != baseline:
        print(f"\nreplay shortlist ({baseline} wrong, {strong['model']} right):")
        print(json.dumps(failure_shortlist(results, baseline, strong["model"]), indent=1))
    errs = sum(a["errors"] for a in table["accuracy"])
    print(f"\n[eval] run {run_id} -> data/evals/{run_id}.json | transport errors: {errs}")
    return 0 if errs == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
