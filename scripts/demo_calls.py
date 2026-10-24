"""Demo driver: send held-out items through the TraceForge proxy, alternating student
and teacher, so verdicts appear live on the dashboard.

  python scripts/demo_calls.py --n 6 --start 40

Uses 127.0.0.1, not localhost: on Windows `localhost` tries IPv6 first and adds 2-3 s.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
MODELS = ["Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--start", type=int, default=40, help="holdout row to start from")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    lines = (REPO / "data/datasets/toolace-fc-v1/holdout.jsonl").read_text(encoding="utf-8").splitlines()
    with httpx.Client(base_url=args.url, timeout=200) as c:
        for i, line in enumerate(lines[args.start: args.start + args.n]):
            model = MODELS[i % 2]
            msgs = json.loads(line)["messages"][:-1]
            t0 = time.time()
            r = c.post("/v1/chat/completions", json={"model": model, "messages": msgs,
                                                     "temperature": 0, "max_tokens": 256})
            ms = int((time.time() - t0) * 1000)
            ans = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else r.text[:80]
            print(f"{r.status_code} {model.split('/')[-1][:24]:24} {ms:>6} ms  {msgs[-1]['content'][:50]!r}")
            print(f"    -> {ans[:100]}")
        time.sleep(2)
        print("\nverdicts:")
        for t in c.get("/traces?limit=%d" % args.n).json()["traces"]:
            print(f"  {t['grade'] or 'ungraded':10} {t['model'].split('/')[-1][:24]:24} {t['grade_detail'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
