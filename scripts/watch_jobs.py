"""Poll fine-tuning jobs until something actionable happens; then exit so the caller wakes.

Exits 0 when any job has a checkpoint or succeeds, 1 when a job fails/cancels,
3 on timeout. Every poll is appended to data/watch.log.

  python scripts/watch_jobs.py ftjob-aaa ftjob-bbb --every 120 --max-minutes 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from traceforge.config import REPO_ROOT, load_settings

LOG = REPO_ROOT / "data" / "watch.log"


def log(line: str) -> None:
    stamped = f"{time.strftime('%H:%M:%S')} {line}"
    print(stamped, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(stamped + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--every", type=int, default=120, help="seconds between polls")
    ap.add_argument("--max-minutes", type=int, default=120)
    args = ap.parse_args()

    s = load_settings()
    base = s.upstream_base_url.rstrip("/")
    deadline = time.time() + args.max_minutes * 60
    last: dict[str, str] = {}

    with httpx.Client(timeout=30,
                      headers={"Authorization": f"Bearer {s.nebius_api_key}"}) as c:
        while time.time() < deadline:
            for jid in args.jobs:
                try:
                    j = c.get(f"{base}/fine_tuning/jobs/{jid}").json()
                    ck = c.get(f"{base}/fine_tuning/jobs/{jid}/checkpoints").json()
                    ev = c.get(f"{base}/fine_tuning/jobs/{jid}/events").json()
                except Exception as e:  # noqa: BLE001 - transient; report, keep polling
                    log(f"{jid[:14]} poll error {type(e).__name__}: {e}")
                    continue
                cks = ck.get("data", [])
                events = sorted(ev.get("data", []), key=lambda e: e["created_at"])
                newest = events[-1]["message"][:90] if events else "-"
                line = (f"{jid[:14]} {j.get('model', '?')[:34]:34} status={j.get('status')} "
                        f"tokens={j.get('trained_tokens')} ckpts={len(cks)} | {newest}")
                if line != last.get(jid):
                    log(line)
                    last[jid] = line
                status = j.get("status")
                if cks or status == "succeeded":
                    log(f"ACTIONABLE {jid}: status={status} checkpoints:")
                    log(json.dumps(cks, indent=1)[:3000])
                    return 0
                if status in ("failed", "cancelled"):
                    log(f"FAILED {jid}: {j.get('error')}")
                    return 1
            time.sleep(args.every)
    log("timeout: no checkpoint yet")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
