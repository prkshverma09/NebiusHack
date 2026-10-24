"""One-shot upload -> fine-tune job CLI (A4 emergency path; contract section 9 gate).

No server needed. Uploads train/val JSONL, creates the job, records the id.

  python scripts/submit_job.py --dataset data/datasets/toolace-fc-v1 --suffix pupil-fc-v1
  python scripts/submit_job.py --status <job_id>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from traceforge.config import load_settings

REPO = Path(__file__).resolve().parent.parent

# Contract A4 values, except learning_rate: 1e-5 -> 1e-4. With packing, ~37
# optimizer steps total; LoRA at 1e-5 would barely move in that budget.
HPARAMS = {
    "lora": True, "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.05,
    "n_epochs": 3, "learning_rate": 1e-4, "batch_size": 8, "packing": True,
    "context_length": 8192,
}


def upload(c: httpx.Client, base: str, path: Path) -> str:
    r = c.post(f"{base}/files",
               files={"file": (path.name, path.read_bytes(), "application/jsonl")},
               data={"purpose": "fine-tune"})
    if r.status_code not in (200, 201):
        raise SystemExit(f"upload {path} failed: HTTP {r.status_code}: {r.text[:400]}")
    fid = r.json()["id"]
    print(f"[PASS] uploaded {path.name} -> {fid}")
    return fid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="dir containing train.jsonl / val.jsonl")
    ap.add_argument("--suffix", default="pupil-fc-v1")
    ap.add_argument("--status", metavar="JOB_ID", help="print job status and exit")
    ap.add_argument("--hp", default="{}",
                    help='JSON overrides merged into HPARAMS, e.g. \'{"batch_size":16}\'')
    args = ap.parse_args()
    hparams = {**HPARAMS, **json.loads(args.hp)}

    s = load_settings()
    base = s.upstream_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {s.nebius_api_key}"}

    with httpx.Client(timeout=300, headers=headers) as c:
        if args.status:
            r = c.get(f"{base}/fine_tuning/jobs/{args.status}")
            print(json.dumps(r.json(), indent=2)[:3000])
            return 0 if r.status_code == 200 else 1

        if not args.dataset:
            ap.error("--dataset is required unless --status is given")
        ds = REPO / args.dataset
        train_id = upload(c, base, ds / "train.jsonl")
        val_id = upload(c, base, ds / "val.jsonl")

        body = {"model": s.student_base, "training_file": train_id,
                "validation_file": val_id, "hyperparameters": hparams,
                "suffix": args.suffix, "seed": 42}
        r = c.post(f"{base}/fine_tuning/jobs", json=body)
        if r.status_code not in (200, 201):
            print(f"[FAIL] job create: HTTP {r.status_code}: {r.text[:800]}")
            return 1
        job = r.json()

    print(f"[PASS] JOB SUBMITTED: {job.get('id')} status={job.get('status')}")
    record = {"submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "job_id": job.get("id"), "dataset": args.dataset,
              "train_file": train_id, "val_file": val_id,
              "request": body, "response": job}
    (REPO / "data" / f"job-{args.suffix}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (REPO / "data" / "STATUS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n- **job submitted {record['submitted_at']}: {job.get('id')}** "
                f"on {s.student_base}, dataset {args.dataset}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
