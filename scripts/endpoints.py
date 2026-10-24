"""Dedicated-endpoint CLI. Replaces the removed serverless LoRA deploy (TF-07: /v0/models -> 410).

Dedicated endpoints bill per GPU-hour while they exist. `delete` is the teardown.

  python scripts/endpoints.py create --name tf-base-llama31-8b
  python scripts/endpoints.py create --name pupil-fc-v1 --weights <custom_weights_id>
  python scripts/endpoints.py list
  python scripts/endpoints.py delete <endpoint_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from traceforge.config import load_settings

REPO = Path(__file__).resolve().parent.parent

# Cheapest config the template allows for this model (data/templates.json, 14:36).
BASE_CONFIG = {
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
    "flavor_name": "base",
    "gpu_type": "gpu-l40s-d",
    "region": "eu-north1",
    "gpu_count": 1,
    "scaling": {"min_replicas": 1, "max_replicas": 1},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--weights", help="custom_weights_id for a fine-tuned model")
    sub.add_parser("list")
    d = sub.add_parser("delete")
    d.add_argument("endpoint_id")
    args = ap.parse_args()

    s = load_settings()
    v0 = s.v0_base_url
    with httpx.Client(timeout=120,
                      headers={"Authorization": f"Bearer {s.nebius_api_key}"}) as cl:
        if args.cmd == "list":
            r = cl.get(f"{v0}/dedicated_endpoints")
        elif args.cmd == "delete":
            r = cl.delete(f"{v0}/dedicated_endpoints/{args.endpoint_id}")
        else:
            body = {"name": args.name, **BASE_CONFIG}
            if args.weights:
                body["custom_weights_id"] = args.weights
            r = cl.post(f"{v0}/dedicated_endpoints", json=body)

    print(f"HTTP {r.status_code}")
    try:
        out = r.json()
        print(json.dumps(out, indent=2)[:3000])
    except ValueError:
        out = None
        print(r.text[:1500])
    if args.cmd == "create" and r.status_code in (200, 201) and out:
        log = REPO / "data" / "endpoints.jsonl"
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"name": args.name, "response": out}) + "\n")
        print(f"[info] recorded in {log} -- remember: it bills until deleted")
    return 0 if r.status_code < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
