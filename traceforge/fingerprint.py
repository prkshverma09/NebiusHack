"""Task fingerprint. Contract section 3.

Two calls with the same system prompt + tools + response_format + X-TF-Task header are
the same task; everything else (the user turn) varies within it.
"""
from __future__ import annotations

import hashlib
import json


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(payload: dict, task_header: str | None = None) -> str:
    system = next((m.get("content") or "" for m in payload.get("messages", [])
                   if m.get("role") == "system"), "")
    key = {
        "system": system,
        "tools": payload.get("tools") or [],
        "response_format": payload.get("response_format") or None,
        # B's scripts/gen_traces.py passes the label in-payload as "_task".
        "task": task_header or payload.get("_task") or "",
    }
    return hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()[:12]
