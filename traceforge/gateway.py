"""OpenAI-compatible proxy (A3). Forwards to Token Factory, returns the upstream bytes
verbatim, writes a trace row, and -- the point of the product -- attaches a verdict.

Verdicts need ground truth. A call whose prompt matches a known item (B's item id is a
hash of the prompt messages, so the request itself identifies it) is graded in the
background right after the response is returned. Other calls stay grade=NULL: unknown,
never assumed correct.

Logging is fail-open: a broken trace store must never break the caller's request.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from traceforge.costs import PriceTable
from traceforge.db import DB
from traceforge.fingerprint import fingerprint
from traceforge.router import pick_upstream

log = logging.getLogger("traceforge.gateway")


def item_id(messages: list[dict]) -> str:
    """Same id as B's eval.load_holdout: sha256 of the prompt messages, sort_keys."""
    prompt = [m for m in messages if m.get("role") != "assistant"] \
        if messages and messages[-1].get("role") == "assistant" else messages
    return hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class GoldIndex:
    """item_id -> (task name, ground truth), built from holdout files."""
    items: dict[str, tuple[str, object]] = field(default_factory=dict)

    @classmethod
    def from_holdouts(cls, paths: dict[str, Path]) -> "GoldIndex":
        from traceforge.eval import load_holdout   # B-owned
        idx = cls()
        for task, p in paths.items():
            if Path(p).exists():
                for it in load_holdout(p):
                    idx.items[it.item_id] = (task, it.ground_truth)
        return idx

    def lookup(self, messages: list[dict]) -> tuple[str, object] | None:
        return self.items.get(item_id(messages))


@dataclass
class Gateway:
    db: DB
    prices: PriceTable
    upstream_base_url: str
    api_key: str
    gold: GoldIndex
    grader_for: callable           # task name -> Grader (B's registry); raises if unknown
    client: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=180)
        return self.client

    async def proxy(self, body: bytes, task_header: str | None = None,
                    session_id: str | None = None) -> tuple[int, bytes, str]:
        """Returns (status, body bytes, content-type) exactly as upstream sent them."""
        try:
            payload = json.loads(body)
        except ValueError:
            return 400, b'{"error":"request body is not JSON"}', "application/json"

        task_id = fingerprint(payload, task_header)
        requested = payload.get("model", "")
        routed = pick_upstream(self.db, task_id, requested)
        if routed != requested:
            payload["model"] = routed
            body = json.dumps(payload).encode()

        t0 = time.time()
        try:
            r = await self._client().post(
                f"{self.upstream_base_url.rstrip('/')}/chat/completions", content=body,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"})
            status, raw = r.status_code, r.content
            ctype = r.headers.get("content-type", "application/json")
        except httpx.HTTPError as e:
            # Unreachable upstream is a recorded failure, not a missing trace.
            status, ctype = 502, "application/json"
            raw = json.dumps({"error": {"type": "upstream_unreachable",
                                        "message": f"{type(e).__name__}: {e}"}}).encode()
        latency_ms = int((time.time() - t0) * 1000)

        # Off the request path: a locked db (B's writer shares the file) must not
        # stall the caller, and the write retries instead of being dropped.
        asyncio.get_running_loop().run_in_executor(
            None, self._record_safely, payload, routed, task_id, task_header,
            session_id, status, raw, latency_ms)
        return status, raw, ctype

    def _record_safely(self, *args) -> None:
        for attempt in range(6):
            try:
                self._record(*args)
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 5:
                    log.exception("trace write failed after %d attempts", attempt + 1)
                    return
                time.sleep(2 ** attempt)
            except Exception:  # noqa: BLE001 - fail-open by design, but loudly
                log.exception("trace logging failed; request still served")
                return

    def _record(self, payload, model, task_id, task_header, session_id,
                status, raw, latency_ms) -> None:
        response = None
        try:
            response = json.loads(raw)
        except ValueError:
            pass
        usage = (response or {}).get("usage") if isinstance(response, dict) else None
        trace_id = self.db.insert_trace(
            task_id=task_id, model=model, request=payload, response=response,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            latency_ms=latency_ms, cost_usd=self.prices.cost_usd(model, usage),
            session_id=session_id, source="proxy", task_name=task_header)
        if status != 200:
            kind = (response or {}).get("error", {}).get("type") if isinstance(response, dict) else None
            self.db.update_grade(trace_id, grader="transport", correct=None,
                                 detail=kind or f"upstream_http_{status}")
            return
        hit = self.gold.lookup(payload.get("messages", []))
        if hit:   # already on a worker thread
            self.grade_trace(trace_id, hit[0], payload["messages"], response, hit[1])

    def grade_trace(self, trace_id, task, messages, response, gold) -> str:
        try:
            v = self.grader_for(task).grade(prompt_messages=messages, response=response,
                                            ground_truth=gold)
            self.db.update_grade(trace_id, grader=task, correct=v.correct, detail=v.detail)
            return "correct" if v.correct else "incorrect"
        except Exception as e:  # noqa: BLE001 - a grader crash is an 'error' verdict, not silence
            self.db.update_grade(trace_id, grader=task, correct=None,
                                 detail=f"grader_error:{type(e).__name__}:{str(e)[:80]}")
            return "error"

    def grade_pending(self, task_id: str | None = None, limit: int = 500) -> dict:
        """POST /grades/run: grade ungraded traces that have known ground truth."""
        counts = {"correct": 0, "incorrect": 0, "error": 0, "no_ground_truth": 0}
        for t in self.db.traces(task_id=task_id, grade="ungraded", limit=limit):
            req = json.loads(t["request_json"])
            hit = self.gold.lookup(req.get("messages", []))
            if not hit or not t["response_json"]:
                counts["no_ground_truth"] += 1
                continue
            counts[self.grade_trace(t["id"], hit[0], req["messages"],
                                    json.loads(t["response_json"]), hit[1])] += 1
        return counts
