"""Shared test doubles (A8). Either hacker may import these.

FakeUpstream  -- an httpx.MockTransport handler that answers like Token Factory
FakeGrader    -- a Grader whose verdict is decided by a predicate on the answer
make_services -- a fully wired Services for TestClient, no env, no network
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from traceforge.config import Settings
from traceforge.costs import PriceTable
from traceforge.db import DB
from traceforge.gateway import Gateway, GoldIndex
from traceforge.schemas import Verdict


class FakeUpstream:
    """Answers chat completions; records every request it saw."""

    def __init__(self, answer: str = "[f(x=1)]", status: int = 200, fail: Exception | None = None):
        self.answer, self.status, self.fail = answer, status, fail
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            raise self.fail
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"type": "fake_error", "message": "x"}})
        return httpx.Response(200, json={
            "id": "fake", "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.answer}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}})


class FakeGrader:
    name = "fake"

    def __init__(self, correct_if=lambda content, gold: content == gold):
        self.correct_if = correct_if

    def grade(self, *, prompt_messages, response, ground_truth) -> Verdict:
        content = response["choices"][0]["message"]["content"]
        ok = self.correct_if(content, ground_truth)
        return Verdict(ok, "fake:match" if ok else "fake:mismatch")


PROMPT = [{"role": "system", "content": "tools: f"}, {"role": "user", "content": "call f"}]


def make_settings(tmp: Path) -> Settings:
    return Settings(nebius_api_key="test-key", db_path=tmp / "tf.db",
                    upstream_base_url="https://upstream.test/v1/",
                    teacher_model="teacher/T", student_base="student/S", port=0)


def make_gateway(tmp: Path, upstream: FakeUpstream, gold: dict | None = None,
                 grader: FakeGrader | None = None) -> Gateway:
    from traceforge.gateway import item_id
    idx = GoldIndex()
    for prompt, (task, truth) in (gold or {}).items():
        idx.items[item_id(json.loads(prompt))] = (task, truth)
    s = make_settings(tmp)
    return Gateway(db=DB(s.db_path), prices=PriceTable({"teacher/T": (2e-7, 6e-7)}),
                   upstream_base_url=s.upstream_base_url, api_key=s.nebius_api_key,
                   gold=idx, grader_for=lambda task: grader or FakeGrader(),
                   client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
