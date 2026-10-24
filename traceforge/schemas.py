"""Shared dataclasses. Contract §5 — verbatim. Hacker B codes against these."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Verdict:
    """Returned by every grader."""
    correct: bool
    detail: str = ""          # machine-readable reason, e.g. "exec_match:false"


class Grader(Protocol):
    """B implements one per task type."""
    name: str

    def grade(self, *, prompt_messages: list[dict], response: dict,
              ground_truth: dict) -> Verdict: ...


@dataclass(frozen=True)
class EvalResult:
    item_id: str
    model: str
    verdict: bool
    detail: str
    latency_ms: int
    tokens: int
