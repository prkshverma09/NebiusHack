"""Grader registry. Contract §6 — A's `/grades/run` looks up by task name."""
from __future__ import annotations

from traceforge.graders.base import Grader, Verdict
from traceforge.graders.bfcl_ast import ToolCallGrader

_REGISTRY: dict[str, Grader] = {
    "bfcl": ToolCallGrader(),
    "xlam": ToolCallGrader(),      # same shape, same matcher (contract §9a.2)
    "glaive": ToolCallGrader(),
    "toolace": ToolCallGrader(),   # DSL gold/answers, parsed in toolace_dsl.py
}

try:                               # STRETCH (B6) — absent until Spider lands
    from traceforge.graders.sql_exec import SQLExecGrader
    _REGISTRY["sql"] = SQLExecGrader()
except ImportError:
    pass


class UnknownGraderError(KeyError):
    """Raised for an unregistered task name — never silently returns a no-op grader."""


def get_grader(task_name: str) -> Grader:
    """Look up a grader. Raises rather than returning a permissive default:
    a missing grader must not read as 'everything graded correct'."""
    key = (task_name or "").strip().lower()
    if key not in _REGISTRY:
        raise UnknownGraderError(
            f"no grader registered for task {task_name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def available_graders() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Grader", "Verdict", "ToolCallGrader", "get_grader",
           "available_graders", "UnknownGraderError"]
