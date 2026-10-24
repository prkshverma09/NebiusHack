"""Tool-call grader — the PRIMARY grader on the critical path (contract §9a.2).

Scores a model's `tool_calls` against ground-truth calls of the form
`{"name": ..., "arguments": {...}}`, which is the shape xlam's `answers`,
glaive's `function_call` and BFCL's `ground_truth` all reduce to.

Four BFCL categories, one code path:
  simple       1 gold call,  1 emitted call
  multiple     1 gold call,  N tools offered (the choice is the test)
  parallel     N gold calls, N emitted calls, order-insensitive
  irrelevance  0 gold calls  -> emitting ANY call is wrong

No LLM-as-judge. Every verdict is a dict comparison with a machine-readable reason.
"""
from __future__ import annotations

from typing import Any

from traceforge.graders.base import Verdict, args_match, parse_arguments
from traceforge.graders.toolace_dsl import DSLParseError, looks_like_dsl, parse_calls


def extract_calls(response: dict) -> list[dict]:
    """Pull `[{"name","arguments"}]` out of an OpenAI-shaped response dict.

    Falls back to the contract §7 degraded form: a JSON tool call serialised
    into `content` when the API would not accept `tool_calls` on a row.
    Returns [] only when the model genuinely emitted no call.
    """
    choices = response.get("choices") or []
    if not choices:
        return []
    msg = choices[0].get("message") or {}

    calls: list[dict] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"name": fn.get("name", ""),
                      "arguments": parse_arguments(fn.get("arguments"))})
    if calls:
        return calls

    content = (msg.get("content") or "").strip()

    # ToolACE DSL in content: [Name(k=v), ...] — what the pupil is trained to emit.
    # Malformed DSL raises DSLParseError; grade() turns that into a distinct verdict.
    if looks_like_dsl(content):
        return parse_calls(content)

    # Degraded path: a bare JSON object in content.
    if content.startswith("{") or content.startswith("["):
        parsed = parse_arguments(content) or _try_list(content)
        for c in (parsed if isinstance(parsed, list) else [parsed]):
            if isinstance(c, dict) and "name" in c:
                calls.append({"name": c.get("name", ""),
                              "arguments": parse_arguments(c.get("arguments", {}))})
    return calls


def _try_list(raw: str) -> list:
    import json
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return v if isinstance(v, list) else []


def normalise_gold(ground_truth: Any) -> list[dict]:
    """Accept the several shapes ground truth arrives in; emit one.

    dict  {"calls": [...]}  | {"name","arguments"}  | {} (irrelevance)
    list  [{"name","arguments"}, ...]
    """
    if ground_truth is None:
        return []
    if isinstance(ground_truth, str):
        # ToolACE gold. A malformed label is a DATA error: let DSLParseError
        # propagate so the eval excludes the item loudly instead of scoring it.
        return parse_calls(ground_truth) if ground_truth.strip() else []
    if isinstance(ground_truth, dict):
        if "calls" in ground_truth:
            return normalise_gold(ground_truth["calls"])
        if "name" in ground_truth:
            return [{"name": ground_truth["name"],
                     "arguments": parse_arguments(ground_truth.get("arguments", {}))}]
        return []
    if isinstance(ground_truth, list):
        out = []
        for c in ground_truth:
            if isinstance(c, dict) and "name" in c:
                out.append({"name": c["name"],
                            "arguments": parse_arguments(c.get("arguments", {}))})
        return out
    return []


class ToolCallGrader:
    """Deterministic AST-equivalent matcher over tool calls."""

    name = "bfcl"

    def grade(self, *, prompt_messages: list[dict], response: dict,
              ground_truth: Any) -> Verdict:
        gold = normalise_gold(ground_truth)
        try:
            got = extract_calls(response)
        except DSLParseError as e:
            return Verdict(False, f"unparseable_call:{str(e)[:80]}")

        # irrelevance: the correct behaviour is to call nothing.
        if not gold:
            if got:
                return Verdict(False, f"irrelevance:called:{got[0]['name']}")
            return Verdict(True, "irrelevance:abstained")

        if not got:
            return Verdict(False, "no_call_emitted")
        if len(got) != len(gold):
            return Verdict(False, f"call_count:{len(got)}!={len(gold)}")

        # Order-insensitive one-to-one matching (greedy is exact here: calls in
        # a parallel set are distinct, so no augmenting path exists).
        remaining = list(range(len(got)))
        details: list[str] = []
        for g in gold:
            hit = None
            for idx in remaining:
                cand = got[idx]
                if cand["name"] != g["name"]:
                    continue
                ok, detail = args_match(g["arguments"], cand["arguments"])
                if ok:
                    hit, note = idx, detail
                    break
                details.append(f"{g['name']}:{detail}")
            if hit is None:
                names = ",".join(c["name"] for c in got)
                reason = details[-1] if details else f"no_match_for:{g['name']}|got:{names}"
                return Verdict(False, reason)
            remaining.remove(hit)
            details.append(f"{g['name']}:{note}")
        return Verdict(True, ";".join(details)[:200])


GRADER = ToolCallGrader()
