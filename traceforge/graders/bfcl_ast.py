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

    name = "toolcall"

    # Overridable seams: BFCLGrader swaps the gold shape and the argument rule.
    def normalise(self, ground_truth: Any) -> list[dict]:
        return normalise_gold(ground_truth)

    def match_args(self, gold_args: dict, got_args: dict) -> tuple[bool, str]:
        return args_match(gold_args, got_args)

    def grade(self, *, prompt_messages: list[dict], response: dict,
              ground_truth: Any) -> Verdict:
        gold = self.normalise(ground_truth)
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

        # Order-insensitive one-to-one matching via augmenting paths. Greedy is NOT
        # exact: extra args are tolerated, so got A(x=1,y=2) can satisfy gold A(x=1)
        # and starve gold A(x=1,y=2) of the only call that fits it.
        ok_edge: dict[tuple[int, int], str] = {}     # (gold_i, got_j) -> match note
        misses: dict[int, str] = {}                  # gold_i -> most informative failure
        for i, g in enumerate(gold):
            for j, cand in enumerate(got):
                if cand["name"] != g["name"]:
                    continue
                ok, detail = self.match_args(g["arguments"], cand["arguments"])
                if ok:
                    ok_edge[(i, j)] = detail
                else:
                    misses.setdefault(i, f"{g['name']}:{detail}")

        owner: dict[int, int] = {}                   # got_j -> gold_i

        def augment(i: int, seen: set[int]) -> bool:
            for j in range(len(got)):
                if (i, j) in ok_edge and j not in seen:
                    seen.add(j)
                    if j not in owner or augment(owner[j], seen):
                        owner[j] = i
                        return True
            return False

        for i in range(len(gold)):
            if not augment(i, set()):
                names = ",".join(c["name"] for c in got)
                # Report THIS gold call's own failure, never another call's success note.
                return Verdict(False, misses.get(i, f"no_match_for:{gold[i]['name']}|got:{names}"))
        notes = [f"{gold[i]['name']}:{ok_edge[(i, j)]}" for j, i in sorted(owner.items(), key=lambda x: x[1])]
        return Verdict(True, ";".join(notes)[:200])


class BFCLGrader(ToolCallGrader):
    """BFCL v3 semantics: each gold argument lists ALL acceptable values.

    Gold: {"bfcl": [{"fn": {"arg": [v1, v2, ""]}}]}  — "" means "may be omitted".
    Unlike ToolACE, an argument the model sends that is NOT in the gold mapping
    fails: BFCL's answer key enumerates every permitted parameter, so an extra
    one is an invented parameter, not an optional one.
    """

    name = "bfcl"

    def normalise(self, ground_truth: Any) -> list[dict]:
        calls = ground_truth.get("bfcl") if isinstance(ground_truth, dict) else None
        if calls is None:
            raise ValueError(f"BFCL gold must be {{'bfcl': [...]}}, got {type(ground_truth).__name__}")
        out = []
        for c in calls:
            (fn, allowed), = c.items()          # exactly one function per gold entry
            out.append({"name": fn, "arguments": allowed})
        return out

    def match_args(self, gold_args: dict, got_args: dict) -> tuple[bool, str]:
        from traceforge.graders.base import _canon
        got_c = {str(k): _canon(v) for k, v in got_args.items()}
        for k, allowed in gold_args.items():
            optional = "" in allowed
            if k not in got_c:
                if optional:
                    continue
                return False, f"missing_args:{k}"
            if got_c[k] not in [_canon(a) for a in allowed if a != ""]:
                return False, f"wrong_values:{k}"
        extra = sorted(set(got_c) - set(gold_args))
        if extra:
            return False, f"extra_args:{','.join(extra)}"
        return True, "match"


GRADER = ToolCallGrader()
