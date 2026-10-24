"""Grader primitives shared by every task grader. Contract §5.

`Verdict` and `Grader` live in `traceforge/schemas.py` (A-owned, contract-verbatim);
they are re-exported here so grader modules have one import site.

The argument-comparison policy below is the single knob that moves the headline
number. It is deliberately in one place so it can be swept, not scattered.
"""
from __future__ import annotations

import json
import re
from typing import Any

from traceforge.schemas import Grader, Verdict  # noqa: F401  (re-export)

__all__ = ["Grader", "Verdict", "args_match", "normalise_scalar", "parse_arguments"]

_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def parse_arguments(raw: Any) -> dict:
    """Tool-call arguments arrive as a JSON string from the API, a dict from datasets.

    Returns {} for anything unparseable — callers distinguish "no args" from
    "malformed" via the verdict detail, never by an empty dict alone.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalise_scalar(v: Any) -> Any:
    """Coerce a scalar to a comparable canonical form.

    POLICY (the knob): models emit "2" where ground truth has 2, and "New York"
    where ground truth has "new york". Both are correct tool calls by any
    reasonable reading, and BFCL's own checker tolerates them. We therefore
    compare numbers numerically and strings case/whitespace-insensitively.

    We do NOT coerce across kinds: True != 1, None != "". Booleans are tagged
    rather than returned bare, because `bool` subclasses `int` in Python and
    `True == 1.0` compares equal — an untagged bool would silently match the
    integer 1.
    """
    if isinstance(v, bool):
        return ("__bool__", v)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if _NUM_RE.match(s):
            return float(s)
        low = s.lower()
        if low in ("true", "false"):
            return ("__bool__", low == "true")
        return low
    return v


def _canon(v: Any) -> Any:
    """Canonicalise a whole argument value, recursing into lists/dicts.

    Lists stay ORDER-SENSITIVE: ["JFK","LAX"] is a different flight from
    ["LAX","JFK"]. Only the mapping of argument names is order-insensitive.
    """
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return normalise_scalar(v)


def args_match(gold: dict, got: dict) -> tuple[bool, str]:
    """Compare argument mappings. Returns (ok, machine-readable detail).

    POLICY (the knob): every gold argument must be present and equal. Extra
    arguments the model volunteered are reported but NOT failed, because gold
    answers routinely omit optional parameters the model is free to fill.
    Flip `_EXTRA_ARGS_FAIL` to make this strict, and re-run the eval — the
    delta between the two settings is itself worth reporting.
    """
    gold_c = {str(k): _canon(v) for k, v in gold.items()}
    got_c = {str(k): _canon(v) for k, v in got.items()}

    missing = [k for k in gold_c if k not in got_c]
    if missing:
        return False, f"missing_args:{','.join(sorted(missing))}"

    wrong = [k for k, v in gold_c.items() if got_c[k] != v]
    if wrong:
        return False, f"wrong_values:{','.join(sorted(wrong))}"

    extra = sorted(set(got_c) - set(gold_c))
    if extra and _EXTRA_ARGS_FAIL:
        return False, f"extra_args:{','.join(extra)}"
    return True, f"match{'+extra:' + ','.join(extra) if extra else ''}"


_EXTRA_ARGS_FAIL = False
