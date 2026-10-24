"""Parser for ToolACE's call DSL:  [Name(k="v", n=1), other.fn(x=[1, {"a": 2}])]

The training/holdout answers A built are in this form (manifest
`assistant_format: toolace_call_dsl_in_content`), and the fine-tuned pupil will
emit it in `content`. This module only PARSES into `[{"name","arguments"}]`;
comparison stays in `base.args_match`, so the matching policy is unchanged.

Why not a regex: names contain spaces and dots ("Video Details by URL",
"golf_course.get_review"), values nest dicts/lists, and strings contain
quotes, commas and brackets. A string-aware scan splits the calls; Python's
`ast` parses each argument list; a whitelist evaluator turns nodes into values
without executing anything.
"""
from __future__ import annotations

import ast
import re
from typing import Any


# Bare keys, optionally $-prefixed (OData-style "$select"), hyphens allowed after the
# first char ("sort-by"). Keywords like from/class allowed. Quoted keys are rejected.
_KEY_RE = re.compile(r"^\$?[A-Za-z_][A-Za-z0-9_.\-]*$")

# Function names: real ToolACE names contain spaces, dots, apostrophes, hyphens,
# commas and balanced parens ("Open-High-Low-Close (OHLC) API"). Broken labels
# look like "{Fetcher.retrieve}=>", "advice|", "{get_x}:", "Get Values-".
# REST-path names are real too: "/scripts/inventory", "/{orgCode}/cards" — so a
# leading "/" is allowed, and only path names may contain {path params}.
_NAME_BAD_CHARS = set("=|;<>")


def _check_name(name: str) -> str:
    is_path = name.startswith("/")
    if (not name or not (is_path or name[0].isalnum() or name[0] == "_")
            or _NAME_BAD_CHARS & set(name) or name[-1] in "-:."
            or (not is_path and set("{}") & set(name))):
        raise DSLParseError(f"bad function name {name[:40]!r}")
    return name


class DSLParseError(ValueError):
    """Raised for text that is not a well-formed call list. Never returns [] for
    malformed input: "no calls" and "unparseable" must stay distinguishable."""


_NAME_CONSTS = {"true": True, "false": False, "null": None, "none": None,
                "True": True, "False": False, "None": None}


def _value(node: ast.AST) -> Any:
    """Evaluate a literal node. Accepts JSON-style true/false/null as well."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_value(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_value(k): _value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Name) and node.id in _NAME_CONSTS:
        return _NAME_CONSTS[node.id]
    raise DSLParseError(f"non-literal argument value: {ast.dump(node)[:80]}")


def _split_top_level(body: str, *, in_args: bool = False) -> list[str]:
    """Split 'A(..), B(..)' on commas at paren/bracket depth 0, outside strings.

    Quotes open a string only at depth > 0: names live at depth 0 and may hold
    an apostrophe ("Get Today's Prices"), which is not a string delimiter.
    """
    parts, buf, depth, quote, esc = [], [], 0, None, False
    for ch in body:
        buf.append(ch)
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"') and (depth > 0 or in_args):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise DSLParseError("unbalanced closing bracket")
        elif ch == "," and depth == 0:
            buf.pop()
            parts.append("".join(buf).strip())
            buf = []
    if quote or depth:
        raise DSLParseError("unterminated string or bracket")
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _args_open_paren(text: str) -> int:
    """Index of the '(' that opens the argument list: the LAST depth-0 '('.

    Names may contain balanced parens ("Open-High-Low-Close (OHLC) API"), so
    the first '(' is not necessarily the argument list.
    """
    last, depth, quote, esc = -1, 0, None, False
    for i, ch in enumerate(text):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"') and depth > 0:
            quote = ch
        elif ch in "([{":
            if ch == "(" and depth == 0:
                last = i
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return last


def _parse_call(text: str) -> dict:
    lp = _args_open_paren(text)
    if lp <= 0 or not text.endswith(")"):
        raise DSLParseError(f"not a call: {text[:60]!r}")
    name = _check_name(text[:lp].strip())
    args: dict = {}
    # Keys are split off by hand rather than via ast: ToolACE uses keys that are
    # Python keywords (from=, class=) or not identifiers at all ($select=).
    for i, part in enumerate(_split_top_level(text[lp + 1:-1], in_args=True)):
        eq = _top_level_eq(part)
        key, raw = (part[:eq].strip(), part[eq + 1:].strip()) if eq > 0 else (f"_pos{i}", part)
        if not key or not raw:
            raise DSLParseError(f"bad argument {part[:40]!r} for {name!r}")
        if not _KEY_RE.match(key):          # rejects quoted keys: "variable_id"=...
            raise DSLParseError(f"bad argument key {key[:30]!r} for {name!r}")
        try:
            args[key] = _value(ast.parse(raw, mode="eval").body)
        except SyntaxError as e:
            raise DSLParseError(f"bad value for {name!r}.{key}: {e.msg}") from e
    return {"name": name, "arguments": args}


def _top_level_eq(part: str) -> int:
    """Index of the first '=' outside strings/brackets, or -1."""
    depth, quote, esc = 0, None, False
    for i, ch in enumerate(part):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            return i
    return -1


def looks_like_dsl(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("[") and t.endswith("]") and "(" in t and not t[1:].lstrip().startswith(("{", '"'))


def parse_calls(text: str) -> list[dict]:
    """'[A(x=1), B()]' -> [{"name":"A","arguments":{"x":1}}, {"name":"B","arguments":{}}]

    '[]' parses to [] (an explicit no-call). Anything malformed raises DSLParseError.
    """
    t = (text or "").strip()
    if not (t.startswith("[") and t.endswith("]")):
        raise DSLParseError("call list must be wrapped in [ ]")
    calls, pending = [], ""
    for part in _split_top_level(t[1:-1]):
        if "(" not in part:                 # "Get Games by Platform, Category, and Sort(...)"
            pending += part + ", "
            continue
        calls.append(_parse_call(pending + part))
        pending = ""
    if pending:
        raise DSLParseError(f"not a call: {pending.rstrip(', ')[:60]!r}")
    return calls
