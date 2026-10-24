"""BFCL any-of grader: unit cases + separation over the real downloaded answer keys."""
from __future__ import annotations

import json

import pytest

from traceforge.graders import get_grader

G = get_grader("bfcl")


def native(calls):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]}}]}


def dsl(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def grade(resp, gt):
    return G.grade(prompt_messages=[], response=resp, ground_truth={"bfcl": gt})


GT = [{"calc_area": {"base": [10], "height": [5], "unit": ["units", "cm", ""]}}]


def test_any_of_value_and_optional_omitted():
    assert grade(native([("calc_area", {"base": 10, "height": 5})]), GT).correct
    assert grade(native([("calc_area", {"base": 10, "height": 5, "unit": "cm"})]), GT).correct
    assert grade(dsl('[calc_area(base=10, height=5, unit="units")]'), GT).correct


def test_value_outside_allowed_set_fails():
    v = grade(native([("calc_area", {"base": 10, "height": 5, "unit": "inches"})]), GT)
    assert not v.correct and v.detail.endswith("wrong_values:unit")


def test_required_missing_fails():
    v = grade(native([("calc_area", {"base": 10})]), GT)
    assert not v.correct and "missing_args:height" in v.detail


def test_invented_parameter_fails():
    v = grade(native([("calc_area", {"base": 10, "height": 5, "colour": "red"})]), GT)
    assert not v.correct and "extra_args:colour" in v.detail


def test_irrelevance():
    assert grade(dsl("I cannot help with that."), []).correct
    assert grade(dsl("[]"), []).correct
    assert not grade(native([("calc_area", {"base": 1, "height": 1})]), []).correct


def test_malformed_gold_raises():
    with pytest.raises(ValueError):
        G.grade(prompt_messages=[], response=dsl("[]"), ground_truth=[{"name": "x"}])


def test_real_answer_keys_separate():
    """Known-good built from every real key must pass; one perturbed value must fail."""
    bfcl = pytest.importorskip("traceforge.data.bfcl")
    try:
        items = bfcl.load_bfcl(per_category=40, categories=("simple", "multiple", "parallel", "parallel_multiple"))
    except Exception as e:  # noqa: BLE001 — offline / no HF access
        pytest.skip(f"BFCL not available: {type(e).__name__}")
    good = bad = bad_total = 0
    for it in items:
        calls = []
        for c in it.ground_truth["bfcl"]:
            (fn, allowed), = c.items()
            calls.append((fn, {k: v[0] for k, v in allowed.items() if v and v[0] != ""}))
        good += grade(native(calls), it.ground_truth["bfcl"]).correct
        for fn, args in calls:
            if args:
                k = next(iter(args))
                args[k] = (args[k] + 9999) if isinstance(args[k], (int, float)) and not isinstance(args[k], bool) \
                    else f"WRONG_{args[k]}"
                bad_total += 1
                bad += not grade(native(calls), it.ground_truth["bfcl"]).correct
                break
    print(f"\nbfcl known-good {good}/{len(items)} | known-bad caught {bad}/{bad_total}")
    assert good == len(items) and bad == bad_total
