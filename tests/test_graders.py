"""Separation tests for the tool-call grader.

Each case asserts BOTH directions: a known-correct call passes and a
deliberately-wrong one fails. A grader that only ever says True would pass
half of these, so half is a failure.
"""
from __future__ import annotations

import pytest

from traceforge.graders import get_grader, UnknownGraderError
from traceforge.graders.bfcl_ast import ToolCallGrader

G = ToolCallGrader()


def resp(*calls, content=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [
            {"id": f"c{i}", "type": "function",
             "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": msg}]}


def g(response, gold):
    return G.grade(prompt_messages=[], response=response, ground_truth=gold)


# --- simple -----------------------------------------------------------------
def test_simple_exact():
    v = g(resp(("get_weather", '{"city": "Paris"}')), [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    assert v.correct, v.detail


def test_simple_wrong_value_fails():
    v = g(resp(("get_weather", '{"city": "Berlin"}')), [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    assert not v.correct and "wrong_values:city" in v.detail


def test_simple_wrong_function_fails():
    v = g(resp(("get_time", '{"city": "Paris"}')), [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    assert not v.correct and "no_match_for:get_weather" in v.detail


def test_missing_required_arg_fails():
    v = g(resp(("book", '{"from": "JFK"}')), [{"name": "book", "arguments": {"from": "JFK", "to": "LAX"}}])
    assert not v.correct and "missing_args:to" in v.detail


# --- the coercion policy (base.normalise_scalar) ----------------------------
def test_numeric_string_coerces():
    v = g(resp(("set", '{"n": "2"}')), [{"name": "set", "arguments": {"n": 2}}])
    assert v.correct, v.detail


def test_case_insensitive_string():
    v = g(resp(("find", '{"city": "new york"}')), [{"name": "find", "arguments": {"city": "New York"}}])
    assert v.correct, v.detail


def test_bool_not_confused_with_int():
    v = g(resp(("flag", '{"x": 1}')), [{"name": "flag", "arguments": {"x": True}}])
    assert not v.correct, v.detail


def test_list_order_is_significant():
    v = g(resp(("route", '{"stops": ["LAX", "JFK"]}')), [{"name": "route", "arguments": {"stops": ["JFK", "LAX"]}}])
    assert not v.correct, v.detail


def test_extra_arg_tolerated_but_reported():
    v = g(resp(("get_weather", '{"city": "Paris", "unit": "c"}')),
          [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    assert v.correct and "extra:unit" in v.detail


# --- parallel ---------------------------------------------------------------
def test_parallel_order_insensitive():
    gold = [{"name": "w", "arguments": {"c": "Paris"}}, {"name": "w", "arguments": {"c": "Rome"}}]
    v = g(resp(("w", '{"c": "Rome"}'), ("w", '{"c": "Paris"}')), gold)
    assert v.correct, v.detail


def test_parallel_missing_one_fails():
    gold = [{"name": "w", "arguments": {"c": "Paris"}}, {"name": "w", "arguments": {"c": "Rome"}}]
    v = g(resp(("w", '{"c": "Paris"}')), gold)
    assert not v.correct and "call_count" in v.detail


def test_parallel_needs_real_matching_not_greedy():
    """Greedy would give got[0] (x=1,y=2) to gold[0] (x=1) and fail gold[1]."""
    gold = [{"name": "A", "arguments": {"x": 1}}, {"name": "A", "arguments": {"x": 1, "y": 2}}]
    v = g(resp(("A", '{"x": 1, "y": 2}'), ("A", '{"x": 1}')), gold)
    assert v.correct, v.detail


def test_failure_reason_is_the_failing_calls_own():
    """Regression: a later call's miss was reported as the earlier call's 'match' note."""
    gold = [{"name": "A", "arguments": {"x": 1}}, {"name": "B", "arguments": {"y": 2}}]
    v = g(resp(("A", '{"x": 1}'), ("B", '{"y": 3}')), gold)
    assert not v.correct and v.detail == "B:wrong_values:y"


# --- irrelevance ------------------------------------------------------------
def test_irrelevance_abstains_correctly():
    v = g(resp(content="I can't help with that."), [])
    assert v.correct and v.detail == "irrelevance:abstained"


def test_irrelevance_calling_anyway_fails():
    v = g(resp(("get_weather", '{"city": "Paris"}')), [])
    assert not v.correct and v.detail.startswith("irrelevance:called")


# --- degraded form (contract §7 fallback) -----------------------------------
def test_json_in_content_is_graded():
    r = resp(content='{"name": "get_weather", "arguments": {"city": "Paris"}}')
    v = g(r, [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    assert v.correct, v.detail


def test_malformed_json_content_fails_not_crashes():
    v = g(resp(content='{"name": "get_weather", "arg'), [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    assert not v.correct


def test_no_call_when_one_expected_fails():
    v = g(resp(content="Sure, let me check."), [{"name": "get_weather", "arguments": {"city": "Paris"}}])
    assert not v.correct and v.detail == "no_call_emitted"


# --- registry ---------------------------------------------------------------
def test_registry_known_and_unknown():
    assert get_grader("bfcl").name == "bfcl"
    assert get_grader("XLAM") is not None
    with pytest.raises(UnknownGraderError):
        get_grader("does_not_exist")
