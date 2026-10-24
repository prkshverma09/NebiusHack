"""ToolACE DSL parser + grader, unit cases and a real-holdout separation check."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceforge.graders import get_grader
from traceforge.graders.toolace_dsl import DSLParseError, parse_calls

G = get_grader("toolace")
HOLDOUT = Path(__file__).resolve().parent.parent / "data/datasets/toolace-fc-v1/holdout.jsonl"


def content(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def native(calls):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
        for i, c in enumerate(calls)]}}]}


def grade(resp, gold):
    return G.grade(prompt_messages=[], response=resp, ground_truth=gold)


# --- parser units -----------------------------------------------------------
def test_example_from_the_brief():
    assert parse_calls('[Market Trends API(trend_type="MARKET_INDEXES", country="us")]') == [
        {"name": "Market Trends API", "arguments": {"trend_type": "MARKET_INDEXES", "country": "us"}}]


def test_no_args_and_dotted_names():
    assert parse_calls("[Get Categories of Jokes(), golf_course.get_review(course_name=\"X\")]") == [
        {"name": "Get Categories of Jokes", "arguments": {}},
        {"name": "golf_course.get_review", "arguments": {"course_name": "X"}}]


def test_nested_and_tricky_strings():
    out = parse_calls('[f(xml="<t u=\'c\'>18</t>, (x)", d={"a": [1, -2.5, true, null]})]')
    assert out == [{"name": "f", "arguments": {"xml": "<t u='c'>18</t>, (x)",
                                                "d": {"a": [1, -2.5, True, None]}}}]


def test_empty_list_is_explicit_no_call():
    assert parse_calls("[]") == []


@pytest.mark.parametrize("bad", ['[f(x="unterminated)]', "[f(x=1]", "f(x=1)", "[f(x=__import__('os'))]"])
def test_malformed_raises(bad):
    with pytest.raises(DSLParseError):
        parse_calls(bad)


def test_unparseable_model_output_is_a_distinct_verdict():
    v = grade(content('[f(x="oops)]'), '[f(x="ok")]')
    assert not v.correct and v.detail.startswith("unparseable_call")


# --- real holdout separation (known-good must pass, known-bad must fail) -----
def _gold_answers():
    if not HOLDOUT.exists():
        pytest.skip("holdout not built")
    rows = [json.loads(l) for l in HOLDOUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["messages"][-1]["content"] for r in rows]


def _perturb(calls):
    """Change the first argument value we can find; None if the call has no args."""
    import copy
    out = copy.deepcopy(calls)
    for c in out:
        for k, v in c["arguments"].items():
            c["arguments"][k] = (v + 7) if isinstance(v, (int, float)) and not isinstance(v, bool) \
                else f"WRONG_{v}"
            return out
    return None


def test_holdout_separation():
    golds = _gold_answers()
    parsed, good_dsl, good_native, bad_caught, bad_total, fails = 0, 0, 0, 0, 0, []
    for gtext in golds:
        try:
            calls = parse_calls(gtext)
        except DSLParseError as e:
            fails.append((gtext[:90], str(e)))
            continue
        parsed += 1
        good_dsl += grade(content(gtext), gtext).correct
        good_native += grade(native(calls), gtext).correct
        wrong = _perturb(calls)
        if wrong is not None:
            bad_total += 1
            bad_caught += not grade(native(wrong), gtext).correct
    print(f"\nparsed {parsed}/{len(golds)} | known-good DSL {good_dsl}/{parsed} | "
          f"known-good native {good_native}/{parsed} | known-bad caught {bad_caught}/{bad_total}")
    for f in fails[:5]:
        print("  PARSE FAIL:", f)
    assert parsed == len(golds), f"{len(fails)} gold answers unparseable"
    assert good_dsl == parsed and good_native == parsed
    assert bad_caught == bad_total


def test_names_with_apostrophe_and_parens():
    """Regressions found on the real holdout (2/250 failed before the fix)."""
    assert parse_calls("[Get Today's Prices(), Open-High-Low-Close (OHLC) API(symbol=\"BTCUSD\")]") == [
        {"name": "Get Today's Prices", "arguments": {}},
        {"name": "Open-High-Low-Close (OHLC) API", "arguments": {"symbol": "BTCUSD"}}]


def test_keyword_and_nonidentifier_keys():
    """from=/class= are Python keywords, $select is not an identifier — all real ToolACE keys."""
    assert parse_calls('[Flights(from="2021-05-25", class="Business"), GetMySkills($select="coding")]') == [
        {"name": "Flights", "arguments": {"from": "2021-05-25", "class": "Business"}},
        {"name": "GetMySkills", "arguments": {"$select": "coding"}}]


def test_rest_path_names():
    """Real holdout names: REST paths, including {path params}."""
    assert parse_calls('[/scripts/inventory(), /{orgCode}/cards(muid="X")]') == [
        {"name": "/scripts/inventory", "arguments": {}},
        {"name": "/{orgCode}/cards", "arguments": {"muid": "X"}}]


def test_comma_inside_function_name():
    assert parse_calls('[Get All Champions(), Get Games by Platform, Category, and Sort(platform="pc")]') == [
        {"name": "Get All Champions", "arguments": {}},
        {"name": "Get Games by Platform, Category, and Sort", "arguments": {"platform": "pc"}}]


@pytest.mark.parametrize("broken", [
    '[database.insert_data=>["database_name"="company_db"]]',
    '[{Fetcher.retrieve}=>(location="Hawaii"; dateRange="x")]',
    '[advice|(air_quality_index:150)]',
    '[Get Values-("variable_id"="654321")]',
    '[{get_789bet_info}:()]',
    '[{Fetcher.retrieve}=>(location="Hawaii")]'])
def test_genuinely_broken_labels_still_rejected(broken):
    """Real malformed ToolACE labels. Loosening the parser must not start accepting these."""
    with pytest.raises(DSLParseError):
        parse_calls(broken)
