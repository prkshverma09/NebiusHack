"""Eval runner + stats against fake models with known accuracy (contract B4 acceptance)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import traceforge.eval as ev
from traceforge.graders import get_grader

GOLD = '[get_weather(city="Paris")]'


class FakeAPIError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class FakeClient:
    """`answers[model]` = set of item indexes the model gets right; `down` models always 503."""

    def __init__(self, answers: dict[str, set[int]], down: set[str] = frozenset()):
        self.answers, self.down, self.calls = answers, down, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **_):
        self.calls += 1
        if model in self.down:
            raise FakeAPIError(503)
        idx = int(messages[-1]["content"].split("#")[1])
        text = GOLD if idx in self.answers[model] else '[get_weather(city="Rome")]'
        msg = {"role": "assistant", "content": text}
        return SimpleNamespace(
            model_dump=lambda: {"choices": [{"message": msg}]},
            usage=SimpleNamespace(total_tokens=10))


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ev, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(ev.time, "sleep", lambda *_: None)


def items(n=20):
    return [ev.EvalItem(f"i{k:02d}", [{"role": "user", "content": f"q #{k}"}], GOLD) for k in range(n)]


def test_mcnemar_known_values():
    assert ev.mcnemar_exact(0, 0) == 1.0
    assert ev.mcnemar_exact(5, 5) == 1.0
    assert ev.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)          # 2 * 0.5**10
    assert ev.mcnemar_exact(0, 10) == ev.mcnemar_exact(10, 0)           # symmetric


def test_fixture_20_items_two_models():
    # base right on 0..9 (50%), pupil right on 0..17 (90%): pupil-only wins = 8, losses = 0
    c = FakeClient({"base": set(range(10)), "pupil": set(range(18))})
    res = ev.run_eval(["base", "pupil"], items(), get_grader("toolace"), c, progress=False)
    t = ev.score_table(res, ["base", "pupil"], "base")
    acc = {a["model"]: a["acc"] for a in t["accuracy"]}
    assert acc == {"base": 0.5, "pupil": 0.9}
    p = t["paired"][0]
    assert (p["b_only"], p["a_only"], p["n_paired"]) == (8, 0, 20)
    assert p["delta"] == pytest.approx(0.4)
    assert p["mcnemar_p"] == pytest.approx(2 / 256)
    lo, hi = p["ci95"]
    assert lo > 0 and lo <= 0.4 <= hi                                   # CI excludes zero
    assert "| `pupil` | +40.0%" in t["markdown"]


def test_outage_is_error_not_incorrect_and_excluded():
    c = FakeClient({"base": set(range(10))}, down={"teacher"})
    res = ev.run_eval(["base", "teacher"], items(), get_grader("toolace"), c, progress=False)
    a = {x["model"]: x for x in ev.score_table(res, ["base", "teacher"], "base")["accuracy"]}
    assert a["teacher"]["errors"] == 20 and a["teacher"]["acc"] is None    # not 0%
    p = ev.paired_stats(res, "base", "teacher")
    assert p["n_paired"] == 0 and p["excluded"] == 20 and p["delta"] is None


def test_4xx_is_not_retried():
    c = FakeClient({"m": set()})
    c.down = set()
    orig = c._create

    def bad(**kw):
        c.calls += 1
        raise FakeAPIError(404)
    c.chat.completions.create = bad
    out = ev.call_model(c, "m", items(1)[0])
    assert out["error"] == "FakeAPIError:404" and c.calls == 1


def test_cache_makes_rerun_free_and_skips_errors():
    c = FakeClient({"base": set(range(5))})
    its = items(5)
    ev.run_eval(["base"], its, get_grader("toolace"), c, progress=False)
    first = c.calls
    ev.run_eval(["base"], its, get_grader("toolace"), c, progress=False)
    assert first == 5 and c.calls == 5                                  # second run fully cached


def test_persist_writes_json(tmp_path):
    c = FakeClient({"base": {0}, "pupil": {0, 1}})
    res = ev.run_eval(["base", "pupil"], items(2), get_grader("toolace"), c, progress=False)
    t = ev.score_table(res, ["base", "pupil"], "base")
    rid = ev.persist(res, t, benchmark="fixture", models=["base", "pupil"])
    assert (ev.RUNS_DIR / f"{rid}.json").exists()
