"""Day-0 verification against the real Nebius Token Factory (A0.2 / TF-07, TF-08).

Answers, with evidence, the four questions the whole plan rests on:

  1. Does the key authenticate, and is the student base model (TF-01) actually listed?
  2. Does a teacher chat call work, and how fast is it?
  3. Does file upload work (the first half of the fine-tune path)?
  4. **Does a LoRA checkpoint deploy path exist at all?**  This decides
     serverless-vs-dedicated NOW rather than at 15:00.

Probe design note: question 4 posts a *deliberately invalid* checkpoint id. That cannot
create or bill anything, and it discriminates between the two findings that matter:
  - route absent   -> 404/405 whose body is not about the checkpoint -> NO serverless deploy
  - route present  -> 4xx complaining about the checkpoint/model     -> deploy path EXISTS
A probe that cannot distinguish those is not a measurement.

Run:  python scripts/smoke_nebius.py
Exit: 0 = every must-pass check green; 1 = a must-pass check failed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from traceforge.config import ConfigError, load_settings

PASS, FAIL, WARN, INFO = "[PASS]", "[FAIL]", "[WARN]", "[info]"
_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str, must_pass: bool = True) -> None:
    tag = PASS if ok else (FAIL if must_pass else WARN)
    print(f"{tag} {name}: {note}", flush=True)
    if must_pass:
        _results.append((name, ok, note))


def section(title: str) -> None:
    print("\n" + "=" * 68 + "\n" + title + "\n" + "=" * 68, flush=True)


def main() -> int:
    try:
        s = load_settings()
    except ConfigError as e:
        print(f"{FAIL} config: {e}")
        return 1

    print(f"key    : ...{s.nebius_api_key[-6:]} ({len(s.nebius_api_key)} chars)")
    print(f"v1 base: {s.upstream_base_url}")
    print(f"v0 base: {s.v0_base_url}")
    print(f"teacher: {s.teacher_model}")
    print(f"student: {s.student_base}")

    auth = {"Authorization": f"Bearer {s.nebius_api_key}"}
    v1 = s.upstream_base_url.rstrip("/")
    v0 = s.v0_base_url
    findings: dict[str, object] = {}
    Path("data").mkdir(exist_ok=True)

    # ---- 1. auth + model catalogue ------------------------------------------------
    section("1. auth + model catalogue  (GET /v1/models)")
    student_listed = False
    try:
        with httpx.Client(timeout=30) as c:
            r = c.get(f"{v1}/models", headers=auth)
        if r.status_code != 200:
            record("auth", False, f"HTTP {r.status_code}: {r.text[:300]}")
        else:
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            record("auth", True, f"{len(ids)} models visible")
            student_listed = s.student_base in ids
            record("student base listed (TF-01)", student_listed, s.student_base)
            record("teacher listed", s.teacher_model in ids, s.teacher_model)
            findings["n_models"] = len(ids)
            eight_b = [i for i in ids if "8B" in i or "8b" in i]
            print(f"{INFO} 8B-class models: {eight_b[:12]}")
            Path("data/models_catalogue.json").write_text(
                json.dumps(sorted(ids), indent=2), encoding="utf-8")
            print(f"{INFO} full catalogue -> data/models_catalogue.json")
    except Exception as e:
        record("auth", False, f"{type(e).__name__}: {e}")

    # ---- 2. teacher chat call -----------------------------------------------------
    section("2. teacher chat call  (POST /v1/chat/completions)")
    try:
        t0 = time.time()
        with httpx.Client(timeout=120) as c:
            r = c.post(f"{v1}/chat/completions", headers=auth, json={
                "model": s.teacher_model,
                "messages": [{"role": "user",
                              "content": "Reply with the single word: ready"}],
                "max_tokens": 16,
                "temperature": 0,
            })
        dt = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            body = r.json()
            txt = (body["choices"][0]["message"].get("content") or "").strip()
            record("teacher chat", True,
                   f"{dt}ms, usage={body.get('usage', {})}, said={txt[:40]!r}")
            findings["teacher_latency_ms"] = dt
        else:
            record("teacher chat", False, f"HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        record("teacher chat", False, f"{type(e).__name__}: {e}")

    # ---- 3. training-file upload --------------------------------------------------
    section("3. training-file upload  (POST /v1/files)")
    file_id = None
    tool_row_accepted = None
    try:
        # Row 2 carries a tool_call: settles contract section 7's open question in one shot.
        tool_call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "get_weather",
                         "arguments": json.dumps({"city": "Paris"})},
        }
        rows = [
            {"messages": [{"role": "user", "content": "ping"},
                          {"role": "assistant", "content": "pong"}]},
            {"messages": [{"role": "system", "content": "You call tools."},
                          {"role": "user", "content": "weather in Paris"},
                          {"role": "assistant", "content": None,
                           "tool_calls": [tool_call]}]},
            {"messages": [{"role": "user", "content": "2+2"},
                          {"role": "assistant", "content": "4"}]},
        ]
        tmp = Path("data/_smoke_dummy.jsonl")
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        with httpx.Client(timeout=120) as c:
            r = c.post(f"{v1}/files", headers=auth,
                       files={"file": (tmp.name, tmp.read_bytes(),
                                       "application/jsonl")},
                       data={"purpose": "fine-tune"})
        if r.status_code in (200, 201):
            file_id = r.json().get("id")
            record("file upload", True, f"id={file_id}")
            tool_row_accepted = True
            findings["tool_calls_in_sft"] = "accepted at upload"
        else:
            record("file upload", False, f"HTTP {r.status_code}: {r.text[:400]}")
            tool_row_accepted = False
    except Exception as e:
        record("file upload", False, f"{type(e).__name__}: {e}")
    print(f"{INFO} tool_calls row accepted at upload: {tool_row_accepted} "
          "(contract section 7 -- validation may still reject at job creation)")

    # ---- 4. fine-tune job API reachable -------------------------------------------
    section("4. fine-tune job API reachable  (GET /v1/fine_tuning/jobs)")
    try:
        with httpx.Client(timeout=60) as c:
            r = c.get(f"{v1}/fine_tuning/jobs", headers=auth, params={"limit": 5})
        if r.status_code == 200:
            jobs = r.json().get("data", [])
            record("jobs API", True, f"{len(jobs)} existing job(s)")
            for j in jobs[:5]:
                print(f"{INFO}   job {j.get('id')} status={j.get('status')} "
                      f"model={j.get('model')} finished={j.get('finished_at')}")
            findings["existing_jobs"] = [
                {"id": j.get("id"), "status": j.get("status")} for j in jobs[:5]]
        else:
            record("jobs API", False, f"HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        record("jobs API", False, f"{type(e).__name__}: {e}")

    # ---- 5. THE DEPLOY PROBE (TF-07) ----------------------------------------------
    section("5. LoRA deploy path  (POST /v0/models)  <-- TF-07, the decision")
    verdict = "unknown"
    try:
        with httpx.Client(timeout=60) as c:
            g = c.get(f"{v0}/models", headers=auth)
        print(f"{INFO} GET {v0}/models -> {g.status_code} {g.text[:300]}")

        with httpx.Client(timeout=60) as c:
            p = c.post(f"{v0}/models", headers=auth, json={
                "name": "tf-smoke-probe-ignore",
                "checkpoint_id": "ftckpt-DOES-NOT-EXIST-smoke-probe",
            })
        body = p.text[:500]
        print(f"{INFO} POST {v0}/models -> {p.status_code} {body}")

        mentions_resource = any(k in body.lower() for k in (
            "checkpoint", "not found", "does not exist", "invalid", "ftckpt"))
        if p.status_code in (200, 201):
            verdict = "EXISTS (accepted a bogus id -- inspect and clean up manually)"
            record("deploy route (TF-07)", True, verdict)
        elif p.status_code in (400, 404, 422) and mentions_resource:
            verdict = "EXISTS -- route live, rejected the fake checkpoint as expected"
            record("deploy route (TF-07)", True, verdict)
        elif p.status_code in (401, 403):
            verdict = f"GATED -- HTTP {p.status_code}; key lacks deploy scope"
            record("deploy route (TF-07)", False, verdict)
        elif p.status_code in (404, 405):
            verdict = (f"ABSENT -- HTTP {p.status_code}, no deploy route. "
                       "PIVOT to dedicated endpoint.")
            record("deploy route (TF-07)", False, verdict)
        else:
            verdict = f"INCONCLUSIVE -- HTTP {p.status_code}"
            record("deploy route (TF-07)", False, verdict)
        findings["deploy_probe"] = {"status": p.status_code, "verdict": verdict,
                                    "get_status": g.status_code, "body": body}
    except Exception as e:
        verdict = f"INCONCLUSIVE -- {type(e).__name__}: {e}"
        record("deploy route (TF-07)", False, verdict)

    # ---- summary ------------------------------------------------------------------
    section("SUMMARY")
    failed = [n for n, ok, _ in _results if not ok]
    for name, ok, note in _results:
        print(f"{'  ok  ' if ok else ' FAIL '} {name}: {note}")
    print(f"\nDEPLOY PATH (TF-07): {verdict}")

    Path("data/STATUS.md").write_text(
        "# Day-0 smoke findings\n\n"
        f"- run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- student base listed: {student_listed}\n"
        f"- training file id: {file_id}\n"
        f"- tool_calls row accepted at upload: {tool_row_accepted}\n"
        f"- **deploy path (TF-07): {verdict}**\n\n"
        "```json\n" + json.dumps(findings, indent=2) + "\n```\n",
        encoding="utf-8")
    print(f"\n{INFO} findings written to data/STATUS.md")

    if failed:
        print(f"\n{FAIL} {len(failed)} must-pass check(s) failed: {failed}")
        return 1
    print(f"\n{PASS} all must-pass checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
