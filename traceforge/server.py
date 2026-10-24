"""FastAPI app (A6). Contract section 6.

  uvicorn traceforge.server:create_app --factory --port 8000

`create_app(services)` is importable with zero env: tests inject Services directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from traceforge.config import REPO_ROOT, Settings, load_settings
from traceforge.costs import PriceTable
from traceforge.db import DB
from traceforge.gateway import Gateway, GoldIndex, load_regions
from traceforge.orchestrator import Orchestrator, OrchestratorError
from traceforge.router import GateError, demote, promote, to_shadow

DASHBOARD = REPO_ROOT / "dashboard" / "index.html"
EVALS_DIR = REPO_ROOT / "data" / "evals"
HOLDOUTS = {"toolace": REPO_ROOT / "data/datasets/toolace-fc-v1/holdout.jsonl"}


@dataclass
class Services:
    settings: Settings
    db: DB
    gateway: Gateway
    orchestrator: Orchestrator

    @classmethod
    def build(cls) -> "Services":
        s = load_settings()
        db = DB(s.db_path)
        prices = PriceTable.fetch(s.upstream_base_url, s.nebius_api_key,
                                  REPO_ROOT / "data" / "pricing.json")
        from traceforge.graders import get_grader   # B-owned registry
        gw = Gateway(db=db, prices=prices, upstream_base_url=s.upstream_base_url,
                     api_key=s.nebius_api_key, gold=GoldIndex.from_holdouts(HOLDOUTS),
                     grader_for=get_grader,
                     regions=load_regions(REPO_ROOT / "data" / "endpoints.jsonl"))
        return cls(settings=s, db=db, gateway=gw, orchestrator=Orchestrator(s))


def _eval_runs() -> list[dict]:
    """B writes every run to data/evals/<id>.json; newest first."""
    runs = []
    for p in sorted(EVALS_DIR.glob("ev-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        runs.append({"eval_run_id": d["eval_run_id"], "benchmark": d.get("benchmark"),
                     "created_at": d.get("created_at"), "n_items": len(d.get("items", [])),
                     "accuracy": d["table"]["accuracy"], "paired": d["table"]["paired"]})
    return runs


def create_app(services: Services | None = None) -> FastAPI:
    svc = services or Services.build()
    app = FastAPI(title="TraceForge")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        status, body, ctype = await svc.gateway.proxy(
            await request.body(), task_header=request.headers.get("x-tf-task"),
            session_id=request.headers.get("x-tf-session"))
        return Response(content=body, status_code=status, media_type=ctype)

    @app.get("/traces")
    def traces(task_id: str | None = None, grade: str | None = None, limit: int = 100):
        rows = svc.db.traces(task_id=task_id, grade=grade, limit=limit)
        for r in rows:   # keep the feed light: the user turn and the answer, not full JSON
            req = json.loads(r.pop("request_json") or "{}")
            resp = json.loads(r.pop("response_json") or "null") or {}
            r["user"] = next((m.get("content") for m in reversed(req.get("messages", []))
                              if m.get("role") == "user"), "")
            ch = (resp.get("choices") or [{}])[0].get("message") or {}
            r["answer"] = ch.get("content")
        return {"traces": rows}

    @app.get("/tasks")
    def tasks():
        return {"tasks": svc.db.tasks()}

    @app.post("/grades/run")
    async def grades_run(request: Request):
        body = await request.json() if await request.body() else {}
        return {"graded": svc.gateway.grade_pending(body.get("task_id"), body.get("limit", 500))}

    @app.get("/models")
    def models():
        return svc.db.models()

    @app.post("/models/{model_id}/promote")
    def promote_model(model_id: str):
        try:
            promote(svc.db, model_id, teacher=svc.settings.teacher_model)
        except GateError as e:
            raise HTTPException(409, str(e)) from e
        return {"ok": True}

    @app.post("/models/{model_id}/demote")
    def demote_model(model_id: str):
        demote(svc.db, model_id)
        return {"ok": True}

    @app.get("/evals")
    def evals():
        return {"runs": _eval_runs()}

    @app.get("/stats/cost")
    def cost():
        """Priced from recorded tokens x live list price for EVERY row. B's runner rows
        carry tokens but no cost_usd; summing cost_usd skipped them and showed the
        teacher as cheaper than the student. Unpriceable rows are counted, not hidden."""
        out = []
        for r in svc.db.cost_by_model():
            usd = svc.gateway.prices.cost_usd(r["model"], {
                "prompt_tokens": r["prompt_tokens"] or 0,
                "completion_tokens": r["completion_tokens"] or 0})
            out.append({**r, "usd": usd,
                        "usd_per_call": usd / r["n"] if usd is not None and r["n"] else None,
                        "priced": usd is not None})
        return {"by_model": out}

    @app.get("/jobs")
    def jobs():
        """Live status of every job submitted via scripts/submit_job.py."""
        out = []
        base = svc.settings.upstream_base_url.rstrip("/")
        hdr = {"Authorization": f"Bearer {svc.settings.nebius_api_key}"}
        for p in sorted((REPO_ROOT / "data").glob("job-*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            row = {"job_id": rec["job_id"], "model": rec["request"]["model"],
                   "submitted_at": rec["submitted_at"]}
            try:
                j = httpx.get(f"{base}/fine_tuning/jobs/{rec['job_id']}", headers=hdr,
                              timeout=15).json()
                row.update(status=j.get("status"), trained_tokens=j.get("trained_tokens"),
                           error=j.get("error"))
            except (httpx.HTTPError, ValueError) as e:
                row.update(status="unknown", error=f"status check failed: {e}")
            out.append(row)
        return {"jobs": out}

    @app.post("/models/{model_id}/shadow")
    def shadow_model_ep(model_id: str):
        try:
            to_shadow(svc.db, model_id)
        except GateError as e:
            raise HTTPException(404, str(e)) from e
        return {"ok": True}

    @app.post("/replay")
    async def replay(request: Request):
        body = await request.json()
        try:
            return await svc.gateway.replay(body["trace_id"], body["model"])
        except KeyError as e:
            raise HTTPException(404, f"no trace {e}") from e

    @app.post("/datasets")
    async def datasets(request: Request):
        """B's verdict-filtered builder over the model's *correct* traces."""
        body = await request.json()
        from traceforge.datasets import DatasetError, build   # B-owned
        try:
            m = build(model=body.get("model", svc.settings.teacher_model), name=body["name"],
                      holdout=HOLDOUTS["toolace"], db_path=svc.settings.db_path,
                      task_id=body.get("task_id"), source=body.get("source"))
        except DatasetError as e:
            raise HTTPException(422, str(e)) from e
        return {"dataset_id": body["name"], **m}

    @app.post("/distill")
    async def distill(request: Request):
        body = await request.json()
        rows = [d for d in svc.db._q("SELECT * FROM datasets WHERE id=? OR name=?",
                                     (body["dataset_id"], body["dataset_id"]))]
        if not rows:
            raise HTTPException(404, f"no dataset {body['dataset_id']}")
        d = rows[0]
        model = body.get("model", svc.settings.student_base)
        try:
            out = svc.orchestrator.submit(
                train=Path(d["train_path"]), val=Path(d["val_path"]) if d["val_path"] else None,
                model=model, suffix=body.get("suffix", "pupil-fc"), overrides=body.get("hp"))
        except OrchestratorError as e:
            raise HTTPException(502, str(e)) from e
        job = out["job"]
        svc.db.upsert_job(nebius_job_id=job["id"], base_model=model, status=job.get("status"),
                          hyperparams=out["request"]["hyperparameters"], dataset_id=d["id"])
        return {"job_id": job["id"], "status": job.get("status")}

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        try:
            st = svc.orchestrator.status(job_id)
        except OrchestratorError as e:
            raise HTTPException(502, str(e)) from e
        svc.db.upsert_job(nebius_job_id=job_id, base_model=st["model"], status=st["status"],
                          trained_tokens=st["trained_tokens"],
                          error=json.dumps(st["error"]) if st["error"] else None)
        ck = st["checkpoints"]
        return {**st, "checkpoint_id": ck[-1]["id"] if ck else None}

    @app.get("/")
    def dashboard():
        if not DASHBOARD.exists():
            return JSONResponse({"error": "dashboard/index.html missing"}, status_code=500)
        return FileResponse(DASHBOARD)

    return app
