"""Nebius fine-tune + deploy calls as a library (A4). The scripts/ CLIs predate this
module and stay standalone; the server's /distill and /jobs use this.

Deploy path is a dedicated endpoint with custom weights (TF-07: /v0/models is 410).
Only flavors whose template has a `base_model_slug` accept custom weights -- for this
project that rules out every Llama-3.1-8B flavor (see plans/FINDINGS-day0.md).
"""
from __future__ import annotations

from pathlib import Path

import httpx

from traceforge.config import Settings

# Contract A4 values except lr (1e-4: with packing the job is only tens of steps).
HPARAMS = {"lora": True, "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.05,
           "n_epochs": 3, "learning_rate": 1e-4, "batch_size": 8, "packing": True,
           "context_length": 8192}

# Per-model minimums the API enforces (measured: Qwen3-30B rejects < 131072 tokens/batch).
MIN_TOKENS_PER_BATCH = {"Qwen/Qwen3-30B-A3B-Instruct-2507": 131072}

# Custom-weights-capable configuration for the student (templates listing, 14:40).
DEPLOY_CONFIG = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507", "flavor_name": "base",
        "gpu_type": "gpu-h200-sxm", "region": "us-central1", "gpu_count": 1},
}


class OrchestratorError(RuntimeError):
    """A Nebius call failed; the message carries the HTTP status and body."""


class Orchestrator:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.s = settings
        self.v1 = settings.upstream_base_url.rstrip("/")
        self.v0 = settings.v0_base_url
        self.client = client or httpx.Client(timeout=300)
        self.headers = {"Authorization": f"Bearer {settings.nebius_api_key}"}

    def _call(self, method: str, url: str, ok=(200, 201), **kw) -> dict:
        r = self.client.request(method, url, headers=self.headers, **kw)
        if r.status_code not in ok:
            raise OrchestratorError(f"{method} {url} -> HTTP {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else {}

    def hyperparams(self, model: str, overrides: dict | None = None) -> dict:
        hp = {**HPARAMS, **(overrides or {})}
        need = MIN_TOKENS_PER_BATCH.get(model)
        if need and hp["batch_size"] * hp["context_length"] < need:
            hp["batch_size"] = -(-need // hp["context_length"])   # ceil
        return hp

    def upload(self, path: Path) -> str:
        return self._call("POST", f"{self.v1}/files",
                          files={"file": (path.name, path.read_bytes(), "application/jsonl")},
                          data={"purpose": "fine-tune"})["id"]

    def submit(self, *, train: Path, val: Path | None, model: str | None = None,
               suffix: str, overrides: dict | None = None) -> dict:
        model = model or self.s.student_base
        body = {"model": model, "training_file": self.upload(train),
                "hyperparameters": self.hyperparams(model, overrides),
                "suffix": suffix, "seed": 42}
        if val:
            body["validation_file"] = self.upload(val)
        job = self._call("POST", f"{self.v1}/fine_tuning/jobs", json=body)
        return {"job": job, "request": body}

    def status(self, job_id: str) -> dict:
        j = self._call("GET", f"{self.v1}/fine_tuning/jobs/{job_id}")
        ck = self._call("GET", f"{self.v1}/fine_tuning/jobs/{job_id}/checkpoints")
        return {"status": j.get("status"), "model": j.get("model"),
                "trained_tokens": j.get("trained_tokens"), "error": j.get("error"),
                "checkpoints": ck.get("data", [])}

    def deploy(self, *, name: str, base_model: str, custom_weights_id: str) -> dict:
        """custom_weights_id must be a `model-artifact_...` id. How a fine-tuning
        checkpoint becomes one is not in the public API (open question for Nebius)."""
        if not custom_weights_id.startswith("model-artifact_"):
            raise OrchestratorError(
                f"custom_weights_id must start with 'model-artifact_', got {custom_weights_id!r}")
        cfg = DEPLOY_CONFIG.get(base_model)
        if cfg is None:
            raise OrchestratorError(f"no custom-weights deploy config for {base_model}")
        return self._call("POST", f"{self.v0}/dedicated_endpoints", json={
            "name": name, **cfg, "scaling": {"min_replicas": 1, "max_replicas": 1},
            "custom_weights_id": custom_weights_id})["endpoint"]
