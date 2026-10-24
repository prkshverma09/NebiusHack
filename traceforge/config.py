"""Environment loading. Contract §2.

Fails loudly on a missing key: a config error must never look like an empty result
(see spec/REVIEW-traceforge-findings.md on silent-absence bugs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when required configuration is absent or unusable."""


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader — no dependency, tolerant of comments and quotes."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str
    db_path: Path
    upstream_base_url: str
    teacher_model: str
    student_base: str
    port: int

    @property
    def v0_base_url(self) -> str:
        """Deploy/model-management API lives under /v0, not /v1 (TF-07)."""
        return self.upstream_base_url.rstrip("/").removesuffix("/v1") + "/v0"


def load_settings(*, require_key: bool = True) -> Settings:
    _load_dotenv()
    key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if require_key and not key:
        raise ConfigError(
            "NEBIUS_API_KEY is not set. Put it in .env at the repo root "
            "(NEBIUS_API_KEY=...) or export it. Nothing can be submitted without it."
        )
    db = Path(os.environ.get("TF_DB_PATH", str(REPO_ROOT / "data" / "traceforge.db")))
    db.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        nebius_api_key=key,
        db_path=db,
        upstream_base_url=os.environ.get(
            "UPSTREAM_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
        ),
        teacher_model=os.environ.get("TEACHER_MODEL", "deepseek-ai/DeepSeek-V3-0324"),
        student_base=os.environ.get("STUDENT_BASE", "meta-llama/Llama-3.1-8B-Instruct"),
        port=int(os.environ.get("TF_PORT", "8000")),
    )
