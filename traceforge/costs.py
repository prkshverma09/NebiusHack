"""Per-token cost from Token Factory's own published prices (GET /v1/models?verbose=true).

Prices are loaded live and snapshotted to data/pricing.json; nothing is hardcoded.
An unknown model costs None, never 0 -- an unpriced call must not read as free.
Dedicated endpoints bill per GPU-hour, so they are deliberately absent from this table.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx


class PriceTable:
    def __init__(self, prices: dict[str, tuple[float, float]]):
        self.prices = prices          # model -> (usd per prompt token, usd per completion token)

    @classmethod
    def fetch(cls, base_url: str, api_key: str, snapshot: Path) -> "PriceTable":
        """Live prices; falls back to the last snapshot if the catalogue is unavailable."""
        try:
            r = httpx.get(f"{base_url.rstrip('/')}/models", params={"verbose": "true"},
                          headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
            r.raise_for_status()
            prices = {}
            for m in r.json()["data"]:
                p = m.get("pricing") or {}
                if p.get("prompt") is not None and p.get("completion") is not None:
                    prices[m["id"]] = (float(p["prompt"]), float(p["completion"]))
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(json.dumps(prices, indent=1), encoding="utf-8")
            return cls(prices)
        except (httpx.HTTPError, ValueError, KeyError):
            if snapshot.exists():
                raw = json.loads(snapshot.read_text(encoding="utf-8"))
                return cls({k: tuple(v) for k, v in raw.items()})
            raise

    def cost_usd(self, model: str, usage: dict | None) -> float | None:
        if not usage or model not in self.prices:
            return None
        pin, pout = self.prices[model]
        return (usage.get("prompt_tokens") or 0) * pin + (usage.get("completion_tokens") or 0) * pout
