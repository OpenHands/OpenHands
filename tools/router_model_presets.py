"""Router-model preset resolver + local-runtime detection.

Presets: local / best-intelligence / cheapest / most-secure / custom.
Each resolves to {provider_key, model, goal, guardrails} plus a live
trade-off projection from the current registry.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from model_registry import RETENTION_NONE, WATERMARK_NONE, ModelRegistry
from router import (
    GOAL_COST,
    GOAL_PRIVACY,
    GOAL_QUALITY,
    QUALITY_FLOOR,
    default_guardrails,
    model_category_score,
)

PRESET_LOCAL = "local"
PRESET_BEST_INTELLIGENCE = "best-intelligence"
PRESET_CHEAPEST = "cheapest"
PRESET_MOST_SECURE = "most-secure"
PRESET_CUSTOM = "custom"
PRESETS = (
    PRESET_LOCAL,
    PRESET_BEST_INTELLIGENCE,
    PRESET_CHEAPEST,
    PRESET_MOST_SECURE,
    PRESET_CUSTOM,
)

RUNTIME_OLLAMA = "ollama"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
PROBE_CACHE_TTL_S = 15.0

FetchFn = Callable[[str, float], str]


class PresetError(ValueError):
    pass


def default_fetch(url: str, timeout_s: float = 2.0) -> str:
    request = Request(url, headers={"User-Agent": "openhands-agent-canvas-router/1.0"})
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _intelligence(model: dict[str, Any]) -> float:
    return (
        model_category_score(model, "reasoning") + model_category_score(model, "coding")
    ) / 2.0


def tradeoffs(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "cost_per_1k": model.get("cost_per_1k"),
        "latency_s_p90": model.get("latency_s_p90"),
        "retention": model.get("retention"),
        "watermark": model.get("watermark"),
        "classification_accuracy_proxy": round(_intelligence(model), 4),
        "offline_capable": bool(model.get("local")),
        "verified": bool(model.get("verified")),
    }


def _reachable(
    model: dict[str, Any],
    connected: set[str],
    runtimes: dict[str, Any],
) -> bool:
    if model.get("local"):
        runtime = model.get("runtime") or RUNTIME_OLLAMA
        info = runtimes.get(runtime) or {}
        if not info.get("alive"):
            return False
        installed = [str(name).lower() for name in info.get("models") or []]
        short = model["id"].split("/", 1)[-1].lower()
        if installed and not any(short in name or name in short for name in installed):
            return False
        return True
    if not connected:
        return True
    return model["provider_key"] in connected


class LocalRuntimeProbe:
    def __init__(
        self,
        *,
        fetch: FetchFn | None = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ttl_s: float = PROBE_CACHE_TTL_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.fetch = fetch or default_fetch
        self.ollama_url = ollama_url.rstrip("/")
        self.ttl_s = ttl_s
        self.clock = clock or time.monotonic
        self._cache: tuple[float, dict[str, Any]] | None = None

    def probe(self) -> dict[str, Any]:
        now = self.clock()
        if self._cache and now - self._cache[0] < self.ttl_s:
            return self._cache[1]
        result = {RUNTIME_OLLAMA: self._probe_ollama()}
        self._cache = (now, result)
        return result

    def _probe_ollama(self) -> dict[str, Any]:
        url = f"{self.ollama_url}/api/tags"
        try:
            raw = self.fetch(url, 2.0)
            payload = json.loads(raw)
            models = []
            for item in payload.get("models") or []:
                name = item.get("name") if isinstance(item, dict) else str(item)
                if name:
                    models.append(str(name))
            return {"alive": True, "models": models, "error": None}
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            return {"alive": False, "models": [], "error": str(exc)}


def best_local_by_category(
    models: list[dict[str, Any]],
    runtimes: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    installed = []
    for model in models:
        if not model.get("local"):
            continue
        if _reachable(model, set(), runtimes):
            installed.append(model)
    best: dict[str, dict[str, Any]] = {}
    for category in ("coding", "reasoning", "ux", "copy"):
        if not installed:
            continue
        winner = max(installed, key=lambda item: model_category_score(item, category))
        best[category] = {
            "id": winner["id"],
            "provider_key": winner["provider_key"],
            "score": model_category_score(winner, category),
        }
    return best


def resolve_preset(
    preset: str,
    registry: ModelRegistry,
    *,
    connected_providers: list[str] | None = None,
    runtimes: dict[str, Any] | None = None,
    custom: dict[str, Any] | None = None,
    quality_floor: float = QUALITY_FLOOR,
) -> dict[str, Any]:
    if preset not in PRESETS:
        raise PresetError(f"unknown preset {preset}")
    snapshot = registry.snapshot()
    models = snapshot["models"]
    connected = set(connected_providers or [])
    runtimes = runtimes or {}
    pool = [item for item in models if _reachable(item, connected, runtimes)]
    if preset == PRESET_LOCAL:
        pool = [item for item in pool if item.get("local")]
        goal = GOAL_QUALITY
        guardrails = default_guardrails()
        chosen = max(pool, key=_intelligence) if pool else None
    elif preset == PRESET_BEST_INTELLIGENCE:
        goal = GOAL_QUALITY
        guardrails = default_guardrails()
        chosen = max(pool, key=_intelligence) if pool else None
    elif preset == PRESET_CHEAPEST:
        goal = GOAL_COST
        guardrails = default_guardrails()
        above = [item for item in pool if _intelligence(item) >= quality_floor]
        use = above or pool
        chosen = min(use, key=lambda item: float(item.get("cost_per_1k") or 0)) if use else None
    elif preset == PRESET_MOST_SECURE:
        goal = GOAL_PRIVACY
        guardrails = {
            **default_guardrails(),
            "forbid_training_retention": True,
            "forbid_watermarking": True,
        }
        secure = [
            item
            for item in pool
            if item.get("retention") == RETENTION_NONE
            and item.get("watermark") == WATERMARK_NONE
        ]
        chosen = max(secure, key=_intelligence) if secure else None
    else:
        custom = custom or {}
        goal = custom.get("goal") or GOAL_QUALITY
        guardrails = {**default_guardrails(), **(custom.get("guardrails") or {})}
        model_id = custom.get("model")
        provider_key = custom.get("provider_key")
        chosen = next(
            (
                item
                for item in models
                if item["id"] == model_id
                and (not provider_key or item["provider_key"] == provider_key)
            ),
            None,
        )
        if chosen is None and model_id:
            chosen = {
                "id": model_id,
                "provider_key": provider_key or "custom",
                "cost_per_1k": None,
                "latency_s_p90": None,
                "retention": None,
                "watermark": None,
                "local": False,
                "verified": False,
                "benchmarks": {},
            }
    if chosen is None:
        raise PresetError(f"preset {preset} has no reachable candidate")
    resolved = {
        "preset": preset,
        "provider_key": chosen["provider_key"],
        "model": chosen["id"],
        "goal": goal,
        "guardrails": guardrails,
        "tradeoffs": tradeoffs(chosen),
        "pool": [
            {
                "id": item["id"],
                "provider_key": item["provider_key"],
                **tradeoffs(item),
            }
            for item in pool
        ],
    }
    return resolved


def local_runtimes_payload(
    registry: ModelRegistry,
    probe: LocalRuntimeProbe,
) -> dict[str, Any]:
    runtimes = probe.probe()
    snapshot = registry.snapshot()
    return {
        "runtimes": runtimes,
        "best_by_category": best_local_by_category(snapshot["models"], runtimes),
    }
