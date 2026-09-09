"""Per-source benchmark adapters, name normalization, and ingest pipeline.

Adapters fetch data-file/API payloads only (no HTML scraping). Fetch is
injectable so tests run fully offline.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from model_registry import (
    BENCHMARK_CATEGORIES,
    ModelRegistry,
    RegistryError,
    utc_now,
    validate_model_entry,
)

SOURCE_SWEBENCH = "swebench"
SOURCE_LMSYS = "lmsys-arena"
SOURCE_ARTIFICIAL_ANALYSIS = "artificial-analysis"
SOURCE_LIVEBENCH = "livebench"
ENABLED_SOURCES = (
    SOURCE_SWEBENCH,
    SOURCE_LMSYS,
    SOURCE_ARTIFICIAL_ANALYSIS,
    SOURCE_LIVEBENCH,
)

DEFAULT_SOURCE_URLS: dict[str, str] = {
    SOURCE_SWEBENCH: (
        "https://huggingface.co/datasets/SWE-bench/experiments/"
        "resolve/main/evaluation/verified/leaderboard.json"
    ),
    SOURCE_LMSYS: (
        "https://huggingface.co/datasets/lmsys/lmsys-arena-leaderboard/"
        "resolve/main/leaderboard.csv"
    ),
    SOURCE_ARTIFICIAL_ANALYSIS: (
        "https://artificialanalysis.ai/api/v2/data/llms/models/free"
    ),
    SOURCE_LIVEBENCH: (
        "https://raw.githubusercontent.com/LiveBench/LiveBench/main/"
        "livebench/all_groups.csv"
    ),
}

DEFAULT_TIMEOUT_S = 20

# Explicit alias table: source display names → canonical registry ids.
MODEL_ALIASES: dict[str, str] = {
    "gpt-4o": "openai/gpt-4o",
    "gpt4o": "openai/gpt-4o",
    "openai/gpt-4o": "openai/gpt-4o",
    "gpt-5.4": "openai/gpt-5.4",
    "openai/gpt-5.4": "openai/gpt-5.4",
    "claude sonnet 4.5": "anthropic/claude-sonnet-4-5",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4-5",
    "anthropic/claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "claude opus 4.6": "anthropic/claude-opus-4-6",
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "anthropic/claude-opus-4-6": "anthropic/claude-opus-4-6",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini 2.5 pro": "google/gemini-2.5-pro",
    "google/gemini-2.5-pro": "google/gemini-2.5-pro",
    "glm-5.2": "openhands/glm-5.2",
    "openhands/glm-5.2": "openhands/glm-5.2",
    "qwen3-coder": "ollama/qwen3-coder:16b",
    "qwen3-coder:16b": "ollama/qwen3-coder:16b",
    "ollama/qwen3-coder:16b": "ollama/qwen3-coder:16b",
    "llama3.1:8b": "ollama/llama3.1:8b",
    "llama-3.1-8b": "ollama/llama3.1:8b",
    "deepseek-r1:14b": "ollama/deepseek-r1:14b",
    "cursor composer": "cursor-cli/composer",
    "cursor-cli/composer": "cursor-cli/composer",
}

PROVIDER_FROM_ID_PREFIX: dict[str, str] = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "google/": "gemini",
    "openhands/": "openhands",
    "ollama/": "ollama",
    "cursor-cli/": "cursor-cli",
    "opencode/": "opencode",
}

FetchFn = Callable[[str, float], str]


class IngestError(RuntimeError):
    """A single source failed; other sources may still succeed."""


def default_fetch(url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    request = Request(url, headers={"User-Agent": "openhands-agent-canvas-router/1.0"})
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return response.read().decode("utf-8")


def normalize_alias_key(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("/", " ").split())


def lookup_canonical_id(name: str) -> str | None:
    raw = name.strip()
    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw]
    compact = raw.lower().replace(" ", "-")
    if compact in MODEL_ALIASES:
        return MODEL_ALIASES[compact]
    spaced = normalize_alias_key(raw)
    if spaced in MODEL_ALIASES:
        return MODEL_ALIASES[spaced]
    if raw.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[raw.lower()]
    return None


def provider_key_for(model_id: str) -> str:
    for prefix, key in PROVIDER_FROM_ID_PREFIX.items():
        if model_id.startswith(prefix):
            return key
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return model_id


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_from_resolved(resolved: Any, total: Any) -> float:
    resolved_n = float(resolved)
    if total in (None, 0, 0.0, "0"):
        return _clamp01(resolved_n if resolved_n <= 1 else resolved_n / 100.0)
    total_n = float(total)
    if total_n <= 0:
        return _clamp01(resolved_n)
    ratio = resolved_n / total_n if resolved_n > 1 or total_n > 1 else resolved_n
    if resolved_n <= 1 and total_n <= 1:
        return _clamp01(resolved_n)
    return _clamp01(resolved_n / total_n)


def _row(
    model_id: str,
    *,
    category: str | None = None,
    score: float | None = None,
    cost_per_1k: float | None = None,
    latency_s_p90: float | None = None,
    update_cost: bool = False,
    update_latency: bool = False,
    source_url: str | None = None,
) -> dict[str, Any]:
    benchmarks: dict[str, Any] = {}
    if category is not None and score is not None:
        if category not in BENCHMARK_CATEGORIES:
            raise IngestError(f"unknown category {category}")
        benchmarks[category] = {"score": _clamp01(float(score)), "provenance": None}
    payload: dict[str, Any] = {
        "id": model_id,
        "provider_key": provider_key_for(model_id),
        "benchmarks": benchmarks,
        "cost_per_1k": 0.0 if cost_per_1k is None else float(cost_per_1k),
        "latency_s_p90": 0.0 if latency_s_p90 is None else float(latency_s_p90),
        "local": model_id.startswith("ollama/"),
        "runtime": "ollama" if model_id.startswith("ollama/") else None,
        "source_url": source_url,
        "verified": False,
        "update_cost": update_cost,
        "update_latency": update_latency,
    }
    return payload


def _parse_json_or_csv(raw: str) -> Any:
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(raw)
    reader = csv.DictReader(io.StringIO(raw))
    return list(reader)


class SourceAdapter:
    source_id: str
    default_url: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    def fetch(self, raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (mapped_rows, unmapped_report)."""
        raise NotImplementedError


class SweBenchAdapter(SourceAdapter):
    source_id = SOURCE_SWEBENCH
    default_url = DEFAULT_SOURCE_URLS[SOURCE_SWEBENCH]

    def fetch(self, raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        payload = json.loads(raw)
        items = payload.get("models", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise IngestError("swebench payload must be a list or {models: []}")
        mapped: list[dict[str, Any]] = []
        unmapped: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            canonical = lookup_canonical_id(name)
            if canonical is None:
                unmapped.append({"name": name, "source": self.source_id})
                continue
            score = _score_from_resolved(
                item.get("resolved", item.get("score", 0)),
                item.get("total"),
            )
            mapped.append(
                _row(
                    canonical,
                    category="coding",
                    score=score,
                    source_url=self.default_url,
                )
            )
        return mapped, unmapped


class LmsysArenaAdapter(SourceAdapter):
    source_id = SOURCE_LMSYS
    default_url = DEFAULT_SOURCE_URLS[SOURCE_LMSYS]

    def fetch(self, raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = _parse_json_or_csv(raw)
        if isinstance(rows, dict):
            rows = rows.get("models") or rows.get("leaderboard") or []
        elos: list[tuple[str, float]] = []
        unmapped: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("model") or item.get("name") or "")
            elo_raw = item.get("elo") or item.get("score") or item.get("rating")
            if elo_raw in (None, ""):
                continue
            canonical = lookup_canonical_id(name)
            if canonical is None:
                unmapped.append({"name": name, "source": self.source_id})
                continue
            elos.append((canonical, float(elo_raw)))
        if not elos:
            return [], unmapped
        lo = min(score for _, score in elos)
        hi = max(score for _, score in elos)
        span = hi - lo if hi != lo else 1.0
        mapped = [
            _row(
                model_id,
                category="reasoning",
                score=(elo - lo) / span,
                source_url=self.default_url,
            )
            for model_id, elo in elos
        ]
        return mapped, unmapped


class ArtificialAnalysisAdapter(SourceAdapter):
    source_id = SOURCE_ARTIFICIAL_ANALYSIS
    default_url = DEFAULT_SOURCE_URLS[SOURCE_ARTIFICIAL_ANALYSIS]

    def fetch(self, raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        payload = json.loads(raw)
        items = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise IngestError("artificial-analysis payload must contain data[]")
        mapped: list[dict[str, Any]] = []
        unmapped: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("id") or "")
            canonical = lookup_canonical_id(name)
            if canonical is None:
                unmapped.append({"name": name, "source": self.source_id})
                continue
            evaluations = item.get("evaluations") or {}
            coding_idx = evaluations.get("artificial_analysis_coding_index")
            intel_idx = evaluations.get("artificial_analysis_intelligence_index")
            pricing = item.get("pricing") or {}
            in_price = pricing.get("price_1m_input_tokens")
            out_price = pricing.get("price_1m_output_tokens")
            cost = None
            if isinstance(in_price, (int, float)) and isinstance(out_price, (int, float)):
                cost = ((float(in_price) + float(out_price)) / 2.0) / 1000.0
            ttft = item.get("median_time_to_first_token_seconds")
            latency = float(ttft) * 30 if isinstance(ttft, (int, float)) else None
            benches: dict[str, Any] = {}
            if isinstance(coding_idx, (int, float)):
                benches["coding"] = {"score": _clamp01(float(coding_idx) / 100.0)}
            if isinstance(intel_idx, (int, float)):
                benches["reasoning"] = {"score": _clamp01(float(intel_idx) / 100.0)}
            row = _row(
                canonical,
                cost_per_1k=cost,
                latency_s_p90=latency,
                update_cost=cost is not None,
                update_latency=latency is not None,
                source_url=self.default_url,
            )
            if benches:
                row["benchmarks"] = benches
            mapped.append(row)
        return mapped, unmapped


class LiveBenchAdapter(SourceAdapter):
    source_id = SOURCE_LIVEBENCH
    default_url = DEFAULT_SOURCE_URLS[SOURCE_LIVEBENCH]

    def fetch(self, raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = _parse_json_or_csv(raw)
        if isinstance(rows, dict):
            rows = rows.get("models") or []
        mapped: list[dict[str, Any]] = []
        unmapped: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("model") or item.get("name") or "")
            canonical = lookup_canonical_id(name)
            if canonical is None:
                unmapped.append({"name": name, "source": self.source_id})
                continue
            benches: dict[str, Any] = {}
            for category in ("coding", "reasoning"):
                if item.get(category) not in (None, ""):
                    benches[category] = {
                        "score": _clamp01(float(item[category])),
                        "provenance": None,
                    }
            row = _row(canonical, source_url=self.default_url)
            row["benchmarks"] = benches
            mapped.append(row)
        return mapped, unmapped


ADAPTERS: dict[str, SourceAdapter] = {
    SOURCE_SWEBENCH: SweBenchAdapter(),
    SOURCE_LMSYS: LmsysArenaAdapter(),
    SOURCE_ARTIFICIAL_ANALYSIS: ArtificialAnalysisAdapter(),
    SOURCE_LIVEBENCH: LiveBenchAdapter(),
}


def fixtures_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model-registry", "fixtures"
    )


def ingest_sources(
    registry: ModelRegistry,
    *,
    sources: list[str] | None = None,
    fetch: FetchFn | None = None,
    urls: dict[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    clock: Callable[[], str] | None = None,
) -> dict[str, Any]:
    fetch_fn = fetch or default_fetch
    wanted = sources or list(ENABLED_SOURCES)
    stamp = (clock or utc_now)()
    results: dict[str, Any] = {"sources": {}, "unmapped": []}
    for source_id in wanted:
        adapter = ADAPTERS.get(source_id)
        if adapter is None:
            results["sources"][source_id] = {
                "ok": False,
                "error": f"unknown source {source_id}",
            }
            continue
        url = (urls or {}).get(source_id) or adapter.default_url
        try:
            raw = fetch_fn(url, timeout_s)
            mapped, unmapped = adapter.fetch(raw)
            for row in mapped:
                validate_model_entry(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("update_cost", "update_latency")
                    }
                )
            status = registry.apply_ingest_rows(
                source_id, mapped, fetched_at=stamp, version=source_id
            )
            results["unmapped"].extend(unmapped)
            results["sources"][source_id] = {
                "ok": True,
                "rows": len(mapped),
                "unmapped": len(unmapped),
                **status,
            }
        except (IngestError, RegistryError, URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            status = registry.mark_source_failure(source_id, str(exc))
            results["sources"][source_id] = {
                "ok": False,
                "error": str(exc),
                **status,
            }
    return results
