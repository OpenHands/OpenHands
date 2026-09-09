"""Model registry: seed load, schema validation, provenance merge, privacy overlay.

Benchmarks and economics live in ``registry.v1.json`` and are upgraded by
ingestion. Retention/watermark live in ``privacy.v1.json`` and are curated —
never ingested from benchmark sites.
"""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

REGISTRY_VERSION = "v1"
PRIVACY_VERSION = "v1"
BENCHMARK_CATEGORIES = ("coding", "reasoning", "ux", "copy")
RETENTION_NONE = "none"
RETENTION_OPT_OUT = "opt-out"
RETENTION_TRAINS = "trains"
RETENTION_VALUES = (RETENTION_NONE, RETENTION_OPT_OUT, RETENTION_TRAINS)
WATERMARK_NONE = "none"
WATERMARK_CONFIGURABLE = "configurable"
WATERMARK_ALWAYS = "always"
WATERMARK_VALUES = (WATERMARK_NONE, WATERMARK_CONFIGURABLE, WATERMARK_ALWAYS)
REGISTRY_FILENAME = "registry.v1.json"
PRIVACY_FILENAME = "privacy.v1.json"
OVERLAY_FILENAME = "registry.overlay.json"
PRIVACY_OVERLAY_FILENAME = "privacy.overlay.json"
SOURCE_STATUS_FILENAME = "sources.json"


class RegistryError(ValueError):
    """Invalid registry or privacy payload."""


def package_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-registry")


def seed_registry_path() -> str:
    return os.path.join(package_dir(), REGISTRY_FILENAME)


def seed_privacy_path() -> str:
    return os.path.join(package_dir(), PRIVACY_FILENAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def is_stale(last_updated: str | None, max_age_days: int, now: datetime | None = None) -> bool:
    parsed = parse_iso(last_updated)
    if parsed is None:
        return True
    current = now or datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (current - parsed).days > max_age_days


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str, data: Any) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegistryError(f"{label} must be an object")
    return value


def validate_provenance(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _require_mapping(value, "provenance")
    source = data.get("source")
    fetched_at = data.get("fetched_at")
    if not isinstance(source, str) or not source.strip():
        raise RegistryError("provenance.source is required")
    if not isinstance(fetched_at, str) or not fetched_at.strip():
        raise RegistryError("provenance.fetched_at is required")
    version = data.get("version")
    if version is not None and not isinstance(version, str):
        raise RegistryError("provenance.version must be a string")
    return {
        "source": source.strip(),
        "fetched_at": fetched_at.strip(),
        "version": version,
    }


def validate_benchmark_entry(value: Any, category: str) -> dict[str, Any]:
    data = _require_mapping(value, f"benchmarks.{category}")
    score = data.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise RegistryError(f"benchmarks.{category}.score must be a number")
    if score < 0 or score > 1:
        raise RegistryError(f"benchmarks.{category}.score must be between 0 and 1")
    return {
        "score": float(score),
        "provenance": validate_provenance(data.get("provenance")),
    }


def validate_model_entry(value: Any) -> dict[str, Any]:
    data = _require_mapping(value, "model")
    model_id = data.get("id")
    provider_key = data.get("provider_key")
    if not isinstance(model_id, str) or not model_id.strip():
        raise RegistryError("model.id is required")
    if not isinstance(provider_key, str) or not provider_key.strip():
        raise RegistryError("model.provider_key is required")
    benches_in = data.get("benchmarks") or {}
    if not isinstance(benches_in, dict):
        raise RegistryError("model.benchmarks must be an object")
    benchmarks: dict[str, Any] = {}
    for category, entry in benches_in.items():
        if category not in BENCHMARK_CATEGORIES:
            raise RegistryError(f"unknown benchmark category: {category}")
        benchmarks[category] = validate_benchmark_entry(entry, category)
    cost = data.get("cost_per_1k", 0)
    latency = data.get("latency_s_p90", 0)
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
        raise RegistryError("cost_per_1k must be a number >= 0")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0:
        raise RegistryError("latency_s_p90 must be a number >= 0")
    local = bool(data.get("local", False))
    runtime = data.get("runtime")
    if runtime is not None and not isinstance(runtime, str):
        raise RegistryError("runtime must be a string")
    source_url = data.get("source_url")
    if source_url is not None and not isinstance(source_url, str):
        raise RegistryError("source_url must be a string")
    verified = bool(data.get("verified", False))
    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise RegistryError("notes must be a string")
    cost_provenance = validate_provenance(data.get("cost_provenance"))
    latency_provenance = validate_provenance(data.get("latency_provenance"))
    return {
        "id": model_id.strip(),
        "provider_key": provider_key.strip(),
        "benchmarks": benchmarks,
        "cost_per_1k": float(cost),
        "latency_s_p90": float(latency),
        "local": local,
        "runtime": runtime.strip() if isinstance(runtime, str) and runtime.strip() else None,
        "source_url": source_url,
        "verified": verified,
        "notes": notes,
        "cost_provenance": cost_provenance,
        "latency_provenance": latency_provenance,
    }


def validate_registry(payload: Any) -> dict[str, Any]:
    data = _require_mapping(payload, "registry")
    version = data.get("version")
    if version != REGISTRY_VERSION:
        raise RegistryError(f"registry version must be {REGISTRY_VERSION}")
    last_updated = data.get("last_updated")
    if not isinstance(last_updated, str) or not last_updated.strip():
        raise RegistryError("last_updated is required")
    models_in = data.get("models")
    if not isinstance(models_in, list) or not models_in:
        raise RegistryError("models must be a non-empty list")
    models = [validate_model_entry(item) for item in models_in]
    ids = [item["id"] for item in models]
    if len(ids) != len(set(ids)):
        raise RegistryError("duplicate model ids in registry")
    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise RegistryError("notes must be a string")
    return {
        "version": REGISTRY_VERSION,
        "last_updated": last_updated.strip(),
        "notes": notes,
        "models": models,
    }


def validate_privacy_model(value: Any) -> dict[str, Any]:
    data = _require_mapping(value, "privacy model")
    model_id = data.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise RegistryError("privacy model.id is required")
    retention = data.get("retention")
    watermark = data.get("watermark")
    if retention not in RETENTION_VALUES:
        raise RegistryError(f"retention must be one of {RETENTION_VALUES}")
    if watermark not in WATERMARK_VALUES:
        raise RegistryError(f"watermark must be one of {WATERMARK_VALUES}")
    researched_at = data.get("researched_at")
    if not isinstance(researched_at, str) or not researched_at.strip():
        raise RegistryError("researched_at is required")
    source_url = data.get("source_url")
    if source_url is not None and not isinstance(source_url, str):
        raise RegistryError("source_url must be a string")
    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise RegistryError("notes must be a string")
    return {
        "id": model_id.strip(),
        "retention": retention,
        "watermark": watermark,
        "researched_at": researched_at.strip(),
        "source_url": source_url,
        "notes": notes,
    }


def validate_privacy(payload: Any) -> dict[str, Any]:
    data = _require_mapping(payload, "privacy")
    version = data.get("version")
    if version != PRIVACY_VERSION:
        raise RegistryError(f"privacy version must be {PRIVACY_VERSION}")
    last_updated = data.get("last_updated")
    if not isinstance(last_updated, str) or not last_updated.strip():
        raise RegistryError("last_updated is required")
    models_in = data.get("models")
    if not isinstance(models_in, list) or not models_in:
        raise RegistryError("privacy models must be a non-empty list")
    models = [validate_privacy_model(item) for item in models_in]
    ids = [item["id"] for item in models]
    if len(ids) != len(set(ids)):
        raise RegistryError("duplicate model ids in privacy file")
    curated = data.get("curated_fields")
    update_url = data.get("update_url")
    if curated is not None and not isinstance(curated, str):
        raise RegistryError("curated_fields must be a string")
    if update_url is not None and not isinstance(update_url, str):
        raise RegistryError("update_url must be a string")
    return {
        "version": PRIVACY_VERSION,
        "last_updated": last_updated.strip(),
        "curated_fields": curated,
        "update_url": update_url,
        "models": models,
    }


def merge_benchmark_entry(
    existing: dict[str, Any] | None, incoming: dict[str, Any]
) -> dict[str, Any]:
    if existing is None:
        return deepcopy(incoming)
    incoming_prov = incoming.get("provenance")
    if incoming_prov:
        return deepcopy(incoming)
    if existing.get("provenance") and not incoming_prov:
        return deepcopy(existing)
    return deepcopy(incoming)


def merge_model(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return deepcopy(incoming)
    merged = deepcopy(existing)
    merged["provider_key"] = incoming["provider_key"]
    merged["local"] = incoming.get("local", existing.get("local", False))
    if incoming.get("runtime") is not None:
        merged["runtime"] = incoming["runtime"]
    if incoming.get("source_url"):
        merged["source_url"] = incoming["source_url"]
    if incoming.get("notes") is not None:
        merged["notes"] = incoming["notes"]
    benches = dict(existing.get("benchmarks") or {})
    for category, entry in (incoming.get("benchmarks") or {}).items():
        benches[category] = merge_benchmark_entry(benches.get(category), entry)
    merged["benchmarks"] = benches
    if incoming.get("verified"):
        merged["verified"] = True
    elif any((entry or {}).get("provenance") for entry in benches.values()):
        merged["verified"] = True
    else:
        merged["verified"] = bool(incoming.get("verified", existing.get("verified", False)))
    if incoming.get("cost_per_1k") is not None and (
        incoming.get("cost_provenance") or not existing.get("cost_provenance")
    ):
        merged["cost_per_1k"] = incoming["cost_per_1k"]
        if incoming.get("cost_provenance"):
            merged["cost_provenance"] = incoming["cost_provenance"]
    if incoming.get("latency_s_p90") is not None and (
        incoming.get("latency_provenance") or not existing.get("latency_provenance")
    ):
        merged["latency_s_p90"] = incoming["latency_s_p90"]
        if incoming.get("latency_provenance"):
            merged["latency_provenance"] = incoming["latency_provenance"]
    return merged


def merge_registry(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    by_id = {item["id"]: deepcopy(item) for item in base.get("models") or []}
    for item in overlay.get("models") or []:
        by_id[item["id"]] = merge_model(by_id.get(item["id"]), item)
    last_updated = overlay.get("last_updated") or base.get("last_updated")
    return {
        "version": REGISTRY_VERSION,
        "last_updated": last_updated,
        "notes": overlay.get("notes", base.get("notes")),
        "models": list(by_id.values()),
    }


def merge_privacy(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    by_id = {item["id"]: deepcopy(item) for item in base.get("models") or []}
    for item in overlay.get("models") or []:
        by_id[item["id"]] = deepcopy(item)
    return {
        "version": PRIVACY_VERSION,
        "last_updated": overlay.get("last_updated") or base.get("last_updated"),
        "curated_fields": overlay.get("curated_fields", base.get("curated_fields")),
        "update_url": overlay.get("update_url", base.get("update_url")),
        "models": list(by_id.values()),
    }


def empty_source_status() -> dict[str, Any]:
    return {}


def default_privacy_for(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "retention": RETENTION_TRAINS,
        "watermark": WATERMARK_ALWAYS,
        "researched_at": None,
        "source_url": None,
        "notes": "unknown — treated as strictest (trains + always watermark)",
    }


class ModelRegistry:
    """In-memory registry with optional on-disk overlay (never wholesale replace)."""

    def __init__(
        self,
        *,
        seed_dir: str | None = None,
        overlay_dir: str | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.seed_dir = seed_dir or package_dir()
        self.overlay_dir = overlay_dir
        self.clock = clock or utc_now
        self._lock = threading.RLock()
        self.registry = validate_registry(
            load_json(os.path.join(self.seed_dir, REGISTRY_FILENAME))
        )
        self.privacy = validate_privacy(
            load_json(os.path.join(self.seed_dir, PRIVACY_FILENAME))
        )
        self.sources: dict[str, Any] = empty_source_status()
        self._load_overlays()

    def _overlay_path(self, filename: str) -> str | None:
        if not self.overlay_dir:
            return None
        return os.path.join(self.overlay_dir, filename)

    def _load_overlays(self) -> None:
        registry_overlay = self._overlay_path(OVERLAY_FILENAME)
        if registry_overlay and os.path.isfile(registry_overlay):
            overlay = validate_registry(load_json(registry_overlay))
            self.registry = merge_registry(self.registry, overlay)
        privacy_overlay = self._overlay_path(PRIVACY_OVERLAY_FILENAME)
        if privacy_overlay and os.path.isfile(privacy_overlay):
            overlay = validate_privacy(load_json(privacy_overlay))
            self.privacy = merge_privacy(self.privacy, overlay)
        sources_path = self._overlay_path(SOURCE_STATUS_FILENAME)
        if sources_path and os.path.isfile(sources_path):
            loaded = load_json(sources_path)
            if isinstance(loaded, dict):
                self.sources = loaded

    def _persist(self) -> None:
        if not self.overlay_dir:
            return
        os.makedirs(self.overlay_dir, exist_ok=True)
        dump_json(os.path.join(self.overlay_dir, OVERLAY_FILENAME), self.registry)
        dump_json(os.path.join(self.overlay_dir, PRIVACY_OVERLAY_FILENAME), self.privacy)
        dump_json(os.path.join(self.overlay_dir, SOURCE_STATUS_FILENAME), self.sources)

    def privacy_by_id(self) -> dict[str, dict[str, Any]]:
        return {item["id"]: item for item in self.privacy["models"]}

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        for item in self.registry["models"]:
            if item["id"] == model_id:
                return deepcopy(item)
        return None

    def merged_models(self) -> list[dict[str, Any]]:
        privacy = self.privacy_by_id()
        rows: list[dict[str, Any]] = []
        for item in self.registry["models"]:
            row = deepcopy(item)
            posture = privacy.get(item["id"]) or default_privacy_for(item["id"])
            row["retention"] = posture["retention"]
            row["watermark"] = posture["watermark"]
            row["privacy_researched_at"] = posture.get("researched_at")
            row["privacy_source_url"] = posture.get("source_url")
            row["privacy_notes"] = posture.get("notes")
            rows.append(row)
        return rows

    def apply_ingest_rows(
        self,
        source: str,
        rows: list[dict[str, Any]],
        *,
        fetched_at: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        stamp = {
            "source": source,
            "fetched_at": fetched_at or self.clock(),
            "version": version,
        }
        incoming_models: list[dict[str, Any]] = []
        for row in rows:
            model = deepcopy(row)
            benches = {}
            for category, entry in (model.get("benchmarks") or {}).items():
                payload = dict(entry)
                payload["provenance"] = stamp
                benches[category] = payload
            model["benchmarks"] = benches
            if model.get("cost_per_1k") is not None and row.get("update_cost"):
                model["cost_provenance"] = stamp
            if model.get("latency_s_p90") is not None and row.get("update_latency"):
                model["latency_provenance"] = stamp
            model["verified"] = True
            incoming_models.append(model)
        overlay = {
            "version": REGISTRY_VERSION,
            "last_updated": stamp["fetched_at"],
            "models": incoming_models,
        }
        validated_models = [validate_model_entry(item) for item in overlay["models"]]
        overlay["models"] = validated_models
        with self._lock:
            self.registry = merge_registry(self.registry, overlay)
            self.registry["last_updated"] = stamp["fetched_at"]
            self.sources[source] = {
                "last_success": stamp["fetched_at"],
                "last_error": None,
                "stale": False,
                "version": version,
                "rows": len(validated_models),
            }
            self._persist()
        return self.sources[source]

    def mark_source_failure(self, source: str, error: str) -> dict[str, Any]:
        with self._lock:
            previous = dict(self.sources.get(source) or {})
            previous["last_error"] = error
            previous["stale"] = True
            if "last_success" not in previous:
                previous["last_success"] = None
            self.sources[source] = previous
            self._persist()
            return previous

    def refresh_privacy(self, payload: Any) -> dict[str, Any]:
        validated = validate_privacy(payload)
        with self._lock:
            self.privacy = merge_privacy(self.privacy, validated)
            self._persist()
            return deepcopy(self.privacy)

    def snapshot(self, *, max_age_days: int = 90) -> dict[str, Any]:
        privacy_stale = is_stale(self.privacy.get("last_updated"), max_age_days)
        bench_stale = is_stale(self.registry.get("last_updated"), max_age_days)
        return {
            "version": self.registry["version"],
            "last_updated": self.registry["last_updated"],
            "privacy_last_updated": self.privacy.get("last_updated"),
            "stale": bench_stale,
            "privacy_stale": privacy_stale,
            "models": self.merged_models(),
            "sources": deepcopy(self.sources),
            "curated_fields": self.privacy.get("curated_fields"),
        }
