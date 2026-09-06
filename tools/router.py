"""Routing engine: taxonomy, classifier, filter → rank → pick, audit traces.

Config lives in SQLite (loop_runner-style). Project YAML is imported
idempotently and mirrored best-effort on GUI writes.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

from model_registry import (
    RETENTION_NONE,
    WATERMARK_NONE,
    ModelRegistry,
    is_stale,
    utc_now,
)

ROUTER_DB_FILENAME = "routing.db"
CLASSIFIER_VERSION = "v1-taxonomy"
CONFIG_KEY = "config"
GOAL_QUALITY = "quality"
GOAL_COST = "cost"
GOAL_SPEED = "speed"
GOAL_PRIVACY = "privacy"
GOALS = (GOAL_QUALITY, GOAL_COST, GOAL_SPEED, GOAL_PRIVACY)
COST_MODE_FLOOR = "floor"
COST_MODE_CAP = "cap"
MODE_WARN = "warn"
MODE_STRICT = "strict"
TARGET_AUTO = "auto"
COMPLEXITY_LOW = "low"
COMPLEXITY_MEDIUM = "medium"
COMPLEXITY_HIGH = "high"
SENSITIVITY_PUBLIC = "public"
SENSITIVITY_DEFAULT = "default"
SENSITIVITY_SENSITIVE = "sensitive"
SENSITIVITY_SENSITIVE_IP = "sensitive-ip"
DEFAULT_SENSITIVITIES = (
    SENSITIVITY_PUBLIC,
    SENSITIVITY_DEFAULT,
    SENSITIVITY_SENSITIVE,
    SENSITIVITY_SENSITIVE_IP,
)
AUDIT_RESOLVE = "resolve"
AUDIT_SWITCH = "switch"
QUALITY_FLOOR = 0.45
FALLBACK_BUDGET = 3
CATEGORY_BY_WORK_TYPE = {
    "coding": "coding",
    "review": "coding",
    "test": "coding",
    "refactor": "coding",
    "ops": "coding",
    "ux": "ux",
    "copy": "copy",
    "docs": "copy",
    "research": "reasoning",
}

DEFAULT_WORK_TYPES: tuple[dict[str, str], ...] = (
    {
        "id": "coding",
        "name": "Coding",
        "description": (
            "Implement features, fix bugs, write algorithms, or change application code."
        ),
    },
    {
        "id": "review",
        "name": "Review",
        "description": "Review diffs, security, correctness, or pull-request feedback.",
    },
    {
        "id": "ux",
        "name": "UX + visual design",
        "description": "Layout, CSS, visual design, accessibility, and UI polish.",
    },
    {
        "id": "copy",
        "name": "Copy / marketing",
        "description": "Marketing text, UI copy, slogans, and customer-facing wording.",
    },
    {
        "id": "docs",
        "name": "Docs",
        "description": "README, API docs, comments, and explanatory writing.",
    },
    {
        "id": "test",
        "name": "Test",
        "description": "Unit, integration, or end-to-end tests and coverage work.",
    },
    {
        "id": "refactor",
        "name": "Refactor",
        "description": "Restructure code without changing user-visible behavior.",
    },
    {
        "id": "research",
        "name": "Research",
        "description": "Investigate options, compare approaches, or explore a codebase.",
    },
    {
        "id": "ops",
        "name": "Ops",
        "description": "CI, deploy, infrastructure, and developer-environment tasks.",
    },
)

DEFAULT_SENSITIVITY_LABELS: tuple[dict[str, str], ...] = (
    {
        "id": SENSITIVITY_PUBLIC,
        "name": "Public",
        "description": "Already public or intended to be published.",
    },
    {
        "id": SENSITIVITY_DEFAULT,
        "name": "Default",
        "description": "Ordinary internal work without special confidentiality.",
    },
    {
        "id": SENSITIVITY_SENSITIVE,
        "name": "Sensitive",
        "description": "Internal secrets, credentials, or private customer data.",
    },
    {
        "id": SENSITIVITY_SENSITIVE_IP,
        "name": "Sensitive IP / proprietary code",
        "description": "Proprietary source, unpublished IP, or code that must not be retained or watermarked.",
    },
)


class RouterError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status
        self.payload: dict[str, Any] = {}


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, ROUTER_DB_FILENAME)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def default_guardrails() -> dict[str, Any]:
    return {
        "forbid_training_retention": False,
        "forbid_watermarking": False,
        "max_cost_usd_per_task": None,
        "max_latency_s": None,
    }


def default_routes() -> list[dict[str, Any]]:
    return [
        {
            "id": "route-sensitive-ip",
            "work_type": None,
            "sensitivity": SENSITIVITY_SENSITIVE_IP,
            "goal": GOAL_PRIVACY,
            "target": TARGET_AUTO,
            "guardrails": {
                **default_guardrails(),
                "forbid_training_retention": True,
                "forbid_watermarking": True,
            },
        },
        {
            "id": "route-ux",
            "work_type": "ux",
            "sensitivity": None,
            "goal": GOAL_QUALITY,
            "target": TARGET_AUTO,
            "guardrails": default_guardrails(),
        },
        {
            "id": "route-copy",
            "work_type": "copy",
            "sensitivity": None,
            "goal": GOAL_QUALITY,
            "target": TARGET_AUTO,
            "guardrails": default_guardrails(),
        },
        {
            "id": "route-default",
            "work_type": None,
            "sensitivity": None,
            "goal": GOAL_QUALITY,
            "target": TARGET_AUTO,
            "guardrails": default_guardrails(),
        },
    ]


def default_config() -> dict[str, Any]:
    return {
        "goal": GOAL_QUALITY,
        "guardrails": default_guardrails(),
        "mode": MODE_WARN,
        "metadata_max_age_days": 90,
        "blend_alpha": 0.0,
        "cost_mode": COST_MODE_FLOOR,
        "quality_floor": QUALITY_FLOOR,
        "struggle_threshold": 2,
        "router_model": {
            "preset": "cheapest",
            "disabled": False,
            "provider_key": None,
            "model": None,
            "goal": GOAL_COST,
            "guardrails": default_guardrails(),
        },
        "routes": default_routes(),
        "work_types": [dict(item) for item in DEFAULT_WORK_TYPES],
        "sensitivities": [dict(item) for item in DEFAULT_SENSITIVITY_LABELS],
        "imported_project_path": None,
    }


def classifier_prompt(work_types: list[dict[str, Any]], sensitivities: list[dict[str, Any]]) -> str:
    work_lines = "\n".join(
        f"- {item['id']}: {item['name']} — {item['description']}" for item in work_types
    )
    sens_lines = "\n".join(
        f"- {item['id']}: {item['name']} — {item['description']}" for item in sensitivities
    )
    return (
        "Classify the task into exactly one work_type id, one sensitivity id, "
        "and a complexity of low, medium, or high.\n"
        "Respond with JSON: "
        '{"work_type":"...","sensitivity":"...","complexity":"...","confidence":0.0,"reason":"..."}\n\n'
        f"Work types:\n{work_lines}\n\nSensitivities:\n{sens_lines}\n"
    )


def parse_classifier_response(
    payload: Any,
    *,
    work_type_ids: set[str],
    sensitivity_ids: set[str],
) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else json.loads(str(payload))
    if not isinstance(data, dict):
        raise RouterError("classifier response must be an object")
    work_type = str(data.get("work_type") or "").strip()
    sensitivity = str(data.get("sensitivity") or "").strip()
    complexity = str(data.get("complexity") or COMPLEXITY_MEDIUM).strip()
    if work_type not in work_type_ids:
        raise RouterError(f"unknown work_type {work_type}")
    if sensitivity not in sensitivity_ids:
        raise RouterError(f"unknown sensitivity {sensitivity}")
    if complexity not in (COMPLEXITY_LOW, COMPLEXITY_MEDIUM, COMPLEXITY_HIGH):
        raise RouterError("complexity must be low, medium, or high")
    confidence = data.get("confidence", 1.0)
    try:
        confidence_f = float(confidence)
    except (TypeError, ValueError) as exc:
        raise RouterError("confidence must be a number") from exc
    return {
        "work_type": work_type,
        "sensitivity": sensitivity,
        "complexity": complexity,
        "confidence": max(0.0, min(1.0, confidence_f)),
        "reason": str(data.get("reason") or ""),
    }


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def fallback_classify(
    task_text: str,
    work_types: list[dict[str, Any]],
    sensitivities: list[dict[str, Any]],
) -> dict[str, Any]:
    tokens = _tokenize(task_text)
    def score_labels(labels: list[dict[str, Any]]) -> tuple[str, float]:
        best_id = labels[0]["id"] if labels else "coding"
        best = -1.0
        for label in labels:
            hay = _tokenize(f"{label['id']} {label['name']} {label['description']}")
            overlap = len(tokens & hay)
            weighted = overlap + (2 if label["id"] in tokens else 0)
            if label["id"] == "coding" and overlap == 0:
                weighted += 0.1
            if weighted > best:
                best = float(weighted)
                best_id = label["id"]
        confidence = 0.35 if best <= 0.1 else min(0.9, 0.4 + best * 0.1)
        return best_id, confidence

    work_type, work_conf = score_labels(work_types)
    sensitivity, sens_conf = score_labels(sensitivities)
    if not tokens:
        sensitivity = SENSITIVITY_DEFAULT
    lowered = task_text.lower()
    if any(word in lowered for word in ("proprietary", "secret", "nda", "internal ip")):
        sensitivity = SENSITIVITY_SENSITIVE_IP
        sens_conf = max(sens_conf, 0.7)
    elif any(word in lowered for word in ("password", "credential", "customer data", "pii")):
        sensitivity = SENSITIVITY_SENSITIVE
        sens_conf = max(sens_conf, 0.7)
    elif any(word in lowered for word in ("public blog", "open source", "oss")):
        sensitivity = SENSITIVITY_PUBLIC
    complexity = COMPLEXITY_MEDIUM
    if len(task_text) > 800 or "multi" in lowered:
        complexity = COMPLEXITY_HIGH
    elif len(task_text) < 80:
        complexity = COMPLEXITY_LOW
    return {
        "work_type": work_type,
        "sensitivity": sensitivity,
        "complexity": complexity,
        "confidence": min(work_conf, sens_conf),
        "reason": "deterministic keyword match from taxonomy descriptions",
        "classifier": "fallback",
        "classifier_version": CLASSIFIER_VERSION,
    }


def route_specificity(route: dict[str, Any]) -> int:
    has_work = bool(route.get("work_type"))
    has_sens = bool(route.get("sensitivity"))
    if has_work and has_sens:
        return 3
    if has_sens and not has_work:
        return 2
    if has_work and not has_sens:
        return 1
    return 0


def match_route(
    routes: list[dict[str, Any]], work_type: str, sensitivity: str
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for route in routes:
        work_ok = route.get("work_type") in (None, "", work_type)
        sens_ok = route.get("sensitivity") in (None, "", sensitivity)
        if work_ok and sens_ok:
            candidates.append(route)
    if not candidates:
        return default_routes()[-1]
    candidates.sort(key=route_specificity, reverse=True)
    return deepcopy(candidates[0])


def category_for(work_type: str) -> str:
    return CATEGORY_BY_WORK_TYPE.get(work_type, "coding")


def model_category_score(model: dict[str, Any], category: str) -> float:
    benches = model.get("benchmarks") or {}
    entry = benches.get(category) or benches.get("coding") or {}
    try:
        return float(entry.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _locked_target(target: Any) -> tuple[str, str] | None:
    if target is None or target == TARGET_AUTO:
        return None
    if isinstance(target, str) and "/" in target:
        provider, model = target.split("/", 1)
        return provider, target if "/" in target else model
    if isinstance(target, dict):
        provider = str(target.get("provider_key") or target.get("provider") or "")
        model = str(target.get("model") or target.get("id") or "")
        if provider and model:
            return provider, model
    return None


class RouterStore:
    """SQLite-backed routing config, taxonomy, and decision-trace audit."""

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        registry: ModelRegistry | None = None,
        classify_fn: Callable[..., dict[str, Any]] | None = None,
        runtime_probe: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], str] | None = None,
        yaml_writer: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.db_path = db_path
        self.registry = registry or ModelRegistry()
        self.classify_fn = classify_fn
        self.runtime_probe = runtime_probe
        self.clock = clock or utc_now
        self.yaml_writer = yaml_writer
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                card_id TEXT,
                run_id TEXT,
                payload TEXT NOT NULL
            );
            """
        )
        self.conn.commit()
        if self._get_kv(CONFIG_KEY) is None:
            self._put_kv(CONFIG_KEY, default_config())

    def _get_kv(self, key: str) -> Any | None:
        row = self.conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def _put_kv(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        self.conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, payload),
        )
        self.conn.commit()

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._get_kv(CONFIG_KEY) or default_config())

    def put_config(self, patch: dict[str, Any], *, project_yaml: str | None = None) -> dict[str, Any]:
        with self._lock:
            current = self.get_config()
            merged = deepcopy(current)
            for key, value in patch.items():
                if key == "guardrails" and isinstance(value, dict):
                    merged["guardrails"] = {**(merged.get("guardrails") or {}), **value}
                elif key == "router_model" and isinstance(value, dict):
                    merged["router_model"] = {**(merged.get("router_model") or {}), **value}
                else:
                    merged[key] = value
            self._validate_config(merged)
            self._put_kv(CONFIG_KEY, merged)
            if project_yaml:
                self._mirror_yaml(project_yaml, merged)
            return deepcopy(merged)

    def _validate_config(self, config: dict[str, Any]) -> None:
        if config.get("mode") not in (MODE_WARN, MODE_STRICT):
            raise RouterError("mode must be warn or strict")
        if config.get("goal") not in GOALS:
            raise RouterError(f"goal must be one of {GOALS}")
        work_ids = {item["id"] for item in config.get("work_types") or []}
        sens_ids = {item["id"] for item in config.get("sensitivities") or []}
        for route in config.get("routes") or []:
            if route.get("goal") not in GOALS:
                raise RouterError("route.goal is invalid")
            work = route.get("work_type")
            sens = route.get("sensitivity")
            if work and work not in work_ids:
                raise RouterError(f"route references unknown work_type {work}")
            if sens and sens not in sens_ids:
                raise RouterError(f"route references unknown sensitivity {sens}")

    def get_taxonomy(self) -> dict[str, Any]:
        config = self.get_config()
        work_types = config.get("work_types") or []
        sensitivities = config.get("sensitivities") or []
        return {
            "work_types": deepcopy(work_types),
            "sensitivities": deepcopy(sensitivities),
            "classifier_prompt": classifier_prompt(work_types, sensitivities),
        }

    def put_taxonomy(self, payload: dict[str, Any]) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        if "work_types" in payload:
            patch["work_types"] = payload["work_types"]
        if "sensitivities" in payload:
            patch["sensitivities"] = payload["sensitivities"]
        self.put_config(patch)
        return self.get_taxonomy()

    def reset_taxonomy(self) -> dict[str, Any]:
        self.put_config(
            {
                "work_types": [dict(item) for item in DEFAULT_WORK_TYPES],
                "sensitivities": [dict(item) for item in DEFAULT_SENSITIVITY_LABELS],
            }
        )
        return self.get_taxonomy()

    def import_project_config(self, path: str) -> dict[str, Any]:
        from project_config import load_project_config

        data = load_project_config(path)
        routing = (data.get("project") or {}).get("routing") or {}
        config = self.get_config()
        if config.get("imported_project_path") == os.path.abspath(path):
            return config
        routes = list(config.get("routes") or [])
        existing = {
            (item.get("work_type"), item.get("sensitivity"), str(item.get("target")))
            for item in routes
        }
        for rule in routing.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            work_type = str(rule.get("task_type") or rule.get("work_type") or "") or None
            provider = rule.get("provider") or rule.get("provider_key")
            model = rule.get("model")
            if not provider or not model:
                continue
            target = {"provider_key": str(provider), "model": str(model)}
            key = (work_type, rule.get("sensitivity"), str(target))
            if key in existing:
                continue
            routes.append(
                {
                    "id": f"imported-{uuid.uuid4().hex[:8]}",
                    "work_type": work_type,
                    "sensitivity": rule.get("sensitivity"),
                    "goal": rule.get("goal") or GOAL_QUALITY,
                    "target": target,
                    "guardrails": default_guardrails(),
                }
            )
            existing.add(key)
        for route in routing.get("routes") or []:
            if isinstance(route, dict) and route.get("id"):
                if not any(item.get("id") == route["id"] for item in routes):
                    routes.append(deepcopy(route))
        default = routing.get("default")
        if isinstance(default, dict) and default.get("provider") and default.get("model"):
            for item in routes:
                if item.get("work_type") is None and item.get("sensitivity") is None:
                    if item.get("target") == TARGET_AUTO:
                        item["target"] = {
                            "provider_key": default["provider"],
                            "model": default["model"],
                        }
                    break
        return self.put_config(
            {
                "routes": routes,
                "imported_project_path": os.path.abspath(path),
            }
        )

    def _mirror_yaml(self, path: str, config: dict[str, Any]) -> None:
        try:
            if self.yaml_writer:
                self.yaml_writer(path, config)
                return
            import yaml  # type: ignore

            existing: dict[str, Any] = {}
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle.read()) or {}
                    if isinstance(loaded, dict):
                        existing = loaded
            project = dict(existing.get("project") or {})
            routing = dict(project.get("routing") or {})
            routing["goal"] = config.get("goal")
            routing["guardrails"] = config.get("guardrails")
            routing["mode"] = config.get("mode")
            routing["routes"] = config.get("routes")
            routing["router_model"] = config.get("router_model")
            project["routing"] = routing
            if "name" not in project:
                project["name"] = os.path.basename(os.path.dirname(path) or "project")
            existing["project"] = project
            tmp = path + ".tmp"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                yaml.safe_dump(existing, handle, sort_keys=False)
            os.replace(tmp, path)
        except (OSError, ImportError, TypeError, ValueError):
            return

    def classify(self, task_text: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = config or self.get_config()
        work_types = config.get("work_types") or []
        sensitivities = config.get("sensitivities") or []
        router_model = config.get("router_model") or {}
        if router_model.get("disabled") or self.classify_fn is None:
            result = fallback_classify(task_text, work_types, sensitivities)
            result["classifier"] = "fallback"
            result["classifier_version"] = CLASSIFIER_VERSION
            return result
        try:
            raw = self.classify_fn(
                task_text,
                classifier_prompt(work_types, sensitivities),
                router_model,
            )
            parsed = parse_classifier_response(
                raw,
                work_type_ids={item["id"] for item in work_types},
                sensitivity_ids={item["id"] for item in sensitivities},
            )
            if parsed["confidence"] < 0.45:
                fallback = fallback_classify(task_text, work_types, sensitivities)
                fallback["low_confidence_model"] = parsed
                return fallback
            parsed["classifier"] = "router-model"
            parsed["classifier_version"] = CLASSIFIER_VERSION
            return parsed
        except (RouterError, TypeError, ValueError, json.JSONDecodeError):
            return fallback_classify(task_text, work_types, sensitivities)

    def list_audit(
        self,
        *,
        card_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses = ["1=1"]
        params: list[Any] = []
        if card_id:
            clauses.append("card_id = ?")
            params.append(card_id)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = " AND ".join(clauses)
        rows = self.conn.execute(
            f"SELECT * FROM audit WHERE {where} ORDER BY created_at DESC "
            "LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        items = []
        for row in rows:
            item = _row_to_dict(row) or {}
            item["payload"] = json.loads(item["payload"])
            items.append(item)
        total = self.conn.execute(
            f"SELECT COUNT(*) AS n FROM audit WHERE {where}", params
        ).fetchone()["n"]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def append_audit(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        card_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        item = {
            "id": uuid.uuid4().hex,
            "created_at": self.clock(),
            "kind": kind,
            "card_id": card_id,
            "run_id": run_id,
            "payload": payload,
        }
        self.conn.execute(
            "INSERT INTO audit(id, created_at, kind, card_id, run_id, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                item["id"],
                item["created_at"],
                kind,
                card_id,
                run_id,
                json.dumps(payload),
            ),
        )
        self.conn.commit()
        return item

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        config = self.get_config()
        skip = list(request.get("skip_targets") or [])
        budget = int(request.get("fallback_budget") or FALLBACK_BUDGET)
        last: dict[str, Any] | None = None
        attempted: list[str] = []
        for _ in range(max(1, budget)):
            result = self._resolve_once(request, config, skip)
            last = result
            decision = result.get("decision") or {}
            target_key = f"{decision.get('provider_key')}/{decision.get('model')}"
            if decision.get("usable", True) and decision.get("model"):
                result["attempted"] = attempted
                return result
            skip.append(
                {
                    "provider_key": decision.get("provider_key"),
                    "model": decision.get("model"),
                }
            )
            attempted.append(target_key)
        if last is None:
            raise RouterError("no routing decision")
        last["attempted"] = attempted
        return last

    def _resolve_once(
        self,
        request: dict[str, Any],
        config: dict[str, Any],
        skip: list[dict[str, Any]],
    ) -> dict[str, Any]:
        task_text = str(request.get("task_text") or request.get("task") or "")
        explicit_work = request.get("work_type")
        explicit_sens = request.get("sensitivity")
        if explicit_work and explicit_sens:
            classification = {
                "work_type": explicit_work,
                "sensitivity": explicit_sens,
                "complexity": request.get("complexity") or COMPLEXITY_MEDIUM,
                "confidence": 1.0,
                "reason": "explicit tags",
                "classifier": "explicit",
                "classifier_version": CLASSIFIER_VERSION,
            }
        else:
            classification = self.classify(task_text, config)
            if explicit_work:
                classification["work_type"] = explicit_work
            if explicit_sens:
                classification["sensitivity"] = explicit_sens
        route = match_route(
            config.get("routes") or [],
            classification["work_type"],
            classification["sensitivity"],
        )
        goal = route.get("goal") or config.get("goal") or GOAL_QUALITY
        guardrails = {
            **default_guardrails(),
            **(config.get("guardrails") or {}),
            **(route.get("guardrails") or {}),
        }
        if classification["sensitivity"] == SENSITIVITY_SENSITIVE_IP:
            guardrails["forbid_watermarking"] = True
            guardrails["forbid_training_retention"] = True
        if goal == GOAL_PRIVACY:
            guardrails["forbid_training_retention"] = True
        mode = config.get("mode") or MODE_WARN
        max_age = int(config.get("metadata_max_age_days") or 90)
        snapshot = self.registry.snapshot(max_age_days=max_age)
        privacy_stale = bool(snapshot.get("privacy_stale"))
        privacy_guarded = bool(
            guardrails.get("forbid_training_retention")
            or guardrails.get("forbid_watermarking")
        )
        if "connected_providers" in request:
            connected: set[str] | None = set(request.get("connected_providers") or [])
        else:
            connected = None
        runtimes = request.get("local_runtimes")
        if runtimes is None and self.runtime_probe:
            runtimes = self.runtime_probe()
        runtimes = runtimes or {}
        locked = _locked_target(route.get("target"))
        models = snapshot["models"]
        dropped: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        skip_keys = {
            f"{item.get('provider_key')}/{item.get('model')}"
            for item in skip
            if isinstance(item, dict)
        }

        stale_block = privacy_guarded and privacy_stale and mode == MODE_STRICT
        if stale_block and not locked:
            reason = "strict mode refuses privacy-guarded auto routes on stale privacy metadata"
            decision = {
                "provider_key": None,
                "model": None,
                "score": None,
                "rule_id": route.get("id"),
                "registry_version": snapshot.get("version"),
                "target": TARGET_AUTO,
                "usable": False,
                "goal": goal,
            }
            trace = self._build_trace(
                task_text, classification, dropped, [], decision, reason, route, snapshot
            )
            stored = self.append_audit(
                AUDIT_RESOLVE,
                {"decision": decision, "trace": trace},
                card_id=request.get("card_id"),
                run_id=request.get("run_id"),
            )
            return {"decision": decision, "trace": trace, "audit_id": stored["id"]}

        for model in models:
            key = f"{model['provider_key']}/{model['id']}"
            if key in skip_keys or model["id"] in skip_keys:
                dropped.append({"id": model["id"], "reason": "skipped failed target"})
                continue
            drop_reason = self._hard_filter(
                model,
                guardrails,
                goal,
                classification["sensitivity"],
                connected,
                runtimes,
                mode,
                privacy_stale,
                locked,
            )
            if drop_reason:
                dropped.append({"id": model["id"], "reason": drop_reason})
                continue
            candidates.append(model)

        if locked:
            provider, model_id = locked
            locked_model = next(
                (item for item in models if item["id"] == model_id),
                {
                    "id": model_id,
                    "provider_key": provider,
                    "benchmarks": {},
                    "cost_per_1k": 0,
                    "latency_s_p90": 0,
                    "retention": RETENTION_NONE,
                    "watermark": WATERMARK_NONE,
                    "local": False,
                    "verified": False,
                },
            )
            drop_reason = self._hard_filter(
                locked_model,
                guardrails,
                goal,
                classification["sensitivity"],
                connected,
                runtimes,
                mode,
                privacy_stale,
                locked,
            )
            if drop_reason and mode == MODE_STRICT:
                reason = f"strict blocked locked route: {drop_reason}"
                decision = {
                    "provider_key": provider,
                    "model": model_id,
                    "score": None,
                    "rule_id": route.get("id"),
                    "registry_version": snapshot.get("version"),
                    "target": {"provider_key": provider, "model": model_id},
                    "usable": False,
                    "goal": goal,
                    "locked": True,
                }
                trace = self._build_trace(
                    task_text,
                    classification,
                    dropped,
                    [],
                    decision,
                    reason,
                    route,
                    snapshot,
                )
                stored = self.append_audit(
                    AUDIT_RESOLVE,
                    {"decision": decision, "trace": trace},
                    card_id=request.get("card_id"),
                    run_id=request.get("run_id"),
                )
                return {"decision": decision, "trace": trace, "audit_id": stored["id"]}
            ranked = [
                self._scored(
                    locked_model, classification["work_type"], goal, config, request
                )
            ]
            chosen = ranked[0]
            reason = self._reason(classification, route, chosen, locked=True)
            decision = {
                "provider_key": provider,
                "model": model_id,
                "score": chosen["score"],
                "score_source": chosen["score_source"],
                "rule_id": route.get("id"),
                "registry_version": snapshot.get("version"),
                "target": {"provider_key": provider, "model": model_id},
                "usable": drop_reason is None,
                "goal": goal,
                "locked": True,
                "warning": drop_reason if drop_reason and mode == MODE_WARN else None,
            }
            trace = self._build_trace(
                task_text, classification, dropped, ranked, decision, reason, route, snapshot
            )
            stored = self.append_audit(
                AUDIT_RESOLVE,
                {"decision": decision, "trace": trace},
                card_id=request.get("card_id"),
                run_id=request.get("run_id"),
            )
            return {"decision": decision, "trace": trace, "audit_id": stored["id"]}

        ranked = [
            self._scored(model, classification["work_type"], goal, config, request)
            for model in candidates
        ]
        ranked.sort(key=lambda item: item["sort_key"])
        if not ranked:
            reason = "no reachable model passed hard filters"
            decision = {
                "provider_key": None,
                "model": None,
                "score": None,
                "rule_id": route.get("id"),
                "registry_version": snapshot.get("version"),
                "target": TARGET_AUTO,
                "usable": False,
                "goal": goal,
            }
            trace = self._build_trace(
                task_text, classification, dropped, [], decision, reason, route, snapshot
            )
            stored = self.append_audit(
                AUDIT_RESOLVE,
                {"decision": decision, "trace": trace},
                card_id=request.get("card_id"),
                run_id=request.get("run_id"),
            )
            return {"decision": decision, "trace": trace, "audit_id": stored["id"]}

        chosen = ranked[0]
        reason = self._reason(classification, route, chosen, locked=False)
        if classification.get("classifier") == "router-model" and classification.get("reason"):
            reason = classification["reason"]
        decision = {
            "provider_key": chosen["provider_key"],
            "model": chosen["id"],
            "score": chosen["score"],
            "score_source": chosen["score_source"],
            "rule_id": route.get("id"),
            "registry_version": snapshot.get("version"),
            "target": TARGET_AUTO,
            "usable": True,
            "goal": goal,
            "locked": False,
        }
        trace = self._build_trace(
            task_text, classification, dropped, ranked[:3], decision, reason, route, snapshot
        )
        stored = self.append_audit(
            AUDIT_RESOLVE,
            {"decision": decision, "trace": trace},
            card_id=request.get("card_id"),
            run_id=request.get("run_id"),
        )
        return {"decision": decision, "trace": trace, "audit_id": stored["id"]}

    def _hard_filter(
        self,
        model: dict[str, Any],
        guardrails: dict[str, Any],
        goal: str,
        sensitivity: str,
        connected: set[str] | None,
        runtimes: dict[str, Any],
        mode: str,
        privacy_stale: bool,
        locked: tuple[str, str] | None,
    ) -> str | None:
        if model.get("local"):
            runtime = model.get("runtime") or "ollama"
            info = runtimes.get(runtime) or {}
            if not info.get("alive"):
                return f"local runtime {runtime} is not reachable"
            installed = {self._normalize_installed(name) for name in info.get("models") or []}
            model_name = model["id"].split("/", 1)[-1]
            if installed and self._normalize_installed(model_name) not in installed:
                if not any(model_name in name or name in model_name for name in installed):
                    return f"{model['id']} is not installed on {runtime}"
        else:
            if connected is not None:
                aliases = {
                    "cursor": "cursor-cli",
                    "claude": "anthropic",
                    "claude-code": "anthropic",
                }
                mapped = aliases.get(model["provider_key"], model["provider_key"])
                connected_mapped = {aliases.get(item, item) for item in connected} | connected
                if (
                    model["provider_key"] not in connected
                    and mapped not in connected
                    and model["provider_key"] not in connected_mapped
                ):
                    return f"provider {model['provider_key']} is not connected"
        if guardrails.get("forbid_training_retention") and model.get("retention") != RETENTION_NONE:
            return f"retention={model.get('retention')} blocked by forbid_training_retention"
        if guardrails.get("forbid_watermarking") and model.get("watermark") != WATERMARK_NONE:
            return f"watermark={model.get('watermark')} blocked by forbid_watermarking"
        if sensitivity == SENSITIVITY_SENSITIVE_IP and model.get("watermark") != WATERMARK_NONE:
            return "sensitive-ip drops watermark≠none"
        if goal == GOAL_PRIVACY and model.get("retention") != RETENTION_NONE:
            return "privacy goal drops retention≠none"
        max_cost = guardrails.get("max_cost_usd_per_task")
        if max_cost is not None and float(model.get("cost_per_1k") or 0) > float(max_cost):
            return f"cost_per_1k {model.get('cost_per_1k')} exceeds max_cost_usd_per_task"
        max_latency = guardrails.get("max_latency_s")
        if max_latency is not None and float(model.get("latency_s_p90") or 0) > float(max_latency):
            return f"latency_s_p90 {model.get('latency_s_p90')} exceeds max_latency_s"
        return None

    @staticmethod
    def _normalize_installed(name: str) -> str:
        return name.split(":")[0].split("/")[-1].lower()

    def _scored(
        self,
        model: dict[str, Any],
        work_type: str,
        goal: str,
        config: dict[str, Any],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        category = category_for(work_type)
        bench = model_category_score(model, category)
        alpha = float(config.get("blend_alpha") or 0)
        empirical = (request.get("empirical_pass_rates") or {}).get(
            f"{work_type}:{model['provider_key']}:{model['id']}"
        )
        if alpha > 0 and isinstance(empirical, (int, float)):
            score = alpha * bench + (1 - alpha) * float(empirical)
            source = "blend"
        else:
            score = bench
            source = f"benchmark:{category}"
        cost = float(model.get("cost_per_1k") or 0)
        latency = float(model.get("latency_s_p90") or 0)
        floor = float(config.get("quality_floor") or QUALITY_FLOOR)
        cost_mode = config.get("cost_mode") or COST_MODE_FLOOR
        if goal == GOAL_COST:
            if cost_mode == COST_MODE_FLOOR and score < floor:
                sort_key = (1, cost, -score, latency)
            else:
                sort_key = (0, cost, -score, latency)
        elif goal == GOAL_SPEED:
            sort_key = (latency, -score, cost)
        else:
            sort_key = (-score, cost, latency)
        row = deepcopy(model)
        row["score"] = round(score, 4)
        row["score_source"] = source
        row["sort_key"] = sort_key
        row["category"] = category
        return row

    def _reason(
        self,
        classification: dict[str, Any],
        route: dict[str, Any],
        chosen: dict[str, Any],
        *,
        locked: bool,
    ) -> str:
        if locked:
            return (
                f"Locked route {route.get('id')} → {chosen.get('provider_key')}/"
                f"{chosen.get('id')} for {classification['work_type']}/"
                f"{classification['sensitivity']}."
            )
        return (
            f"Auto-picked {chosen.get('provider_key')}/{chosen.get('id')} "
            f"(score {chosen.get('score')} from {chosen.get('score_source')}) "
            f"for {classification['work_type']}/{classification['sensitivity']} "
            f"via {route.get('id')}."
        )

    def _build_trace(
        self,
        task_text: str,
        classification: dict[str, Any],
        dropped: list[dict[str, Any]],
        ranked: list[dict[str, Any]],
        decision: dict[str, Any],
        reason: str,
        route: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        top = []
        for item in ranked:
            top.append(
                {
                    "id": item.get("id"),
                    "provider_key": item.get("provider_key"),
                    "score": item.get("score"),
                    "score_source": item.get("score_source"),
                    "cost_per_1k": item.get("cost_per_1k"),
                    "latency_s_p90": item.get("latency_s_p90"),
                    "retention": item.get("retention"),
                    "watermark": item.get("watermark"),
                }
            )
        return {
            "task_text": task_text,
            "classification": deepcopy(classification),
            "classifier_version": classification.get("classifier_version")
            or CLASSIFIER_VERSION,
            "route_id": route.get("id"),
            "filters": dropped,
            "ranked": top,
            "chosen": deepcopy(decision),
            "reason": reason,
            "registry_last_updated": snapshot.get("last_updated"),
            "privacy_last_updated": snapshot.get("privacy_last_updated"),
            "privacy_stale": snapshot.get("privacy_stale"),
        }

    def record_switch(
        self,
        *,
        from_target: dict[str, Any],
        to_target: dict[str, Any],
        reason: str,
        card_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self.append_audit(
            AUDIT_SWITCH,
            {
                "from": from_target,
                "to": to_target,
                "reason": reason,
            },
            card_id=card_id,
            run_id=run_id,
        )
