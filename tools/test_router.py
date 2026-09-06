"""Router engine tests: taxonomy, classify, filter/rank/pick, traces, API."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from model_registry import ModelRegistry  # noqa: E402
from router import (  # noqa: E402
    CLASSIFIER_VERSION,
    GOAL_COST,
    GOAL_PRIVACY,
    MODE_STRICT,
    MODE_WARN,
    TARGET_AUTO,
    RouterError,
    RouterStore,
    fallback_classify,
    match_route,
    parse_classifier_response,
)
from router_api import RouterService, handle_request  # noqa: E402
from router_model_presets import (  # noqa: E402
    PRESET_BEST_INTELLIGENCE,
    PRESET_CHEAPEST,
    PRESET_LOCAL,
    PRESET_MOST_SECURE,
    LocalRuntimeProbe,
    resolve_preset,
)


CONNECTED = [
    "openai",
    "anthropic",
    "openhands",
    "gemini",
    "cursor-cli",
    "opencode",
]


def _store(test: unittest.TestCase | None = None, **kwargs: object) -> RouterStore:
    store = RouterStore(db_path=":memory:", registry=ModelRegistry(), **kwargs)  # type: ignore[arg-type]
    if test is not None:
        test.addCleanup(store.close)
    return store


class TaxonomyTests(unittest.TestCase):
    def test_default_taxonomy_and_prompt(self) -> None:
        store = _store(self)
        tax = store.get_taxonomy()
        self.assertEqual(len(tax["work_types"]), 9)
        self.assertEqual(len(tax["sensitivities"]), 4)
        self.assertIn("coding", tax["classifier_prompt"])
        self.assertIn("sensitive-ip", tax["classifier_prompt"])
        store.put_taxonomy(
            {
                "work_types": tax["work_types"]
                + [{"id": "legal", "name": "Legal", "description": "Contracts and terms."}]
            }
        )
        updated = store.get_taxonomy()
        self.assertIn("legal", {item["id"] for item in updated["work_types"]})
        self.assertIn("Contracts", updated["classifier_prompt"])
        store.reset_taxonomy()
        self.assertEqual(len(store.get_taxonomy()["work_types"]), 9)

    def test_classifier_contract_and_fallback(self) -> None:
        store = _store(self)
        tax = store.get_taxonomy()
        parsed = parse_classifier_response(
            {
                "work_type": "ux",
                "sensitivity": "default",
                "complexity": "low",
                "confidence": 0.9,
                "reason": "layout task",
            },
            work_type_ids={item["id"] for item in tax["work_types"]},
            sensitivity_ids={item["id"] for item in tax["sensitivities"]},
        )
        self.assertEqual(parsed["work_type"], "ux")
        with self.assertRaises(RouterError):
            parse_classifier_response(
                {"work_type": "nope", "sensitivity": "default", "complexity": "low"},
                work_type_ids={item["id"] for item in tax["work_types"]},
                sensitivity_ids={item["id"] for item in tax["sensitivities"]},
            )
        fallback = fallback_classify(
            "Rewrite the marketing homepage copy for launch",
            tax["work_types"],
            tax["sensitivities"],
        )
        self.assertEqual(fallback["work_type"], "copy")
        self.assertEqual(fallback["classifier"], "fallback")

    def test_low_confidence_and_disabled_use_fallback(self) -> None:
        def noisy(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "work_type": "ops",
                "sensitivity": "public",
                "complexity": "high",
                "confidence": 0.1,
            }

        store = RouterStore(
            ":memory:", registry=ModelRegistry(), classify_fn=noisy
        )
        self.addCleanup(store.close)
        result = store.classify("fix the login button CSS")
        self.assertEqual(result["classifier"], "fallback")
        store.put_config({"router_model": {"preset": "cheapest", "disabled": True}})
        disabled = store.classify("anything")
        self.assertEqual(disabled["classifier"], "fallback")


class ResolveTests(unittest.TestCase):
    def test_locked_beats_auto_and_specificity(self) -> None:
        store = _store(self)
        config = store.get_config()
        config["routes"] = [
            {
                "id": "specific",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "goal": "privacy",
                "target": {"provider_key": "ollama", "model": "ollama/llama3.2:3b"},
                "guardrails": {
                    "forbid_training_retention": True,
                    "forbid_watermarking": True,
                    "max_cost_usd_per_task": None,
                    "max_latency_s": None,
                },
            },
            {
                "id": "sens-only",
                "work_type": None,
                "sensitivity": "sensitive-ip",
                "goal": "privacy",
                "target": TARGET_AUTO,
                "guardrails": {
                    "forbid_training_retention": True,
                    "forbid_watermarking": True,
                    "max_cost_usd_per_task": None,
                    "max_latency_s": None,
                },
            },
            {
                "id": "route-default",
                "work_type": None,
                "sensitivity": None,
                "goal": "quality",
                "target": TARGET_AUTO,
                "guardrails": {
                    "forbid_training_retention": False,
                    "forbid_watermarking": False,
                    "max_cost_usd_per_task": None,
                    "max_latency_s": None,
                },
            },
        ]
        store.put_config({"routes": config["routes"]})
        matched = match_route(config["routes"], "coding", "sensitive-ip")
        self.assertEqual(matched["id"], "specific")
        result = store.resolve(
            {
                "task_text": "implement proprietary billing",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "connected_providers": CONNECTED,
                "local_runtimes": {
                    "ollama": {"alive": True, "models": ["llama3.2:3b"]}
                },
            }
        )
        self.assertTrue(result["decision"]["locked"])
        self.assertEqual(result["decision"]["model"], "ollama/llama3.2:3b")

    def test_privacy_and_sensitive_ip_filters(self) -> None:
        store = _store(self)
        result = store.resolve(
            {
                "task_text": "refactor proprietary compiler internals",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "connected_providers": CONNECTED,
                "local_runtimes": {},
            }
        )
        chosen = result["decision"]["model"]
        self.assertIsNotNone(chosen)
        model = store.registry.privacy_by_id()[chosen]
        self.assertEqual(model["retention"], "none")
        self.assertEqual(model["watermark"], "none")
        dropped = {item["id"]: item["reason"] for item in result["trace"]["filters"]}
        self.assertIn("google/gemini-2.5-pro", dropped)

    def test_quality_ranks_by_category_score(self) -> None:
        store = _store(self)
        result = store.resolve(
            {
                "task_text": "implement a parser",
                "work_type": "coding",
                "sensitivity": "default",
                "connected_providers": ["anthropic", "openai", "openhands"],
                "local_runtimes": {},
            }
        )
        self.assertTrue(result["decision"]["usable"])
        self.assertGreaterEqual(len(result["trace"]["ranked"]), 1)
        scores = [item["score"] for item in result["trace"]["ranked"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("benchmark:coding", result["trace"]["ranked"][0]["score_source"])

    def test_cost_goal_picks_cheapest_above_floor(self) -> None:
        store = _store(self)
        config = store.get_config()
        for route in config["routes"]:
            if route["id"] == "route-default":
                route["goal"] = GOAL_COST
        store.put_config({"routes": config["routes"], "quality_floor": 0.4})
        result = store.resolve(
            {
                "task_text": "small fix",
                "work_type": "coding",
                "sensitivity": "default",
                "connected_providers": ["openhands", "openai", "anthropic"],
                "local_runtimes": {},
            }
        )
        self.assertEqual(result["decision"]["provider_key"], "openhands")

    def test_reachability_gates_unconnected(self) -> None:
        store = _store(self)
        result = store.resolve(
            {
                "task_text": "write tests",
                "work_type": "test",
                "sensitivity": "default",
                "connected_providers": ["openhands"],
                "local_runtimes": {},
            }
        )
        self.assertEqual(result["decision"]["provider_key"], "openhands")
        dropped_ids = {item["id"] for item in result["trace"]["filters"]}
        self.assertIn("openai/gpt-4o", dropped_ids)

    def test_local_only_when_runtime_alive_and_installed(self) -> None:
        store = _store(self)
        dead = store.resolve(
            {
                "task_text": "offline coding",
                "work_type": "coding",
                "sensitivity": "default",
                "connected_providers": [],
                "local_runtimes": {"ollama": {"alive": False, "models": []}},
            }
        )
        self.assertFalse(dead["decision"]["usable"])
        live = store.resolve(
            {
                "task_text": "offline coding",
                "work_type": "coding",
                "sensitivity": "public",
                "connected_providers": [],
                "local_runtimes": {
                    "ollama": {"alive": True, "models": ["qwen3-coder:16b"]}
                },
            }
        )
        self.assertTrue(live["decision"]["usable"])
        self.assertTrue(str(live["decision"]["model"]).startswith("ollama/"))

    def test_decision_trace_round_trip(self) -> None:
        store = _store(self)
        result = store.resolve(
            {
                "task_text": "design a settings form",
                "work_type": "ux",
                "sensitivity": "default",
                "connected_providers": CONNECTED,
                "card_id": "card-1",
                "run_id": "run-1",
                "local_runtimes": {},
            }
        )
        trace = result["trace"]
        self.assertEqual(trace["task_text"], "design a settings form")
        self.assertEqual(trace["classification"]["work_type"], "ux")
        self.assertEqual(trace["classifier_version"], CLASSIFIER_VERSION)
        self.assertIn("filters", trace)
        self.assertIn("ranked", trace)
        self.assertIn("chosen", trace)
        self.assertTrue(trace["reason"])
        audit = store.list_audit(card_id="card-1")
        self.assertEqual(audit["total"], 1)
        self.assertEqual(audit["items"][0]["payload"]["trace"]["reason"], trace["reason"])

    def test_strict_blocks_stale_privacy_auto(self) -> None:
        store = _store(self)
        store.registry.privacy["last_updated"] = "2020-01-01T00:00:00Z"
        store.put_config({"mode": MODE_STRICT, "metadata_max_age_days": 30})
        result = store.resolve(
            {
                "task_text": "secret sauce",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "connected_providers": CONNECTED,
                "local_runtimes": {},
            }
        )
        self.assertFalse(result["decision"]["usable"])
        self.assertIn("stale", result["trace"]["reason"])

    def test_warn_allows_stale_privacy(self) -> None:
        store = _store(self)
        store.registry.privacy["last_updated"] = "2020-01-01T00:00:00Z"
        store.put_config({"mode": MODE_WARN, "metadata_max_age_days": 30})
        result = store.resolve(
            {
                "task_text": "secret sauce",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "connected_providers": CONNECTED,
                "local_runtimes": {},
            }
        )
        self.assertTrue(result["decision"]["usable"])

    def test_strict_blocks_guardrail_violating_manual_route(self) -> None:
        store = _store(self)
        config = store.get_config()
        config["mode"] = MODE_STRICT
        config["routes"] = [
            {
                "id": "bad-lock",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "goal": GOAL_PRIVACY,
                "target": {
                    "provider_key": "gemini",
                    "model": "google/gemini-2.5-pro",
                },
                "guardrails": {
                    "forbid_training_retention": True,
                    "forbid_watermarking": True,
                    "max_cost_usd_per_task": None,
                    "max_latency_s": None,
                },
            }
        ]
        store.put_config(config)
        result = store.resolve(
            {
                "task_text": "proprietary",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "connected_providers": CONNECTED,
                "local_runtimes": {},
            }
        )
        self.assertFalse(result["decision"]["usable"])
        self.assertIn("strict blocked", result["trace"]["reason"])

    def test_fallback_chain_skips_failed_target(self) -> None:
        store = _store(self)
        first = store.resolve(
            {
                "task_text": "implement feature",
                "work_type": "coding",
                "sensitivity": "default",
                "connected_providers": ["anthropic", "openai", "openhands"],
                "local_runtimes": {},
            }
        )
        skipped = {
            "provider_key": first["decision"]["provider_key"],
            "model": first["decision"]["model"],
        }
        second = store.resolve(
            {
                "task_text": "implement feature",
                "work_type": "coding",
                "sensitivity": "default",
                "connected_providers": ["anthropic", "openai", "openhands"],
                "skip_targets": [skipped],
                "local_runtimes": {},
            }
        )
        self.assertNotEqual(second["decision"]["model"], first["decision"]["model"])
        dropped = {item["id"] for item in second["trace"]["filters"]}
        self.assertIn(first["decision"]["model"], dropped)

    def test_preset_resolution(self) -> None:
        registry = ModelRegistry()
        runtimes = {
            "ollama": {"alive": True, "models": ["qwen3-coder:16b", "llama3.2:3b"]}
        }
        local = resolve_preset(
            PRESET_LOCAL,
            registry,
            connected_providers=[],
            runtimes=runtimes,
        )
        self.assertTrue(local["model"].startswith("ollama/"))
        self.assertTrue(local["tradeoffs"]["offline_capable"])
        best = resolve_preset(
            PRESET_BEST_INTELLIGENCE,
            registry,
            connected_providers=["anthropic", "openai"],
            runtimes={},
        )
        self.assertIn(best["provider_key"], ("anthropic", "openai"))
        cheap = resolve_preset(
            PRESET_CHEAPEST,
            registry,
            connected_providers=["openhands", "openai"],
            runtimes={},
        )
        self.assertEqual(cheap["provider_key"], "openhands")
        secure = resolve_preset(
            PRESET_MOST_SECURE,
            registry,
            connected_providers=CONNECTED,
            runtimes={},
        )
        self.assertEqual(secure["tradeoffs"]["retention"], "none")
        self.assertEqual(secure["tradeoffs"]["watermark"], "none")

    def test_yaml_import_is_idempotent(self) -> None:
        store = _store(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.yaml")
            os.makedirs(tmp, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "project:\n"
                    "  name: demo\n"
                    "  routing:\n"
                    "    rules:\n"
                    "      - task_type: coding\n"
                    "        provider: cursor\n"
                    "        model: composer\n"
                    "    default:\n"
                    "      provider: claude\n"
                    "      model: sonnet\n"
                )
            first = store.import_project_config(path)
            count = len(first["routes"])
            second = store.import_project_config(path)
            self.assertEqual(len(second["routes"]), count)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        probe = LocalRuntimeProbe(
            fetch=lambda *_args, **_kwargs: json.dumps({"models": []})
        )
        self.store = _store(self)
        self.service = RouterService(self.store, probe=probe, fetch=lambda *_a, **_k: "")

    def test_resolve_and_audit_endpoints(self) -> None:
        status, data = handle_request(
            self.service,
            "POST",
            "/api/routing/resolve",
            {
                "task_text": "write unit tests",
                "work_type": "test",
                "sensitivity": "default",
                "connected_providers": CONNECTED,
                "local_runtimes": {},
            },
        )
        self.assertEqual(status, 200)
        self.assertIn("decision", data)
        self.assertIn("trace", data)
        status, audit = handle_request(self.service, "GET", "/api/routing/audit?limit=10")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(audit["total"], 1)

    def test_taxonomy_and_config_round_trip(self) -> None:
        status, tax = handle_request(self.service, "GET", "/api/routing/taxonomy")
        self.assertEqual(status, 200)
        status, _ = handle_request(
            self.service,
            "PUT",
            "/api/routing/config",
            {"mode": MODE_STRICT},
        )
        self.assertEqual(status, 200)
        status, config = handle_request(self.service, "GET", "/api/routing/config")
        self.assertEqual(config["mode"], MODE_STRICT)

    def test_privacy_refresh_rejects_malformed(self) -> None:
        status, body = handle_request(
            self.service,
            "POST",
            "/api/routing/registry/privacy-refresh",
            {"version": "v1", "last_updated": "x", "models": [{"id": "x"}]},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
