"""Tests for the model registry seed, schema, merge, and privacy refresh."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from model_registry import (  # noqa: E402
    REGISTRY_VERSION,
    ModelRegistry,
    RegistryError,
    is_stale,
    load_json,
    merge_registry,
    seed_privacy_path,
    seed_registry_path,
    validate_privacy,
    validate_registry,
)


class SeedAndSchemaTests(unittest.TestCase):
    def test_seed_registry_loads_and_validates(self) -> None:
        payload = load_json(seed_registry_path())
        validated = validate_registry(payload)
        self.assertEqual(validated["version"], REGISTRY_VERSION)
        ids = {item["id"] for item in validated["models"]}
        self.assertIn("openhands/glm-5.2", ids)
        self.assertIn("openai/gpt-4o", ids)
        self.assertIn("anthropic/claude-sonnet-4-5", ids)
        self.assertIn("google/gemini-2.5-pro", ids)
        self.assertIn("cursor-cli/composer", ids)
        self.assertIn("opencode/anthropic/claude-sonnet-4-6", ids)
        self.assertIn("ollama/qwen3-coder:16b", ids)
        self.assertIn("ollama/llama3.1:8b", ids)
        self.assertIn("ollama/deepseek-r1:14b", ids)
        for item in validated["models"]:
            self.assertFalse(item["verified"], msg=item["id"])
            self.assertTrue(item["source_url"])
            for entry in item["benchmarks"].values():
                self.assertIsNone(entry["provenance"])

    def test_seed_privacy_is_curated_not_benchmark(self) -> None:
        payload = load_json(seed_privacy_path())
        validated = validate_privacy(payload)
        self.assertIn("manually researched", validated["curated_fields"])
        gemini = next(item for item in validated["models"] if item["id"] == "google/gemini-2.5-pro")
        self.assertEqual(gemini["watermark"], "always")
        local = next(item for item in validated["models"] if item["id"].startswith("ollama/"))
        self.assertEqual(local["retention"], "none")
        self.assertEqual(local["watermark"], "none")

    def test_malformed_privacy_refresh_is_rejected(self) -> None:
        registry = ModelRegistry()
        before = registry.privacy["last_updated"]
        fixture = os.path.join(
            TOOLS_DIR, "model-registry", "fixtures", "privacy-malformed.json"
        )
        with self.assertRaises(RegistryError):
            registry.refresh_privacy(load_json(fixture))
        self.assertEqual(registry.privacy["last_updated"], before)

    def test_merge_by_provenance_does_not_clobber_ingested_with_seed(self) -> None:
        base = validate_registry(load_json(seed_registry_path()))
        overlay = {
            "version": "v1",
            "last_updated": "2026-09-06T12:00:00Z",
            "models": [
                {
                    "id": "openai/gpt-4o",
                    "provider_key": "openai",
                    "benchmarks": {
                        "coding": {
                            "score": 0.91,
                            "provenance": {
                                "source": "swebench",
                                "fetched_at": "2026-09-06T12:00:00Z",
                                "version": "swebench",
                            },
                        }
                    },
                    "cost_per_1k": 0.00375,
                    "latency_s_p90": 12,
                    "verified": True,
                }
            ],
        }
        merged = merge_registry(base, overlay)
        gpt = next(item for item in merged["models"] if item["id"] == "openai/gpt-4o")
        self.assertEqual(gpt["benchmarks"]["coding"]["score"], 0.91)
        self.assertEqual(gpt["benchmarks"]["coding"]["provenance"]["source"], "swebench")
        reseed = merge_registry(merged, base)
        gpt2 = next(item for item in reseed["models"] if item["id"] == "openai/gpt-4o")
        self.assertEqual(gpt2["benchmarks"]["coding"]["score"], 0.91)
        self.assertIsNotNone(gpt2["benchmarks"]["coding"]["provenance"])

    def test_overlay_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(overlay_dir=tmp)
            registry.apply_ingest_rows(
                "swebench",
                [
                    {
                        "id": "openai/gpt-4o",
                        "provider_key": "openai",
                        "benchmarks": {"coding": {"score": 0.7, "provenance": None}},
                        "cost_per_1k": 0.01,
                        "latency_s_p90": 10,
                        "verified": False,
                    }
                ],
            )
            reloaded = ModelRegistry(overlay_dir=tmp)
            gpt = reloaded.get_model("openai/gpt-4o")
            assert gpt is not None
            self.assertTrue(gpt["verified"])
            self.assertEqual(gpt["benchmarks"]["coding"]["provenance"]["source"], "swebench")

    def test_stale_helper(self) -> None:
        from datetime import datetime, timezone

        now = datetime(2026, 9, 6, tzinfo=timezone.utc)
        self.assertTrue(is_stale("2026-01-01T00:00:00Z", 30, now=now))
        self.assertFalse(is_stale("2026-09-01T00:00:00Z", 30, now=now))


if __name__ == "__main__":
    unittest.main()
