"""Offline ingest adapter tests."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from benchmark_ingest import (  # noqa: E402
    SOURCE_ARTIFICIAL_ANALYSIS,
    SOURCE_LIVEBENCH,
    SOURCE_LMSYS,
    SOURCE_SWEBENCH,
    fixtures_dir,
    ingest_sources,
    lookup_canonical_id,
)
from model_registry import ModelRegistry  # noqa: E402


def _read(name: str) -> str:
    with open(os.path.join(fixtures_dir(), name), encoding="utf-8") as handle:
        return handle.read()


FIXTURES = {
    "swebench": _read("swebench.json"),
    "lmsys": _read("lmsys.csv"),
    "aa": _read("artificial-analysis.json"),
    "livebench": _read("livebench.csv"),
}


class AliasTests(unittest.TestCase):
    def test_alias_maps_display_names(self) -> None:
        self.assertEqual(lookup_canonical_id("GPT-4o"), "openai/gpt-4o")
        self.assertEqual(lookup_canonical_id("Claude Sonnet 4.5"), "anthropic/claude-sonnet-4-5")
        self.assertIsNone(lookup_canonical_id("mystery-lab-9"))


class AdapterFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ModelRegistry()

        def fetch(url: str, _timeout: float) -> str:
            if "SWE-bench" in url or url.endswith("swebench.json"):
                return FIXTURES["swebench"]
            if "lmsys" in url:
                return FIXTURES["lmsys"]
            if "artificialanalysis" in url:
                return FIXTURES["aa"]
            if "LiveBench" in url or "livebench" in url:
                return FIXTURES["livebench"]
            raise AssertionError(f"unexpected url {url}")

        self.fetch = fetch

    def test_swebench_maps_coding_and_reports_unmapped(self) -> None:
        result = ingest_sources(
            self.registry,
            sources=[SOURCE_SWEBENCH],
            fetch=self.fetch,
        )
        self.assertTrue(result["sources"][SOURCE_SWEBENCH]["ok"])
        names = {item["name"] for item in result["unmapped"]}
        self.assertIn("mystery-lab-9", names)
        gpt = self.registry.get_model("openai/gpt-4o")
        assert gpt is not None
        self.assertEqual(gpt["benchmarks"]["coding"]["provenance"]["source"], SOURCE_SWEBENCH)
        self.assertTrue(gpt["verified"])

    def test_lmsys_maps_reasoning(self) -> None:
        result = ingest_sources(
            self.registry, sources=[SOURCE_LMSYS], fetch=self.fetch
        )
        self.assertTrue(result["sources"][SOURCE_LMSYS]["ok"])
        self.assertIn("unknown-chatbot", {item["name"] for item in result["unmapped"]})
        claude = self.registry.get_model("anthropic/claude-sonnet-4-5")
        assert claude is not None
        self.assertIn("reasoning", claude["benchmarks"])
        self.assertIsNotNone(claude["benchmarks"]["reasoning"]["provenance"])

    def test_artificial_analysis_updates_cost_and_latency(self) -> None:
        ingest_sources(
            self.registry, sources=[SOURCE_ARTIFICIAL_ANALYSIS], fetch=self.fetch
        )
        gpt = self.registry.get_model("openai/gpt-4o")
        assert gpt is not None
        self.assertAlmostEqual(gpt["cost_per_1k"], (2.5 + 10.0) / 2 / 1000, places=6)
        self.assertGreater(gpt["latency_s_p90"], 0)
        self.assertEqual(gpt["cost_provenance"]["source"], SOURCE_ARTIFICIAL_ANALYSIS)

    def test_livebench_column_mapping(self) -> None:
        result = ingest_sources(
            self.registry, sources=[SOURCE_LIVEBENCH], fetch=self.fetch
        )
        self.assertTrue(result["sources"][SOURCE_LIVEBENCH]["ok"])
        self.assertIn(
            "totally-unknown-model", {item["name"] for item in result["unmapped"]}
        )

    def test_source_failure_keeps_last_good_and_marks_stale(self) -> None:
        ingest_sources(
            self.registry, sources=[SOURCE_SWEBENCH], fetch=self.fetch
        )
        good = self.registry.get_model("openai/gpt-4o")
        assert good is not None
        good_score = good["benchmarks"]["coding"]["score"]

        def boom(url: str, timeout: float) -> str:
            raise OSError("network down")

        result = ingest_sources(
            self.registry, sources=[SOURCE_SWEBENCH], fetch=boom
        )
        self.assertFalse(result["sources"][SOURCE_SWEBENCH]["ok"])
        status = self.registry.sources[SOURCE_SWEBENCH]
        self.assertTrue(status["stale"])
        self.assertEqual(status["last_error"], "network down")
        self.assertIsNotNone(status["last_success"])
        still = self.registry.get_model("openai/gpt-4o")
        assert still is not None
        self.assertEqual(still["benchmarks"]["coding"]["score"], good_score)

    def test_malformed_payload_does_not_apply(self) -> None:
        before = self.registry.get_model("openai/gpt-4o")
        assert before is not None
        score = before["benchmarks"]["coding"]["score"]

        def bad(_url: str, _timeout: float) -> str:
            return "not-json-or-csv<<<"

        ingest_sources(self.registry, sources=[SOURCE_SWEBENCH], fetch=bad)
        after = self.registry.get_model("openai/gpt-4o")
        assert after is not None
        self.assertEqual(after["benchmarks"]["coding"]["score"], score)
        self.assertTrue(self.registry.sources[SOURCE_SWEBENCH]["stale"])


if __name__ == "__main__":
    unittest.main()
