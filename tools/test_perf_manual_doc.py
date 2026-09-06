"""Integration tests for perf, manual, and doc loops.

Run from the repo root:

    python3 tools/test_perf_manual_doc.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from doc_loop import DOC_LOOP_NAME, DocLoopService  # noqa: E402
from loop_runner import (  # noqa: E402
    STATUS_ABORTED,
    STATUS_AWAITING_INPUT,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    LoopStore,
)
from manual_loop import MANUAL_LOOP_NAME, ManualLoopService  # noqa: E402
from perf_loop import PERF_LOOP_NAME, detect_metrics  # noqa: E402
from perf_manual_doc_api import PerfManualDocService, handle_request  # noqa: E402


class PerfManualDocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp()
        self.loops = LoopStore(":memory:")
        self.service = PerfManualDocService(self.loops)

    def tearDown(self) -> None:
        self.service.close()
        self.loops.close()

    def test_perf_detection_crosses_threshold(self) -> None:
        log_path = os.path.join(self.workdir, "bench.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("request took 12ms\nrequest took 900ms\n")
        with open(log_path, encoding="utf-8") as handle:
            log_text = handle.read()
        detections, failures = detect_metrics(
            log_text,
            [{"pattern": r"took ([0-9.]+)ms", "threshold": 500, "metric": "duration"}],
        )
        self.assertEqual(detections[0]["value"], 900.0)
        self.assertEqual(len(failures), 1)

        definition = self.service.perf.setup(
            "proj-1",
            {
                "cmd": f"python3 -c \"print(open({log_path!r}).read())\"",
                "rules": [
                    {
                        "pattern": r"took ([0-9.]+)ms",
                        "threshold": 500,
                        "metric": "duration",
                    }
                ],
            },
        )
        run = self.loops.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(run["status"], STATUS_FAILED)
        self.assertIn("900", run["stages"][1]["last_output"])

    def test_manual_loop_blocks_then_feedback(self) -> None:
        flag = os.path.join(self.workdir, "FAIL")
        open(flag, "w", encoding="utf-8").close()
        cmd = (
            "python3 -c \"import pathlib,sys; "
            f"p=pathlib.Path({flag!r});"
            "sys.exit(1 if p.exists() else 0)\""
        )
        self.service.manual.setup("proj-1", {"cmd": cmd})
        run = self.service.manual.start("proj-1", self.workdir)
        self.assertEqual(run["status"], STATUS_AWAITING_INPUT)

        approved = self.loops.submit_feedback(run["id"], approve=True, note="try again")
        self.assertEqual(approved["status"], STATUS_AWAITING_INPUT)
        os.remove(flag)
        passed = self.loops.submit_feedback(run["id"], approve=True)
        self.assertEqual(passed["status"], STATUS_PASSED)

        open(flag, "w", encoding="utf-8").close()
        blocked = self.service.manual.start("proj-1", self.workdir)
        rejected = self.loops.submit_feedback(blocked["id"], approve=False, note="stop")
        self.assertEqual(rejected["status"], STATUS_ABORTED)

    def test_doc_loop_skips_without_credentials_and_syncs_with_mock(self) -> None:
        posted: list[dict[str, Any]] = []

        def secrets(name: str) -> str | None:
            return None

        def http_post(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
            posted.append({"url": url, "body": body, "headers": headers})
            return 200, "ok"

        docs = DocLoopService(
            loop_store=self.loops, secrets_get=secrets, http_post=http_post
        )
        definition = docs.setup(
            "proj-1",
            {
                "gen_cmd": "python3 -c \"print('# docs')\"",
                "adapter": "notion",
                "base_url": "http://docs.test",
            },
        )
        skipped = self.loops.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(skipped["status"], STATUS_PASSED)
        self.assertEqual(skipped["stages"][1]["status"], STATUS_SKIPPED)
        self.assertIn("NOTION_API_KEY", skipped["stages"][1]["last_output"])

        docs.secrets_get = lambda name: "secret-token"
        synced = self.loops.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(synced["status"], STATUS_PASSED)
        self.assertEqual(posted[0]["url"], "http://docs.test/v1/pages")
        self.assertEqual(posted[0]["headers"]["Authorization"], "Bearer secret-token")
        self.assertIn("# docs", posted[0]["body"]["content"])

    def test_setup_is_idempotent_and_status_routes(self) -> None:
        status, created = handle_request(
            self.service,
            "POST",
            "/api/perf-manual-doc/projects/proj-1/setup",
            {
                "perf": {"cmd": "true"},
                "manual": {"cmd": "true"},
                "doc": {"gen_cmd": "true"},
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["perf"]["name"], PERF_LOOP_NAME)
        self.assertEqual(created["manual"]["name"], MANUAL_LOOP_NAME)
        self.assertEqual(created["doc"]["name"], DOC_LOOP_NAME)
        status, again = handle_request(
            self.service,
            "POST",
            "/api/perf-manual-doc/projects/proj-1/setup",
            {},
        )
        self.assertEqual(status, 201)
        self.assertEqual(again["perf"]["id"], created["perf"]["id"])
        self.assertEqual(again["manual"]["id"], created["manual"]["id"])
        self.assertEqual(again["doc"]["id"], created["doc"]["id"])
        status, report = handle_request(
            self.service, "GET", "/api/perf-manual-doc/projects/proj-1/status"
        )
        self.assertEqual(status, 200)
        self.assertEqual(report["perf"]["definition"]["id"], created["perf"]["id"])
        status, rules = handle_request(
            self.service, "GET", "/api/perf-manual-doc/perf-rules"
        )
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(rules), 1)


if __name__ == "__main__":
    unittest.main()
