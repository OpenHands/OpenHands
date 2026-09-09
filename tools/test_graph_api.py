"""HTTP API tests for the codebase graph."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from codebase_graph import GraphStore  # noqa: E402
from graph_api import GraphService, handle_request  # noqa: E402
from parser_fallback import parse_source as fallback_parse  # noqa: E402
from test_codebase_graph import _repo  # noqa: E402


def _service(test: unittest.TestCase) -> GraphService:
    store = GraphStore(db_path=":memory:", parse_file=fallback_parse)
    test.addCleanup(store.close)
    return GraphService(store)


class GraphApiTests(unittest.TestCase):
    def test_index_status_query_clear_retrigger(self) -> None:
        service = _service(self)
        root = _repo()
        status, payload = handle_request(
            service, "POST", "/api/graph/index", {"root": root, "full": True}
        )
        self.assertEqual(status, 200)
        self.assertGreater(payload["files_indexed"], 0)
        status, snapshot = handle_request(
            service, "GET", f"/api/graph/index/status?root={root}"
        )
        self.assertEqual(status, 200)
        self.assertIn("coverage", snapshot)
        self.assertFalse(snapshot["running"])
        status, callers = handle_request(
            service,
            "GET",
            f"/api/graph/query?q=callers&symbol=greet&root={root}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(callers["source"], "graph")
        self.assertIn("main", {item["name"] for item in callers["result"]})
        status, defs = handle_request(
            service,
            "GET",
            f"/api/graph/definitions?symbol=gre&root={root}",
        )
        self.assertEqual(status, 200)
        self.assertIn("greet", {item["name"] for item in defs["result"]})
        status, cleared = handle_request(
            service, "DELETE", "/api/graph/index", {"root": root}
        )
        self.assertEqual(status, 200)
        self.assertEqual(cleared["files_indexed"], 0)
        status, rebuilt = handle_request(
            service, "POST", "/api/graph/index/retrigger", {"root": root}
        )
        self.assertEqual(status, 200)
        self.assertGreater(rebuilt["files_indexed"], 0)

    def test_config_crud(self) -> None:
        service = _service(self)
        status, config = handle_request(service, "GET", "/api/graph/config")
        self.assertEqual(status, 200)
        self.assertTrue(config["enabled"])
        status, updated = handle_request(
            service, "PUT", "/api/graph/config", {"max_context_files": 4}
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["max_context_files"], 4)

    def test_unknown_query_is_400(self) -> None:
        service = _service(self)
        status, payload = handle_request(service, "GET", "/api/graph/query?q=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_import_project_yaml(self) -> None:
        service = _service(self)
        yaml_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(yaml_dir, ".openhands"), exist_ok=True)
        path = os.path.join(yaml_dir, ".openhands", "project.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "project:\n  name: Demo\n  graph:\n    max_context_files: 9\n"
            )
        status, config = handle_request(
            service, "POST", "/api/graph/import-project-config", {"path": path}
        )
        self.assertEqual(status, 200)
        self.assertEqual(config["max_context_files"], 9)
        status, again = handle_request(
            service, "POST", "/api/graph/import-project-config", {"path": path}
        )
        self.assertEqual(again, config)


if __name__ == "__main__":
    unittest.main()
