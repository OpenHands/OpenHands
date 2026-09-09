"""Graph store, incremental indexer, and query tests."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from codebase_graph import (  # noqa: E402
    EDGE_CALL,
    EDGE_IMPORT,
    QUERY_CALLERS,
    QUERY_DEPS,
    QUERY_USAGES,
    SOURCE_GRAPH,
    GraphStore,
)
from parser_fallback import parse_source as fallback_parse  # noqa: E402

HELPER_PY = '''\
def greet():
    print("hi")
'''

APP_PY = '''\
from helper import greet
from missinglib import ghost

def main():
    greet()
'''

LIB_JS = '''\
export function foo() {
  return 1;
}
'''

APP_JS = '''\
import { foo } from "./lib";
import { greet } from "./helper";

export function run() {
  foo();
}
'''

README_MD = '''\
# Docs

See [app](./app.py).
'''


def _write(root: str, rel: str, contents: str) -> str:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(contents)
    return path


def _repo() -> str:
    root = tempfile.mkdtemp(prefix="graph-idx-")
    _write(root, "helper.py", HELPER_PY)
    _write(root, "app.py", APP_PY)
    _write(root, "lib.js", LIB_JS)
    _write(root, "app.js", APP_JS)
    _write(root, "README.md", README_MD)
    return root


def _store(test: unittest.TestCase) -> GraphStore:
    store = GraphStore(db_path=":memory:", parse_file=fallback_parse)
    test.addCleanup(store.close)
    return store


class IndexerTests(unittest.TestCase):
    def test_creates_nodes_and_edges_idempotently(self) -> None:
        store = _store(self)
        root = _repo()
        first = store.index(root, full=True)
        nodes = store.list_nodes(root)
        edges = store.list_edges(root)
        self.assertGreater(first["files_indexed"], 0)
        self.assertTrue(any(n["kind"] == "function" and n["name"] == "greet" for n in nodes))
        self.assertTrue(any(e["kind"] == EDGE_CALL for e in edges))
        self.assertTrue(any(e["kind"] == EDGE_IMPORT for e in edges))
        second = store.index(root, full=False)
        self.assertEqual(second["files_parsed"], 0)
        self.assertEqual(len(store.list_nodes(root)), len(nodes))
        self.assertEqual(len(store.list_edges(root)), len(edges))

    def test_incremental_only_reparses_changed_mtimes(self) -> None:
        parsed: list[str] = []

        def tracking_parse(path: str, source: str, language: str):
            parsed.append(os.path.basename(path))
            return fallback_parse(path, source, language)

        store = GraphStore(db_path=":memory:", parse_file=tracking_parse)
        self.addCleanup(store.close)
        root = _repo()
        store.index(root, full=True)
        parsed.clear()
        store.index(root, full=False)
        self.assertEqual(parsed, [])
        helper = os.path.join(root, "helper.py")
        time.sleep(0.05)
        os.utime(helper, None)
        store.index(root, full=False)
        self.assertEqual(parsed, ["helper.py"])

    def test_full_rebuild_replaces_stale_rows(self) -> None:
        store = _store(self)
        root = _repo()
        store.index(root, full=True)
        helper = os.path.join(root, "helper.py")
        with open(helper, "w", encoding="utf-8") as handle:
            handle.write("def other():\n    return 1\n")
        store.index(root, full=True)
        names = {n["name"] for n in store.list_nodes(root) if n["kind"] == "function"}
        self.assertIn("other", names)
        self.assertNotIn("greet", names)

    def test_cross_language_imports_and_external_flag(self) -> None:
        store = _store(self)
        root = _repo()
        store.index(root, full=True)
        nodes = store.list_nodes(root)
        helper = next(
            n
            for n in nodes
            if n["kind"] == "file" and (n.get("path") or "").endswith("helper.py")
        )
        self.assertFalse(helper["external"])
        missing = next(n for n in nodes if n["name"] == "missinglib")
        self.assertTrue(missing["external"])
        js_helper = next(
            n
            for n in nodes
            if n["kind"] in {"file", "module"}
            and (n.get("path") or "").endswith("helper.py")
        )
        self.assertFalse(js_helper["external"])

    def test_tree_sitter_import_error_does_not_fail_index(self) -> None:
        store = GraphStore(db_path=":memory:")
        self.addCleanup(store.close)
        root = _repo()
        with mock.patch.dict(sys.modules, {"tree_sitter": None}):
            status = store.index(root, full=True)
        self.assertGreater(status["files_indexed"], 0)
        self.assertIsNone(status.get("last_error") or None)


class QueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _store(self)
        self.root = _repo()
        self.store.index(self.root, full=True)

    def test_callers_reverses_call_edges(self) -> None:
        payload = self.store.query(QUERY_CALLERS, symbol="greet", root=self.root)
        self.assertEqual(payload["source"], SOURCE_GRAPH)
        self.assertEqual(payload["query"], QUERY_CALLERS)
        names = {item["name"] for item in payload["result"]}
        self.assertIn("main", names)
        self.assertIn("coverage", payload)
        self.assertIn("last_full_index_at", payload)

    def test_deps_follow_import_edges(self) -> None:
        payload = self.store.query(
            QUERY_DEPS, file="app.py", root=self.root
        )
        names = {item["name"] for item in payload["result"]}
        self.assertIn("helper", names)
        self.assertIn("missinglib", names)

    def test_usages_match_reference_and_call_names(self) -> None:
        payload = self.store.query(QUERY_USAGES, symbol="foo", root=self.root)
        self.assertTrue(payload["result"])
        self.assertTrue(
            any(item.get("kind") in {EDGE_CALL, "reference", "call"} for item in payload["result"])
        )

    def test_definitions_fuzzy_lookup(self) -> None:
        payload = self.store.definitions("gre", root=self.root)
        names = {item["name"] for item in payload["result"]}
        self.assertIn("greet", names)


class ConfigTests(unittest.TestCase):
    def test_config_crud_and_project_yaml_import_idempotent(self) -> None:
        store = _store(self)
        default = store.get_config()
        self.assertTrue(default["enabled"])
        updated = store.put_config({"max_context_files": 3, "strict": True})
        self.assertEqual(updated["max_context_files"], 3)
        self.assertTrue(updated["strict"])
        yaml_dir = tempfile.mkdtemp()
        yaml_path = os.path.join(yaml_dir, "project.yaml")
        os.makedirs(os.path.join(yaml_dir, ".openhands"), exist_ok=True)
        yaml_path = os.path.join(yaml_dir, ".openhands", "project.yaml")
        with open(yaml_path, "w", encoding="utf-8") as handle:
            handle.write(
                "project:\n"
                "  name: Demo\n"
                "  graph:\n"
                "    enabled: true\n"
                "    max_context_files: 7\n"
                "    graph_budget_lines: 120\n"
            )
        first = store.import_project_config(yaml_path)
        self.assertEqual(first["max_context_files"], 7)
        self.assertEqual(first["graph_budget_lines"], 120)
        second = store.import_project_config(yaml_path)
        self.assertEqual(first, second)

    def test_clear_and_retrigger(self) -> None:
        store = _store(self)
        root = _repo()
        store.index(root, full=True)
        self.assertGreater(len(store.list_nodes(root)), 0)
        store.clear(root)
        self.assertEqual(store.list_nodes(root), [])
        retriggered = store.retrigger(root)
        self.assertGreater(retriggered["files_indexed"], 0)
        self.assertGreater(len(store.list_nodes(root)), 0)


if __name__ == "__main__":
    unittest.main()
