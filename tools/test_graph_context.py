"""Relevance ranking and graph-context composition. No LLM, no randomness."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from codebase_graph import GraphStore  # noqa: E402
from feature_developer import FeatureDeveloper, TICKET_PASSED  # noqa: E402
from graph_agent_hooks import (  # noqa: E402
    GRAPH_CONTEXT_FILENAME,
    GRAPH_NOTE_PREFIX,
    apply_dispatch_graph_context,
)
from graph_context import (  # noqa: E402
    GRAPH_CONTEXT_OPEN,
    attach_graph_context,
    compose_graph_context_block,
    select_relevant_files,
    suggest_edit_target,
)
from kanban import KanbanStore  # noqa: E402
from loop_runner import LoopStore  # noqa: E402
from loop_triggers import TRIGGER_MANUAL, LoopTriggerService  # noqa: E402
from parser_fallback import parse_source as fallback_parse  # noqa: E402
from test_codebase_graph import _repo  # noqa: E402


class RelevanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = GraphStore(db_path=":memory:", parse_file=fallback_parse)
        self.addCleanup(self.store.close)
        self.root = _repo()
        self.store.index(self.root, full=True)

    def test_selects_definition_file_for_named_symbol(self) -> None:
        files = select_relevant_files(
            self.store,
            self.root,
            "Please fix the greet function",
        )
        paths = [item["file"] for item in files]
        self.assertIn("helper.py", paths)
        helper = next(item for item in files if item["file"] == "helper.py")
        self.assertIn("defined here", helper["reason"])

    def test_walk_caps_depth(self) -> None:
        files = select_relevant_files(
            self.store,
            self.root,
            "run the app",
            seeds=["app.py"],
            walk_depth=2,
        )
        self.assertTrue(files)
        self.assertLessEqual(max(item.get("depth", 0) for item in files), 2)

    def test_budget_respected(self) -> None:
        files = select_relevant_files(
            self.store,
            self.root,
            "greet foo run main",
            budget={"max_files": 1, "budget_lines": 400},
        )
        self.assertEqual(len(files), 1)
        total_lines = sum(item["lines"] for item in files)
        self.assertLessEqual(total_lines, 400)

    def test_deterministic(self) -> None:
        a = select_relevant_files(self.store, self.root, "greet helper")
        b = select_relevant_files(self.store, self.root, "greet helper")
        self.assertEqual(a, b)

    def test_disabled_does_not_block(self) -> None:
        spec = attach_graph_context(
            {
                "task_text": "fix greet",
                "root": self.root,
                "graph_enabled": False,
            },
            store=self.store,
        )
        self.assertNotIn("<GRAPH_CONTEXT>", spec.get("spec_text") or "")
        self.assertFalse(spec.get("graph_blocked"))

    def test_stale_strict_skips_warn_pins(self) -> None:
        self.store.put_config({"stale_after_minutes": 0, "strict": True})
        skipped = attach_graph_context(
            {"task_text": "fix greet", "root": self.root},
            store=self.store,
        )
        self.assertTrue(skipped.get("graph_stale_banner"))
        self.assertNotIn("<GRAPH_CONTEXT>", skipped.get("spec_text") or "")
        self.store.put_config({"strict": False})
        warned = attach_graph_context(
            {"task_text": "fix greet", "root": self.root},
            store=self.store,
        )
        self.assertIn("<GRAPH_CONTEXT>", warned["spec_text"])
        self.assertIn("stale", warned["spec_text"].lower())

    def test_suggest_edit_target_narrows_span(self) -> None:
        target = suggest_edit_target(
            self.store, "change greet", "helper.py", self.root
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target["name"], "greet")
        self.assertGreaterEqual(target["start_line"], 1)

    def test_compose_block_lists_reasons(self) -> None:
        files = [{"file": "helper.py", "relevance": 400, "reason": "defined here", "lines": 3}]
        block = compose_graph_context_block(files)
        self.assertIn("helper.py", block)
        self.assertIn("defined here", block)


class DispatchHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = GraphStore(db_path=":memory:", parse_file=fallback_parse)
        self.addCleanup(self.store.close)
        self.root = _repo()
        self.store.index(self.root, full=True)
        self.kanban = KanbanStore(":memory:")
        self.addCleanup(self.kanban.close)
        board = self.kanban.create_board("graph")
        self.card = self.kanban.create_card(
            board["columns"][0]["id"],
            "Fix greet",
            description="Please fix the greet function",
        )

    def test_apply_dispatch_writes_file_and_card_note(self) -> None:
        attached = apply_dispatch_graph_context(
            {
                "task_text": "Please fix the greet function",
                "root": self.root,
                "worktree_dir": self.root,
                "card_id": self.card["id"],
            },
            store=self.store,
            kanban_store=self.kanban,
        )
        self.assertIn(GRAPH_CONTEXT_OPEN, attached["prompt"])
        self.assertIn("helper.py", attached["prompt"])
        path = os.path.join(self.root, GRAPH_CONTEXT_FILENAME)
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            self.assertIn("helper.py", handle.read())
        log = self.kanban.get_card(self.card["id"])["activity_log"]
        self.assertTrue(
            any(
                str(item.get("message") or "").startswith(GRAPH_NOTE_PREFIX)
                for item in log
            )
        )

    def test_loop_trigger_prompt_includes_graph_context(self) -> None:
        loops = LoopStore(":memory:")
        self.addCleanup(loops.close)
        definition = loops.create_definition(
            name="echo",
            project_id="proj-1",
            stages=[
                {
                    "name": "echo",
                    "cmd": "python3 -c \"print('ok')\"",
                    "iterative": False,
                }
            ],
        )
        service = LoopTriggerService(
            db_path=":memory:",
            loop_store=loops,
            graph_store=self.store,
            kanban_store=self.kanban,
        )
        self.addCleanup(service.close)
        trigger = service.create_trigger(
            project_id="proj-1",
            loop_definition_id=definition["id"],
            trigger_type=TRIGGER_MANUAL,
        )
        result = service.fire_trigger(
            trigger["id"],
            {
                "worktree_dir": self.root,
                "task_text": "Please fix the greet function",
                "card_id": self.card["id"],
            },
        )
        self.assertIn(GRAPH_CONTEXT_OPEN, result.get("prompt") or "")
        self.assertIn("helper.py", result["prompt"])
        log = self.kanban.get_card(self.card["id"])["activity_log"]
        self.assertTrue(
            any(
                str(item.get("message") or "").startswith(GRAPH_NOTE_PREFIX)
                for item in log
            )
        )

    def test_feature_developer_run_appends_graph_block(self) -> None:
        seen: list[str] = []

        def implement(run: dict, ticket: dict) -> dict:
            seen.append(str(run.get("spec_text") or ""))
            return {
                "status": TICKET_PASSED,
                "actual_usd": 0.1,
                "branch_name": "feat/greet",
            }

        def llm(_prompt: str) -> str:
            return """
            {"features":[{"name":"Greet","epics":[{"name":"G","tickets":[
              {"title":"Please fix the greet function","description":"Update greet","acceptance":["ok"]}
            ]}]}]}
            """

        dev = FeatureDeveloper(
            kanban_store=self.kanban,
            graph_store=self.store,
            implement_fn=implement,
            llm_complete=llm,
        )
        self.addCleanup(dev.close)
        run = dev.start_run("proj-1", "Please fix the greet function")
        self.assertTrue(seen)
        self.assertIn(GRAPH_CONTEXT_OPEN, seen[0])
        self.assertIn("helper.py", seen[0])
        self.assertIn(GRAPH_CONTEXT_OPEN, run["spec_text"])
        ticket = run["tickets"][0]
        log = self.kanban.get_card(ticket["card_id"])["activity_log"]
        self.assertTrue(
            any(
                str(item.get("message") or "").startswith(GRAPH_NOTE_PREFIX)
                for item in log
            )
        )


if __name__ == "__main__":
    unittest.main()
