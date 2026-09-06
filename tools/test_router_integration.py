"""Resolve → dispatch → escalate → resume integration tests.

No real LLM, CLI binary, local runtime, or network.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from cost_estimator import record_routing_outcome, reset_routing_outcomes  # noqa: E402
from kanban import KanbanStore  # noqa: E402
from loop_runner import LoopStore  # noqa: E402
from loop_triggers import LoopTriggerService, TRIGGER_MANUAL  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402
from provider_adapters import adapter_for  # noqa: E402
from router import MODE_STRICT, MODE_WARN, RouterStore  # noqa: E402
from router_runtime import (  # noqa: E402
    TRACE_MARKER,
    build_resume_prompt,
    dispatch_resolved,
    escalate_on_struggle,
    persist_dispatch_trace,
    read_resume_state,
    reset_struggle,
    resolve_for_dispatch,
)

CONNECTED = ["cursor-cli", "anthropic", "opencode"]


def _store(test: unittest.TestCase) -> RouterStore:
    store = RouterStore(db_path=":memory:", registry=ModelRegistry())
    test.addCleanup(store.close)
    return store


class ResolveDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_struggle()
        reset_routing_outcomes()

    def tearDown(self) -> None:
        reset_struggle()
        reset_routing_outcomes()

    def test_resolve_picks_auto_target_and_dispatches_adapter(self) -> None:
        store = _store(self)
        result = resolve_for_dispatch(
            store,
            task_text="implement a parser",
            work_type="coding",
            sensitivity="default",
            connected_providers=CONNECTED,
            local_runtimes={},
            run_id="run-dispatch",
        )
        decision = result["decision"]
        self.assertTrue(decision["usable"])
        self.assertTrue(decision["provider_key"])
        self.assertTrue(result["trace"]["reason"])
        adapter = adapter_for(decision["provider_key"])
        self.assertIsNotNone(adapter)
        adapter.find_binary = lambda: "/bin/fake"  # type: ignore[method-assign]
        captured: list[list[str]] = []

        def runner(cmd: list[str], cwd: str | None, timeout_s: float) -> dict[str, object]:
            captured.append(cmd)
            return {"ok": True, "stdout": "ok", "stderr": "", "cmd": cmd}

        output = dispatch_resolved(result, "implement a parser", runner=runner)
        self.assertTrue(output["ok"])
        self.assertTrue(captured)
        audit = store.list_audit(run_id="run-dispatch")
        self.assertGreaterEqual(audit["total"], 1)
        self.assertEqual(audit["items"][0]["kind"], "resolve")

    def test_struggle_escalation_switches_target_and_records_audit(self) -> None:
        store = _store(self)
        worktree = tempfile.mkdtemp()
        first = resolve_for_dispatch(
            store,
            task_text="implement a parser",
            work_type="coding",
            sensitivity="default",
            connected_providers=CONNECTED,
            local_runtimes={},
            run_id="run-struggle",
        )
        failed_target = {
            "provider_key": first["decision"]["provider_key"],
            "model": first["decision"]["model"],
        }
        first_try = escalate_on_struggle(
            store,
            task_text="implement a parser",
            failed_result=first,
            failed_output="stage lint failed",
            worktree_dir=worktree,
            branch="feat/parser",
            ticket="implement a parser",
            run_id="run-struggle",
            connected_providers=CONNECTED,
            local_runtimes={},
            stage_type="lint",
        )
        self.assertFalse(first_try.get("switched"))
        nxt = escalate_on_struggle(
            store,
            task_text="implement a parser",
            failed_result=first,
            failed_output="stage lint failed again",
            worktree_dir=worktree,
            branch="feat/parser",
            ticket="implement a parser",
            run_id="run-struggle",
            connected_providers=CONNECTED,
            local_runtimes={},
            stage_type="lint",
        )
        self.assertTrue(nxt.get("switched"))
        self.assertNotEqual(
            (nxt["decision"]["provider_key"], nxt["decision"]["model"]),
            (failed_target["provider_key"], failed_target["model"]),
        )
        switches = store.list_audit(run_id="run-struggle", kind="switch")
        self.assertGreaterEqual(switches["total"], 1)
        payload = switches["items"][0]["payload"]
        self.assertEqual(payload["reason"], "struggle")
        self.assertEqual(payload["from"]["model"], failed_target["model"])
        self.assertEqual(payload["to"]["model"], nxt["decision"]["model"])

    def test_resumed_provider_prompt_includes_branch_and_failed_output(self) -> None:
        store = _store(self)
        worktree = tempfile.mkdtemp()
        first = resolve_for_dispatch(
            store,
            task_text="fix the login form",
            work_type="coding",
            sensitivity="default",
            connected_providers=CONNECTED,
            local_runtimes={},
            run_id="run-resume",
        )
        escalate_on_struggle(
            store,
            task_text="fix the login form",
            failed_result=first,
            failed_output="TypeError: boom",
            worktree_dir=worktree,
            branch="feat/login",
            ticket="fix the login form",
            run_id="run-resume",
            connected_providers=CONNECTED,
            stage_type="typecheck",
        )
        nxt = escalate_on_struggle(
            store,
            task_text="fix the login form",
            failed_result=first,
            failed_output="TypeError: boom",
            worktree_dir=worktree,
            branch="feat/login",
            ticket="fix the login form",
            run_id="run-resume",
            connected_providers=CONNECTED,
            stage_type="typecheck",
        )
        prompt = nxt["resume_prompt"]
        self.assertIn("Resume context", prompt)
        self.assertIn("feat/login", prompt)
        self.assertIn("TypeError: boom", prompt)
        self.assertIn("fix the login form", prompt)
        state = read_resume_state(worktree)
        self.assertIsNotNone(state)
        self.assertEqual(state["failed_output"], "TypeError: boom")
        rebuilt = build_resume_prompt("fix the login form", state)
        self.assertIn("TypeError: boom", rebuilt)

    def test_strict_refuses_stale_privacy_and_guardrail_manual(self) -> None:
        store = _store(self)
        store.registry.privacy["last_updated"] = "2020-01-01T00:00:00Z"
        store.put_config({"mode": MODE_STRICT, "metadata_max_age_days": 30})
        stale = resolve_for_dispatch(
            store,
            task_text="secret sauce",
            work_type="coding",
            sensitivity="sensitive-ip",
            connected_providers=CONNECTED + ["gemini"],
            local_runtimes={},
        )
        self.assertFalse(stale["decision"]["usable"])
        self.assertIn("stale", stale["trace"]["reason"])

        store.put_config({"mode": MODE_WARN, "metadata_max_age_days": 30})
        warned = resolve_for_dispatch(
            store,
            task_text="secret sauce",
            work_type="coding",
            sensitivity="sensitive-ip",
            connected_providers=CONNECTED + ["gemini"],
            local_runtimes={},
        )
        self.assertTrue(warned["decision"]["usable"])

        store.registry.privacy["last_updated"] = "2026-09-06T00:00:00Z"
        config = store.get_config()
        config["mode"] = MODE_STRICT
        config["routes"] = [
            {
                "id": "bad-lock",
                "work_type": "coding",
                "sensitivity": "sensitive-ip",
                "goal": "privacy",
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
        blocked = resolve_for_dispatch(
            store,
            task_text="secret sauce",
            work_type="coding",
            sensitivity="sensitive-ip",
            connected_providers=CONNECTED + ["gemini"],
            local_runtimes={},
        )
        self.assertFalse(blocked["decision"]["usable"])

    def test_blender_combines_benchmark_and_empirical_pass_rate(self) -> None:
        store = _store(self)
        store.put_config({"blend_alpha": 0.5, "goal": "quality"})
        for _ in range(4):
            record_routing_outcome("coding", "openhands", "openhands/glm-5.2", True)
        blended = resolve_for_dispatch(
            store,
            task_text="implement a parser",
            work_type="coding",
            sensitivity="default",
            connected_providers=["openhands", "anthropic", "openai"],
            local_runtimes={},
        )
        self.assertEqual(blended["decision"]["model"], "openhands/glm-5.2")
        self.assertEqual(blended["decision"]["score_source"], "blend")
        ranked = blended["trace"]["ranked"]
        glm = next(item for item in ranked if item["id"] == "openhands/glm-5.2")
        self.assertEqual(glm["score_source"], "blend")
        self.assertGreater(glm["score"], 0.78)

    def test_trace_persisted_on_every_auto_dispatch(self) -> None:
        store = _store(self)
        worktree = tempfile.mkdtemp()
        kanban = KanbanStore(":memory:")
        self.addCleanup(kanban.close)
        board = kanban.create_board("Work", project_id="proj-1")
        card = kanban.create_card(board["columns"][0]["id"], title="Ship parser")
        result = resolve_for_dispatch(
            store,
            task_text="implement a parser work_type:coding sensitivity:default",
            connected_providers=CONNECTED,
            local_runtimes={},
            card_id=card["id"],
            run_id="run-trace",
        )
        persist_dispatch_trace(
            result,
            worktree_dir=worktree,
            kanban_store=kanban,
            card_id=card["id"],
        )
        decision_path = Path(worktree) / ".openhands" / "routing-decision.json"
        self.assertTrue(decision_path.is_file())
        updated = kanban.get_card(card["id"])
        self.assertIn(TRACE_MARKER.strip(), updated["description"])
        self.assertIn(result["trace"]["reason"], updated["description"])
        audit = store.list_audit(card_id=card["id"])
        self.assertEqual(audit["total"], 1)
        self.assertEqual(
            audit["items"][0]["payload"]["trace"]["reason"],
            result["trace"]["reason"],
        )

    def test_loop_trigger_resolves_at_fire_time(self) -> None:
        store = _store(self)
        worktree = tempfile.mkdtemp()
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
            router_store=store,
        )
        self.addCleanup(service.close)
        trigger = service.create_trigger(
            project_id="proj-1",
            loop_definition_id=definition["id"],
            trigger_type=TRIGGER_MANUAL,
        )
        fired = service.fire_trigger(
            trigger["id"],
            {
                "worktree_dir": worktree,
                "task_text": "implement a parser",
                "connected_providers": CONNECTED,
                "local_runtimes": {},
            },
        )
        self.assertIn("routing", fired)
        self.assertTrue(fired["routing"]["decision"]["usable"])
        self.assertTrue(fired["routing"]["trace"]["reason"])
        audit = store.list_audit(run_id=fired["run"]["id"])
        self.assertGreaterEqual(audit["total"], 1)


if __name__ == "__main__":
    unittest.main()
