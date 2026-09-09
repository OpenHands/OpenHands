"""Manual-in-the-loop wait/resume around a failing stage."""

from __future__ import annotations

from typing import Any

from loop_runner import (
    LoopStore,
    ON_FAILURE_AUTO_FIX,
    STATUS_RUNNING,
    register_stage_handler,
)

MANUAL_LOOP_NAME = "manual-loop"
MANUAL_STAGES: list[dict[str, Any]] = [
    {"name": "work", "cmd": None, "iterative": True},
]


class ManualLoopError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class ManualLoopService:
    def __init__(self, loop_store: LoopStore | None = None) -> None:
        self.loop_store = loop_store or LoopStore()
        self._owns_loop_store = loop_store is None
        register_stage_handler(MANUAL_LOOP_NAME, "work", self._handle_work)

    def close(self) -> None:
        if self._owns_loop_store:
            self.loop_store.close()

    def setup(self, project_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        if not project_id:
            raise ManualLoopError("project_id is required")
        payload = dict(config or {})
        payload.setdefault("cmd", "true")
        existing = self._definition_for(project_id)
        if existing is not None:
            return self.loop_store.set_definition_config(existing["id"], payload)
        return self.loop_store.create_definition(
            name=MANUAL_LOOP_NAME,
            project_id=project_id,
            stages=MANUAL_STAGES,
            on_failure=ON_FAILURE_AUTO_FIX,
            config=payload,
        )

    def status(self, project_id: str) -> dict[str, Any]:
        definition = self._definition_for(project_id)
        if definition is None:
            raise ManualLoopError(
                f"manual-loop is not set up for project {project_id}", 404
            )
        runs = self.loop_store.list_runs(definition["id"])
        return {"definition": definition, "last_run": runs[-1] if runs else None}

    def start(self, project_id: str, worktree_dir: str) -> dict[str, Any]:
        definition = self.setup(project_id, self._config_for(project_id))
        run = self.loop_store.start_run(definition["id"], worktree_dir=worktree_dir)
        if run["status"] == STATUS_RUNNING:
            return self.loop_store.pause_for_input(run["id"])
        return run

    def _config_for(self, project_id: str) -> dict[str, Any]:
        definition = self._definition_for(project_id)
        return dict(definition["config"] if definition else {})

    def _definition_for(self, project_id: str) -> dict[str, Any] | None:
        for definition in self.loop_store.list_definitions():
            if definition["project_id"] == project_id and definition["name"] == MANUAL_LOOP_NAME:
                return definition
        return None

    def _handle_work(
        self,
        run_id: str,
        stage: dict[str, Any],
        worktree_dir: str,
        definition: dict[str, Any],
    ) -> tuple[bool, str]:
        del run_id, stage
        import subprocess

        cmd = str((definition.get("config") or {}).get("cmd") or "true")
        result = subprocess.run(
            cmd,
            cwd=worktree_dir,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        return result.returncode == 0, output
