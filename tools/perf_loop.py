"""Performance loop: run a benchmark command and fail on metric rules."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from loop_runner import (
    LoopStore,
    ON_FAILURE_STOP,
    register_stage_handler,
)

PERF_LOOP_NAME = "perf-loop"
PERF_STAGES: list[dict[str, Any]] = [
    {"name": "bench", "cmd": None, "iterative": False},
    {"name": "detect", "cmd": None, "iterative": False},
]
DEFAULT_PERF_RULES: list[dict[str, Any]] = [
    {"pattern": r"took ([0-9.]+)ms", "threshold": 500, "metric": "duration"},
    {"pattern": r"ERROR", "threshold": 0, "metric": "count"},
]


class PerfLoopError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def detect_metrics(output: str, rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detections: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rule in rules:
        metric = str(rule.get("metric") or "count")
        threshold = float(rule.get("threshold") or 0)
        compiled = re.compile(str(rule.get("pattern") or ""))
        matches = list(compiled.finditer(output or ""))
        if metric == "duration":
            values: list[float] = []
            for match in matches:
                raw = match.group(1) if match.lastindex else match.group(0)
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
            value = max(values) if values else 0.0
        else:
            value = float(len(matches))
        detection = {
            "pattern": rule.get("pattern"),
            "metric": metric,
            "threshold": threshold,
            "value": value,
        }
        detections.append(detection)
        if value > threshold:
            failures.append(detection)
    return detections, failures


class PerfLoopService:
    def __init__(self, loop_store: LoopStore | None = None) -> None:
        self.loop_store = loop_store or LoopStore()
        self._owns_loop_store = loop_store is None
        register_stage_handler(PERF_LOOP_NAME, "bench", self._handle_bench)
        register_stage_handler(PERF_LOOP_NAME, "detect", self._handle_detect)

    def close(self) -> None:
        if self._owns_loop_store:
            self.loop_store.close()

    def setup(self, project_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        if not project_id:
            raise PerfLoopError("project_id is required")
        payload = dict(config or {})
        payload.setdefault("cmd", "true")
        payload.setdefault("rules", list(DEFAULT_PERF_RULES))
        existing = self._definition_for(project_id)
        if existing is not None:
            return self.loop_store.set_definition_config(existing["id"], payload)
        return self.loop_store.create_definition(
            name=PERF_LOOP_NAME,
            project_id=project_id,
            stages=PERF_STAGES,
            max_iterations=1,
            on_failure=ON_FAILURE_STOP,
            config=payload,
        )

    def status(self, project_id: str) -> dict[str, Any]:
        definition = self._definition_for(project_id)
        if definition is None:
            raise PerfLoopError(f"perf-loop is not set up for project {project_id}", 404)
        runs = self.loop_store.list_runs(definition["id"])
        return {"definition": definition, "last_run": runs[-1] if runs else None}

    def _definition_for(self, project_id: str) -> dict[str, Any] | None:
        for definition in self.loop_store.list_definitions():
            if definition["project_id"] == project_id and definition["name"] == PERF_LOOP_NAME:
                return definition
        return None

    def _handle_bench(
        self,
        run_id: str,
        stage: dict[str, Any],
        worktree_dir: str,
        definition: dict[str, Any],
    ) -> tuple[bool, str]:
        del run_id, stage
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

    def _handle_detect(
        self,
        run_id: str,
        stage: dict[str, Any],
        worktree_dir: str,
        definition: dict[str, Any],
    ) -> tuple[bool, str]:
        del stage, worktree_dir
        run = self.loop_store.get_run(run_id)
        bench = next(
            (item for item in run["stages"] if item["stage_name"] == "bench"),
            None,
        )
        output = (bench or {}).get("last_output") or ""
        rules = list((definition.get("config") or {}).get("rules") or DEFAULT_PERF_RULES)
        detections, failures = detect_metrics(output, rules)
        payload = json.dumps({"detections": detections, "failures": failures})
        return (not failures), payload
