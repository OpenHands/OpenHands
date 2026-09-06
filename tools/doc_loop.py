"""Doc generation loop with optional Notion/AppFlowy sync."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Callable
from urllib.request import Request, urlopen

from loop_runner import (
    LoopStore,
    ON_FAILURE_STOP,
    STATUS_SKIPPED,
    register_stage_handler,
)

DOC_LOOP_NAME = "doc-loop"
DOC_STAGES: list[dict[str, Any]] = [
    {"name": "generator", "cmd": None, "iterative": False},
    {"name": "docsync", "cmd": None, "iterative": False},
]
NOTION_SECRET = "NOTION_API_KEY"
APPFLOWY_SECRET = "APPFLOWY_TOKEN"
ADAPTER_NOTION = "notion"
ADAPTER_APPFLOWY = "appflowy"
ADAPTER_ENDPOINTS = {
    ADAPTER_NOTION: "/v1/pages",
    ADAPTER_APPFLOWY: "/api/docs",
}
SecretsGet = Callable[[str], str | None]
HttpPost = Callable[[str, dict[str, Any], dict[str, str]], tuple[int, str]]


class DocLoopError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def default_secrets_get(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def default_http_post(
    url: str, body: dict[str, Any], headers: dict[str, str]
) -> tuple[int, str]:
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return response.status, response.read().decode("utf-8")


class DocLoopService:
    def __init__(
        self,
        loop_store: LoopStore | None = None,
        secrets_get: SecretsGet | None = None,
        http_post: HttpPost | None = None,
    ) -> None:
        self.loop_store = loop_store or LoopStore()
        self.secrets_get = secrets_get or default_secrets_get
        self.http_post = http_post or default_http_post
        self._owns_loop_store = loop_store is None
        register_stage_handler(DOC_LOOP_NAME, "generator", self._handle_generator)
        register_stage_handler(DOC_LOOP_NAME, "docsync", self._handle_docsync)

    def close(self) -> None:
        if self._owns_loop_store:
            self.loop_store.close()

    def setup(self, project_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        if not project_id:
            raise DocLoopError("project_id is required")
        payload = dict(config or {})
        payload.setdefault("gen_cmd", "true")
        payload.setdefault("adapter", ADAPTER_NOTION)
        existing = self._definition_for(project_id)
        if existing is not None:
            return self.loop_store.set_definition_config(existing["id"], payload)
        return self.loop_store.create_definition(
            name=DOC_LOOP_NAME,
            project_id=project_id,
            stages=DOC_STAGES,
            max_iterations=1,
            on_failure=ON_FAILURE_STOP,
            config=payload,
        )

    def status(self, project_id: str) -> dict[str, Any]:
        definition = self._definition_for(project_id)
        if definition is None:
            raise DocLoopError(f"doc-loop is not set up for project {project_id}", 404)
        runs = self.loop_store.list_runs(definition["id"])
        return {"definition": definition, "last_run": runs[-1] if runs else None}

    def _definition_for(self, project_id: str) -> dict[str, Any] | None:
        for definition in self.loop_store.list_definitions():
            if definition["project_id"] == project_id and definition["name"] == DOC_LOOP_NAME:
                return definition
        return None

    def _handle_generator(
        self,
        run_id: str,
        stage: dict[str, Any],
        worktree_dir: str,
        definition: dict[str, Any],
    ) -> tuple[bool, str]:
        del run_id, stage
        cmd = str((definition.get("config") or {}).get("gen_cmd") or "true")
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

    def _handle_docsync(
        self,
        run_id: str,
        stage: dict[str, Any],
        worktree_dir: str,
        definition: dict[str, Any],
    ) -> tuple[Any, str]:
        del stage, worktree_dir
        config = definition.get("config") or {}
        adapter = str(config.get("adapter") or ADAPTER_NOTION)
        secret_name = APPFLOWY_SECRET if adapter == ADAPTER_APPFLOWY else NOTION_SECRET
        token = self.secrets_get(secret_name)
        if not token:
            return STATUS_SKIPPED, f"skipped: missing {secret_name}"
        run = self.loop_store.get_run(run_id)
        generated = next(
            (item for item in run["stages"] if item["stage_name"] == "generator"),
            None,
        )
        docs = (generated or {}).get("last_output") or ""
        base_url = str(config.get("base_url") or "").rstrip("/")
        path = str(config.get("endpoint") or ADAPTER_ENDPOINTS.get(adapter) or "")
        url = f"{base_url}{path}" if base_url else path
        status, body = self.http_post(
            url,
            {"adapter": adapter, "content": docs},
            {"Authorization": f"Bearer {token}"},
        )
        return status < 400, body
