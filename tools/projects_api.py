"""REST handlers for the local project / worktree store.

The dispatcher is stdlib-only so unit tests and a tiny HTTP server can
share one implementation. Until these routes are mounted on the
agent-server, run:

    python3 tools/projects_api.py --host 127.0.0.1 --port 18004
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from projects import (
    ProjectError,
    ProjectStore,
    default_db_path,
    default_projects_root,
)

JsonBody = dict[str, Any] | None
Handler = Callable[[ProjectStore, dict[str, str], JsonBody], tuple[int, Any]]

PROJECTS_PATH = "/api/projects"
PROJECT_PATH_RE = re.compile(r"^/api/projects/(?P<project_id>[^/]+)$")
PROJECT_WORKTREES_PATH_RE = re.compile(
    r"^/api/projects/(?P<project_id>[^/]+)/worktrees$"
)
WORKTREE_PATH_RE = re.compile(
    r"^/api/projects/(?P<project_id>[^/]+)/worktrees/(?P<worktree_id>[^/]+)$"
)
WORKTREE_ASSIGN_PATH_RE = re.compile(
    r"^/api/projects/(?P<project_id>[^/]+)/worktrees/(?P<worktree_id>[^/]+)/assign$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _list_projects(
    store: ProjectStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_projects()


def _create_project(
    store: ProjectStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    project = store.create_project(
        name=str(payload.get("name") or ""),
        description=payload.get("description"),
        repo_url=payload.get("repo_url"),
        local_path=payload.get("local_path"),
        default_branch=payload.get("default_branch"),
        default_agent_profile=payload.get("default_agent_profile"),
        kanban_board_id=payload.get("kanban_board_id"),
        cost_cap=payload.get("cost_cap"),
        status=payload.get("status"),
    )
    return 201, project


def _get_project(
    store: ProjectStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.get_project(params["project_id"])


def _update_project(
    store: ProjectStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    return 200, store.update_project(params["project_id"], **_json_body(body))


def _delete_project(
    store: ProjectStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    store.delete_project(params["project_id"])
    return 204, None


def _list_worktrees(
    store: ProjectStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_worktrees(params["project_id"])


def _create_worktree(
    store: ProjectStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    worktree = store.create_worktree(
        params["project_id"],
        branch_name=str(payload.get("branch_name") or ""),
        status=payload.get("status"),
    )
    return 201, worktree


def _delete_worktree(
    store: ProjectStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    store.remove_worktree(params["project_id"], params["worktree_id"])
    return 204, None


def _assign_worktree(
    store: ProjectStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    return 200, store.assign_worktree(
        params["project_id"],
        params["worktree_id"],
        str(payload.get("agent_session_id") or ""),
    )


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{PROJECTS_PATH}$"), _list_projects),
    ("POST", re.compile(rf"^{PROJECTS_PATH}$"), _create_project),
    ("GET", PROJECT_WORKTREES_PATH_RE, _list_worktrees),
    ("POST", PROJECT_WORKTREES_PATH_RE, _create_worktree),
    ("POST", WORKTREE_ASSIGN_PATH_RE, _assign_worktree),
    ("DELETE", WORKTREE_PATH_RE, _delete_worktree),
    ("GET", PROJECT_PATH_RE, _get_project),
    ("PATCH", PROJECT_PATH_RE, _update_project),
    ("DELETE", PROJECT_PATH_RE, _delete_project),
)


def handle_request(
    store: ProjectStore,
    method: str,
    path: str,
    body: JsonBody = None,
) -> tuple[int, Any]:
    parsed = urlparse(path)
    pathname = parsed.path
    try:
        for route_method, pattern, handler in ROUTES:
            if route_method != method:
                continue
            match = pattern.match(pathname)
            if match is None:
                continue
            return handler(store, match.groupdict(), body)
        return 404, {"error": f"No route for {method} {pathname}"}
    except ProjectError as exc:
        return exc.status, {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class ProjectRequestHandler(BaseHTTPRequestHandler):
    server: "ProjectHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        payload: JsonBody = json.loads(raw) if raw else None
        status, data = handle_request(
            self.server.store, self.command, self.path, payload
        )
        body = b"" if data is None else json.dumps(data).encode()
        self.send_response(status)
        if body:
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
        else:
            self.send_header("Content-Length", "0")
        self.end_headers()
        if body:
            self.wfile.write(body)


class ProjectHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: ProjectStore,
    ) -> None:
        super().__init__(server_address, ProjectRequestHandler)
        self.store = store


def serve_projects(
    host: str,
    port: int,
    store: ProjectStore | None = None,
    db_path: str | None = None,
    projects_root: str | None = None,
) -> ProjectHTTPServer:
    if store is None:
        store = ProjectStore(
            db_path or default_db_path(),
            projects_root=projects_root or default_projects_root(),
        )
    return ProjectHTTPServer((host, port), store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local projects HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18004)
    parser.add_argument("--db", default=None)
    parser.add_argument("--projects-root", default=None)
    args = parser.parse_args()
    server = serve_projects(
        args.host,
        args.port,
        db_path=args.db,
        projects_root=args.projects_root,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
