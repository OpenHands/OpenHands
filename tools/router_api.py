"""REST handlers for the model router.

Until these routes are mounted on the agent-server, run:

    python3 tools/router_api.py --host 127.0.0.1 --port 18011
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from benchmark_ingest import ENABLED_SOURCES, ingest_sources
from model_registry import ModelRegistry, RegistryError
from router import RouterError, RouterStore, default_db_path
from router_model_presets import (
    LocalRuntimeProbe,
    PresetError,
    local_runtimes_payload,
    resolve_preset,
)

JsonBody = dict[str, Any] | None
Handler = Callable[["RouterService", dict[str, str], JsonBody], tuple[int, Any]]

CONFIG_PATH = "/api/routing/config"
IMPORT_PATH = "/api/routing/import-project-config"
TAXONOMY_PATH = "/api/routing/taxonomy"
REGISTRY_PATH = "/api/routing/registry"
INGEST_PATH = "/api/routing/benchmarks/ingest"
SOURCES_PATH = "/api/routing/benchmarks/sources"
PRIVACY_REFRESH_PATH = "/api/routing/registry/privacy-refresh"
ROUTER_MODEL_PATH = "/api/routing/router-model"
LOCAL_RUNTIMES_PATH = "/api/routing/local-runtimes"
RESOLVE_PATH = "/api/routing/resolve"
AUDIT_PATH = "/api/routing/audit"


class RouterService:
    def __init__(
        self,
        store: RouterStore,
        *,
        probe: LocalRuntimeProbe | None = None,
        fetch: Callable[[str, float], str] | None = None,
    ) -> None:
        self.store = store
        self.registry = store.registry
        self.probe = probe or LocalRuntimeProbe()
        self.fetch = fetch

    def close(self) -> None:
        self.store.close()


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_config(service: RouterService, _params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    return 200, service.store.get_config()


def _put_config(service: RouterService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    payload = _json_body(body)
    project_yaml = payload.pop("project_yaml", None)
    return 200, service.store.put_config(payload, project_yaml=project_yaml)


def _import_project(service: RouterService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    path = str(_json_body(body).get("path") or "")
    if not path:
        raise RouterError("path is required")
    return 200, service.store.import_project_config(path)


def _get_taxonomy(service: RouterService, _params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    return 200, service.store.get_taxonomy()


def _put_taxonomy(service: RouterService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    payload = _json_body(body)
    if payload.get("reset"):
        return 200, service.store.reset_taxonomy()
    return 200, service.store.put_taxonomy(payload)


def _get_registry(service: RouterService, params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    connected = _csv(params.get("connected_providers"))
    snapshot = service.registry.snapshot()
    runtimes = service.probe.probe()
    models = []
    for item in snapshot["models"]:
        row = dict(item)
        if row.get("local"):
            runtime = row.get("runtime") or "ollama"
            info = runtimes.get(runtime) or {}
            row["reachable"] = bool(info.get("alive"))
        else:
            row["reachable"] = (not connected) or row["provider_key"] in connected
        models.append(row)
    snapshot["models"] = models
    snapshot["connected_providers"] = connected
    return 200, snapshot


def _ingest(service: RouterService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    payload = _json_body(body)
    sources = payload.get("sources")
    if sources is not None and not isinstance(sources, list):
        raise RouterError("sources must be a list")
    result = ingest_sources(
        service.registry,
        sources=sources,
        fetch=service.fetch,
    )
    return 200, result


def _get_sources(service: RouterService, _params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    status = dict(service.registry.sources)
    items = []
    for source_id in ENABLED_SOURCES:
        row = dict(status.get(source_id) or {})
        row["id"] = source_id
        items.append(row)
    return 200, {"sources": items}


def _privacy_refresh(service: RouterService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    return 200, service.registry.refresh_privacy(_json_body(body))


def _get_router_model(service: RouterService, params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    config = service.store.get_config()
    router_model = config.get("router_model") or {}
    preset = str(params.get("preset") or router_model.get("preset") or "cheapest")
    connected = _csv(params.get("connected_providers"))
    runtimes = service.probe.probe()
    resolved = resolve_preset(
        preset,
        service.registry,
        connected_providers=connected or None,
        runtimes=runtimes,
        custom=router_model,
    )
    return 200, {"config": router_model, "resolved": resolved, "runtimes": runtimes}


def _put_router_model(service: RouterService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    payload = _json_body(body)
    updated = service.store.put_config({"router_model": payload})
    runtimes = service.probe.probe()
    router_model = updated.get("router_model") or {}
    resolved = resolve_preset(
        str(router_model.get("preset") or "cheapest"),
        service.registry,
        runtimes=runtimes,
        custom=router_model,
    )
    return 200, {"config": router_model, "resolved": resolved}


def _local_runtimes(service: RouterService, _params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    return 200, local_runtimes_payload(service.registry, service.probe)


def _resolve(service: RouterService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    payload = _json_body(body)
    if "local_runtimes" not in payload:
        payload["local_runtimes"] = service.probe.probe()
    return 200, service.store.resolve(payload)


def _audit(service: RouterService, params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    return 200, service.store.list_audit(
        card_id=params.get("card_id") or None,
        run_id=params.get("run_id") or None,
        kind=params.get("kind") or None,
        limit=int(params.get("limit") or 50),
        offset=int(params.get("offset") or 0),
    )


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{CONFIG_PATH}$"), _get_config),
    ("PUT", re.compile(rf"^{CONFIG_PATH}$"), _put_config),
    ("POST", re.compile(rf"^{IMPORT_PATH}$"), _import_project),
    ("GET", re.compile(rf"^{TAXONOMY_PATH}$"), _get_taxonomy),
    ("PUT", re.compile(rf"^{TAXONOMY_PATH}$"), _put_taxonomy),
    ("GET", re.compile(rf"^{REGISTRY_PATH}$"), _get_registry),
    ("POST", re.compile(rf"^{INGEST_PATH}$"), _ingest),
    ("GET", re.compile(rf"^{SOURCES_PATH}$"), _get_sources),
    ("POST", re.compile(rf"^{PRIVACY_REFRESH_PATH}$"), _privacy_refresh),
    ("GET", re.compile(rf"^{ROUTER_MODEL_PATH}$"), _get_router_model),
    ("PUT", re.compile(rf"^{ROUTER_MODEL_PATH}$"), _put_router_model),
    ("GET", re.compile(rf"^{LOCAL_RUNTIMES_PATH}$"), _local_runtimes),
    ("POST", re.compile(rf"^{RESOLVE_PATH}$"), _resolve),
    ("GET", re.compile(rf"^{AUDIT_PATH}$"), _audit),
)


def handle_request(
    service: RouterService,
    method: str,
    path: str,
    body: JsonBody = None,
) -> tuple[int, Any]:
    parsed = urlparse(path)
    pathname = parsed.path
    params = _query(path)
    try:
        for route_method, pattern, handler in ROUTES:
            if route_method != method:
                continue
            match = pattern.match(pathname)
            if match is None:
                continue
            params.update(match.groupdict())
            return handler(service, params, body)
        return 404, {"error": f"No route for {method} {pathname}"}
    except RouterError as exc:
        return exc.status, {"error": str(exc), **exc.payload}
    except (RegistryError, PresetError, TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class RouterRequestHandler(BaseHTTPRequestHandler):
    server: "RouterHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        payload: JsonBody = json.loads(raw) if raw else None
        status, data = handle_request(
            self.server.service, self.command, self.path, payload
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


class RouterHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: RouterService,
    ) -> None:
        super().__init__(server_address, RouterRequestHandler)
        self.service = service


def serve_router(
    host: str,
    port: int,
    service: RouterService | None = None,
) -> RouterHTTPServer:
    if service is None:
        store = RouterStore(default_db_path(), registry=ModelRegistry())
        service = RouterService(store)
    return RouterHTTPServer((host, port), service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-router HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18011)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    store = RouterStore(args.db or default_db_path(), registry=ModelRegistry())
    service = RouterService(store)
    server = serve_router(args.host, args.port, service=service)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
