import os
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from server.auth.auth_error import NoCredentialsError
from server.logger import logger

from openhands.app_server.user_auth import get_access_token
from openhands.app_server.utils.http_session import httpx_verify_option

AGENT_CANVAS_INTERNAL_URL_ENV = "AGENT_CANVAS_INTERNAL_URL"
AGENT_CANVAS_PATH_PREFIX = "/canvas"
AGENT_CANVAS_PROXY_TIMEOUT_SECONDS = 30.0
HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _agent_canvas_internal_url() -> str | None:
    value = os.getenv(AGENT_CANVAS_INTERNAL_URL_ENV, "").strip().rstrip("/")
    return value or None


def _return_to_path(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    return path


def _login_redirect(request: Request) -> RedirectResponse:
    return_to = quote(_return_to_path(request), safe="")
    return RedirectResponse(f"/login?returnTo={return_to}", status_code=302)


async def _is_authenticated(request: Request) -> bool:
    try:
        token = await get_access_token(request)
    except NoCredentialsError:
        return False
    return token is not None


def _proxy_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_RESPONSE_HEADERS
    }


async def _proxy_agent_canvas(request: Request, path: str) -> Response:
    target = _agent_canvas_internal_url()
    if not target:
        logger.warning("agent_canvas_proxy:not_configured")
        return Response("Agent Canvas proxy is not configured", status_code=404)

    upstream_path = f"{AGENT_CANVAS_PATH_PREFIX}{path}"
    if request.url.query:
        upstream_path = f"{upstream_path}?{request.url.query}"
    upstream_url = f"{target}{upstream_path}"

    async with httpx.AsyncClient(
        verify=httpx_verify_option(), timeout=AGENT_CANVAS_PROXY_TIMEOUT_SECONDS
    ) as client:
        upstream_response = await client.request(
            request.method,
            upstream_url,
            headers={
                key: value
                for key, value in request.headers.items()
                if key.lower()
                in {"accept", "if-modified-since", "if-none-match", "range"}
            },
        )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_proxy_response_headers(upstream_response),
    )


def add_agent_canvas_proxy_routes(app: FastAPI) -> None:
    if not _agent_canvas_internal_url():
        return

    @app.api_route(AGENT_CANVAS_PATH_PREFIX, methods=["GET", "HEAD"])
    async def agent_canvas_root(request: Request):
        if not await _is_authenticated(request):
            return _login_redirect(request)
        return await _proxy_agent_canvas(request, "")

    @app.api_route(f"{AGENT_CANVAS_PATH_PREFIX}/{{path:path}}", methods=["GET", "HEAD"])
    async def agent_canvas_path(request: Request, path: str):
        if not await _is_authenticated(request):
            return _login_redirect(request)
        return await _proxy_agent_canvas(request, f"/{path}")
