from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from server.auth.auth_error import NoCredentialsError
from server.routes.agent_canvas import add_agent_canvas_proxy_routes


async def _authenticated(_request):
    return SecretStr("access-token")


async def _unauthenticated(_request):
    raise NoCredentialsError("missing credentials")


async def _unexpected_auth(_request):
    raise AssertionError("static assets should not authenticate")


def test_canvas_redirects_unauthenticated_users_to_login(monkeypatch):
    monkeypatch.setenv("AGENT_CANVAS_INTERNAL_URL", "http://agent-canvas:8000")
    app = FastAPI()
    add_agent_canvas_proxy_routes(app)
    client = TestClient(app)

    with patch("server.routes.agent_canvas.get_access_token", _unauthenticated):
        response = client.get("/canvas/settings?tab=llm", follow_redirects=False)

    assert response.status_code == 302
    assert (
        response.headers["location"]
        == "/login?returnTo=%2Fcanvas%2Fsettings%3Ftab%3Dllm"
    )


def test_canvas_proxies_authenticated_requests(monkeypatch):
    monkeypatch.setenv("AGENT_CANVAS_INTERNAL_URL", "http://agent-canvas:8000")
    calls = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, headers=None):
            calls.append({"method": method, "url": url, "headers": headers})
            return httpx.Response(
                200,
                content=b"<html>canvas</html>",
                headers={"content-type": "text/html", "content-length": "19"},
            )

    app = FastAPI()
    add_agent_canvas_proxy_routes(app)
    client = TestClient(app)

    with (
        patch("server.routes.agent_canvas.get_access_token", _authenticated),
        patch("server.routes.agent_canvas.httpx.AsyncClient", FakeAsyncClient),
    ):
        response = client.get("/canvas/settings?tab=llm")

    assert response.status_code == 200
    assert response.text == "<html>canvas</html>"
    assert calls == [
        {
            "method": "GET",
            "url": "http://agent-canvas:8000/canvas/settings?tab=llm",
            "headers": {"accept": "*/*"},
        }
    ]


def test_canvas_proxies_static_assets_without_authentication(monkeypatch):
    monkeypatch.setenv("AGENT_CANVAS_INTERNAL_URL", "http://agent-canvas:8000")
    calls = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, headers=None):
            calls.append({"method": method, "url": url, "headers": headers})
            return httpx.Response(
                200,
                content=b"console.log('canvas')",
                headers={"content-type": "application/javascript"},
            )

    app = FastAPI()
    add_agent_canvas_proxy_routes(app)
    client = TestClient(app)

    with (
        patch("server.routes.agent_canvas.get_access_token", _unexpected_auth),
        patch("server.routes.agent_canvas.httpx.AsyncClient", FakeAsyncClient),
    ):
        response = client.get("/canvas/assets/app.js?v=1")

    assert response.status_code == 200
    assert response.text == "console.log('canvas')"
    assert calls == [
        {
            "method": "GET",
            "url": "http://agent-canvas:8000/canvas/assets/app.js?v=1",
            "headers": {"accept": "*/*"},
        }
    ]
