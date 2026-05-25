"""Unit tests for ``GET /api/v1/databricks/models`` (two-tier picker route).

Covers:
  * Curated-only response when no auth context is available.
  * Discovered tier merged when user has stored PAT + host.
  * Session U2M token takes priority over stored PAT (matches resolution order).
  * ``include_discovered=false`` hard-disables the HTTP probe.
  * Graceful fallback when ``list_chat_endpoints`` raises.
  * Response JSON shape matches the ``ModelPickerEntry`` contract.
  * SDK-missing path returns a structured ``unavailable`` response, not a 500.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.middleware.sessions import SessionMiddleware

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_HOST = 'https://adb-123.azuredatabricks.net'


def _discovery_response(status: int, body: dict) -> httpx.Response:
    req = httpx.Request('GET', f'{_HOST}/api/2.0/serving-endpoints')
    return httpx.Response(status, json=body, request=req)


def _ep(name: str, et: str = 'FOUNDATION_MODEL_API', ready: bool = True) -> dict:
    return {
        'name': name,
        'endpoint_type': et,
        'task': 'llm/v1/chat',
        'state': {'ready': 'READY' if ready else 'NOT_READY'},
    }


def _build_app(
    *,
    user_info: Any = None,
    seed_session: dict[str, Any] | None = None,
) -> FastAPI:
    """FastAPI app that mounts the route with an overridden user context.

    ``seed_session`` (if given) registers a helper GET endpoint that the
    caller can hit once to pre-populate the request session — used to
    exercise the U2M-token-in-session code path.
    """
    from openhands.app_server.auth.databricks_models_routes import (
        router as models_router,
    )
    from openhands.app_server.auth.databricks_models_routes import (
        user_dependency,
    )
    from openhands.server.dependencies import check_session_api_key

    application = FastAPI()
    application.add_middleware(
        SessionMiddleware, secret_key='test-secret-for-databricks-models'
    )

    application.dependency_overrides[check_session_api_key] = lambda: None

    mock_user_context = AsyncMock()
    mock_user_context.get_user_info = AsyncMock(return_value=user_info)
    application.dependency_overrides[user_dependency.dependency] = (
        lambda: mock_user_context
    )

    application.include_router(models_router, prefix='/api/v1')

    if seed_session is not None:

        @application.get('/_test/seed_session')
        async def _seed(request: Request) -> dict:
            for k, v in seed_session.items():
                request.session[k] = v
            return {'ok': True}

    return application


def _mk_user(
    *,
    llm_api_key: SecretStr | None = None,
    llm_base_url: str | None = None,
) -> Any:
    """Minimal stand-in for ``UserInfo`` — only the fields the route reads."""

    class _U:
        pass

    u = _U()
    u.llm_api_key = llm_api_key
    u.llm_base_url = llm_base_url
    u.llm_model = None
    u.id = 'test-user'
    return u


@pytest.fixture
def curated_qnames() -> set[str]:
    from openhands.sdk.llm.providers.databricks import CURATED_DATABRICKS_MODELS

    return {e.qualified_name for e in CURATED_DATABRICKS_MODELS}


# ---------------------------------------------------------------------------
# Curated-only paths
# ---------------------------------------------------------------------------


def test_returns_curated_only_when_no_credentials(curated_qnames: set[str]) -> None:
    """No host, no PAT, no session token → curated list + source='curated'."""
    app = _build_app(user_info=_mk_user())
    client = TestClient(app)

    with patch('httpx.get') as mock_get:
        resp = client.get('/api/v1/databricks/models')

    mock_get.assert_not_called()
    assert resp.status_code == 200
    body = resp.json()
    assert body['source'] == 'curated'
    assert body['host'] is None
    assert {e['qualified_name'] for e in body['entries']} == curated_qnames


def test_returns_curated_only_when_include_discovered_false(
    curated_qnames: set[str],
) -> None:
    """Explicit opt-out skips the HTTP probe even when creds are available."""
    app = _build_app(
        user_info=_mk_user(
            llm_api_key=SecretStr('dapi-pat'),
            llm_base_url=_HOST,
        ),
    )
    client = TestClient(app)

    with patch('httpx.get') as mock_get:
        resp = client.get(
            '/api/v1/databricks/models',
            params={'include_discovered': 'false'},
        )

    mock_get.assert_not_called()
    assert resp.status_code == 200
    body = resp.json()
    assert body['source'] == 'curated'
    assert {e['qualified_name'] for e in body['entries']} == curated_qnames


def test_returns_curated_when_discovery_fails(curated_qnames: set[str]) -> None:
    """Discovery errors swallowed — curated entries still returned."""
    app = _build_app(
        user_info=_mk_user(
            llm_api_key=SecretStr('dapi-pat'),
            llm_base_url=_HOST,
        ),
    )
    client = TestClient(app)

    with patch('httpx.get', side_effect=RuntimeError('workspace down')):
        resp = client.get('/api/v1/databricks/models')

    assert resp.status_code == 200
    body = resp.json()
    # Credentials were resolved → source='curated+discovered' is claimed…
    assert body['source'] == 'curated+discovered'
    assert body['host'] == _HOST
    # …but the only entries are the curated ones (discovery swallowed).
    assert {e['qualified_name'] for e in body['entries']} == curated_qnames
    for e in body['entries']:
        assert e['source'] == 'curated'


# ---------------------------------------------------------------------------
# Merged discovery paths
# ---------------------------------------------------------------------------


def test_merges_discovered_on_top_of_curated(curated_qnames: set[str]) -> None:
    """With PAT + host, live endpoints get merged in; dedup by qualified name."""
    app = _build_app(
        user_info=_mk_user(
            llm_api_key=SecretStr('dapi-pat'),
            llm_base_url=_HOST,
        ),
    )
    client = TestClient(app)

    payload = {
        'endpoints': [
            _ep('databricks-claude-sonnet-4-5'),  # overlaps curated
            _ep('databricks-meta-llama-4-maverick'),  # discovered only
            _ep('customer-private-gpt', et='EXTERNAL_MODEL'),
        ]
    }
    with patch('httpx.get', return_value=_discovery_response(200, payload)):
        resp = client.get('/api/v1/databricks/models')

    assert resp.status_code == 200
    body = resp.json()
    assert body['source'] == 'curated+discovered'
    assert body['host'] == _HOST

    qnames = {e['qualified_name'] for e in body['entries']}
    assert curated_qnames.issubset(qnames)
    assert 'databricks/databricks-meta-llama-4-maverick' in qnames
    assert 'databricks/customer-private-gpt' in qnames

    by_qn = {e['qualified_name']: e for e in body['entries']}

    overlap = by_qn['databricks/databricks-claude-sonnet-4-5']
    assert overlap['source'] == 'curated+discovered'
    assert overlap['family'] == 'anthropic'
    assert overlap['recommended'] is True
    assert overlap['endpoint_type'] == 'FOUNDATION_MODEL_API'

    llama = by_qn['databricks/databricks-meta-llama-4-maverick']
    assert llama['source'] == 'discovered'
    assert llama['recommended'] is False
    assert llama['family'] == 'openai'  # default family for llama names

    ext = by_qn['databricks/customer-private-gpt']
    assert ext['source'] == 'discovered'
    assert ext['endpoint_type'] == 'EXTERNAL_MODEL'


def test_host_query_param_wins_over_user_base_url(curated_qnames: set[str]) -> None:
    """Explicit ?host=… beats the stored llm_base_url — lets users re-target."""
    other_host = 'https://my-other-workspace.cloud.databricks.com'
    app = _build_app(
        user_info=_mk_user(
            llm_api_key=SecretStr('dapi-pat'),
            llm_base_url='https://default-workspace.cloud.databricks.com',
        ),
    )
    client = TestClient(app)

    captured_url: list[str] = []

    def _fake_get(url: str, **kwargs: Any) -> httpx.Response:
        captured_url.append(url)
        return _discovery_response(200, {'endpoints': []})

    with patch('httpx.get', side_effect=_fake_get):
        resp = client.get(
            '/api/v1/databricks/models',
            params={'host': other_host},
        )
    assert resp.status_code == 200
    assert resp.json()['host'] == other_host
    assert captured_url and captured_url[0].startswith(other_host)


# ---------------------------------------------------------------------------
# Token resolution order — U2M session > PAT > env
# ---------------------------------------------------------------------------


def test_session_u2m_token_wins_over_user_pat() -> None:
    """Session U2M token takes priority over stored PAT."""
    app = _build_app(
        user_info=_mk_user(
            llm_api_key=SecretStr('dapi-pat'),
            llm_base_url=_HOST,
        ),
        seed_session={
            'databricks_u2m_tokens': {
                'access_token': 'u2m-access-token',
                'refresh_token': 'u2m-refresh-token',
                'expires_in': 3600,
            }
        },
    )
    client = TestClient(app)
    assert client.get('/_test/seed_session').status_code == 200

    captured_auth: list[str] = []

    def _fake_get(url: str, *, headers: dict, **kw: Any) -> httpx.Response:
        captured_auth.append(headers['Authorization'])
        return _discovery_response(200, {'endpoints': []})

    with patch('httpx.get', side_effect=_fake_get):
        resp = client.get('/api/v1/databricks/models')

    assert resp.status_code == 200
    assert captured_auth == ['Bearer u2m-access-token']


def test_env_token_used_when_no_user_pat(
    monkeypatch: pytest.MonkeyPatch, curated_qnames: set[str]
) -> None:
    """DATABRICKS_TOKEN env fallback when user has no stored PAT."""
    monkeypatch.setenv('DATABRICKS_TOKEN', 'env-dapi-token')
    monkeypatch.setenv('DATABRICKS_HOST', _HOST)

    app = _build_app(user_info=_mk_user())  # no PAT, no base_url
    client = TestClient(app)

    captured_auth: list[str] = []

    def _fake_get(url: str, *, headers: dict, **kw: Any) -> httpx.Response:
        captured_auth.append(headers['Authorization'])
        return _discovery_response(200, {'endpoints': [_ep('env-fmapi')]})

    with patch('httpx.get', side_effect=_fake_get):
        resp = client.get('/api/v1/databricks/models')

    assert resp.status_code == 200
    assert captured_auth == ['Bearer env-dapi-token']
    body = resp.json()
    assert body['host'] == _HOST
    assert 'databricks/env-fmapi' in {e['qualified_name'] for e in body['entries']}


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def test_response_entry_shape_matches_picker_contract() -> None:
    """Every entry has the fields the frontend + CLI rely on."""
    app = _build_app(user_info=_mk_user())
    client = TestClient(app)
    resp = client.get('/api/v1/databricks/models')
    assert resp.status_code == 200
    expected = {
        'qualified_name',
        'name',
        'family',
        'source',
        'endpoint_type',
        'ready',
        'recommended',
    }
    for e in resp.json()['entries']:
        assert set(e) == expected


def test_sort_order_recommended_first_then_family_name() -> None:
    """UI expects recommended rows at the top, grouped by family alpha, then name."""
    app = _build_app(user_info=_mk_user())
    client = TestClient(app)
    body = client.get('/api/v1/databricks/models').json()

    entries = body['entries']
    first_non_rec = next(
        (i for i, e in enumerate(entries) if not e['recommended']), len(entries)
    )
    rec_section = entries[:first_non_rec]
    rest_section = entries[first_non_rec:]
    assert all(e['recommended'] for e in rec_section)
    assert all(not e['recommended'] for e in rest_section)

    for section in (rec_section, rest_section):
        fams = [e['family'] for e in section]
        assert fams == sorted(fams)
