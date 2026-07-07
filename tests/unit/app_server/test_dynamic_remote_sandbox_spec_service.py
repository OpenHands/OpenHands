"""Tests for DynamicRemoteSandboxSpecService and its injector."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.datastructures import State

from openhands.app_server.sandbox.dynamic_remote_sandbox_spec_service import (
    DynamicRemoteSandboxSpecService,
    DynamicRemoteSandboxSpecServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIGS_RESPONSE = {
    'configs': [
        {
            'name': 'v1_current',
            'image': 'ghcr.io/openhands/agent-server:1.0.0',
            'command': ['--port', '8000'],
            'environment': {'FOO': 'bar'},
            'working_dir': '/workspace',
        },
        {
            'name': 'v1_legacy',
            'image': 'ghcr.io/openhands/agent-server:0.9.0',
            'command': ['--port', '9000'],
            'environment': {},
            'working_dir': '/home/user',
        },
        {
            'name': 'nightly',
            'image': 'ghcr.io/openhands/agent-server:nightly',
            'command': None,
            'environment': {'DEBUG': '1'},
            'working_dir': '/workspace',
        },
    ]
}


def _make_http_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response with the given JSON payload."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    # raise_for_status raises only for 4xx/5xx
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f'HTTP {status_code}',
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_service(
    *,
    api_url: str = 'https://runtime-api.example.com',
    api_key: str = 'test-key',
    default_spec_name: str = '',
    cache_ttl_seconds: int = 60,
) -> DynamicRemoteSandboxSpecService:
    return DynamicRemoteSandboxSpecService(
        api_url=api_url,
        api_key=api_key,
        default_spec_name=default_spec_name,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def _make_async_client_mock(response: MagicMock) -> MagicMock:
    """Return a mock that behaves like `async with httpx.AsyncClient() as client`."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, client


# ---------------------------------------------------------------------------
# _fetch_specs
# ---------------------------------------------------------------------------


class TestFetchSpecs:
    async def test_calls_correct_url_and_headers(self):
        """_fetch_specs must call /api/warm-runtime-configs with X-API-Key header."""
        ctx, client = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service(
            api_url='https://runtime-api.example.com', api_key='secret-key'
        )

        with patch('httpx.AsyncClient', return_value=ctx):
            await service._fetch_specs()

        client.get.assert_called_once_with(
            'https://runtime-api.example.com/api/warm-runtime-configs',
            headers={'X-API-Key': 'secret-key'},
            timeout=10.0,
        )

    async def test_maps_fields_to_sandbox_spec_info(self):
        """Each config item must be mapped to a SandboxSpecInfo correctly."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            specs = await service._fetch_specs()

        assert len(specs) == 3
        first = specs[0]
        assert first.id == 'ghcr.io/openhands/agent-server:1.0.0'
        assert first.command == ['--port', '8000']
        assert first.initial_env == {'FOO': 'bar'}
        assert first.working_dir == '/workspace'

    async def test_populates_name_to_spec_mapping(self):
        """The name→spec dict must be keyed by the config 'name', not the image URL."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            await service._fetch_specs()

        assert set(service._name_to_spec) == {'v1_current', 'v1_legacy', 'nightly'}
        assert (
            service._name_to_spec['nightly'].id
            == 'ghcr.io/openhands/agent-server:nightly'
        )

    async def test_caches_results_within_ttl(self):
        """A second call within the TTL must not issue another HTTP request."""
        ctx, client = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service(cache_ttl_seconds=60)

        with patch('httpx.AsyncClient', return_value=ctx):
            await service._fetch_specs()
            await service._fetch_specs()

        assert client.get.call_count == 1

    async def test_refreshes_after_ttl_expires(self):
        """A call after the TTL must re-fetch from runtime-api."""
        ctx, client = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service(cache_ttl_seconds=60)

        # Pre-populate cache with an already-expired timestamp
        service._cached_specs = [
            SandboxSpecInfo(id='old-image:1', command=None, working_dir='/old')
        ]
        service._cache_expires_at = 0.0  # expired

        with patch('httpx.AsyncClient', return_value=ctx):
            specs = await service._fetch_specs()

        assert client.get.call_count == 1
        assert len(specs) == 3  # fresh data, not the single stale entry

    async def test_empty_configs_falls_back_to_default_specs(self):
        """An empty configs list must fall back to get_default_sandbox_specs().

        Without a warm runtime we still want the service to be usable, so we
        hand back the same default specs the static RemoteSandboxSpecService uses.
        """
        from openhands.app_server.sandbox.remote_sandbox_spec_service import (
            get_default_sandbox_specs,
        )

        ctx, _ = _make_async_client_mock(_make_http_response({'configs': []}))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            specs = await service._fetch_specs()

        # Compare field-by-field since SandboxSpecInfo has an auto-generated
        # created_at that differs between calls.
        expected = get_default_sandbox_specs()
        assert len(specs) == len(expected)
        for actual, want in zip(specs, expected, strict=False):
            assert actual.id == want.id
            assert actual.command == want.command
            assert actual.initial_env == want.initial_env
            assert actual.working_dir == want.working_dir
        # No warm runtime names means the default-name lookup will miss, which
        # is the intended signal for get_default_sandbox_spec to fall back too.
        assert service._name_to_spec == {}

    async def test_fallback_specs_are_cached(self):
        """Default fallback specs must be cached like normal results."""
        from openhands.app_server.sandbox.remote_sandbox_spec_service import (
            get_default_sandbox_specs,
        )

        ctx, client = _make_async_client_mock(_make_http_response({'configs': []}))
        service = _make_service(cache_ttl_seconds=60)

        with patch('httpx.AsyncClient', return_value=ctx):
            first = await service._fetch_specs()
            second = await service._fetch_specs()

        assert client.get.call_count == 1
        assert first is second
        expected = get_default_sandbox_specs()
        assert len(first) == len(expected)
        for actual, want in zip(first, expected, strict=False):
            assert actual.id == want.id
            assert actual.command == want.command
            assert actual.initial_env == want.initial_env
            assert actual.working_dir == want.working_dir

    async def test_http_error_propagates(self):
        """An HTTP 500 from runtime-api must propagate as an exception."""
        ctx, _ = _make_async_client_mock(_make_http_response({}, status_code=500))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            with pytest.raises(httpx.HTTPStatusError):
                await service._fetch_specs()


# ---------------------------------------------------------------------------
# search_sandbox_specs
# ---------------------------------------------------------------------------


class TestSearchSandboxSpecs:
    async def test_returns_all_specs_without_pagination(self):
        """Default call returns all specs and no next_page_id."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            page = await service.search_sandbox_specs()

        assert len(page.items) == 3
        assert page.next_page_id is None

    async def test_respects_limit_and_sets_next_page_id(self):
        """When limit < total, next_page_id points to the next batch."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            page = await service.search_sandbox_specs(limit=2)

        assert len(page.items) == 2
        assert page.items[0].id == 'ghcr.io/openhands/agent-server:1.0.0'
        assert page.items[1].id == 'ghcr.io/openhands/agent-server:0.9.0'
        assert page.next_page_id == '2'

    async def test_page_id_offsets_start_index(self):
        """Passing page_id='2' returns the third item onward."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            page = await service.search_sandbox_specs(page_id='2', limit=100)

        assert len(page.items) == 1
        assert page.items[0].id == 'ghcr.io/openhands/agent-server:nightly'
        assert page.next_page_id is None

    async def test_no_next_page_id_when_results_fit_exactly(self):
        """next_page_id is None when end_idx == len(specs)."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            # 3 specs, limit 3 from start → no next page
            page = await service.search_sandbox_specs(limit=3)

        assert len(page.items) == 3
        assert page.next_page_id is None


# ---------------------------------------------------------------------------
# get_sandbox_spec
# ---------------------------------------------------------------------------


class TestGetSandboxSpec:
    async def test_returns_spec_by_image_id(self):
        """get_sandbox_spec must find a spec by its image URL (the id field)."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            spec = await service.get_sandbox_spec(
                'ghcr.io/openhands/agent-server:0.9.0'
            )

        assert spec is not None
        assert spec.id == 'ghcr.io/openhands/agent-server:0.9.0'
        assert spec.working_dir == '/home/user'

    async def test_returns_none_for_unknown_id(self):
        """get_sandbox_spec must return None when the id is not found."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service()

        with patch('httpx.AsyncClient', return_value=ctx):
            spec = await service.get_sandbox_spec('does-not-exist:latest')

        assert spec is None


# ---------------------------------------------------------------------------
# get_default_sandbox_spec
# ---------------------------------------------------------------------------


class TestGetDefaultSandboxSpec:
    async def test_returns_spec_matching_default_spec_name(self):
        """When default_spec_name matches a config name, that spec is returned."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service(default_spec_name='v1_legacy')

        with patch('httpx.AsyncClient', return_value=ctx):
            spec = await service.get_default_sandbox_spec()

        assert spec.id == 'ghcr.io/openhands/agent-server:0.9.0'

    async def test_falls_back_to_first_spec_when_name_not_found(self):
        """When default_spec_name is set but absent from results, use the first spec."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service(default_spec_name='unknown-name')

        with patch('httpx.AsyncClient', return_value=ctx):
            spec = await service.get_default_sandbox_spec()

        assert spec.id == 'ghcr.io/openhands/agent-server:1.0.0'

    async def test_falls_back_to_first_spec_when_no_default_name(self):
        """When default_spec_name is empty, the first spec is returned."""
        ctx, _ = _make_async_client_mock(_make_http_response(_CONFIGS_RESPONSE))
        service = _make_service(default_spec_name='')

        with patch('httpx.AsyncClient', return_value=ctx):
            spec = await service.get_default_sandbox_spec()

        assert spec.id == 'ghcr.io/openhands/agent-server:1.0.0'

    async def test_falls_back_to_default_spec_when_no_warm_runtimes(self):
        """get_default_sandbox_spec must return a default spec (not raise) when
        the runtime-api endpoint returns no warm runtimes."""
        from openhands.app_server.sandbox.remote_sandbox_spec_service import (
            get_default_sandbox_specs,
        )

        ctx, _ = _make_async_client_mock(_make_http_response({'configs': []}))
        service = _make_service(default_spec_name='v1_current')

        with patch('httpx.AsyncClient', return_value=ctx):
            spec = await service.get_default_sandbox_spec()

        # The configured warm-runtime name doesn't exist, so we get the first
        # of the fallback default specs rather than raising.
        expected = get_default_sandbox_specs()
        assert spec.id == expected[0].id


# ---------------------------------------------------------------------------
# DynamicRemoteSandboxSpecServiceInjector
# ---------------------------------------------------------------------------


class TestDynamicRemoteSandboxSpecServiceInjector:
    def test_reads_api_url_from_env(self, monkeypatch):
        """api_url defaults to SANDBOX_REMOTE_RUNTIME_API_URL."""
        monkeypatch.setenv(
            'SANDBOX_REMOTE_RUNTIME_API_URL', 'https://env-url.example.com'
        )
        monkeypatch.setenv('SANDBOX_API_KEY', '')
        injector = DynamicRemoteSandboxSpecServiceInjector()
        assert injector.api_url == 'https://env-url.example.com'

    def test_reads_api_key_from_env(self, monkeypatch):
        """api_key defaults to SANDBOX_API_KEY."""
        monkeypatch.setenv('SANDBOX_API_KEY', 'env-api-key')
        monkeypatch.setenv('SANDBOX_REMOTE_RUNTIME_API_URL', '')
        injector = DynamicRemoteSandboxSpecServiceInjector()
        assert injector.api_key == 'env-api-key'

    def test_defaults_when_env_vars_absent(self, monkeypatch):
        """When env vars are unset, api_url and api_key default to empty strings."""
        monkeypatch.delenv('SANDBOX_REMOTE_RUNTIME_API_URL', raising=False)
        monkeypatch.delenv('SANDBOX_API_KEY', raising=False)
        injector = DynamicRemoteSandboxSpecServiceInjector()
        assert injector.api_url == ''
        assert injector.api_key == ''
        assert injector.default_spec_name == ''
        assert injector.cache_ttl_seconds == 60

    def test_custom_fields_are_stored(self):
        """Explicitly supplied fields are stored as-is."""
        injector = DynamicRemoteSandboxSpecServiceInjector(
            api_url='https://custom.example.com',
            api_key='my-key',
            default_spec_name='v1_current',
            cache_ttl_seconds=120,
        )
        assert injector.api_url == 'https://custom.example.com'
        assert injector.api_key == 'my-key'
        assert injector.default_spec_name == 'v1_current'
        assert injector.cache_ttl_seconds == 120

    async def test_inject_yields_dynamic_service_with_correct_params(self):
        """inject() must yield exactly one DynamicRemoteSandboxSpecService."""
        injector = DynamicRemoteSandboxSpecServiceInjector(
            api_url='https://rt.example.com',
            api_key='k',
            default_spec_name='nightly',
            cache_ttl_seconds=30,
        )
        state = State()
        services = []
        async for svc in injector.inject(state):
            services.append(svc)

        assert len(services) == 1
        svc = services[0]
        assert isinstance(svc, DynamicRemoteSandboxSpecService)
        assert svc.api_url == 'https://rt.example.com'
        assert svc.api_key == 'k'
        assert svc.default_spec_name == 'nightly'
        assert svc.cache_ttl_seconds == 30

    async def test_inject_returns_same_service_each_call(self):
        """inject() must return the same service instance on every call.

        The injector is a long-lived singleton. Re-using the same
        DynamicRemoteSandboxSpecService instance is what allows the TTL
        cache to survive across requests; creating a new instance each time
        would silently discard the cache.
        """
        injector = DynamicRemoteSandboxSpecServiceInjector(
            api_url='https://rt.example.com', api_key='k'
        )
        state = State()

        first = None
        async for svc in injector.inject(state):
            first = svc

        second = None
        async for svc in injector.inject(state):
            second = svc

        assert first is second
