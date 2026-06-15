"""Unit tests for the LiteLLM-proxy-backed model discovery service."""

import logging

import httpx
import pytest
from server.verified_models import litellm_proxy_model_router as router_module
from server.verified_models.litellm_proxy_model_router import (
    LiteLLMProxyModelService,
    LiteLLMProxyModelServiceInjector,
    _derive_default_model,
)

from openhands.app_server.utils.llm import ModelsResponse

LOGGER_NAME = 'server.verified_models.litellm_proxy_model_router'

HAPPY_PAYLOAD = {
    'data': [
        {
            'model_name': 'claude-sonnet-4-5',
            'litellm_params': {'model': 'anthropic/claude-sonnet-4-5'},
            'model_info': {'openhands_default': True},
        },
        {
            'model_name': 'hidden-model',
            'litellm_params': {'model': 'openai/gpt-5'},
            'model_info': {
                'openhands_hidden': True,
                'openhands_canonical': 'claude-sonnet-4-5',
            },
        },
        {
            # Duplicate of the first entry — a load-balanced deployment.
            'model_name': 'claude-sonnet-4-5',
            'litellm_params': {'model': 'bedrock/claude-sonnet-4-5'},
            'model_info': {},
        },
        {
            'model_name': 'gpt-5',
            'litellm_params': {'model': 'openai/gpt-5'},
            'model_info': {},
        },
        {
            # Entry without a model_name is skipped.
            'litellm_params': {'model': 'openai/gpt-4o'},
        },
        'not-a-dict',
    ]
}


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f'HTTP {self.status_code}',
                request=httpx.Request('GET', 'http://litellm.test/model/info'),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._payload


def make_fake_client(response=None, error=None):
    """Build a fake httpx.AsyncClient class that records calls."""

    calls: list[dict] = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, headers=None):
            calls.append({'url': url, 'headers': headers or {}})
            if error is not None:
                raise error
            return response

    return FakeAsyncClient, calls


@pytest.fixture(autouse=True)
def reset_cache():
    LiteLLMProxyModelService._reset_cache()
    yield
    LiteLLMProxyModelService._reset_cache()


@pytest.fixture(autouse=True)
def proxy_env(monkeypatch):
    monkeypatch.setattr(router_module, 'LITE_LLM_API_URL', 'http://litellm.test')
    monkeypatch.setattr(router_module, 'LITE_LLM_API_KEY', 'test-master-key')
    monkeypatch.setattr(
        router_module,
        'get_default_litellm_model',
        lambda: 'litellm_proxy/claude-sonnet-4-5',
    )


@pytest.fixture(autouse=True)
def byok_off(monkeypatch):
    """Default BYOK off so the proxy-only tests are unaffected; the union
    tests opt in via ``byok_on``. (BYOK defaults *on* in real config, so
    without this every test would hit the real web-client config lookup.)"""

    async def _off(self):
        return False

    monkeypatch.setattr(LiteLLMProxyModelService, '_byok_enabled', _off)


# A deterministic stand-in for the SDK catalogue. Two bare names collide with
# the proxy's (``gpt-5``, ``claude-sonnet-4-5``) to exercise the verified-flag
# guard; an ``openhands/``-prefixed entry must be dropped from the union.
STUB_CATALOGUE = ModelsResponse(
    models=[
        'anthropic/claude-sonnet-4-5',
        'openai/gpt-5',
        'openai/gpt-4o',
        'gemini/gemini-2.0-flash',
        'openhands/should-be-dropped',
    ],
    verified_models=['anthropic/claude-sonnet-4-5'],
    verified_providers=['anthropic'],
    default_model='anthropic/claude-sonnet-4-5',
    hidden_models=[],
    hidden_model_canonicals={},
)


def byok_on(monkeypatch, catalogue=STUB_CATALOGUE):
    """Turn BYOK on and pin the SDK catalogue the union pulls from."""

    async def _on(self):
        return True

    monkeypatch.setattr(LiteLLMProxyModelService, '_byok_enabled', _on)
    monkeypatch.setattr(
        router_module, 'get_supported_llm_models', lambda *a, **k: catalogue
    )


@pytest.fixture
def fake_clock(monkeypatch):
    """Controllable monotonic clock injected into the service."""
    state = {'now': 1000.0}
    monkeypatch.setattr(
        LiteLLMProxyModelService, '_now', staticmethod(lambda: state['now'])
    )
    return state


def install_client(monkeypatch, response=None, error=None):
    fake_client, calls = make_fake_client(response=response, error=error)
    monkeypatch.setattr(router_module.httpx, 'AsyncClient', fake_client)
    return calls


class TestHappyPath:
    async def test_models_response_shape(self, monkeypatch):
        calls = install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        service = LiteLLMProxyModelService()

        response = await service._get_models_response()

        # Hidden models kept out of the visible list, duplicates removed,
        # proxy order preserved, every model exposed under the public
        # openhands/ prefix.
        assert response.models == ['openhands/claude-sonnet-4-5', 'openhands/gpt-5']
        assert response.verified_models == ['claude-sonnet-4-5', 'gpt-5']
        assert response.verified_providers == ['openhands']
        # Hidden alias routes are still reported so saved settings that
        # reference them count as available.
        assert response.hidden_models == ['openhands/hidden-model']
        # The alias carries its canonical mapping (canonical is visible).
        assert response.hidden_model_canonicals == {
            'openhands/hidden-model': 'openhands/claude-sonnet-4-5'
        }
        # litellm_proxy/ prefix translated to openhands/.
        assert response.default_model == 'openhands/claude-sonnet-4-5'

        assert len(calls) == 1
        assert calls[0]['url'] == 'http://litellm.test/model/info'
        assert calls[0]['headers']['Authorization'] == 'Bearer test-master-key'

    async def test_search_llm_models_uses_proxy_models(self, monkeypatch):
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        service = LiteLLMProxyModelService()

        page = await service.search_llm_models()

        names = [(m.provider, m.name, m.hidden, m.canonical) for m in page.items]
        assert names == [
            ('openhands', 'claude-sonnet-4-5', False, None),
            ('openhands', 'gpt-5', False, None),
            # Hidden alias routes ride along flagged hidden=True so the
            # frontend can exclude them from dropdown options while still
            # treating saved settings that reference them as available;
            # canonical names the visible model the alias routes to.
            ('openhands', 'hidden-model', True, 'claude-sonnet-4-5'),
        ]

    async def test_hidden_only_when_all_duplicate_entries_hidden(self, monkeypatch):
        # A load-balanced group is hidden only when every deployment that
        # carries the name is hidden; one visible entry wins.
        payload = {
            'data': [
                {
                    'model_name': 'mixed-model',
                    'model_info': {'openhands_hidden': True},
                },
                {'model_name': 'mixed-model', 'model_info': {}},
                {
                    'model_name': 'legacy-alias',
                    'model_info': {'openhands_hidden': True},
                },
                {
                    # Duplicate hidden entry — reported once.
                    'model_name': 'legacy-alias',
                    'model_info': {'openhands_hidden': True},
                },
            ]
        }
        install_client(monkeypatch, response=FakeResponse(payload))
        service = LiteLLMProxyModelService()

        response = await service._get_models_response()

        assert response.models == ['openhands/mixed-model']
        assert response.hidden_models == ['openhands/legacy-alias']
        # No openhands_canonical tags anywhere -> no mappings.
        assert response.hidden_model_canonicals == {}

    async def test_canonical_skipped_when_not_visible(self, monkeypatch):
        # A canonical tag pointing at a name the proxy does not serve
        # visibly is dropped; a tag on a visible entry is ignored.
        payload = {
            'data': [
                {'model_name': 'real-model', 'model_info': {}},
                {
                    'model_name': 'alias-to-missing',
                    'model_info': {
                        'openhands_hidden': True,
                        'openhands_canonical': 'removed-model',
                    },
                },
                {
                    'model_name': 'alias-to-hidden',
                    'model_info': {
                        'openhands_hidden': True,
                        'openhands_canonical': 'other-alias',
                    },
                },
                {
                    'model_name': 'other-alias',
                    'model_info': {'openhands_hidden': True},
                },
            ]
        }
        install_client(monkeypatch, response=FakeResponse(payload))
        service = LiteLLMProxyModelService()

        response = await service._get_models_response()

        assert response.models == ['openhands/real-model']
        assert response.hidden_models == [
            'openhands/alias-to-missing',
            'openhands/alias-to-hidden',
            'openhands/other-alias',
        ]
        assert response.hidden_model_canonicals == {}

        page = await service.search_llm_models()
        assert all(m.canonical is None for m in page.items)

    async def test_canonical_mapped_for_visible_target(self, monkeypatch):
        payload = {
            'data': [
                {'model_name': 'real-model', 'model_info': {}},
                {
                    'model_name': 'legacy-alias',
                    'model_info': {
                        'openhands_hidden': True,
                        'openhands_canonical': 'real-model',
                    },
                },
                {
                    # Hidden alias without a canonical tag -> no mapping.
                    'model_name': 'untagged-alias',
                    'model_info': {'openhands_hidden': True},
                },
            ]
        }
        install_client(monkeypatch, response=FakeResponse(payload))
        service = LiteLLMProxyModelService()

        response = await service._get_models_response()

        assert response.hidden_model_canonicals == {
            'openhands/legacy-alias': 'openhands/real-model'
        }

    async def test_search_providers_only_openhands(self, monkeypatch):
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        service = LiteLLMProxyModelService()

        page = await service.search_providers()

        assert [(p.name, p.verified) for p in page.items] == [('openhands', True)]

    async def test_no_auth_header_without_key(self, monkeypatch):
        monkeypatch.setattr(router_module, 'LITE_LLM_API_KEY', None)
        calls = install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        service = LiteLLMProxyModelService()

        await service._get_models_response()

        assert 'Authorization' not in calls[0]['headers']


class TestDefaultModelTranslation:
    def test_litellm_proxy_prefix_translated(self, monkeypatch):
        monkeypatch.setattr(
            router_module, 'get_default_litellm_model', lambda: 'litellm_proxy/gpt-5'
        )
        assert _derive_default_model() == 'openhands/gpt-5'

    def test_openhands_prefix_preserved(self, monkeypatch):
        monkeypatch.setattr(
            router_module, 'get_default_litellm_model', lambda: 'openhands/gpt-5'
        )
        assert _derive_default_model() == 'openhands/gpt-5'

    def test_bare_name_prefixed(self, monkeypatch):
        monkeypatch.setattr(router_module, 'get_default_litellm_model', lambda: 'gpt-5')
        assert _derive_default_model() == 'openhands/gpt-5'


class TestFailurePaths:
    async def test_cold_failure_returns_empty_without_raising(
        self, monkeypatch, caplog
    ):
        install_client(monkeypatch, error=httpx.ConnectError('boom'))
        service = LiteLLMProxyModelService()

        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            response = await service._get_models_response()

        assert response.models == []
        assert response.verified_models == []
        assert response.hidden_models == []
        assert response.hidden_model_canonicals == {}
        # The env-derived default is still communicated.
        assert response.default_model == 'openhands/claude-sonnet-4-5'
        assert any(
            'no previous result is cached' in record.message
            for record in caplog.records
        )

    async def test_cold_failure_is_not_cached(self, monkeypatch):
        install_client(monkeypatch, error=httpx.ConnectError('boom'))
        service = LiteLLMProxyModelService()
        assert (await service._get_models_response()).models == []

        # The proxy recovers; the very next request succeeds (the failure
        # result was not cached).
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        response = await LiteLLMProxyModelService()._get_models_response()
        assert response.models == ['openhands/claude-sonnet-4-5', 'openhands/gpt-5']

    async def test_http_error_status_returns_empty(self, monkeypatch):
        install_client(monkeypatch, response=FakeResponse({}, status_code=500))
        service = LiteLLMProxyModelService()

        response = await service._get_models_response()

        assert response.models == []

    async def test_stale_cache_served_on_failure(self, monkeypatch, fake_clock, caplog):
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        first = await LiteLLMProxyModelService()._get_models_response()
        assert first.models == ['openhands/claude-sonnet-4-5', 'openhands/gpt-5']

        # Far beyond the TTL the proxy starts failing — the last-good result
        # is served regardless of age.
        fake_clock['now'] += 10_000.0
        install_client(monkeypatch, error=httpx.ConnectError('boom'))
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            second = await LiteLLMProxyModelService()._get_models_response()

        assert second is first
        assert any(
            'serving the last successful result' in record.message
            for record in caplog.records
        )


class TestCacheTtl:
    async def test_within_ttl_no_refetch(self, monkeypatch, fake_clock):
        calls = install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        first = await LiteLLMProxyModelService()._get_models_response()

        fake_clock['now'] += router_module.CACHE_TTL_SECONDS - 1.0
        second = await LiteLLMProxyModelService()._get_models_response()

        assert second is first
        assert len(calls) == 1

    async def test_after_ttl_refetches(self, monkeypatch, fake_clock):
        calls = install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))
        await LiteLLMProxyModelService()._get_models_response()

        fake_clock['now'] += router_module.CACHE_TTL_SECONDS + 1.0
        updated_payload = {'data': [{'model_name': 'new-model', 'model_info': {}}]}
        calls2 = install_client(monkeypatch, response=FakeResponse(updated_payload))
        response = await LiteLLMProxyModelService()._get_models_response()

        assert len(calls) == 1
        assert len(calls2) == 1
        assert response.models == ['openhands/new-model']


class TestByokUnion:
    """BYOK on unions the SDK catalogue so users can bring their own provider
    (SaaS parity); BYOK off keeps the managed proxy list only."""

    async def test_byok_off_ignores_catalogue(self, monkeypatch):
        # Catalogue is available but BYOK is off (autouse default) -> proxy only.
        monkeypatch.setattr(
            router_module, 'get_supported_llm_models', lambda *a, **k: STUB_CATALOGUE
        )
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))

        response = await LiteLLMProxyModelService()._get_models_response()

        assert response.models == ['openhands/claude-sonnet-4-5', 'openhands/gpt-5']

    async def test_byok_on_unions_catalogue(self, monkeypatch):
        byok_on(monkeypatch)
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))

        response = await LiteLLMProxyModelService()._get_models_response()

        # Managed proxy models first (order preserved), then catalogue extras;
        # the openhands/-prefixed catalogue entry is dropped.
        assert response.models == [
            'openhands/claude-sonnet-4-5',
            'openhands/gpt-5',
            'anthropic/claude-sonnet-4-5',
            'openai/gpt-5',
            'openai/gpt-4o',
            'gemini/gemini-2.0-flash',
        ]
        assert len(response.models) == len(set(response.models))
        # Proxy stays the verified/default authority; catalogue opinions ignored.
        assert response.verified_models == ['claude-sonnet-4-5', 'gpt-5']
        assert response.verified_providers == ['openhands']
        assert response.default_model == 'openhands/claude-sonnet-4-5'

    async def test_byok_on_only_managed_models_verified(self, monkeypatch):
        # Regression: a catalogue model whose bare name collides with a proxy
        # one (openai/gpt-5 vs proxy gpt-5; anthropic/claude-sonnet-4-5 vs
        # proxy claude-sonnet-4-5) must NOT be marked verified.
        byok_on(monkeypatch)
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))

        page = await LiteLLMProxyModelService().search_llm_models()
        verified = {
            (m.provider, m.name) for m in page.items if m.verified and not m.hidden
        }

        assert verified == {('openhands', 'claude-sonnet-4-5'), ('openhands', 'gpt-5')}

    async def test_byok_on_exposes_catalogue_providers(self, monkeypatch):
        byok_on(monkeypatch)
        install_client(monkeypatch, response=FakeResponse(HAPPY_PAYLOAD))

        page = await LiteLLMProxyModelService().search_providers()

        # openhands stays the verified managed provider, listed first; catalogue
        # providers ride along unverified so the picker shows for BYOK users.
        assert [(p.name, p.verified) for p in page.items] == [
            ('openhands', True),
            ('anthropic', False),
            ('gemini', False),
            ('openai', False),
        ]

    async def test_byok_on_serves_catalogue_when_proxy_down(self, monkeypatch):
        # Proxy unreachable + BYOK on: no managed models, but the catalogue is
        # still offered so a BYOK user can bring their own key.
        byok_on(monkeypatch)
        install_client(monkeypatch, error=httpx.ConnectError('boom'))

        response = await LiteLLMProxyModelService()._get_models_response()

        assert response.models == [
            'anthropic/claude-sonnet-4-5',
            'openai/gpt-5',
            'openai/gpt-4o',
            'gemini/gemini-2.0-flash',
        ]
        assert response.verified_models == []


class TestInjector:
    async def test_injector_yields_service(self):
        injector = LiteLLMProxyModelServiceInjector()
        generator = injector.inject(None)  # type: ignore[arg-type]
        service = await generator.__anext__()
        assert isinstance(service, LiteLLMProxyModelService)
        with pytest.raises(StopAsyncIteration):
            await generator.__anext__()
