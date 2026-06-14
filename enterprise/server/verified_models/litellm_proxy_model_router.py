"""LiteLLM-proxy-backed LLM model discovery for self-hosted enterprise (OHE).

Self-hosted enterprise installs bundle a LiteLLM proxy, and the set of models
that actually works for a customer is exactly the set configured on that
proxy. This service discovers models from the proxy's ``GET /model/info``
endpoint instead of unioning the static litellm catalogue and the SaaS
``verified_models`` database table, so the model dropdown only shows models
that genuinely exist on the customer's proxy.

Activate via the environment variable::

    OH_LLM_MODEL_KIND=server.verified_models.litellm_proxy_model_router.LiteLLMProxyModelServiceInjector
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, ClassVar

import httpx
from fastapi import Request
from server.constants import (
    LITE_LLM_API_KEY,
    LITE_LLM_API_URL,
    get_default_litellm_model,
)

from openhands.app_server.config_api.default_llm_model_service import (
    DefaultLLMModelService,
)
from openhands.app_server.config_api.llm_model_service import (
    LLMModelService,
    LLMModelServiceInjector,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.http_session import httpx_verify_option
from openhands.app_server.utils.llm import ModelsResponse, get_supported_llm_models

_logger = logging.getLogger(__name__)

# Timeout (seconds) for the proxy /model/info request.
MODEL_INFO_TIMEOUT = 5.0
# How long (seconds) a successful discovery result is reused before refetching.
CACHE_TTL_SECONDS = 60.0

# The proxy-transport prefix used by LITELLM_DEFAULT_MODEL.
_LITELLM_PROXY_PREFIX = 'litellm_proxy/'
# The public/stored prefix used by the app. The SDK translates
# ``openhands/`` -> ``litellm_proxy/`` at the transport boundary.
_OPENHANDS_PREFIX = 'openhands/'


def _derive_default_model() -> str:
    """Translate the env-derived default model to the public prefix.

    ``LITELLM_DEFAULT_MODEL`` holds the default as ``litellm_proxy/<name>``;
    the app stores and displays models as ``openhands/<name>``. An already-
    ``openhands/``-prefixed default is kept as-is; a bare name is prefixed.
    """
    default = get_default_litellm_model()
    if default.startswith(_LITELLM_PROXY_PREFIX):
        return _OPENHANDS_PREFIX + default[len(_LITELLM_PROXY_PREFIX) :]
    if default.startswith(_OPENHANDS_PREFIX):
        return default
    return _OPENHANDS_PREFIX + default


class LiteLLMProxyModelService(DefaultLLMModelService):
    """Model discovery backed by the bundled LiteLLM proxy.

    Inherits filtering, pagination, and provider logic from
    ``DefaultLLMModelService`` — only the model list source is different.

    Results are cached at class level (the injector creates a new service per
    request) with a short TTL. On fetch failure the last-good result is
    served regardless of age; if there has never been a successful fetch an
    empty model list is returned. Discovery never raises out of the request
    path just because the proxy is briefly unreachable.
    """

    # Shared across instances; one service instance is created per request.
    _shared_response: ClassVar[ModelsResponse | None] = None
    _shared_fetched_at: ClassVar[float] = 0.0
    _fetch_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    def _reset_cache(cls) -> None:
        """Clear the shared cache (intended for tests)."""
        cls._shared_response = None
        cls._shared_fetched_at = 0.0

    @staticmethod
    def _now() -> float:
        """Monotonic clock; isolated for tests."""
        return time.monotonic()

    async def _fetch_proxy_model_names(
        self,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Fetch ``(visible, hidden, canonicals)`` model names from the proxy.

        Entries flagged ``model_info.openhands_hidden`` (legacy alias routes
        the proxy still serves after a rename) are collected separately: they
        must not be offered as dropdown options, but a saved setting that
        references one is still valid. A hidden entry may carry
        ``model_info.openhands_canonical`` naming the visible route it
        aliases; ``canonicals`` maps hidden name -> canonical name, dropping
        mappings whose target is not among the visible names. A duplicated
        ``model_name`` is an intentional load-balanced deployment — it is
        reported once, and is hidden only when *every* entry carrying that
        name is hidden. Proxy order is preserved. ``litellm_params`` are
        never propagated.
        """
        url = LITE_LLM_API_URL.rstrip('/') + '/model/info'
        headers: dict[str, str] = {}
        if LITE_LLM_API_KEY:
            headers['Authorization'] = f'Bearer {LITE_LLM_API_KEY}'
        async with httpx.AsyncClient(
            verify=httpx_verify_option(), timeout=MODEL_INFO_TIMEOUT
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()

        order: list[str] = []
        hidden_by_name: dict[str, bool] = {}
        canonical_by_name: dict[str, str] = {}
        for entry in payload.get('data') or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get('model_name')
            if not name or not isinstance(name, str):
                continue
            model_info: Any = entry.get('model_info') or {}
            hidden = isinstance(model_info, dict) and bool(
                model_info.get('openhands_hidden')
            )
            if hidden and name not in canonical_by_name:
                canonical = model_info.get('openhands_canonical')
                if canonical and isinstance(canonical, str):
                    canonical_by_name[name] = canonical
            if name not in hidden_by_name:
                order.append(name)
                hidden_by_name[name] = hidden
            elif not hidden:
                # Any visible deployment makes the name visible.
                hidden_by_name[name] = False
        visible = [name for name in order if not hidden_by_name[name]]
        hidden_names = [name for name in order if hidden_by_name[name]]
        visible_set = set(visible)
        canonicals = {
            name: canonical_by_name[name]
            for name in hidden_names
            if canonical_by_name.get(name) in visible_set
        }
        return visible, hidden_names, canonicals

    @staticmethod
    def _build_response(
        model_names: list[str],
        hidden_model_names: list[str] | None = None,
        hidden_model_canonicals: dict[str, str] | None = None,
    ) -> ModelsResponse:
        return ModelsResponse(
            models=[_OPENHANDS_PREFIX + name for name in model_names],
            verified_models=list(model_names),
            verified_providers=['openhands'],
            default_model=_derive_default_model(),
            hidden_models=[
                _OPENHANDS_PREFIX + name for name in hidden_model_names or []
            ],
            hidden_model_canonicals={
                _OPENHANDS_PREFIX + alias: _OPENHANDS_PREFIX + canonical
                for alias, canonical in (hidden_model_canonicals or {}).items()
            },
        )

    async def _byok_enabled(self) -> bool:
        """Whether BYOK is on. Read from the resolved web-client config so
        discovery and the UI gate agree on the same value."""
        from openhands.app_server.config import get_global_config

        config = await get_global_config().web_client.get_web_client_config()
        return config.feature_flags.allow_user_llm_configuration

    async def _union_with_catalogue(
        self, proxy_response: ModelsResponse
    ) -> ModelsResponse:
        """BYOK on: add the full SDK provider catalogue so users can pick any
        provider and bring their own key (matches SaaS). Proxy models stay the
        managed/verified set; catalogue models land unverified ("Other"). BYOK
        off: return the managed proxy list unchanged."""
        if not await self._byok_enabled():
            return proxy_response
        catalogue = get_supported_llm_models()
        proxy_models = set(proxy_response.models)
        extra = [
            m
            for m in catalogue.models
            if not m.startswith(_OPENHANDS_PREFIX) and m not in proxy_models
        ]
        return ModelsResponse(
            models=proxy_response.models + extra,
            verified_models=proxy_response.verified_models,
            verified_providers=proxy_response.verified_providers,
            default_model=proxy_response.default_model,
            hidden_models=proxy_response.hidden_models,
            hidden_model_canonicals=proxy_response.hidden_model_canonicals,
        )

    def _is_model_verified(
        self, model_name: str, name: str, models_response: ModelsResponse
    ) -> bool:
        # Managed proxy models are verified; the unioned SDK catalogue (BYOK)
        # is not. Require the openhands/ prefix so a catalogue model whose bare
        # name collides with a proxy one (openai/gpt-5 vs proxy gpt-5) isn't
        # falsely verified.
        return model_name.startswith(_OPENHANDS_PREFIX) and name in set(
            models_response.verified_models
        )

    async def _get_models_response(
        self,
        verified_models: list[str] | None = None,
    ) -> ModelsResponse:
        cls = LiteLLMProxyModelService
        response = cls._shared_response
        if (
            response is not None
            and self._now() - cls._shared_fetched_at < CACHE_TTL_SECONDS
        ):
            return response

        async with cls._fetch_lock:
            # Re-check after acquiring the lock — a concurrent request may
            # have refreshed the cache while this one was waiting.
            response = cls._shared_response
            if (
                response is not None
                and self._now() - cls._shared_fetched_at < CACHE_TTL_SECONDS
            ):
                return response

            try:
                (
                    model_names,
                    hidden_model_names,
                    hidden_model_canonicals,
                ) = await self._fetch_proxy_model_names()
            except Exception:
                if response is not None:
                    # Serve the last-good result regardless of age. The next
                    # request retries because the timestamp is not refreshed.
                    _logger.warning(
                        'Failed to fetch models from the LiteLLM proxy; '
                        'serving the last successful result',
                        exc_info=True,
                    )
                    return response
                _logger.error(
                    'Failed to fetch models from the LiteLLM proxy and no '
                    'previous result is cached; returning an empty model list',
                    exc_info=True,
                )
                # Not cached, so the next request retries immediately. Still
                # union the catalogue so BYOK users can bring a key while the
                # proxy is unreachable.
                return await self._union_with_catalogue(self._build_response([]))

            response = await self._union_with_catalogue(
                self._build_response(
                    model_names, hidden_model_names, hidden_model_canonicals
                )
            )
            cls._shared_response = response
            cls._shared_fetched_at = self._now()
            return response


class LiteLLMProxyModelServiceInjector(LLMModelServiceInjector):
    """Injector that provides the LiteLLM-proxy-backed model service.

    Activate via the environment variable::

        OH_LLM_MODEL_KIND=server.verified_models.litellm_proxy_model_router.LiteLLMProxyModelServiceInjector
    """

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[LLMModelService, None]:
        yield LiteLLMProxyModelService()
