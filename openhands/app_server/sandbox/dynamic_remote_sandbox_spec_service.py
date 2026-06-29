"""Sandbox spec service that resolves specs from runtime-api warm runtime configs."""

import os
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

import httpx
from fastapi import Request
from pydantic import Field, PrivateAttr

from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.sandbox_spec_models import (
    SandboxSpecInfo,
    SandboxSpecInfoPage,
)
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
    SandboxSpecServiceInjector,
)
from openhands.app_server.services.injector import InjectorState


@dataclass
class DynamicRemoteSandboxSpecService(SandboxSpecService):
    """Sandbox spec service backed by the runtime-api warm runtime configs endpoint.

    Fetches the list of available warm runtime configurations and exposes each
    as a SandboxSpecInfo. SandboxSpecInfo.id is the container image URL so that
    it flows correctly through RemoteSandboxService to runtime-api for pod creation
    and warm-runtime matching.

    Results are cached for `cache_ttl_seconds` to avoid hammering the endpoint on
    every conversation start.
    """

    api_url: str
    api_key: str
    default_spec_name: str
    cache_ttl_seconds: int = 60
    _cached_specs: list[SandboxSpecInfo] = field(default_factory=list, init=False)
    _name_to_spec: dict[str, SandboxSpecInfo] = field(default_factory=dict, init=False)
    _cache_expires_at: float = field(default=0.0, init=False)

    async def _fetch_specs(self) -> list[SandboxSpecInfo]:
        """Return specs from cache, or re-fetch from runtime-api if the TTL has expired."""
        now = time.monotonic()
        if self._cached_specs and now < self._cache_expires_at:
            return self._cached_specs

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.api_url}/api/warm-runtime-configs',
                headers={'X-API-Key': self.api_key},
                timeout=10.0,
            )
            response.raise_for_status()

        name_to_spec: dict[str, SandboxSpecInfo] = {}
        specs: list[SandboxSpecInfo] = []
        for config in response.json().get('configs', []):
            spec = SandboxSpecInfo(
                id=config['image'],
                command=config['command'],
                initial_env=config['environment'],
                working_dir=config['working_dir'],
            )
            specs.append(spec)
            name_to_spec[config['name']] = spec

        self._cached_specs = specs
        self._name_to_spec = name_to_spec
        self._cache_expires_at = now + self.cache_ttl_seconds
        return specs

    async def search_sandbox_specs(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxSpecInfoPage:
        specs = await self._fetch_specs()
        start_idx = int(page_id) if page_id else 0
        end_idx = start_idx + limit
        return SandboxSpecInfoPage(
            items=specs[start_idx:end_idx],
            next_page_id=str(end_idx) if end_idx < len(specs) else None,
        )

    async def get_sandbox_spec(self, sandbox_spec_id: str) -> SandboxSpecInfo | None:
        specs = await self._fetch_specs()
        return next((s for s in specs if s.id == sandbox_spec_id), None)

    async def get_default_sandbox_spec(self) -> SandboxSpecInfo:
        specs = await self._fetch_specs()
        if not specs:
            raise SandboxError('No warm runtime configs available from runtime-api.')
        if self.default_spec_name:
            spec = self._name_to_spec.get(self.default_spec_name)
            if spec is not None:
                return spec
        return specs[0]


class DynamicRemoteSandboxSpecServiceInjector(SandboxSpecServiceInjector):
    """Injector for DynamicRemoteSandboxSpecService.

    Enable via environment variable:
        OH_SANDBOX_SPEC_KIND=openhands.app_server.sandbox.dynamic_remote_sandbox_spec_service.DynamicRemoteSandboxSpecServiceInjector

    The api_url and api_key default to the standard SANDBOX_REMOTE_RUNTIME_API_URL /
    SANDBOX_API_KEY variables used by RemoteSandboxServiceInjector, so no extra
    credential configuration is needed when running with RUNTIME=remote.

    Set OH_SANDBOX_SPEC_DEFAULT_SPEC_NAME to the warm runtime config name
    (e.g. "v1_current") to control which image is used by default.
    """

    api_url: str = Field(
        default_factory=lambda: os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL', ''),
        description='Runtime-api base URL. Defaults to SANDBOX_REMOTE_RUNTIME_API_URL.',
    )
    api_key: str = Field(
        default_factory=lambda: os.environ.get('SANDBOX_API_KEY', ''),
        description='Runtime-api API key. Defaults to SANDBOX_API_KEY.',
    )
    default_spec_name: str = Field(
        default='',
        description=(
            'Name of the warm runtime config to use as the default sandbox spec. '
            'If empty or not found, the first config returned by runtime-api is used.'
        ),
    )
    cache_ttl_seconds: int = Field(
        default=60,
        description='Seconds to cache the warm runtime config list before re-fetching.',
    )

    # Shared across all requests — the injector is a long-lived singleton in the
    # global config, so this attribute persists and the TTL cache actually works.
    _service: DynamicRemoteSandboxSpecService | None = PrivateAttr(default=None)

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxSpecService, None]:
        if self._service is None:
            self._service = DynamicRemoteSandboxSpecService(
                api_url=self.api_url,
                api_key=self.api_key,
                default_spec_name=self.default_spec_name,
                cache_ttl_seconds=self.cache_ttl_seconds,
            )
        yield self._service
