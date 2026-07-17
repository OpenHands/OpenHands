from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.provider import ProviderHandler, ProviderToken
from openhands.app_server.integrations.service_types import (
    AuthenticationError,
    ProviderType,
    UnknownException,
    User,
)


def _handler_with_tokens(*providers: ProviderType) -> ProviderHandler:
    return ProviderHandler(
        MappingProxyType(
            {
                provider: ProviderToken(token=SecretStr(f'{provider.value}-token'))
                for provider in providers
            }
        )
    )


@pytest.mark.asyncio
async def test_get_user_reraises_transient_provider_errors(monkeypatch):
    handler = _handler_with_tokens(ProviderType.GITHUB)
    service = AsyncMock()
    service.get_user.side_effect = UnknownException(
        'Unknown error: GitHub API returned 503'
    )
    monkeypatch.setattr(handler, 'get_service', lambda provider: service)

    with pytest.raises(UnknownException, match='GitHub API returned 503'):
        await handler.get_user()


@pytest.mark.asyncio
async def test_get_user_keeps_authentication_error_for_invalid_tokens(monkeypatch):
    handler = _handler_with_tokens(ProviderType.GITHUB)
    service = AsyncMock()
    service.get_user.side_effect = AuthenticationError('Invalid github token')
    monkeypatch.setattr(handler, 'get_service', lambda provider: service)

    with pytest.raises(AuthenticationError, match='Need valid provider token'):
        await handler.get_user()


@pytest.mark.asyncio
async def test_get_user_still_tries_next_provider_after_failure(monkeypatch):
    handler = _handler_with_tokens(ProviderType.GITHUB, ProviderType.GITLAB)
    gitlab_user = User(
        id='gitlab-user-id',
        login='gitlab-user',
        avatar_url='https://example.com/avatar.png',
    )
    services = {
        ProviderType.GITHUB: AsyncMock(),
        ProviderType.GITLAB: AsyncMock(),
    }
    services[ProviderType.GITHUB].get_user.side_effect = UnknownException(
        'GitHub unavailable'
    )
    services[ProviderType.GITLAB].get_user.return_value = gitlab_user
    monkeypatch.setattr(handler, 'get_service', lambda provider: services[provider])

    assert await handler.get_user() == gitlab_user
    services[ProviderType.GITHUB].get_user.assert_awaited_once()
    services[ProviderType.GITLAB].get_user.assert_awaited_once()
