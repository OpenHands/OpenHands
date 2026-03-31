"""Tests for the GET /api/user/git-organizations endpoint.

This endpoint returns git organizations for the user's active provider
in SaaS mode (single provider at a time).
"""

from types import MappingProxyType
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType


@pytest.fixture
def github_provider_tokens():
    return MappingProxyType(
        {ProviderType.GITHUB: ProviderToken(token=SecretStr('gh-token'))}
    )


@pytest.fixture
def gitlab_provider_tokens():
    return MappingProxyType(
        {ProviderType.GITLAB: ProviderToken(token=SecretStr('gl-token'))}
    )


@pytest.fixture
def bitbucket_provider_tokens():
    return MappingProxyType(
        {ProviderType.BITBUCKET: ProviderToken(token=SecretStr('bb-token'))}
    )


@pytest.mark.asyncio
async def test_github_returns_organizations(github_provider_tokens):
    """User signed in with GitHub sees their GitHub organizations."""
    from server.routes.user import saas_get_user_git_organizations

    with patch('server.routes.user.ProviderHandler') as MockHandler:
        mock_client = MockHandler.return_value
        mock_client.get_github_organizations = AsyncMock(
            return_value=['All-Hands-AI', 'OpenHands']
        )

        result = await saas_get_user_git_organizations(
            provider_tokens=github_provider_tokens,
            access_token=SecretStr('token'),
            user_id='user-1',
        )

        assert result == {
            'provider': 'github',
            'organizations': ['All-Hands-AI', 'OpenHands'],
        }


@pytest.mark.asyncio
async def test_gitlab_returns_groups(gitlab_provider_tokens):
    """User signed in with GitLab sees their GitLab groups."""
    from server.routes.user import saas_get_user_git_organizations

    with patch('server.routes.user.ProviderHandler') as MockHandler:
        mock_client = MockHandler.return_value
        mock_client.get_gitlab_groups = AsyncMock(
            return_value=['my-team', 'open-source']
        )

        result = await saas_get_user_git_organizations(
            provider_tokens=gitlab_provider_tokens,
            access_token=SecretStr('token'),
            user_id='user-1',
        )

        assert result == {
            'provider': 'gitlab',
            'organizations': ['my-team', 'open-source'],
        }


@pytest.mark.asyncio
async def test_bitbucket_returns_workspaces(bitbucket_provider_tokens):
    """User signed in with Bitbucket sees their Bitbucket workspaces."""
    from server.routes.user import saas_get_user_git_organizations

    with patch('server.routes.user.ProviderHandler') as MockHandler:
        mock_client = MockHandler.return_value
        mock_client.get_bitbucket_workspaces = AsyncMock(return_value=['my-workspace'])

        result = await saas_get_user_git_organizations(
            provider_tokens=bitbucket_provider_tokens,
            access_token=SecretStr('token'),
            user_id='user-1',
        )

        assert result == {
            'provider': 'bitbucket',
            'organizations': ['my-workspace'],
        }


@pytest.mark.asyncio
async def test_no_provider_tokens_falls_back_to_idp(mock_check_idp):
    """When no provider tokens exist, falls back to IDP check."""
    from server.routes.user import saas_get_user_git_organizations

    mock_check_idp.return_value = {}

    result = await saas_get_user_git_organizations(
        provider_tokens=None,
        access_token=SecretStr('token'),
        user_id='user-1',
    )

    assert result == {}
    mock_check_idp.assert_called_once()


@pytest.fixture
def mock_check_idp():
    with patch('server.routes.user._check_idp', new_callable=AsyncMock) as mock_fn:
        yield mock_fn
