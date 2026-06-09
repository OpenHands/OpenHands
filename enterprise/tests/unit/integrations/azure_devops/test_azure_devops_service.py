"""Unit tests for SaaSAzureDevOpsService."""

from unittest.mock import AsyncMock, patch

import pytest
from integrations.azure_devops.azure_devops_service import SaaSAzureDevOpsService
from pydantic import SecretStr

from openhands.app_server.integrations.service_types import ProviderType


@pytest.mark.asyncio
async def test_get_latest_token_updates_cached_token_for_retry_headers():
    service = SaaSAzureDevOpsService(
        external_auth_token=SecretStr('keycloak-token'),
        token=SecretStr('expired-token'),
    )

    with patch.object(
        service.token_manager,
        'get_idp_token',
        new_callable=AsyncMock,
        return_value='fresh-token',
    ) as mock_get_idp_token:
        token = await service.get_latest_token()

    assert token is not None
    assert token.get_secret_value() == 'fresh-token'
    assert service.token.get_secret_value() == 'fresh-token'
    mock_get_idp_token.assert_awaited_once_with(
        'keycloak-token',
        idp=ProviderType.AZURE_DEVOPS,
    )


@pytest.mark.asyncio
async def test_get_latest_token_updates_cached_token_from_external_auth_id():
    service = SaaSAzureDevOpsService(
        external_auth_id='external-auth-id',
        token=SecretStr('expired-token'),
    )

    with (
        patch.object(
            service.token_manager,
            'load_offline_token',
            new_callable=AsyncMock,
            return_value='offline-token',
        ) as mock_load_offline_token,
        patch.object(
            service.token_manager,
            'get_idp_token_from_offline_token',
            new_callable=AsyncMock,
            return_value='fresh-token',
        ) as mock_get_idp_token_from_offline_token,
    ):
        token = await service.get_latest_token()

    assert token is not None
    assert token.get_secret_value() == 'fresh-token'
    assert service.token.get_secret_value() == 'fresh-token'
    mock_load_offline_token.assert_awaited_once_with('external-auth-id')
    mock_get_idp_token_from_offline_token.assert_awaited_once_with(
        'offline-token',
        ProviderType.AZURE_DEVOPS,
    )


@pytest.mark.asyncio
async def test_get_latest_token_updates_cached_token_from_user_id():
    service = SaaSAzureDevOpsService(
        user_id='azure-user-id',
        token=SecretStr('expired-token'),
    )

    with patch.object(
        service.token_manager,
        'get_idp_token_from_idp_user_id',
        new_callable=AsyncMock,
        return_value='fresh-token',
    ) as mock_get_idp_token_from_user_id:
        token = await service.get_latest_token()

    assert token is not None
    assert token.get_secret_value() == 'fresh-token'
    assert service.token.get_secret_value() == 'fresh-token'
    mock_get_idp_token_from_user_id.assert_awaited_once_with(
        'azure-user-id',
        ProviderType.AZURE_DEVOPS,
    )


@pytest.mark.asyncio
async def test_get_latest_token_leaves_cached_token_when_refresh_unavailable():
    service = SaaSAzureDevOpsService(token=SecretStr('stored-token'))

    token = await service.get_latest_token()

    assert token is None
    assert service.token.get_secret_value() == 'stored-token'
