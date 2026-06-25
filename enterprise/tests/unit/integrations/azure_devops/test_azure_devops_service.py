"""Unit tests for SaaSAzureDevOpsService."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from integrations.azure_devops.azure_devops_service import SaaSAzureDevOpsService
from pydantic import SecretStr

from openhands.app_server.integrations.service_types import ProviderType, RequestMethod


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


@pytest.mark.asyncio
async def test_pr_comment_urls_do_not_duplicate_organization():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(return_value=({'id': 1}, {}))  # type: ignore[method-assign]

    await service.add_pr_comment_to_thread(
        'alonaking/My Project/My Repo',
        12,
        5,
        'hello',
    )

    url = service._make_request.await_args.kwargs['url']
    assert url.startswith(
        'https://dev.azure.com/alonaking/My%20Project/_apis/git/repositories/My%20Repo/'
    )
    assert 'alonaking/alonaking' not in url


@pytest.mark.asyncio
async def test_work_item_comment_urls_do_not_duplicate_organization():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(return_value=({'id': 1}, {}))  # type: ignore[method-assign]

    await service.add_work_item_comment(
        'alonaking/My Project/My Repo',
        42,
        'hello',
    )

    url = service._make_request.await_args.kwargs['url']
    assert url.startswith(
        'https://dev.azure.com/alonaking/My%20Project/_apis/wit/workItems/42/'
    )
    assert 'alonaking/alonaking' not in url


@pytest.mark.asyncio
async def test_make_request_accepts_successful_empty_response_body():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    request = httpx.Request('DELETE', 'https://dev.azure.com/alonaking/_apis/hooks/1')
    response = httpx.Response(204, request=request)

    with patch.object(
        service,
        'execute_request',
        new_callable=AsyncMock,
        return_value=response,
    ):
        body, headers = await service._make_request(
            url='https://dev.azure.com/alonaking/_apis/hooks/1',
            method=RequestMethod.DELETE,
        )

    assert body == {}
    assert headers == {}


@pytest.mark.asyncio
async def test_create_pr_comment_service_hook_posts_expected_payload():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=({'id': 'subscription-id'}, {})
    )

    response = await service.create_pr_comment_service_hook(
        project_id='project-id',
        webhook_url='https://app.example.com/integration/azure-devops/events',
        webhook_secret='secret',
    )

    assert response == {'id': 'subscription-id'}
    kwargs = service._make_request.await_args.kwargs
    assert kwargs['url'] == (
        'https://dev.azure.com/alonaking/_apis/hooks/subscriptions?api-version=7.1'
    )
    assert kwargs['method'] == RequestMethod.POST
    assert kwargs['params']['eventType'] == 'ms.vss-code.git-pullrequest-comment-event'
    assert kwargs['params']['resourceVersion'] == '2.0'
    assert kwargs['params']['publisherInputs'] == {'projectId': 'project-id'}
    assert kwargs['params']['consumerInputs']['url'] == (
        'https://app.example.com/integration/azure-devops/events'
    )
    assert kwargs['params']['consumerInputs']['basicAuthUsername'] == 'openhands'
    assert kwargs['params']['consumerInputs']['basicAuthPassword'] == 'secret'


@pytest.mark.asyncio
async def test_create_work_item_comment_service_hook_posts_expected_payload():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=({'id': 'subscription-id'}, {})
    )

    response = await service.create_work_item_comment_service_hook(
        project_id='project-id',
        webhook_url='https://app.example.com/integration/azure-devops/events',
        webhook_secret='secret',
    )

    assert response == {'id': 'subscription-id'}
    kwargs = service._make_request.await_args.kwargs
    assert kwargs['url'] == (
        'https://dev.azure.com/alonaking/_apis/hooks/subscriptions?api-version=7.1'
    )
    assert kwargs['method'] == RequestMethod.POST
    assert kwargs['params']['eventType'] == 'workitem.commented'
    assert kwargs['params']['resourceVersion'] == '1.0'
    assert kwargs['params']['publisherInputs'] == {'projectId': 'project-id'}
    assert kwargs['params']['consumerInputs']['basicAuthUsername'] == 'openhands'
    assert kwargs['params']['consumerInputs']['basicAuthPassword'] == 'secret'
