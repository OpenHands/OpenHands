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
    assert kwargs['params']['publisherInputs'] == {}
    assert kwargs['params']['consumerInputs']['url'] == (
        'https://app.example.com/integration/azure-devops/events'
    )
    assert kwargs['params']['consumerInputs']['basicAuthUsername'] == 'openhands'
    assert kwargs['params']['consumerInputs']['basicAuthPassword'] == 'secret'


@pytest.mark.asyncio
async def test_has_contribute_access_posts_repo_token_and_returns_evaluation():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=({'evaluations': [{'value': True}]}, {})
    )

    result = await service.has_contribute_access('proj-1', 'repo-1')

    assert result is True
    kwargs = service._make_request.await_args.kwargs
    assert kwargs['url'] == (
        'https://dev.azure.com/alonaking/_apis/security/permissionevaluationbatch'
        '?api-version=7.1-preview.1'
    )
    assert kwargs['method'] == RequestMethod.POST
    evaluation = kwargs['params']['evaluations'][0]
    assert evaluation['token'] == 'repoV2/proj-1/repo-1'
    assert evaluation['permissions'] == 4


@pytest.mark.asyncio
async def test_has_contribute_access_fails_closed_on_missing_ids():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock()  # type: ignore[method-assign]

    assert await service.has_contribute_access('', 'repo-1') is False
    service._make_request.assert_not_called()


@pytest.mark.asyncio
async def test_has_contribute_access_fails_closed_on_error():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(side_effect=RuntimeError('boom'))  # type: ignore[method-assign]

    assert await service.has_contribute_access('proj-1', 'repo-1') is False


@pytest.mark.asyncio
async def test_get_project_repositories_returns_value_list():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=({'value': [{'id': 'repo-1', 'name': 'svc'}]}, {})
    )

    repos = await service.get_project_repositories('My Project')

    assert repos == [{'id': 'repo-1', 'name': 'svc'}]
    call = service._make_request.await_args
    url = call.kwargs.get('url') or call.args[0]
    assert url == (
        'https://dev.azure.com/alonaking/My%20Project/_apis/git/repositories'
        '?api-version=7.1'
    )


@pytest.mark.asyncio
async def test_create_work_item_comment_service_hook_posts_expected_payload():
    service = SaaSAzureDevOpsService(token=SecretStr('token'), base_domain='alonaking')
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=({'id': 'subscription-id'}, {})
    )

    response = await service.create_work_item_comment_service_hook(
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
    assert kwargs['params']['publisherInputs'] == {}
    assert kwargs['params']['consumerInputs']['basicAuthUsername'] == 'openhands'
    assert kwargs['params']['consumerInputs']['basicAuthPassword'] == 'secret'
