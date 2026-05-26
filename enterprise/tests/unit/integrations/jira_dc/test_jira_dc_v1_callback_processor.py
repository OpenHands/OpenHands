from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.jira_dc.jira_dc_v1_callback_processor import (
    JiraDcV1CallbackProcessor,
)


@pytest.mark.asyncio
@patch('integrations.jira_dc.jira_dc_v1_callback_processor.httpx.AsyncClient')
@patch('integrations.jira_dc.jira_dc_v1_callback_processor.TokenManager')
@patch(
    'integrations.jira_dc.jira_dc_v1_callback_processor.JiraDcIntegrationStore.get_instance'
)
async def test_post_summary_resolves_service_account_at_callback_time(
    mock_store_get_instance,
    mock_token_manager_cls,
    mock_async_client,
    sample_jira_dc_workspace,
):
    store = MagicMock()
    store.get_workspace_by_name = AsyncMock(return_value=sample_jira_dc_workspace)
    mock_store_get_instance.return_value = store

    token_manager = MagicMock()
    token_manager.decrypt_text.return_value = 'runtime-pat'
    mock_token_manager_cls.return_value = token_manager

    response = MagicMock()
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.post.return_value = response
    mock_async_client.return_value.__aenter__.return_value = client

    processor = JiraDcV1CallbackProcessor(
        issue_key='PROJ-123',
        workspace_name='jira.company.com',
        base_api_url='https://jira.company.com',
    )

    await processor._post_summary_to_jira_dc('Summary')

    store.get_workspace_by_name.assert_awaited_once_with('jira.company.com')
    token_manager.decrypt_text.assert_called_once_with('encrypted_api_key')
    client.post.assert_awaited_once()
    _, kwargs = client.post.call_args
    assert kwargs['headers'] == {'Authorization': 'Bearer runtime-pat'}
    assert kwargs['json']['body'].startswith('OpenHands resolved this issue:')
