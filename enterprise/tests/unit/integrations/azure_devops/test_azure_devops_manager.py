"""Tests for AzureDevOpsManager.receive_message write-access gating."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from integrations.azure_devops.azure_devops_manager import AzureDevOpsManager
from integrations.models import Message, SourceType


def _pr_comment_message(body: str = '@openhands please fix this') -> Message:
    return Message(
        source=SourceType.AZURE_DEVOPS,
        message={
            'event_key': 'ms.vss-code.git-pullrequest-comment-event',
            'payload': {
                'eventType': 'ms.vss-code.git-pullrequest-comment-event',
                'resourceContainers': {
                    'account': {'baseUrl': 'https://dev.azure.com/alonaking/'}
                },
                'resource': {
                    'comment': {
                        'id': 2,
                        'author': {
                            'id': 'ado-user-id',
                            'displayName': 'Alice Example',
                            'uniqueName': 'alice@example.com',
                        },
                        'content': body,
                    },
                    'pullRequest': {
                        'pullRequestId': 7,
                        'sourceRefName': 'refs/heads/feature/x',
                        'repository': {
                            'id': 'repo-1',
                            'name': 'Repo',
                            'project': {'id': 'proj-1', 'name': 'Project'},
                        },
                    },
                },
            },
        },
    )


@pytest.mark.asyncio
async def test_receive_message_skips_when_commenter_lacks_write_access(monkeypatch):
    manager = AzureDevOpsManager(AsyncMock())
    manager._resolve_mentioner_keycloak_id = AsyncMock(return_value='kc-alice')  # type: ignore[method-assign]
    manager.start_job = AsyncMock()  # type: ignore[method-assign]

    fake_service = MagicMock()
    fake_service.has_contribute_access = AsyncMock(return_value=False)
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_manager.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    await manager.receive_message(_pr_comment_message())

    fake_service.has_contribute_access.assert_awaited_once_with('proj-1', 'repo-1')
    manager.start_job.assert_not_called()


@pytest.mark.asyncio
async def test_receive_message_dispatches_when_commenter_has_write_access(monkeypatch):
    manager = AzureDevOpsManager(AsyncMock())
    manager._resolve_mentioner_keycloak_id = AsyncMock(return_value='kc-alice')  # type: ignore[method-assign]
    manager.start_job = AsyncMock()  # type: ignore[method-assign]

    fake_service = MagicMock()
    fake_service.has_contribute_access = AsyncMock(return_value=True)
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_manager.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    await manager.receive_message(_pr_comment_message())

    fake_service.has_contribute_access.assert_awaited_once_with('proj-1', 'repo-1')
    manager.start_job.assert_awaited_once()
