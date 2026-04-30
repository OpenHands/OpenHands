"""Tests for BitbucketDCManager.receive_message and send_message dispatch."""

from unittest.mock import AsyncMock, patch

import pytest
from integrations.bitbucket_data_center.bitbucket_dc_manager import BitbucketDCManager
from integrations.bitbucket_data_center.bitbucket_dc_view import (
    BitbucketDCInlinePRComment,
    BitbucketDCPRComment,
)
from integrations.models import Message, SourceType
from integrations.types import UserData


def _comment_message(*, body: str = 'Hey @openhands fix') -> Message:
    return Message(
        source=SourceType.BITBUCKET_DATA_CENTER,
        message={
            'payload': {
                'actor': {
                    'id': 1001,
                    'name': 'alice',
                    'slug': 'alice',
                    'displayName': 'Alice',
                },
                'pullRequest': {
                    'id': 7,
                    'fromRef': {
                        'displayId': 'feature/x',
                        'id': 'refs/heads/feature/x',
                    },
                    'toRef': {
                        'repository': {
                            'slug': 'myrepo',
                            'public': False,
                            'project': {'key': 'PROJ'},
                        }
                    },
                },
                'comment': {'id': 99, 'text': body},
            },
            'event_key': 'pr:comment:added',
            'installation_id': 'PROJ/myrepo',
        },
    )


def _pr_comment_view(parent_id: int | None = 42) -> BitbucketDCPRComment:
    return BitbucketDCPRComment(
        installation_id='PROJ/myrepo',
        issue_number=7,
        project_key='PROJ',
        repo_slug='myrepo',
        full_repo_name='PROJ/myrepo',
        is_public_repo=False,
        user_info=UserData(
            user_id='alice', username='Alice', keycloak_user_id='kc-installer'
        ),
        raw_payload=_comment_message(),
        conversation_id='',
        should_extract=True,
        send_summary_instruction=True,
        title='',
        description='',
        previous_comments=[],
        branch_name='feature/x',
        comment_body='Hey @openhands fix',
        parent_comment_id=parent_id,
    )


def _inline_view() -> BitbucketDCInlinePRComment:
    return BitbucketDCInlinePRComment(
        installation_id='PROJ/myrepo',
        issue_number=7,
        project_key='PROJ',
        repo_slug='myrepo',
        full_repo_name='PROJ/myrepo',
        is_public_repo=False,
        user_info=UserData(
            user_id='alice', username='Alice', keycloak_user_id='kc-installer'
        ),
        raw_payload=_comment_message(),
        conversation_id='',
        should_extract=True,
        send_summary_instruction=True,
        title='',
        description='',
        previous_comments=[],
        branch_name='feature/x',
        comment_body='@openhands rename',
        parent_comment_id=None,
        file_location='src/x.py',
        line_number=12,
        line_type='ADDED',
        file_type='TO',
    )


@pytest.mark.asyncio
async def test_receive_message_dispatches_when_commenter_has_write_access():
    manager = BitbucketDCManager(AsyncMock())

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
    ), patch.object(
        manager, '_commenter_has_write_access', return_value=True
    ), patch.object(manager, 'start_job', new=AsyncMock()) as mock_start:
        await manager.receive_message(_comment_message())

    mock_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_message_skips_when_commenter_lacks_write_access():
    manager = BitbucketDCManager(AsyncMock())

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
    ), patch.object(
        manager, '_commenter_has_write_access', return_value=False
    ), patch.object(manager, 'start_job', new=AsyncMock()) as mock_start:
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_receive_message_skips_when_no_installer_recorded_for_repo():
    manager = BitbucketDCManager(AsyncMock())

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value=None
    ), patch.object(manager, 'start_job', new=AsyncMock()) as mock_start:
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_replies_inline_with_anchor_for_inline_view():
    manager = BitbucketDCManager(AsyncMock())
    fake_service = AsyncMock()
    with patch(
        'integrations.bitbucket_data_center.bitbucket_dc_service.SaaSBitbucketDCService',
        return_value=fake_service,
    ):
        await manager.send_message('Done', _inline_view())

    kwargs = fake_service.reply_to_pr_comment.call_args.kwargs
    assert kwargs['anchor'] == {
        'path': 'src/x.py',
        'line': 12,
        'lineType': 'ADDED',
        'fileType': 'TO',
    }


@pytest.mark.asyncio
async def test_send_message_replies_via_parent_id_for_pr_comment_view():
    manager = BitbucketDCManager(AsyncMock())
    fake_service = AsyncMock()
    with patch(
        'integrations.bitbucket_data_center.bitbucket_dc_service.SaaSBitbucketDCService',
        return_value=fake_service,
    ):
        await manager.send_message('I am on it!', _pr_comment_view(parent_id=42))

    kwargs = fake_service.reply_to_pr_comment.call_args.kwargs
    assert kwargs['parent_comment_id'] == 42
    assert 'anchor' not in kwargs


def test_confirm_incoming_source_type_raises_on_wrong_source():
    manager = BitbucketDCManager(AsyncMock())
    with pytest.raises(ValueError):
        manager._confirm_incoming_source_type(
            Message(source=SourceType.BITBUCKET, message={})
        )
