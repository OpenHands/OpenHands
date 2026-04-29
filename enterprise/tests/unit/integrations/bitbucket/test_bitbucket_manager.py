"""Tests for BitbucketManager.receive_message and send_message dispatch."""

from unittest.mock import AsyncMock, patch

import pytest
from integrations.bitbucket.bitbucket_manager import BitbucketManager
from integrations.bitbucket.bitbucket_view import (
    BitbucketInlinePRComment,
    BitbucketPRComment,
)
from integrations.models import Message, SourceType
from integrations.types import UserData


def _comment_message(*, body: str = 'Hey @openhands fix') -> Message:
    return Message(
        source=SourceType.BITBUCKET,
        message={
            'payload': {
                'actor': {
                    'account_id': '712020:bdadedc7',
                    'uuid': '{abc}',
                    'display_name': 'Alice',
                    'nickname': 'alice',
                },
                'repository': {'full_name': 'ws/repo', 'is_private': True},
                'pullrequest': {
                    'id': 7,
                    'source': {'branch': {'name': 'feature/x'}},
                },
                'comment': {'id': 99, 'content': {'raw': body}},
            },
            'event_key': 'pullrequest:comment_created',
            'installation_id': 'install-uuid',
        },
    )


def _pr_comment_view(parent_id: int | None = 42) -> BitbucketPRComment:
    return BitbucketPRComment(
        installation_id='install-uuid',
        issue_number=7,
        workspace='ws',
        repo_slug='repo',
        full_repo_name='ws/repo',
        is_public_repo=False,
        user_info=UserData(
            user_id='712020:bdadedc7', username='alice', keycloak_user_id='kc-installer'
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


def _inline_view() -> BitbucketInlinePRComment:
    return BitbucketInlinePRComment(
        installation_id='install-uuid',
        issue_number=7,
        workspace='ws',
        repo_slug='repo',
        full_repo_name='ws/repo',
        is_public_repo=False,
        user_info=UserData(
            user_id='712020:bdadedc7', username='alice', keycloak_user_id='kc-installer'
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
    )


@pytest.mark.asyncio
async def test_receive_message_dispatches_when_commenter_has_write_access() -> None:
    manager = BitbucketManager(AsyncMock())

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
    ), patch.object(
        manager, '_commenter_has_write_access', return_value=True
    ), patch.object(manager, 'start_job', new=AsyncMock()) as mock_start:
        await manager.receive_message(_comment_message())

    mock_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_message_skips_when_commenter_lacks_write_access() -> None:
    manager = BitbucketManager(AsyncMock())

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
    ), patch.object(
        manager, '_commenter_has_write_access', return_value=False
    ), patch.object(manager, 'start_job', new=AsyncMock()) as mock_start:
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_receive_message_skips_when_no_installer_recorded_for_webhook() -> None:
    manager = BitbucketManager(AsyncMock())

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value=None
    ), patch.object(manager, 'start_job', new=AsyncMock()) as mock_start:
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_commenter_has_write_access_uses_installer_token_and_actor_account_id() -> (
    None
):
    manager = BitbucketManager(AsyncMock())
    fake_service = AsyncMock()
    fake_service.user_has_write_access_for = AsyncMock(return_value=True)

    with patch(
        'integrations.bitbucket.bitbucket_service.SaaSBitBucketService',
        return_value=fake_service,
    ) as mock_cls:
        result = await manager._commenter_has_write_access(
            _comment_message(), installer_user_id='kc-installer'
        )

    assert result is True
    assert mock_cls.call_args.kwargs == {'external_auth_id': 'kc-installer'}
    fake_service.user_has_write_access_for.assert_awaited_once_with(
        'ws', 'repo', '712020:bdadedc7'
    )


@pytest.mark.asyncio
async def test_send_message_replies_via_parent_id_for_pr_comment_view() -> None:
    manager = BitbucketManager(AsyncMock())
    fake_service = AsyncMock()
    with patch(
        'integrations.bitbucket.bitbucket_service.SaaSBitBucketService',
        return_value=fake_service,
    ):
        await manager.send_message('I am on it!', _pr_comment_view(parent_id=42))

    fake_service.reply_to_pr_comment.assert_awaited_once()
    kwargs = fake_service.reply_to_pr_comment.call_args.kwargs
    assert kwargs['parent_comment_id'] == 42
    assert 'inline' not in kwargs


@pytest.mark.asyncio
async def test_send_message_replies_inline_for_inline_view() -> None:
    manager = BitbucketManager(AsyncMock())
    fake_service = AsyncMock()
    with patch(
        'integrations.bitbucket.bitbucket_service.SaaSBitBucketService',
        return_value=fake_service,
    ):
        await manager.send_message('Done', _inline_view())

    kwargs = fake_service.reply_to_pr_comment.call_args.kwargs
    assert kwargs['inline'] == {'path': 'src/x.py', 'to': 12}


def test_confirm_incoming_source_type_raises_on_wrong_source() -> None:
    manager = BitbucketManager(AsyncMock())
    with pytest.raises(ValueError):
        manager._confirm_incoming_source_type(
            Message(source=SourceType.GITHUB, message={})
        )
