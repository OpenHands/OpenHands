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
        # ``user_info.keycloak_user_id`` is the @-mentioning user, set by
        # ``receive_message`` after the Keycloak lookup. The installer's
        # id is carried separately on the view.
        user_info=UserData(
            user_id='alice', username='Alice', keycloak_user_id='kc-alice'
        ),
        raw_payload=_comment_message(),
        conversation_id='',
        should_extract=True,
        send_summary_instruction=True,
        title='',
        description='',
        previous_comments=[],
        branch_name='feature/x',
        installer_keycloak_user_id='kc-installer',
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
            user_id='alice', username='Alice', keycloak_user_id='kc-alice'
        ),
        raw_payload=_comment_message(),
        conversation_id='',
        should_extract=True,
        send_summary_instruction=True,
        title='',
        description='',
        previous_comments=[],
        branch_name='feature/x',
        installer_keycloak_user_id='kc-installer',
        comment_body='@openhands rename',
        parent_comment_id=None,
        file_location='src/x.py',
        line_number=12,
        line_type='ADDED',
        file_type='TO',
    )


@pytest.mark.asyncio
async def test_receive_message_runs_job_as_mentioner_when_linked_in_keycloak():
    """When the @-mentioning user has an OHE account, the view's
    ``user_info.keycloak_user_id`` is the mentioner and the installer's
    id is carried alongside on ``installer_keycloak_user_id``.
    """
    token_manager = AsyncMock()
    token_manager.get_user_id_from_idp_user_id = AsyncMock(return_value='kc-alice')
    manager = BitbucketDCManager(token_manager)

    captured: dict = {}

    async def fake_start_job(view):
        captured['view'] = view

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
    ), patch.object(
        manager, '_commenter_has_write_access', return_value=True
    ), patch.object(manager, 'start_job', new=fake_start_job):
        await manager.receive_message(_comment_message())

    view = captured['view']
    assert view.user_info.keycloak_user_id == 'kc-alice'
    assert view.installer_keycloak_user_id == 'kc-installer'
    # Regression guard for the slug-vs-numeric-id bug: the mentioner must be
    # resolved by their NUMERIC BBDC id (actor['id'] == 1001), which is what
    # Keycloak's `bitbucket_data_center_id` attribute stores (the OIDC `sub`
    # claim) -- NOT the slug 'alice'. Looking up by slug never matched and
    # silently fell back to the webhook installer.
    token_manager.get_user_id_from_idp_user_id.assert_awaited_once()
    assert token_manager.get_user_id_from_idp_user_id.await_args.args[0] == '1001'


@pytest.mark.asyncio
async def test_receive_message_falls_back_to_installer_when_mentioner_has_no_account():
    """A mentioner with no OHE account leaves the view running as the
    installer, so the resolver still works for unenrolled BBDC users.
    """
    token_manager = AsyncMock()
    token_manager.get_user_id_from_idp_user_id = AsyncMock(return_value=None)
    manager = BitbucketDCManager(token_manager)

    captured: dict = {}

    async def fake_start_job(view):
        captured['view'] = view

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
    ), patch.object(
        manager, '_commenter_has_write_access', return_value=True
    ), patch.object(manager, 'start_job', new=fake_start_job):
        await manager.receive_message(_comment_message())

    view = captured['view']
    assert view.user_info.keycloak_user_id == 'kc-installer'
    assert view.installer_keycloak_user_id == 'kc-installer'


@pytest.mark.asyncio
async def test_receive_message_falls_back_to_installer_when_keycloak_lookup_raises():
    """Transient Keycloak errors must not block the resolver — fall
    back to the installer rather than dropping the event.
    """
    token_manager = AsyncMock()
    token_manager.get_user_id_from_idp_user_id = AsyncMock(
        side_effect=RuntimeError('keycloak unreachable')
    )
    manager = BitbucketDCManager(token_manager)

    captured: dict = {}

    async def fake_start_job(view):
        captured['view'] = view

    with patch.object(
        manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
    ), patch.object(
        manager, '_commenter_has_write_access', return_value=True
    ), patch.object(manager, 'start_job', new=fake_start_job):
        await manager.receive_message(_comment_message())

    view = captured['view']
    assert view.user_info.keycloak_user_id == 'kc-installer'


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


@pytest.mark.asyncio
async def test_send_message_uses_view_user_info_keycloak_id():
    """send_message constructs the BBDC service with the view's
    user_info.keycloak_user_id (the mentioner) rather than the
    installer, so replies post under the mentioner's BBDC account.
    """
    manager = BitbucketDCManager(AsyncMock())

    with patch(
        'integrations.bitbucket_data_center.bitbucket_dc_service.SaaSBitbucketDCService'
    ) as service_cls:
        service_cls.return_value = AsyncMock()
        await manager.send_message('Done', _pr_comment_view())

    service_cls.assert_called_once_with(external_auth_id='kc-alice')
