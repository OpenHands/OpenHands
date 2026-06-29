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
        comment_id=99,
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
        comment_id=99,
        comment_body='@openhands rename',
        parent_comment_id=None,
        file_location='src/x.py',
        line_number=12,
        line_type='ADDED',
        file_type='TO',
    )


@pytest.mark.asyncio
async def test_receive_message_runs_job_as_mentioner_when_linked_in_keycloak():
    """Use the linked mentioner as the resolver user.

    When the @-mentioning user has an OHE account, the view's
    ``user_info.keycloak_user_id`` is the mentioner and the installer's id is
    carried alongside on ``installer_keycloak_user_id``.
    """
    token_manager = AsyncMock()
    token_manager.get_user_id_from_idp_user_id = AsyncMock(return_value='kc-alice')
    manager = BitbucketDCManager(token_manager)

    captured: dict = {}

    async def fake_start_job(view):
        captured['view'] = view

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_USERNAME',
            'openhands-bot',
        ),
        patch.object(
            manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
        ),
        patch.object(manager, '_commenter_has_write_access', return_value=True),
        patch.object(manager, '_add_eyes_reaction', new=AsyncMock()),
        patch.object(manager, 'start_job', new=fake_start_job),
    ):
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
async def test_receive_message_asks_unenrolled_mentioner_to_sign_up():
    """Ask unenrolled mentioners to sign up.

    A mentioner with no OHE account is not run as the installer. We mirror the
    GitHub manager: refuse the job and reply asking them to sign up, so every
    job runs as (and is billed to) the actual requester.
    """
    token_manager = AsyncMock()
    token_manager.get_user_id_from_idp_user_id = AsyncMock(return_value=None)
    manager = BitbucketDCManager(token_manager)

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_USERNAME',
            'openhands-bot',
        ),
        patch.object(
            manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
        ),
        patch.object(manager, '_commenter_has_write_access', return_value=True),
        patch.object(manager, 'start_job', new=AsyncMock()) as mock_start,
        patch.object(
            manager, '_send_user_not_found_message', new=AsyncMock()
        ) as mock_not_found,
    ):
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()
    mock_not_found.assert_awaited_once()
    # Posted under the installer's token, mentioning the actual commenter slug.
    assert mock_not_found.await_args.args[1] == 'kc-installer'
    assert mock_not_found.await_args.args[2] == 'alice'


@pytest.mark.asyncio
async def test_receive_message_drops_event_when_keycloak_lookup_raises():
    """Drop events when mentioner lookup fails.

    A transient Keycloak error leaves enrollment status unknown. We drop the
    event rather than guess: neither silently running as the installer nor
    wrongly telling a possibly enrolled user to sign up.
    """
    token_manager = AsyncMock()
    token_manager.get_user_id_from_idp_user_id = AsyncMock(
        side_effect=RuntimeError('keycloak unreachable')
    )
    manager = BitbucketDCManager(token_manager)

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_USERNAME',
            'openhands-bot',
        ),
        patch.object(
            manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
        ),
        patch.object(manager, '_commenter_has_write_access', return_value=True),
        patch.object(manager, 'start_job', new=AsyncMock()) as mock_start,
        patch.object(
            manager, '_send_user_not_found_message', new=AsyncMock()
        ) as mock_not_found,
    ):
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()
    mock_not_found.assert_not_called()


@pytest.mark.asyncio
async def test_receive_message_skips_when_bot_token_unset():
    """Drop the event when no bot token is configured.

    Without the bot PAT there is no safe identity to post results, so the gate
    drops the event before any installer/permission/lookup work.
    """
    token_manager = AsyncMock()
    manager = BitbucketDCManager(token_manager)

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            '',
        ),
        patch.object(
            manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
        ),
        patch.object(manager, '_commenter_has_write_access', return_value=True),
        patch.object(manager, 'start_job', new=AsyncMock()) as mock_start,
        patch.object(
            manager, '_send_user_not_found_message', new=AsyncMock()
        ) as mock_not_found,
    ):
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()
    mock_not_found.assert_not_called()
    token_manager.get_user_id_from_idp_user_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_receive_message_skips_when_bot_username_unset():
    """Drop the event when the bot username is missing even if the token is set.

    Token + username are one required unit: without the username the self-author
    guard can't run, so a bot reply could re-trigger a job -- so gate on both.
    """
    token_manager = AsyncMock()
    manager = BitbucketDCManager(token_manager)

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_USERNAME',
            '',
        ),
        patch.object(
            manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
        ),
        patch.object(manager, '_commenter_has_write_access', return_value=True),
        patch.object(manager, 'start_job', new=AsyncMock()) as mock_start,
        patch.object(
            manager, '_send_user_not_found_message', new=AsyncMock()
        ) as mock_not_found,
    ):
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()
    mock_not_found.assert_not_called()
    token_manager.get_user_id_from_idp_user_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_slug', ['openhands', 'OpenHands'])
async def test_receive_message_skips_event_authored_by_bot(actor_slug):
    """Skip events the bot account authored (case-insensitive).

    The agent's reply is posted via the bot PAT and can contain "@openhands";
    the self-author guard stops it from re-triggering a job.
    """
    token_manager = AsyncMock()
    manager = BitbucketDCManager(token_manager)
    message = _comment_message()
    message.message['payload']['actor']['slug'] = actor_slug

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_USERNAME',
            'openhands',
        ),
        patch.object(
            manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
        ),
        patch.object(manager, '_commenter_has_write_access', return_value=True),
        patch.object(manager, 'start_job', new=AsyncMock()) as mock_start,
    ):
        await manager.receive_message(message)

    mock_start.assert_not_called()
    token_manager.get_user_id_from_idp_user_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_user_not_found_message_replies_as_installer():
    """Post the sign-up reply as the installer.

    The sign-up reply is built from the payload with the installer as the
    posting identity and carries the user-not-found copy.
    """
    manager = BitbucketDCManager(AsyncMock())
    sentinel_view = object()

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BitbucketDCFactory.create_bitbucket_dc_view_from_payload',
            new=AsyncMock(return_value=sentinel_view),
        ) as mock_factory,
        patch.object(manager, 'send_message', new=AsyncMock()) as mock_send,
    ):
        await manager._send_user_not_found_message(
            _comment_message(), 'kc-installer', 'alice'
        )

    mock_factory.assert_awaited_once()
    assert mock_factory.await_args.kwargs['keycloak_user_id'] == 'kc-installer'
    assert (
        mock_factory.await_args.kwargs['installer_keycloak_user_id'] == 'kc-installer'
    )
    mock_send.assert_awaited_once()
    body, view = mock_send.await_args.args
    assert view is sentinel_view
    assert 'sign up' in body.lower()
    assert '@alice' in body


@pytest.mark.asyncio
async def test_receive_message_skips_when_commenter_lacks_write_access():
    manager = BitbucketDCManager(AsyncMock())

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_USERNAME',
            'openhands-bot',
        ),
        patch.object(
            manager.webhook_store, 'get_webhook_user_id', return_value='kc-installer'
        ),
        patch.object(manager, '_commenter_has_write_access', return_value=False),
        patch.object(manager, 'start_job', new=AsyncMock()) as mock_start,
    ):
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_receive_message_skips_when_no_installer_recorded_for_repo():
    manager = BitbucketDCManager(AsyncMock())

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_manager.BITBUCKET_DATA_CENTER_BOT_USERNAME',
            'openhands-bot',
        ),
        patch.object(manager.webhook_store, 'get_webhook_user_id', return_value=None),
        patch.object(manager, 'start_job', new=AsyncMock()) as mock_start,
    ):
        await manager.receive_message(_comment_message())

    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_replies_inline_with_anchor_for_inline_view():
    manager = BitbucketDCManager(AsyncMock())
    fake_service = AsyncMock()
    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.SaaSBitbucketDCService',
            return_value=fake_service,
        ),
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
    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.SaaSBitbucketDCService',
            return_value=fake_service,
        ),
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
async def test_send_message_posts_as_bot_not_mentioner():
    """Post replies as the bot, never the mentioner.

    send_message builds the posting service via ``bitbucket_dc_posting_service``,
    which auths as the bot account. The service is constructed with no per-user
    ``external_auth_id`` and the raw bot token is set directly.
    """
    manager = BitbucketDCManager(AsyncMock())
    fake_service = AsyncMock()

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat-123',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.SaaSBitbucketDCService',
            return_value=fake_service,
        ) as service_cls,
    ):
        await manager.send_message('Done', _pr_comment_view())

    service_cls.assert_called_once_with()
    assert 'external_auth_id' not in service_cls.call_args.kwargs
    assert fake_service.token.get_secret_value() == 'bot-pat-123'


def test_posting_service_uses_bot_token_when_set():
    """Use the bot token when it is configured.

    ``_posting_service`` takes no args and always auths as the bot, never the
    per-user token.
    """
    manager = BitbucketDCManager(AsyncMock())

    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat-123',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.SaaSBitbucketDCService'
        ) as service_cls,
    ):
        svc = manager._posting_service()

    # Built with no per-user auth; the raw bot token is set directly so the
    # service sends Bearer. NOT via external_auth_id, and NOT via the token=
    # constructor arg (which would downgrade it to x-token-auth HTTP Basic,
    # rejected by Bitbucket Data Center).
    service_cls.assert_called_once_with()
    assert 'external_auth_id' not in service_cls.call_args.kwargs
    assert svc.token.get_secret_value() == 'bot-pat-123'


def test_posting_service_raises_when_bot_token_unset():
    """Raise instead of falling back to a user token.

    Without a bot token there is no safe posting identity (a user-authored
    reply with "@openhands" would re-fire the webhook), so building the posting
    service must raise rather than fall back.
    """
    manager = BitbucketDCManager(AsyncMock())

    with patch(
        'integrations.bitbucket_data_center.bitbucket_dc_service_account.BITBUCKET_DATA_CENTER_BOT_TOKEN',
        '',
    ):
        with pytest.raises(RuntimeError):
            manager._posting_service()
