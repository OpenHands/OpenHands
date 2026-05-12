"""Tests for the Bitbucket Cloud resolver factory and view dispatch."""

import pytest
from integrations.bitbucket.bitbucket_view import (
    BitbucketFactory,
    BitbucketInlinePRComment,
    BitbucketPRComment,
)
from integrations.models import Message, SourceType


def _make_message(
    *,
    body: str,
    inline: dict | None = None,
    event_key: str = 'pullrequest:comment_created',
    parent_id: int | None = None,
) -> Message:
    payload: dict = {
        'actor': {'uuid': '{abc}', 'display_name': 'Alice', 'nickname': 'alice'},
        'repository': {'full_name': 'ws/repo', 'is_private': True},
        'pullrequest': {
            'id': 7,
            'source': {'branch': {'name': 'feature/x'}},
        },
        'comment': {
            'id': 99,
            'content': {'raw': body},
        },
    }
    if inline is not None:
        payload['comment']['inline'] = inline
    if parent_id is not None:
        payload['comment']['parent'] = {'id': parent_id}

    return Message(
        source=SourceType.BITBUCKET,
        message={
            'payload': payload,
            'event_key': event_key,
            'installation_id': 'install-uuid',
        },
    )


@pytest.mark.asyncio
async def test_factory_creates_pr_comment_view_for_top_level_mention() -> None:
    msg = _make_message(body='Hey @openhands please fix typo', parent_id=42)

    view = await BitbucketFactory.create_bitbucket_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert isinstance(view, BitbucketPRComment)
    assert not isinstance(view, BitbucketInlinePRComment)
    assert view.parent_comment_id == 42
    assert view.full_repo_name == 'ws/repo'
    assert view.branch_name == 'feature/x'


@pytest.mark.asyncio
async def test_factory_creates_inline_pr_comment_view_when_inline_present() -> None:
    msg = _make_message(
        body='@openhands rename this',
        inline={'path': 'src/x.py', 'to': 12},
    )

    view = await BitbucketFactory.create_bitbucket_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert isinstance(view, BitbucketInlinePRComment)
    assert view.file_location == 'src/x.py'
    assert view.line_number == 12


def test_is_pr_comment_returns_false_when_mention_absent() -> None:
    msg = _make_message(body='lgtm, ship it')
    assert BitbucketFactory.is_pr_comment(msg) is False


def test_is_pr_comment_returns_false_for_unknown_event_key() -> None:
    msg = _make_message(body='@openhands fix', event_key='repo:push')
    assert BitbucketFactory.is_pr_comment(msg) is False


@pytest.mark.asyncio
async def test_factory_assigns_installer_keycloak_user_id_to_user_info() -> None:
    msg = _make_message(body='@openhands fix')

    view = await BitbucketFactory.create_bitbucket_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert view.user_info.keycloak_user_id == 'kc-installer'


@pytest.mark.asyncio
async def test_factory_records_actor_account_id_when_present() -> None:
    msg = _make_message(body='@openhands fix')
    msg.message['payload']['actor']['account_id'] = '712020:bdadedc7'
    msg.message['payload']['actor']['uuid'] = '{9339e48a-a6ce-4791-99f2}'

    view = await BitbucketFactory.create_bitbucket_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert view.user_info.user_id == '712020:bdadedc7'


@pytest.mark.asyncio
async def test_factory_strips_braces_from_uuid_when_account_id_missing() -> None:
    msg = _make_message(body='@openhands fix')
    msg.message['payload']['actor'].pop('account_id', None)
    msg.message['payload']['actor']['uuid'] = '{9339e48a-a6ce-4791-99f2}'

    view = await BitbucketFactory.create_bitbucket_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert view.user_info.user_id == '9339e48a-a6ce-4791-99f2'
