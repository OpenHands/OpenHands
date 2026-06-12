"""Tests for the Bitbucket Data Center resolver factory and view dispatch."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from integrations.bitbucket_data_center.bitbucket_dc_view import (
    BitbucketDCFactory,
    BitbucketDCInlinePRComment,
    BitbucketDCPRComment,
)
from integrations.models import Message, SourceType
from jinja2 import Environment, FileSystemLoader


@pytest.fixture
def jinja_env() -> Environment:
    repo_root = Path(__file__).resolve().parents[5]
    return Environment(
        loader=FileSystemLoader(
            str(
                repo_root
                / 'openhands/app_server/integrations/templates/resolver/bitbucket'
            )
        )
    )


def _make_message(
    *,
    body: str,
    anchor: dict | None = None,
    event_key: str = 'pr:comment:added',
    parent_id: int | None = None,
) -> Message:
    payload: dict = {
        'actor': {
            'id': 1001,
            'name': 'alice',
            'slug': 'alice',
            'displayName': 'Alice',
        },
        'pullRequest': {
            'id': 7,
            'fromRef': {'displayId': 'feature/x', 'id': 'refs/heads/feature/x'},
            'toRef': {
                'repository': {
                    'slug': 'myrepo',
                    'public': False,
                    'project': {'key': 'PROJ'},
                }
            },
        },
        'comment': {'id': 99, 'text': body},
    }
    if anchor is not None:
        payload['commentAnchor'] = anchor
    if parent_id is not None:
        payload['comment']['parent'] = {'id': parent_id}

    return Message(
        source=SourceType.BITBUCKET_DATA_CENTER,
        message={
            'payload': payload,
            'event_key': event_key,
            'installation_id': 'PROJ/myrepo',
        },
    )


@pytest.mark.asyncio
async def test_factory_creates_pr_comment_view_for_pr_comment_added_with_mention():
    msg = _make_message(body='Hey @openhands please fix typo', parent_id=42)

    view = await BitbucketDCFactory.create_bitbucket_dc_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert isinstance(view, BitbucketDCPRComment)
    assert not isinstance(view, BitbucketDCInlinePRComment)
    assert view.comment_id == 99
    assert view.parent_comment_id == 42
    assert view.full_repo_name == 'PROJ/myrepo'
    assert view.branch_name == 'feature/x'


@pytest.mark.asyncio
async def test_factory_creates_inline_view_when_anchor_present():
    msg = _make_message(
        body='@openhands rename this',
        anchor={
            'path': 'src/x.py',
            'line': 12,
            'lineType': 'ADDED',
            'fileType': 'TO',
        },
    )

    view = await BitbucketDCFactory.create_bitbucket_dc_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert isinstance(view, BitbucketDCInlinePRComment)
    assert view.comment_id == 99
    assert view.file_location == 'src/x.py'
    assert view.line_number == 12
    assert view.line_type == 'ADDED'
    assert view.file_type == 'TO'


def test_is_pr_comment_returns_false_when_mention_absent():
    msg = _make_message(body='lgtm, ship it')
    assert BitbucketDCFactory.is_pr_comment(msg) is False


def test_is_pr_comment_returns_false_for_unknown_event_key():
    msg = _make_message(body='@openhands fix', event_key='repo:refs_changed')
    assert BitbucketDCFactory.is_pr_comment(msg) is False


@pytest.mark.asyncio
async def test_factory_records_actor_slug_and_assigns_keycloak_user_id():
    msg = _make_message(body='@openhands fix')

    view = await BitbucketDCFactory.create_bitbucket_dc_view_from_payload(
        msg, keycloak_user_id='kc-installer'
    )

    assert view.user_info.user_id == 'alice'
    assert view.user_info.username == 'Alice'
    assert view.user_info.keycloak_user_id == 'kc-installer'
    # Single-arg form (no explicit installer) keeps the mentioner id on
    # both fields for backward compatibility.
    assert view.installer_keycloak_user_id == 'kc-installer'


@pytest.mark.asyncio
async def test_factory_keeps_mentioner_and_installer_distinct_when_passed():
    """Keep the mentioner and installer identities distinct.

    The mentioner runs the job; the installer's id is carried alongside on
    ``installer_keycloak_user_id`` for the bits that need elevated permissions.
    """
    msg = _make_message(body='@openhands fix')

    view = await BitbucketDCFactory.create_bitbucket_dc_view_from_payload(
        msg,
        keycloak_user_id='kc-alice',
        installer_keycloak_user_id='kc-installer',
    )

    assert view.user_info.keycloak_user_id == 'kc-alice'
    assert view.installer_keycloak_user_id == 'kc-installer'


@pytest.mark.asyncio
async def test_pr_comment_instructions_include_context_and_actionable_comment(
    jinja_env,
):
    msg = _make_message(body='@openhands please update the tests')
    view = await BitbucketDCFactory.create_bitbucket_dc_view_from_payload(
        msg, keycloak_user_id='kc-alice'
    )

    async def _load_context():
        view.title = 'PR title'
        view.description = 'PR body'
        view.previous_comments = [
            MagicMock(author='bob', created_at='2026-01-01', body='old thread')
        ]

    view._load_resolver_context = AsyncMock(side_effect=_load_context)  # type: ignore[method-assign]

    user_instructions, conversation_instructions = await view._get_instructions(
        jinja_env
    )

    assert conversation_instructions == ''
    assert 'PR title' in user_instructions
    assert 'PR body' in user_instructions
    assert '@openhands please update the tests' in user_instructions
    assert 'old thread' in user_instructions
    assert 'The comment above is the actionable request' in user_instructions
