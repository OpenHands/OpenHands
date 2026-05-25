"""Tests for the Jira DC view factory's conversation-creation strategy."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from integrations.jira_dc.jira_dc_view import (
    JiraDcExistingConversationView,
    JiraDcFactory,
    JiraDcNewConversationView,
)

from openhands.app_server.integrations.service_types import ProviderType, Repository


@pytest.mark.asyncio
async def test_factory_always_creates_new_conversation(
    sample_job_context,
    sample_user_auth,
    sample_jira_dc_user,
    sample_jira_dc_workspace,
    jira_dc_conversation,
):
    """Every @openhands mention starts a fresh conversation (matches GitHub/BBDC).

    JDC used to reuse the existing conversation for (issue, user), but that path
    sends the message into a possibly-recycled sandbox and 404s ("Sorry, there was
    an unexpected error starting the job."). The factory must always return a
    JiraDcNewConversationView and must never consult the conversation-reuse lookup,
    even when a prior conversation exists for this (issue, user).
    """
    with patch('integrations.jira_dc.jira_dc_view.integration_store') as mock_store:
        # A prior conversation exists -- the old code would have reused it.
        mock_store.get_user_conversations_by_issue_id = AsyncMock(
            return_value=jira_dc_conversation
        )

        view = await JiraDcFactory.create_jira_dc_view_from_payload(
            job_context=sample_job_context,
            saas_user_auth=sample_user_auth,
            jira_dc_user=sample_jira_dc_user,
            jira_dc_workspace=sample_jira_dc_workspace,
        )

    assert isinstance(view, JiraDcNewConversationView)
    assert not isinstance(view, JiraDcExistingConversationView)
    # The reuse lookup must not be consulted at all.
    mock_store.get_user_conversations_by_issue_id.assert_not_called()
    # A fresh view starts with no conversation id (assigned at creation time).
    assert view.conversation_id == ''
    assert view.selected_repo is None


@pytest.mark.asyncio
async def test_new_conversation_resolves_org_from_selected_repo_claim(
    new_conversation_view,
):
    """Jira DC @mentions use the shared repo-claim org routing path."""
    resolved_org_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    new_conversation_view.saas_user_auth.get_provider_tokens.return_value = {
        ProviderType.GITHUB: MagicMock()
    }
    repository = Repository(
        id='1',
        full_name='company/repo1',
        stargazers_count=0,
        git_provider=ProviderType.GITHUB,
        is_public=False,
    )

    with (
        patch('integrations.jira_dc.jira_dc_view.ProviderHandler') as handler_cls,
        patch(
            'integrations.jira_dc.jira_dc_view.resolve_org_for_repo',
            new=AsyncMock(return_value=resolved_org_id),
        ) as resolve_org,
    ):
        handler = MagicMock()
        handler.verify_repo_provider = AsyncMock(return_value=repository)
        handler_cls.return_value = handler

        result = await new_conversation_view._get_resolved_org_id()

    assert result == resolved_org_id
    resolve_org.assert_awaited_once_with(
        provider=ProviderType.GITHUB.value,
        full_repo_name='company/repo1',
        keycloak_user_id='test_keycloak_id',
    )
