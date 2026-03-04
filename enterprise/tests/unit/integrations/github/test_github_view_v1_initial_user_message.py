from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from integrations.github.github_view import (
    GithubInlinePRComment,
    GithubIssueComment,
    GithubPRComment,
)
from integrations.types import UserData
from jinja2 import Environment, FileSystemLoader

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTaskStatus,
)
from openhands.storage.data_models.conversation_metadata import ConversationMetadata


@pytest.fixture
def jinja_env() -> Environment:
    repo_root = Path(__file__).resolve().parents[5]
    return Environment(
        loader=FileSystemLoader(
            str(repo_root / 'openhands/integrations/templates/resolver/github')
        )
    )


@asynccontextmanager
async def _fake_app_conversation_service_ctx(fake_service):
    yield fake_service


class _FakeAppConversationService:
    def __init__(self):
        self.requests = []

    async def start_app_conversation(self, request):
        self.requests.append(request)
        yield MagicMock(status=AppConversationStartTaskStatus.READY, detail=None)


def _build_conversation_metadata() -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=str(uuid4()),
        selected_repository='test-owner/test-repo',
    )


def _build_user_data() -> UserData:
    return UserData(user_id=1, username='test-user', keycloak_user_id='kc-user')


@pytest.mark.asyncio
class TestGithubViewV1InitialUserMessage:
    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_issue_comment_v1_injects_context_into_initial_user_message(
        self,
        mock_get_app_conversation_service,
        jinja_env,
    ):
        view = GithubIssueComment(
            installation_id=123,
            issue_number=42,
            full_repo_name='test-owner/test-repo',
            is_public_repo=False,
            user_info=_build_user_data(),
            raw_payload=MagicMock(),
            conversation_id='conv',
            uuid=None,
            should_extract=False,
            send_summary_instruction=False,
            title='ignored',
            description='ignored',
            previous_comments=[],
            v1_enabled=True,
            comment_body='please fix this',
            comment_id=999,
        )

        async def _load_context():
            view.title = 'Issue title'
            view.description = 'Issue body'
            view.previous_comments = [MagicMock(author='alice', body='old comment 1')]

        view._load_resolver_context = AsyncMock(side_effect=_load_context)  # type: ignore[method-assign]

        fake_service = _FakeAppConversationService()
        mock_get_app_conversation_service.return_value = (
            _fake_app_conversation_service_ctx(fake_service)
        )

        await view._create_v1_conversation(
            jinja_env=jinja_env,
            saas_user_auth=MagicMock(),
            conversation_metadata=_build_conversation_metadata(),
        )

        assert len(fake_service.requests) == 1
        req = fake_service.requests[0]
        assert req.system_message_suffix is None

        text = req.initial_message.content[0].text
        assert 'Issue title' in text
        assert 'Issue body' in text
        assert 'please fix this' in text
        assert 'old comment 1' in text

    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_pr_comment_v1_injects_context_and_comment_into_initial_user_message(
        self,
        mock_get_app_conversation_service,
        jinja_env,
    ):
        view = GithubPRComment(
            installation_id=123,
            issue_number=7,
            full_repo_name='test-owner/test-repo',
            is_public_repo=False,
            user_info=_build_user_data(),
            raw_payload=MagicMock(),
            conversation_id='conv',
            uuid=None,
            should_extract=False,
            send_summary_instruction=False,
            title='ignored',
            description='ignored',
            previous_comments=[],
            v1_enabled=True,
            comment_body='nit: rename variable',
            comment_id=1001,
            branch_name='feature-branch',
        )

        async def _load_context():
            view.title = 'PR title'
            view.description = 'PR body'
            view.previous_comments = [
                MagicMock(author='bob', created_at='2026-01-01', body='old thread')
            ]

        view._load_resolver_context = AsyncMock(side_effect=_load_context)  # type: ignore[method-assign]

        fake_service = _FakeAppConversationService()
        mock_get_app_conversation_service.return_value = (
            _fake_app_conversation_service_ctx(fake_service)
        )

        await view._create_v1_conversation(
            jinja_env=jinja_env,
            saas_user_auth=MagicMock(),
            conversation_metadata=_build_conversation_metadata(),
        )

        assert len(fake_service.requests) == 1
        req = fake_service.requests[0]
        assert req.system_message_suffix is None

        text = req.initial_message.content[0].text
        assert 'feature-branch' in text
        assert 'PR title' in text
        assert 'PR body' in text
        assert 'nit: rename variable' in text
        assert 'old thread' in text

    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_inline_pr_comment_v1_includes_file_context(
        self, mock_get_service, jinja_env
    ):
        view = GithubInlinePRComment(
            installation_id=123,
            issue_number=7,
            full_repo_name='test-owner/test-repo',
            is_public_repo=False,
            user_info=_build_user_data(),
            raw_payload=MagicMock(),
            conversation_id='conv',
            uuid=None,
            should_extract=False,
            send_summary_instruction=False,
            title='ignored',
            description='ignored',
            previous_comments=[],
            v1_enabled=True,
            comment_body='please add a null check',
            comment_id=1002,
            branch_name='feature-branch',
            file_location='src/app.py',
            line_number=123,
            comment_node_id='node',
        )

        async def _load_context():
            view.title = 'PR title'
            view.description = 'PR body'
            view.previous_comments = []

        view._load_resolver_context = AsyncMock(side_effect=_load_context)  # type: ignore[method-assign]

        fake_service = _FakeAppConversationService()
        mock_get_service.return_value = _fake_app_conversation_service_ctx(fake_service)

        await view._create_v1_conversation(
            jinja_env=jinja_env,
            saas_user_auth=MagicMock(),
            conversation_metadata=_build_conversation_metadata(),
        )

        req = fake_service.requests[0]
        assert req.system_message_suffix is None
        text = req.initial_message.content[0].text
        assert 'src/app.py' in text
        assert '123' in text
        assert 'please add a null check' in text
