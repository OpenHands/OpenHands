from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from integrations.azure_devops.azure_devops_view import (
    AzureDevOpsFactory,
    AzureDevOpsPRComment,
    AzureDevOpsWorkItemComment,
    _select_project_repo,
    actor_email,
    mark_openhands_comment,
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
                / 'openhands/app_server/integrations/templates/resolver/azure_devops'
            )
        )
    )


def _make_pr_message(body: str = '@openhands please fix this') -> Message:
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
                        '_links': {
                            'threads': {
                                'href': 'https://dev.azure.com/alonaking/Project/_apis/git/repositories/repo/pullRequests/7/threads/5'
                            }
                        },
                    },
                    'pullRequest': {
                        'pullRequestId': 7,
                        'sourceRefName': 'refs/heads/feature/x',
                        'repository': {
                            'id': 'repo-1',
                            'name': 'Repo',
                            'project': {'id': 'proj-1', 'name': 'Project'},
                            'remoteUrl': 'https://dev.azure.com/alonaking/Project/_git/Repo',
                        },
                    },
                },
            },
        },
    )


def _make_work_item_message(body: str = '@openhands please fix work item') -> Message:
    return Message(
        source=SourceType.AZURE_DEVOPS,
        message={
            'event_key': 'workitem.commented',
            'payload': {
                'eventType': 'workitem.commented',
                'resourceContainers': {
                    'account': {'baseUrl': 'https://dev.azure.com/alonaking/'}
                },
                'resource': {
                    'id': 42,
                    'revisedBy': {
                        'id': 'ado-revised-user-id',
                        'displayName': 'Alice Revised',
                        'uniqueName': 'alice.revised@example.com',
                    },
                    'fields': {
                        'System.TeamProject': 'Project',
                        'System.ChangedBy': {
                            'id': 'ado-user-id',
                            'displayName': 'Alice Example',
                            'uniqueName': 'alice@example.com',
                        },
                        'System.History': body,
                    },
                    '_links': {
                        'self': {
                            'href': 'https://dev.azure.com/alonaking/Project/_apis/wit/workItems/42'
                        }
                    },
                },
            },
        },
    )


def test_is_pr_comment_requires_exact_mention():
    assert AzureDevOpsFactory.is_pr_comment(_make_pr_message()) is True
    assert AzureDevOpsFactory.is_pr_comment(_make_pr_message('hello')) is False
    assert (
        AzureDevOpsFactory.is_pr_comment(
            _make_pr_message(mark_openhands_comment('@openhands generated summary'))
        )
        is False
    )


def test_is_work_item_comment_requires_exact_mention():
    assert AzureDevOpsFactory.is_work_item_comment(_make_work_item_message()) is True
    assert (
        AzureDevOpsFactory.is_work_item_comment(_make_work_item_message('hello'))
        is False
    )
    assert (
        AzureDevOpsFactory.is_work_item_comment(
            _make_work_item_message(
                mark_openhands_comment('@openhands generated summary')
            )
        )
        is False
    )


@pytest.mark.asyncio
async def test_factory_creates_pr_comment_view():
    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_pr_message(),
        keycloak_user_id='kc-alice',
    )

    assert isinstance(view, AzureDevOpsPRComment)
    assert view.full_repo_name == 'alonaking/Project/Repo'
    assert view.issue_number == 7
    assert view.comment_id == 2
    assert view.thread_id == 5
    assert view.branch_name == 'feature/x'
    assert view.user_info.user_id == 'ado-user-id'
    assert view.user_info.keycloak_user_id == 'kc-alice'
    assert view.project_id == 'proj-1'
    assert view.repository_id == 'repo-1'


@pytest.mark.asyncio
async def test_factory_work_item_view_derives_real_repo(monkeypatch):
    # System.TeamProject ('Project') matches a repo name -> use the matched repo.
    fake_service = MagicMock()
    fake_service.get_project_repositories = AsyncMock(
        return_value=[
            {'id': 'repo-9', 'name': 'service-api', 'project': {'id': 'proj-9'}},
            {'id': 'repo-1', 'name': 'Project', 'project': {'id': 'proj-9'}},
        ]
    )
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_view.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_work_item_message(),
        keycloak_user_id='kc-alice',
    )

    assert isinstance(view, AzureDevOpsWorkItemComment)
    assert view.full_repo_name == 'alonaking/Project/Project'
    assert view.repository_id == 'repo-1'
    assert view.project_id == 'proj-9'
    assert view.issue_number == 42
    assert view.comment_body == '@openhands please fix work item'
    assert view.user_info.user_id == 'ado-revised-user-id'
    assert view.user_info.username == 'Alice Revised'


@pytest.mark.asyncio
async def test_factory_work_item_view_uses_sole_repo(monkeypatch):
    # No name match but exactly one repo -> use it.
    fake_service = MagicMock()
    fake_service.get_project_repositories = AsyncMock(
        return_value=[
            {'id': 'repo-9', 'name': 'service-api', 'project': {'id': 'proj-9'}}
        ]
    )
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_view.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_work_item_message(),
        keycloak_user_id='kc-alice',
    )

    assert isinstance(view, AzureDevOpsWorkItemComment)
    assert view.full_repo_name == 'alonaking/Project/service-api'
    assert view.repository_id == 'repo-9'


@pytest.mark.asyncio
async def test_factory_work_item_view_returns_none_when_ambiguous(monkeypatch):
    # Multiple repos, none matching the project name -> fail closed (skip).
    fake_service = MagicMock()
    fake_service.get_project_repositories = AsyncMock(
        return_value=[
            {'id': 'repo-9', 'name': 'service-api', 'project': {'id': 'proj-9'}},
            {'id': 'repo-2', 'name': 'docs', 'project': {'id': 'proj-9'}},
        ]
    )
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_view.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_work_item_message(),
        keycloak_user_id='kc-alice',
    )

    assert view is None


@pytest.mark.asyncio
async def test_factory_work_item_view_returns_none_when_no_repos(monkeypatch):
    fake_service = MagicMock()
    fake_service.get_project_repositories = AsyncMock(return_value=[])
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_view.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_work_item_message(),
        keycloak_user_id='kc-alice',
    )

    assert view is None


def test_select_project_repo_resolution():
    name_match = {'id': 'r1', 'name': 'Project'}
    other = {'id': 'r2', 'name': 'docs'}

    # Name match wins even among several repos.
    assert _select_project_repo([other, name_match], 'project') is name_match
    # Sole repo is used when nothing matches by name.
    assert _select_project_repo([other], 'Project') is other
    # Multiple repos with no name match -> None (ambiguous).
    assert _select_project_repo([other, {'id': 'r3', 'name': 'api'}], 'Project') is None
    # No repos -> None.
    assert _select_project_repo([], 'Project') is None


@pytest.mark.asyncio
async def test_pr_comment_instructions_include_actionable_comment(jinja_env):
    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_pr_message(),
        keycloak_user_id='kc-alice',
    )

    async def _load_context():
        view.title = 'PR title'
        view.description = 'PR body'
        view.previous_comments = [
            MagicMock(author='bob', created_at='2026-01-01', body='older comment')
        ]

    view._load_resolver_context = AsyncMock(side_effect=_load_context)  # type: ignore[method-assign]

    user_instructions, conversation_instructions = await view._get_instructions(
        jinja_env
    )

    assert '@openhands please fix this' in user_instructions
    assert 'PR title' in conversation_instructions
    assert 'PR body' in conversation_instructions
    assert 'older comment' in conversation_instructions


@pytest.mark.asyncio
async def test_work_item_load_resolver_context_uses_work_item_not_pr(monkeypatch):
    # Work item #42 can collide with PR #42; ensure WI context is loaded, not PR.
    fake_service = MagicMock()
    fake_service.get_project_repositories = AsyncMock(
        return_value=[{'id': 'repo-1', 'name': 'Project', 'project': {'id': 'proj-9'}}]
    )
    fake_service.get_work_item_comments = AsyncMock(
        return_value=[MagicMock(author='wi-user', body='work item comment')]
    )
    fake_service.get_work_item_title_and_body = AsyncMock(
        return_value=('Work item title', 'Work item body')
    )
    fake_service.get_issue_or_pr_comments = AsyncMock(
        return_value=[MagicMock(author='pr-user', body='pr comment')]
    )
    fake_service.get_issue_or_pr_title_and_body = AsyncMock(
        return_value=('PR title', 'PR body')
    )
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_view.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_work_item_message(),
        keycloak_user_id='kc-alice',
    )
    assert isinstance(view, AzureDevOpsWorkItemComment)

    await view._load_resolver_context()

    assert view.title == 'Work item title'
    assert view.description == 'Work item body'
    assert view.previous_comments[0].body == 'work item comment'
    fake_service.get_work_item_title_and_body.assert_awaited_once_with(42)
    fake_service.get_work_item_comments.assert_awaited_once()
    fake_service.get_issue_or_pr_comments.assert_not_called()
    fake_service.get_issue_or_pr_title_and_body.assert_not_called()


@pytest.mark.asyncio
async def test_pr_comment_load_resolver_context_uses_pr_not_work_item(monkeypatch):
    # A brand-new PR whose only comment is the @openhands mention returns an empty
    # list from get_pr_comments. The loader must NOT fall back to work-item
    # discussion in that case; it must stay on the PR code path.
    fake_service = MagicMock()
    fake_service.get_pr_comments = AsyncMock(return_value=[])  # empty — new PR
    fake_service.get_issue_or_pr_title_and_body = AsyncMock(
        return_value=('PR title', 'PR body')
    )
    fake_service.get_work_item_comments = AsyncMock(
        return_value=[MagicMock(author='wi-user', body='work item comment')]
    )
    fake_service.get_issue_or_pr_comments = AsyncMock(
        return_value=[MagicMock(author='pr-user', body='pr comment')]
    )
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_view.AzureDevOpsServiceImpl',
        lambda external_auth_id: fake_service,
    )

    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_pr_message(),
        keycloak_user_id='kc-alice',
    )
    assert isinstance(view, AzureDevOpsPRComment)

    await view._load_resolver_context()

    assert view.title == 'PR title'
    assert view.description == 'PR body'
    assert view.previous_comments == []  # empty list, not work-item fallback
    fake_service.get_pr_comments.assert_awaited_once()
    fake_service.get_issue_or_pr_title_and_body.assert_awaited_once()
    fake_service.get_work_item_comments.assert_not_called()
    fake_service.get_issue_or_pr_comments.assert_not_called()


@pytest.mark.asyncio
async def test_factory_returns_none_when_org_empty(monkeypatch):
    # No resolvable org -> skip rather than build a malformed '/project/repo'.
    monkeypatch.setattr(
        'integrations.azure_devops.azure_devops_view._extract_org',
        lambda payload: '',
    )

    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_pr_message(),
        keycloak_user_id='kc-alice',
    )

    assert view is None


@pytest.mark.asyncio
async def test_pr_view_get_branch_name_returns_branch():
    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_pr_message(),
        keycloak_user_id='kc-alice',
    )

    assert isinstance(view, AzureDevOpsPRComment)
    assert view._get_branch_name() == 'feature/x'


def test_actor_email_extracts_email_from_unique_name_or_display_name():
    assert actor_email({'uniqueName': 'alice@example.com'}) == 'alice@example.com'
    assert (
        actor_email({'displayName': 'Alice Example <alice@example.com>'})
        == 'alice@example.com'
    )
    assert actor_email({'displayName': 'Alice Example'}) == ''
