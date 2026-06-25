from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from integrations.azure_devops.azure_devops_view import (
    AzureDevOpsFactory,
    AzureDevOpsPRComment,
    AzureDevOpsWorkItemComment,
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
                            'name': 'Repo',
                            'project': {'name': 'Project'},
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


@pytest.mark.asyncio
async def test_factory_creates_work_item_comment_view():
    view = await AzureDevOpsFactory.create_azure_devops_view_from_payload(
        _make_work_item_message(),
        keycloak_user_id='kc-alice',
    )

    assert isinstance(view, AzureDevOpsWorkItemComment)
    assert view.full_repo_name == 'alonaking/Project/Project'
    assert view.issue_number == 42
    assert view.comment_body == '@openhands please fix work item'
    assert view.user_info.user_id == 'ado-revised-user-id'
    assert view.user_info.username == 'Alice Revised'


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


def test_actor_email_extracts_email_from_unique_name_or_display_name():
    assert actor_email({'uniqueName': 'alice@example.com'}) == 'alice@example.com'
    assert (
        actor_email({'displayName': 'Alice Example <alice@example.com>'})
        == 'alice@example.com'
    )
    assert actor_email({'displayName': 'Alice Example'}) == ''
