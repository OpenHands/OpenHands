"""Tests for Forgejo integration with send_pull_request."""

from unittest.mock import MagicMock, patch

from openhands.integrations.service_types import ProviderType as ServiceProviderType
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.send_pull_request import PR_SIGNATURE, send_pull_request


@patch('openhands.resolver.send_pull_request.ServiceContextIssue')
@patch('openhands.resolver.send_pull_request.ForgejoIssueHandler')
@patch('subprocess.run')
def test_send_pull_request_forgejo(
    mock_run, mock_forgejo_handler, mock_service_context
):
    """Ensure we can build and submit a Forgejo pull request."""
    mock_run.return_value = MagicMock(returncode=0)

    handler_instance = MagicMock()
    mock_forgejo_handler.return_value = handler_instance

    service_context_instance = MagicMock()
    service_context_instance.get_branch_name.return_value = 'openhands-fix-issue-7'
    service_context_instance.branch_exists.return_value = True
    service_context_instance.get_default_branch_name.return_value = 'main'
    service_context_instance.get_clone_url.return_value = (
        'https://codeberg.org/example/repo.git'
    )
    service_context_instance.create_pull_request.return_value = {
        'html_url': 'https://codeberg.org/example/repo/pulls/42',
        'number': 42,
    }
    service_context_instance._strategy = MagicMock()
    mock_service_context.return_value = service_context_instance

    issue = Issue(
        number=7,
        title='Fix the Forgejo PR flow',
        owner='example',
        repo='repo',
        body='Details about the fix',
        created_at='2024-01-01T00:00:00Z',
        updated_at='2024-01-01T00:00:00Z',
        closed_at=None,
        head_branch='feature-branch',
        thread_ids=None,
    )

    result = send_pull_request(
        issue=issue,
        token='forgejo-token',
        username=None,
        platform=ServiceProviderType.FORGEJO,
        patch_dir='/tmp',
        pr_type='ready',
        pr_title='Fix the Forgejo PR flow',
        target_branch='main',
    )

    assert result == 'https://codeberg.org/example/repo/pulls/42'

    mock_forgejo_handler.assert_called_once_with(
        'example', 'repo', 'forgejo-token', None, 'codeberg.org'
    )
    mock_service_context.assert_called_once_with(handler_instance, None)

    expected_payload = {
        'title': 'Fix the Forgejo PR flow',
        'body': f'This pull request fixes #7.\n\n{PR_SIGNATURE}',
        'head': 'openhands-fix-issue-7',
        'base': 'main',
        'draft': False,
    }
    service_context_instance.create_pull_request.assert_called_once_with(
        expected_payload
    )

    mock_run.assert_called()
