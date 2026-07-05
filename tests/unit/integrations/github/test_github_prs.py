"""Tests for GitHubPRsMixin.create_pr documented defaults."""

import inspect
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.github.github_service import GitHubService


def test_create_pr_draft_default_matches_docstring():
    # The docstring states draft defaults to True; keep signature and prose aligned.
    sig = inspect.signature(GitHubService.create_pr)
    assert sig.parameters['draft'].default is True
    # title is required (no default); the docstring must not call it optional.
    assert sig.parameters['title'].default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_create_pr_defaults_to_draft_when_not_specified():
    service = GitHubService(token=SecretStr('t'))
    mock_response = {'number': 1, 'html_url': 'https://github.com/o/r/pull/1'}

    with patch.object(
        service, '_make_request', return_value=(mock_response, {})
    ) as mock_req:
        await service.create_pr('owner/repo', 'feature', 'main', 'My PR')

    payload = mock_req.call_args.kwargs['params']
    assert payload['draft'] is True
