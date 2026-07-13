import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'enterprise'))

from enterprise.server.services.org_conversation_service import OrgConversationService


def _make_metadata(**overrides):
    base = {
        'conversation_id': 'conv-1',
        'title': 'Test',
        'llm_model': 'gpt-test',
        'agent_kind': 'openhands',
        'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
        'last_updated_at': datetime(2026, 1, 2, tzinfo=timezone.utc),
        'sandbox_id': 'sandbox-1',
        'execution_status': 'running',
        'selected_repository': 'repo',
        'selected_branch': 'main',
        'git_provider': 'github',
        'trigger': 'manual',
        'pr_number': [101],
        'tags': {'team': 'alpha'},
        'accumulated_cost': 1.25,
        'prompt_tokens': 100,
        'completion_tokens': 200,
        'cache_read_tokens': 10,
        'cache_write_tokens': 5,
        'context_window': 0,
        'per_turn_token': 0,
        'max_budget_per_task': None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_saas_metadata(user_id='00000000-0000-0000-0000-000000000001'):
    return SimpleNamespace(user_id=user_id)


@pytest.mark.parametrize(
    ('pr_map', 'expected'),
    [
        ({}, None),
        ({('github', 'repo', 101): None}, None),
        ({('github', 'repo', 101): True}, True),
        ({('github', 'repo', 101): False}, False),
        ({('github', 'repo', 101): False, ('github', 'repo', 102): True}, True),
        ({('github', 'repo', 101): False, ('github', 'repo', 102): None}, False),
    ],
)
def test_resolve_pr_merged(pr_map, expected):
    service = OrgConversationService(db_session=None)
    metadata = _make_metadata(pr_number=[101, 102])
    assert service._resolve_pr_merged(metadata, pr_map) == expected


def test_resolve_pr_merged_missing_repo_or_provider():
    service = OrgConversationService(db_session=None)
    metadata = _make_metadata(selected_repository=None)
    assert service._resolve_pr_merged(metadata, {}) is None


def test_build_conversation_response_includes_pr_merged():
    service = OrgConversationService(db_session=None)
    metadata = _make_metadata()
    saas_metadata = _make_saas_metadata()
    response = service._build_conversation_response(
        metadata,
        saas_metadata,
        user=None,
        sandbox_info=None,
        pr_merged=True,
    )
    assert response.pr_merged is True
    assert response.pr_number == [101]
