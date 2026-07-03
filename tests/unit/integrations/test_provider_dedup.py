from __future__ import annotations

from types import MappingProxyType

from pydantic import SecretStr

from openhands.app_server.integrations.provider import (
    ProviderHandler,
    ProviderToken,
    ProviderType,
    Repository,
)


def _repo(repo_id: str, full_name: str) -> Repository:
    return Repository(
        id=repo_id,
        full_name=full_name,
        git_provider=ProviderType.GITHUB,
        is_public=True,
    )


def _handler() -> ProviderHandler:
    tokens = MappingProxyType(
        {ProviderType.GITHUB: ProviderToken(token=SecretStr('test'))}
    )
    return ProviderHandler(provider_tokens=tokens)


def test_deduplicate_repositories_removes_duplicate_full_names():
    """Repos sharing a full_name collapse to the first occurrence."""
    handler = _handler()
    repos = [
        _repo('1', 'owner/repo'),
        _repo('2', 'owner/repo'),  # duplicate full_name, distinct id
        _repo('3', 'owner/other'),
    ]

    result = handler._deduplicate_repositories(repos)

    assert [r.full_name for r in result] == ['owner/repo', 'owner/other']
    assert [r.id for r in result] == ['1', '3']  # first occurrence wins


def test_deduplicate_repositories_keeps_distinct_full_names():
    """Distinct full_names are all preserved, order intact."""
    handler = _handler()
    repos = [_repo('1', 'a/one'), _repo('2', 'b/two'), _repo('3', 'c/three')]

    result = handler._deduplicate_repositories(repos)

    assert [r.full_name for r in result] == ['a/one', 'b/two', 'c/three']
