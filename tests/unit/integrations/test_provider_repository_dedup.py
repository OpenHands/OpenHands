"""Tests for ProviderHandler repository deduplication."""

from types import MappingProxyType

from pydantic import SecretStr

from openhands.integrations.provider import ProviderHandler, ProviderToken
from openhands.integrations.service_types import OwnerType, ProviderType, Repository


def _make_handler() -> ProviderHandler:
    return ProviderHandler(
        provider_tokens=MappingProxyType(
            {
                ProviderType.GITHUB: ProviderToken(token=SecretStr('test-token')),
            }
        )
    )


def _repo(
    repo_id: str,
    full_name: str,
    *,
    provider: ProviderType = ProviderType.GITHUB,
) -> Repository:
    return Repository(
        id=repo_id,
        full_name=full_name,
        git_provider=provider,
        is_public=True,
        owner_type=OwnerType.ORGANIZATION,
    )


def test_deduplicate_repositories_keeps_first_occurrence_per_full_name():
    """Same full_name with different ids must collapse to one row (merge path)."""
    handler = _make_handler()
    first = _repo('1', 'acme/app')
    second = _repo('999', 'acme/app')

    result = handler._deduplicate_repositories([first, second])

    assert result == [first]


def test_deduplicate_repositories_preserves_distinct_full_names():
    handler = _make_handler()
    a = _repo('1', 'acme/app')
    b = _repo('2', 'acme/other')

    result = handler._deduplicate_repositories([a, b])

    assert result == [a, b]


def test_deduplicate_repositories_empty_input():
    handler = _make_handler()
    assert handler._deduplicate_repositories([]) == []
