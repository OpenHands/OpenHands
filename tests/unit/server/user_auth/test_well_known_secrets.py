from types import MappingProxyType

from pydantic import SecretStr

from openhands.integrations.provider import CustomSecret, ProviderToken
from openhands.integrations.service_types import ProviderType
from openhands.server.user_auth.default_user_auth import (
    _resolve_github_token_from_custom_secret,
)
from openhands.storage.data_models.secrets import (
    WELL_KNOWN_SECRET_GITHUB_TOKEN,
    Secrets,
)


def test_resolve_github_token_creates_provider_token():
    """When github-token custom secret exists and no GitHub provider token,
    a GitHub provider token should be created."""
    secrets = Secrets(
        provider_tokens=MappingProxyType({}),
        custom_secrets=MappingProxyType(
            {
                WELL_KNOWN_SECRET_GITHUB_TOKEN: CustomSecret(
                    secret=SecretStr('ghp_test123')
                ),
            }
        ),
    )

    result = _resolve_github_token_from_custom_secret(secrets)

    assert ProviderType.GITHUB in result.provider_tokens
    assert (
        result.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'ghp_test123'
    )


def test_resolve_github_token_preserves_existing_token():
    """When a GitHub provider token already exists, the custom secret should
    NOT override it."""
    existing_token = ProviderToken(token=SecretStr('existing-token'))
    secrets = Secrets(
        provider_tokens=MappingProxyType({ProviderType.GITHUB: existing_token}),
        custom_secrets=MappingProxyType(
            {
                WELL_KNOWN_SECRET_GITHUB_TOKEN: CustomSecret(
                    secret=SecretStr('ghp_custom')
                ),
            }
        ),
    )

    result = _resolve_github_token_from_custom_secret(secrets)

    assert (
        result.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'existing-token'
    )


def test_resolve_github_token_no_custom_secret():
    """When no github-token custom secret exists, secrets should pass through
    unchanged."""
    secrets = Secrets(
        provider_tokens=MappingProxyType({}),
        custom_secrets=MappingProxyType({}),
    )

    result = _resolve_github_token_from_custom_secret(secrets)

    assert ProviderType.GITHUB not in result.provider_tokens


def test_resolve_github_token_empty_existing_token():
    """When GitHub provider token exists but has empty value, the custom secret
    should be used."""
    empty_token = ProviderToken(token=SecretStr(''))
    secrets = Secrets(
        provider_tokens=MappingProxyType({ProviderType.GITHUB: empty_token}),
        custom_secrets=MappingProxyType(
            {
                WELL_KNOWN_SECRET_GITHUB_TOKEN: CustomSecret(
                    secret=SecretStr('ghp_replacement')
                ),
            }
        ),
    )

    result = _resolve_github_token_from_custom_secret(secrets)

    assert (
        result.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'ghp_replacement'
    )


def test_resolve_github_token_preserves_other_providers():
    """Other provider tokens should be unaffected by GitHub token resolution."""
    gitlab_token = ProviderToken(token=SecretStr('glpat-test'))
    secrets = Secrets(
        provider_tokens=MappingProxyType({ProviderType.GITLAB: gitlab_token}),
        custom_secrets=MappingProxyType(
            {
                WELL_KNOWN_SECRET_GITHUB_TOKEN: CustomSecret(
                    secret=SecretStr('ghp_test')
                ),
            }
        ),
    )

    result = _resolve_github_token_from_custom_secret(secrets)

    assert ProviderType.GITHUB in result.provider_tokens
    assert ProviderType.GITLAB in result.provider_tokens
    assert (
        result.provider_tokens[ProviderType.GITLAB].token.get_secret_value()
        == 'glpat-test'
    )
