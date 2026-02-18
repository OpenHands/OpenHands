from types import MappingProxyType

from pydantic import SecretStr

from openhands.integrations.provider import CustomSecret
from openhands.storage.data_models.secrets import (
    WELL_KNOWN_SECRET_GITHUB_TOKEN,
    WELL_KNOWN_SECRET_LLM_API_KEY,
    WELL_KNOWN_SECRET_NEON_API_KEY,
    Secrets,
)


def test_get_env_vars_returns_custom_secrets():
    """Custom secrets should be returned as env vars keyed by their name."""
    secrets = Secrets(
        provider_tokens=MappingProxyType({}),
        custom_secrets=MappingProxyType(
            {
                WELL_KNOWN_SECRET_GITHUB_TOKEN: CustomSecret(
                    secret=SecretStr('ghp_token123')
                ),
                'my-other-secret': CustomSecret(secret=SecretStr('other-val')),
            }
        ),
    )

    env_vars = secrets.get_env_vars()

    assert env_vars[WELL_KNOWN_SECRET_GITHUB_TOKEN] == 'ghp_token123'
    assert env_vars['my-other-secret'] == 'other-val'
    assert len(env_vars) == 2


def test_get_env_vars_neon_api_key_alias():
    """neon-api-key custom secret should also be forwarded as NEON_API_KEY."""
    secrets = Secrets(
        provider_tokens=MappingProxyType({}),
        custom_secrets=MappingProxyType(
            {
                WELL_KNOWN_SECRET_NEON_API_KEY: CustomSecret(
                    secret=SecretStr('neon-key-123')
                ),
            }
        ),
    )

    env_vars = secrets.get_env_vars()

    assert env_vars[WELL_KNOWN_SECRET_NEON_API_KEY] == 'neon-key-123'
    assert env_vars['NEON_API_KEY'] == 'neon-key-123'
    assert len(env_vars) == 2


def test_get_env_vars_empty():
    """When no custom secrets are present, get_env_vars returns an empty dict."""
    secrets = Secrets(
        provider_tokens=MappingProxyType({}),
        custom_secrets=MappingProxyType({}),
    )

    env_vars = secrets.get_env_vars()
    assert env_vars == {}


def test_well_known_secret_constants():
    """Verify the well-known secret name constants."""
    assert WELL_KNOWN_SECRET_LLM_API_KEY == 'anthropic-api-key'
    assert WELL_KNOWN_SECRET_GITHUB_TOKEN == 'github-token'
    assert WELL_KNOWN_SECRET_NEON_API_KEY == 'neon-api-key'
