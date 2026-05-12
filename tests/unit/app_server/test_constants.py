"""Tests for openhands.app_server.constants module."""

import os
from unittest.mock import patch

import pytest

from openhands.app_server.constants import (
    BLOCKED_SECRET_NAMES,
    BLOCKED_SECRET_PREFIXES,
    MAX_API_SECRET_NAME_LENGTH,
    MAX_API_SECRET_VALUE_LENGTH,
    MAX_API_SECRETS_COUNT,
    validate_secret_name,
    validate_secrets_dict,
)


class TestValidateSecretName:
    """Tests for validate_secret_name function."""

    def test_valid_secret_name(self):
        """Valid secret names should not raise."""
        validate_secret_name('MY_API_KEY')
        validate_secret_name('github_token')  # case-insensitive for overridable
        validate_secret_name('CUSTOM_SECRET')
        validate_secret_name('a')  # short names are OK

    def test_blocked_exact_name(self):
        """Blocked names should raise ValueError."""
        for name in BLOCKED_SECRET_NAMES:
            with pytest.raises(ValueError, match='reserved for internal use'):
                validate_secret_name(name)

    def test_blocked_name_case_insensitive(self):
        """Blocked name check should be case-insensitive."""
        # Pick one from the set
        blocked_name = next(iter(BLOCKED_SECRET_NAMES))
        with pytest.raises(ValueError, match='reserved for internal use'):
            validate_secret_name(blocked_name.lower())

    def test_blocked_prefix(self):
        """Names starting with blocked prefixes should raise."""
        for prefix in BLOCKED_SECRET_PREFIXES:
            with pytest.raises(ValueError, match=f"reserved prefix '{prefix}'"):
                validate_secret_name(f'{prefix}SOME_VAR')

    def test_blocked_prefix_case_insensitive(self):
        """Blocked prefix check should be case-insensitive."""
        with pytest.raises(ValueError, match='reserved prefix'):
            validate_secret_name('llm_api_key')

    def test_name_too_long(self):
        """Names exceeding max length should raise."""
        long_name = 'A' * (MAX_API_SECRET_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match='exceeds maximum length'):
            validate_secret_name(long_name)

    def test_name_at_max_length(self):
        """Names at exactly max length should be accepted."""
        max_name = 'A' * MAX_API_SECRET_NAME_LENGTH
        validate_secret_name(max_name)  # Should not raise

    def test_overridable_secrets_allowed(self):
        """Overridable system secrets (like GITHUB_TOKEN) should be allowed."""
        validate_secret_name('GITHUB_TOKEN')
        validate_secret_name('GITLAB_TOKEN')
        validate_secret_name('AWS_ACCESS_KEY_ID')


class TestValidateSecretsDict:
    """Tests for validate_secrets_dict function."""

    def test_none_secrets(self):
        """None should be accepted without error."""
        validate_secrets_dict(None)

    def test_empty_dict(self):
        """Empty dict should be accepted."""
        validate_secrets_dict({})

    def test_valid_secrets(self):
        """Valid secrets dict should not raise."""
        validate_secrets_dict(
            {
                'KEY1': 'value1',
                'KEY2': 'value2',
            }
        )

    def test_too_many_secrets(self):
        """Exceeding max count should raise."""
        secrets = {f'KEY_{i}': f'value_{i}' for i in range(MAX_API_SECRETS_COUNT + 1)}
        with pytest.raises(ValueError, match='Too many secrets'):
            validate_secrets_dict(secrets)

    def test_at_max_count(self):
        """Exactly max count should be accepted."""
        secrets = {f'KEY_{i}': f'value_{i}' for i in range(MAX_API_SECRETS_COUNT)}
        validate_secrets_dict(secrets)  # Should not raise

    def test_value_too_long(self):
        """Secret value exceeding max length should raise."""
        long_value = 'x' * (MAX_API_SECRET_VALUE_LENGTH + 1)
        with pytest.raises(ValueError, match='value exceeds maximum length'):
            validate_secrets_dict({'KEY': long_value})

    def test_value_at_max_length(self):
        """Value at exactly max length should be accepted."""
        max_value = 'x' * MAX_API_SECRET_VALUE_LENGTH
        validate_secrets_dict({'KEY': max_value})  # Should not raise

    def test_unicode_value_bytes(self):
        """Value length should be checked in bytes, not characters."""
        # Each emoji is 4 bytes in UTF-8
        emoji_count = (MAX_API_SECRET_VALUE_LENGTH // 4) + 1
        unicode_value = '🔐' * emoji_count
        with pytest.raises(ValueError, match='value exceeds maximum length'):
            validate_secrets_dict({'KEY': unicode_value})

    def test_secretstr_values(self):
        """Should handle Pydantic SecretStr values."""
        from pydantic import SecretStr

        validate_secrets_dict(
            {
                'KEY1': SecretStr('value1'),
                'KEY2': SecretStr('value2'),
            }
        )

    def test_secretstr_value_too_long(self):
        """Should check SecretStr value length correctly."""
        from pydantic import SecretStr

        long_value = SecretStr('x' * (MAX_API_SECRET_VALUE_LENGTH + 1))
        with pytest.raises(ValueError, match='value exceeds maximum length'):
            validate_secrets_dict({'KEY': long_value})


class TestConfigurableLimits:
    """Tests for environment variable configuration of limits."""

    def test_default_values(self):
        """Default values should be sensible."""
        assert MAX_API_SECRETS_COUNT == 50
        assert MAX_API_SECRET_NAME_LENGTH == 256
        assert MAX_API_SECRET_VALUE_LENGTH == 65536

    def test_env_override_count(self):
        """OH_MAX_API_SECRETS_COUNT should override default."""
        # This tests that the pattern works; the actual module-level
        # value is set at import time, so we verify the env var is read
        with patch.dict(os.environ, {'OH_MAX_API_SECRETS_COUNT': '100'}):
            # Re-import to pick up new env
            import importlib

            import openhands.app_server.constants as constants

            importlib.reload(constants)
            assert constants.MAX_API_SECRETS_COUNT == 100
            # Reset for other tests
            importlib.reload(constants)
