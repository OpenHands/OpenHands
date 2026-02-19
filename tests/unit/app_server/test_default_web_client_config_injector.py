"""Tests for DefaultWebClientConfigInjector.

This module tests environment variable handling in DefaultWebClientConfigInjector.
"""

import os
from unittest.mock import patch


class TestGetPosthogClientKey:
    """Test cases for _get_posthog_client_key helper function."""

    OSS_DEFAULT_KEY = 'phc_3ESMmY9SgqEAGBB6sMGK5ayYHkeUuknH2vP6FmWH9RA'

    def test_returns_env_var_when_set(self):
        """When POSTHOG_CLIENT_KEY is set, return that value."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_posthog_client_key,
        )

        with patch.dict(os.environ, {'POSTHOG_CLIENT_KEY': 'phc_saas_key_123'}):
            result = _get_posthog_client_key()
            assert result == 'phc_saas_key_123'

    def test_returns_oss_default_when_env_var_unset(self):
        """When POSTHOG_CLIENT_KEY is not set, return the OSS default key."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_posthog_client_key,
        )

        with patch.dict(os.environ, {}, clear=True):
            # Ensure POSTHOG_CLIENT_KEY is not in environment
            os.environ.pop('POSTHOG_CLIENT_KEY', None)
            result = _get_posthog_client_key()
            assert result == self.OSS_DEFAULT_KEY

    def test_returns_oss_default_when_env_var_empty(self):
        """When POSTHOG_CLIENT_KEY is empty string, return the OSS default key."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_posthog_client_key,
        )

        with patch.dict(os.environ, {'POSTHOG_CLIENT_KEY': ''}):
            result = _get_posthog_client_key()
            assert result == self.OSS_DEFAULT_KEY

    def test_strips_whitespace_from_env_var(self):
        """When POSTHOG_CLIENT_KEY has whitespace, strip it."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_posthog_client_key,
        )

        with patch.dict(os.environ, {'POSTHOG_CLIENT_KEY': '  phc_trimmed_key  '}):
            result = _get_posthog_client_key()
            assert result == 'phc_trimmed_key'

    def test_returns_oss_default_when_env_var_only_whitespace(self):
        """When POSTHOG_CLIENT_KEY is only whitespace, return the OSS default key."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_posthog_client_key,
        )

        with patch.dict(os.environ, {'POSTHOG_CLIENT_KEY': '   '}):
            result = _get_posthog_client_key()
            assert result == self.OSS_DEFAULT_KEY


class TestGetAuthUrl:
    """Test cases for _get_auth_url helper function."""

    def test_returns_env_var_when_set(self):
        """When AUTH_URL is set, return that value."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_auth_url,
        )

        with patch.dict(os.environ, {'AUTH_URL': 'https://auth.example.com'}):
            result = _get_auth_url()
            assert result == 'https://auth.example.com'

    def test_returns_none_when_env_var_unset(self):
        """When AUTH_URL is not set, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_auth_url,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('AUTH_URL', None)
            result = _get_auth_url()
            assert result is None

    def test_returns_none_when_env_var_empty(self):
        """When AUTH_URL is empty string, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_auth_url,
        )

        with patch.dict(os.environ, {'AUTH_URL': ''}):
            result = _get_auth_url()
            assert result is None

    def test_strips_whitespace_from_env_var(self):
        """When AUTH_URL has whitespace, strip it."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_auth_url,
        )

        with patch.dict(os.environ, {'AUTH_URL': '  https://auth.example.com  '}):
            result = _get_auth_url()
            assert result == 'https://auth.example.com'

    def test_returns_none_when_env_var_only_whitespace(self):
        """When AUTH_URL is only whitespace, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_auth_url,
        )

        with patch.dict(os.environ, {'AUTH_URL': '   '}):
            result = _get_auth_url()
            assert result is None


class TestGetFeatureFlags:
    """Test cases for _get_feature_flags helper function."""

    def test_returns_all_false_when_no_env_vars_set(self):
        """When no feature flag env vars are set, all flags default to False."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing feature flag env vars
            for var in [
                'ENABLE_BILLING',
                'HIDE_LLM_SETTINGS',
                'ENABLE_JIRA',
                'ENABLE_JIRA_DC',
                'ENABLE_LINEAR',
            ]:
                os.environ.pop(var, None)
            result = _get_feature_flags()
            assert result.enable_billing is False
            assert result.hide_llm_settings is False
            assert result.enable_jira is False
            assert result.enable_jira_dc is False
            assert result.enable_linear is False

    def test_enable_billing_true_when_env_var_true(self):
        """When ENABLE_BILLING is 'true', enable_billing flag is True."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {'ENABLE_BILLING': 'true'}):
            result = _get_feature_flags()
            assert result.enable_billing is True

    def test_enable_billing_false_when_env_var_false(self):
        """When ENABLE_BILLING is 'false', enable_billing flag is False."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {'ENABLE_BILLING': 'false'}):
            result = _get_feature_flags()
            assert result.enable_billing is False

    def test_enable_billing_false_when_env_var_other_value(self):
        """When ENABLE_BILLING is any value other than 'true', enable_billing is False."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {'ENABLE_BILLING': 'yes'}):
            result = _get_feature_flags()
            assert result.enable_billing is False

    def test_hide_llm_settings_true_when_env_var_true(self):
        """When HIDE_LLM_SETTINGS is 'true', hide_llm_settings flag is True."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {'HIDE_LLM_SETTINGS': 'true'}):
            result = _get_feature_flags()
            assert result.hide_llm_settings is True

    def test_enable_jira_true_when_env_var_true(self):
        """When ENABLE_JIRA is 'true', enable_jira flag is True."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {'ENABLE_JIRA': 'true'}):
            result = _get_feature_flags()
            assert result.enable_jira is True

    def test_enable_jira_dc_true_when_env_var_true(self):
        """When ENABLE_JIRA_DC is 'true', enable_jira_dc flag is True."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {'ENABLE_JIRA_DC': 'true'}):
            result = _get_feature_flags()
            assert result.enable_jira_dc is True

    def test_enable_linear_true_when_env_var_true(self):
        """When ENABLE_LINEAR is 'true', enable_linear flag is True."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(os.environ, {'ENABLE_LINEAR': 'true'}):
            result = _get_feature_flags()
            assert result.enable_linear is True

    def test_multiple_flags_can_be_set(self):
        """Multiple feature flags can be enabled simultaneously."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_feature_flags,
        )

        with patch.dict(
            os.environ,
            {
                'ENABLE_BILLING': 'true',
                'HIDE_LLM_SETTINGS': 'true',
                'ENABLE_JIRA': 'false',
                'ENABLE_LINEAR': 'true',
            },
        ):
            result = _get_feature_flags()
            assert result.enable_billing is True
            assert result.hide_llm_settings is True
            assert result.enable_jira is False
            assert result.enable_jira_dc is False
            assert result.enable_linear is True


class TestGetMaintenanceStartTime:
    """Test cases for _get_maintenance_start_time helper function."""

    def test_returns_datetime_when_valid_iso_timestamp_set(self):
        """When MAINTENANCE_START_TIME is a valid ISO 8601 timestamp, return parsed datetime."""
        from datetime import datetime, timezone

        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_maintenance_start_time,
        )

        with patch.dict(os.environ, {'MAINTENANCE_START_TIME': '2026-03-15T10:00:00Z'}):
            result = _get_maintenance_start_time()
            assert result == datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_returns_none_when_env_var_unset(self):
        """When MAINTENANCE_START_TIME is not set, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_maintenance_start_time,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('MAINTENANCE_START_TIME', None)
            result = _get_maintenance_start_time()
            assert result is None

    def test_returns_none_when_env_var_empty(self):
        """When MAINTENANCE_START_TIME is empty string, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_maintenance_start_time,
        )

        with patch.dict(os.environ, {'MAINTENANCE_START_TIME': ''}):
            result = _get_maintenance_start_time()
            assert result is None

    def test_returns_none_when_env_var_invalid(self):
        """When MAINTENANCE_START_TIME is invalid format, return None (graceful fallback)."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_maintenance_start_time,
        )

        with patch.dict(
            os.environ, {'MAINTENANCE_START_TIME': 'not-a-valid-timestamp'}
        ):
            result = _get_maintenance_start_time()
            assert result is None

    def test_strips_whitespace_from_env_var(self):
        """When MAINTENANCE_START_TIME has whitespace, strip it before parsing."""
        from datetime import datetime, timezone

        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_maintenance_start_time,
        )

        with patch.dict(
            os.environ, {'MAINTENANCE_START_TIME': '  2026-03-15T10:00:00Z  '}
        ):
            result = _get_maintenance_start_time()
            assert result == datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)


class TestGetProvidersConfigured:
    """Test cases for _get_providers_configured helper function."""

    def test_returns_empty_list_when_no_env_vars_set(self):
        """When no provider env vars are set, return empty list."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )

        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing provider env vars
            for var in [
                'GITHUB_APP_CLIENT_ID',
                'GITLAB_APP_CLIENT_ID',
                'BITBUCKET_APP_CLIENT_ID',
                'ENABLE_ENTERPRISE_SSO',
            ]:
                os.environ.pop(var, None)
            result = _get_providers_configured()
            assert result == []

    def test_includes_github_when_client_id_set(self):
        """When GITHUB_APP_CLIENT_ID is set, include GitHub in providers."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )
        from openhands.integrations.service_types import ProviderType

        with patch.dict(os.environ, {'GITHUB_APP_CLIENT_ID': 'some-client-id'}):
            result = _get_providers_configured()
            assert ProviderType.GITHUB in result

    def test_includes_gitlab_when_client_id_set(self):
        """When GITLAB_APP_CLIENT_ID is set, include GitLab in providers."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )
        from openhands.integrations.service_types import ProviderType

        with patch.dict(os.environ, {'GITLAB_APP_CLIENT_ID': 'some-client-id'}):
            result = _get_providers_configured()
            assert ProviderType.GITLAB in result

    def test_includes_bitbucket_when_client_id_set(self):
        """When BITBUCKET_APP_CLIENT_ID is set, include Bitbucket in providers."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )
        from openhands.integrations.service_types import ProviderType

        with patch.dict(os.environ, {'BITBUCKET_APP_CLIENT_ID': 'some-client-id'}):
            result = _get_providers_configured()
            assert ProviderType.BITBUCKET in result

    def test_includes_enterprise_sso_when_enabled(self):
        """When ENABLE_ENTERPRISE_SSO is set, include Enterprise SSO in providers."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )
        from openhands.integrations.service_types import ProviderType

        with patch.dict(os.environ, {'ENABLE_ENTERPRISE_SSO': 'true'}):
            result = _get_providers_configured()
            assert ProviderType.ENTERPRISE_SSO in result

    def test_excludes_provider_when_env_var_empty(self):
        """When env var is empty string, do not include provider."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )
        from openhands.integrations.service_types import ProviderType

        with patch.dict(os.environ, {'GITHUB_APP_CLIENT_ID': ''}):
            result = _get_providers_configured()
            assert ProviderType.GITHUB not in result

    def test_excludes_provider_when_env_var_only_whitespace(self):
        """When env var is only whitespace, do not include provider."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )
        from openhands.integrations.service_types import ProviderType

        with patch.dict(os.environ, {'GITHUB_APP_CLIENT_ID': '   '}):
            result = _get_providers_configured()
            assert ProviderType.GITHUB not in result

    def test_includes_multiple_providers(self):
        """Multiple providers can be configured simultaneously."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_providers_configured,
        )
        from openhands.integrations.service_types import ProviderType

        with patch.dict(
            os.environ,
            {
                'GITHUB_APP_CLIENT_ID': 'github-id',
                'GITLAB_APP_CLIENT_ID': 'gitlab-id',
                'BITBUCKET_APP_CLIENT_ID': '',
                'ENABLE_ENTERPRISE_SSO': 'enabled',
            },
        ):
            result = _get_providers_configured()
            assert ProviderType.GITHUB in result
            assert ProviderType.GITLAB in result
            assert ProviderType.BITBUCKET not in result
            assert ProviderType.ENTERPRISE_SSO in result
            assert len(result) == 3


class TestGetGithubAppSlug:
    """Test cases for _get_github_app_slug helper function."""

    def test_returns_env_var_when_set(self):
        """When GITHUB_APP_SLUG is set, return that value."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_github_app_slug,
        )

        with patch.dict(os.environ, {'GITHUB_APP_SLUG': 'openhands-app'}):
            result = _get_github_app_slug()
            assert result == 'openhands-app'

    def test_returns_none_when_env_var_unset(self):
        """When GITHUB_APP_SLUG is not set, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_github_app_slug,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GITHUB_APP_SLUG', None)
            result = _get_github_app_slug()
            assert result is None

    def test_returns_none_when_env_var_empty(self):
        """When GITHUB_APP_SLUG is empty string, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_github_app_slug,
        )

        with patch.dict(os.environ, {'GITHUB_APP_SLUG': ''}):
            result = _get_github_app_slug()
            assert result is None

    def test_strips_whitespace_from_env_var(self):
        """When GITHUB_APP_SLUG has whitespace, strip it."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_github_app_slug,
        )

        with patch.dict(os.environ, {'GITHUB_APP_SLUG': '  openhands-app  '}):
            result = _get_github_app_slug()
            assert result == 'openhands-app'

    def test_returns_none_when_env_var_only_whitespace(self):
        """When GITHUB_APP_SLUG is only whitespace, return None."""
        from openhands.app_server.web_client.default_web_client_config_injector import (
            _get_github_app_slug,
        )

        with patch.dict(os.environ, {'GITHUB_APP_SLUG': '   '}):
            result = _get_github_app_slug()
            assert result is None
