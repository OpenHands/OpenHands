"""Tests for config_from_env event service provider selection.

This module tests the event service provider selection logic in config_from_env,
which determines which EventServiceInjector to use based on environment variables.
"""

import os
from unittest.mock import patch

import pytest

# Note: We need to clear the global config cache before each test
# to ensure environment variable changes take effect


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset the global config before and after each test."""
    import openhands.app_server.config as config_module

    original_config = config_module._global_config
    config_module._global_config = None
    yield
    config_module._global_config = original_config


def _get_clean_env():
    """Get a base environment dict with essential system vars preserved."""
    # Preserve essential system environment variables
    env = {}
    for key in ['PATH', 'HOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'TMPDIR', 'TMP', 'TEMP']:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


class TestConfigFromEnvEventServiceSelection:
    """Test cases for event service provider selection in config_from_env."""

    def test_defaults_to_filesystem_when_no_env_set(self):
        """Test that FilesystemEventServiceInjector is used when no FILE_STORE is set."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.filesystem_event_service import (
            FilesystemEventServiceInjector,
        )

        env = _get_clean_env()
        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()
            assert isinstance(config.event, FilesystemEventServiceInjector)

    def test_uses_google_cloud_when_file_store_google_cloud(self):
        """Test that GoogleCloudEventServiceInjector is used when FILE_STORE=google_cloud."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.google_cloud_event_service import (
            GoogleCloudEventServiceInjector,
        )

        env = _get_clean_env()
        env['FILE_STORE'] = 'google_cloud'
        env['FILE_STORE_PATH'] = 'test-gcp-bucket'

        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()

            assert isinstance(config.event, GoogleCloudEventServiceInjector)
            assert config.event.bucket_name == 'test-gcp-bucket'

    def test_uses_gcp_when_provider_gcp(self):
        """Test that GoogleCloudEventServiceInjector is used when SHARED_EVENT_STORAGE_PROVIDER=gcp."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.google_cloud_event_service import (
            GoogleCloudEventServiceInjector,
        )

        env = _get_clean_env()
        env['SHARED_EVENT_STORAGE_PROVIDER'] = 'gcp'
        env['FILE_STORE_PATH'] = 'test-gcp-bucket'

        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()
            assert isinstance(config.event, GoogleCloudEventServiceInjector)

    def test_uses_aws_when_provider_aws(self):
        """Test that AwsEventServiceInjector is used when SHARED_EVENT_STORAGE_PROVIDER=aws."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.aws_event_service import (
            AwsEventServiceInjector,
        )

        env = _get_clean_env()
        env['SHARED_EVENT_STORAGE_PROVIDER'] = 'aws'
        env['FILE_STORE_PATH'] = 'test-aws-bucket'

        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()

            assert isinstance(config.event, AwsEventServiceInjector)
            assert config.event.bucket_name == 'test-aws-bucket'

    def test_aws_requires_file_store_path(self):
        """Test that AWS provider requires FILE_STORE_PATH to be set."""
        from openhands.app_server.config import config_from_env

        env = _get_clean_env()
        env['SHARED_EVENT_STORAGE_PROVIDER'] = 'aws'

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError) as exc_info:
                config_from_env()

            assert 'FILE_STORE_PATH' in str(exc_info.value)
            assert 'required' in str(exc_info.value).lower()

    def test_provider_takes_precedence_over_file_store(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER takes precedence over FILE_STORE."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.aws_event_service import (
            AwsEventServiceInjector,
        )

        env = _get_clean_env()
        env['SHARED_EVENT_STORAGE_PROVIDER'] = 'aws'
        env['FILE_STORE'] = 'google_cloud'
        env['FILE_STORE_PATH'] = 'test-bucket'

        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()

            # Should use AWS because SHARED_EVENT_STORAGE_PROVIDER takes precedence
            assert isinstance(config.event, AwsEventServiceInjector)

    def test_provider_gcp_takes_precedence_over_file_store_s3(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER=gcp takes precedence over FILE_STORE=s3."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.google_cloud_event_service import (
            GoogleCloudEventServiceInjector,
        )

        env = _get_clean_env()
        env['SHARED_EVENT_STORAGE_PROVIDER'] = 'gcp'
        env['FILE_STORE'] = 's3'
        env['FILE_STORE_PATH'] = 'test-bucket'

        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()

            # Should use GCP because SHARED_EVENT_STORAGE_PROVIDER takes precedence
            assert isinstance(config.event, GoogleCloudEventServiceInjector)

    def test_provider_is_case_insensitive(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER is case insensitive."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.aws_event_service import (
            AwsEventServiceInjector,
        )

        env = _get_clean_env()
        env['SHARED_EVENT_STORAGE_PROVIDER'] = 'AWS'
        env['FILE_STORE_PATH'] = 'test-bucket'

        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()
            assert isinstance(config.event, AwsEventServiceInjector)

    def test_provider_gcp_is_case_insensitive(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER=GCP is case insensitive."""
        from openhands.app_server.config import config_from_env
        from openhands.app_server.event.google_cloud_event_service import (
            GoogleCloudEventServiceInjector,
        )

        env = _get_clean_env()
        env['SHARED_EVENT_STORAGE_PROVIDER'] = 'GCP'
        env['FILE_STORE_PATH'] = 'test-bucket'

        with patch.dict(os.environ, env, clear=True):
            config = config_from_env()
            assert isinstance(config.event, GoogleCloudEventServiceInjector)
