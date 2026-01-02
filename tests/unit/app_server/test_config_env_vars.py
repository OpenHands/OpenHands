"""Tests for config_from_env OH_SANDBOX_PULL_IF_MISSING environment variable.

This tests the environment variable that controls Docker image pulling on startup.
Related to GitHub issue #11878.
"""

import os
from unittest.mock import patch

import pytest


class TestOHSandboxPullIfMissingEnvVar:
    """Test cases for OH_SANDBOX_PULL_IF_MISSING environment variable."""

    @pytest.fixture(autouse=True)
    def reset_global_config(self):
        """Reset global config before each test."""
        import openhands.app_server.config as config_module

        config_module._global_config = None
        yield
        config_module._global_config = None

    @patch.dict(os.environ, {}, clear=False)
    def test_pull_if_missing_defaults_to_true(self):
        """Test that pull_if_missing defaults to True when env var is not set."""
        # Remove the env var if it exists
        os.environ.pop('OH_SANDBOX_PULL_IF_MISSING', None)
        os.environ.pop('RUNTIME', None)  # Ensure Docker mode

        from openhands.app_server.config import config_from_env
        from openhands.app_server.sandbox.docker_sandbox_spec_service import (
            DockerSandboxSpecServiceInjector,
        )

        config = config_from_env()

        assert isinstance(config.sandbox_spec, DockerSandboxSpecServiceInjector)
        assert config.sandbox_spec.pull_if_missing is True

    @patch.dict(os.environ, {'OH_SANDBOX_PULL_IF_MISSING': 'false'}, clear=False)
    def test_pull_if_missing_false_when_env_var_is_false(self):
        """Test that pull_if_missing is False when env var is 'false'."""
        os.environ.pop('RUNTIME', None)  # Ensure Docker mode

        from openhands.app_server.config import config_from_env
        from openhands.app_server.sandbox.docker_sandbox_spec_service import (
            DockerSandboxSpecServiceInjector,
        )

        config = config_from_env()

        assert isinstance(config.sandbox_spec, DockerSandboxSpecServiceInjector)
        assert config.sandbox_spec.pull_if_missing is False

    @patch.dict(os.environ, {'OH_SANDBOX_PULL_IF_MISSING': 'FALSE'}, clear=False)
    def test_pull_if_missing_false_case_insensitive(self):
        """Test that env var comparison is case insensitive."""
        os.environ.pop('RUNTIME', None)  # Ensure Docker mode

        from openhands.app_server.config import config_from_env
        from openhands.app_server.sandbox.docker_sandbox_spec_service import (
            DockerSandboxSpecServiceInjector,
        )

        config = config_from_env()

        assert isinstance(config.sandbox_spec, DockerSandboxSpecServiceInjector)
        assert config.sandbox_spec.pull_if_missing is False

    @patch.dict(os.environ, {'OH_SANDBOX_PULL_IF_MISSING': 'true'}, clear=False)
    def test_pull_if_missing_true_when_env_var_is_true(self):
        """Test that pull_if_missing is True when env var is 'true'."""
        os.environ.pop('RUNTIME', None)  # Ensure Docker mode

        from openhands.app_server.config import config_from_env
        from openhands.app_server.sandbox.docker_sandbox_spec_service import (
            DockerSandboxSpecServiceInjector,
        )

        config = config_from_env()

        assert isinstance(config.sandbox_spec, DockerSandboxSpecServiceInjector)
        assert config.sandbox_spec.pull_if_missing is True

    @patch.dict(os.environ, {'OH_SANDBOX_PULL_IF_MISSING': 'TRUE'}, clear=False)
    def test_pull_if_missing_true_case_insensitive(self):
        """Test that TRUE (uppercase) also works."""
        os.environ.pop('RUNTIME', None)  # Ensure Docker mode

        from openhands.app_server.config import config_from_env
        from openhands.app_server.sandbox.docker_sandbox_spec_service import (
            DockerSandboxSpecServiceInjector,
        )

        config = config_from_env()

        assert isinstance(config.sandbox_spec, DockerSandboxSpecServiceInjector)
        assert config.sandbox_spec.pull_if_missing is True

    @patch.dict(os.environ, {'OH_SANDBOX_PULL_IF_MISSING': '0'}, clear=False)
    def test_pull_if_missing_false_for_non_true_values(self):
        """Test that non-'true' values result in False."""
        os.environ.pop('RUNTIME', None)  # Ensure Docker mode

        from openhands.app_server.config import config_from_env
        from openhands.app_server.sandbox.docker_sandbox_spec_service import (
            DockerSandboxSpecServiceInjector,
        )

        config = config_from_env()

        assert isinstance(config.sandbox_spec, DockerSandboxSpecServiceInjector)
        assert config.sandbox_spec.pull_if_missing is False
