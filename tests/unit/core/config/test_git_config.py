"""Tests for git configuration functionality."""

import os
from unittest.mock import patch

from openhands.core.config import OpenHandsConfig, load_from_env


class TestGitConfig:
    """Test git configuration functionality."""

    def test_default_git_config(self):
        """Test that default git configuration is set correctly."""
        config = OpenHandsConfig()
        assert config.git_user_name == 'openhands'
        assert config.git_user_email == 'openhands@all-hands.dev'

    def test_git_config_from_env_vars(self):
        """Test that git configuration can be set via environment variables."""
        with patch.dict(
            os.environ,
            {'GIT_USER_NAME': 'testuser', 'GIT_USER_EMAIL': 'testuser@example.com'},
        ):
            config = OpenHandsConfig()
            load_from_env(config, os.environ)

            assert config.git_user_name == 'testuser'
            assert config.git_user_email == 'testuser@example.com'

    def test_git_config_empty_values(self):
        """Test behavior with empty git configuration values."""
        with patch.dict(os.environ, {'GIT_USER_NAME': '', 'GIT_USER_EMAIL': ''}):
            config = OpenHandsConfig()
            load_from_env(config, os.environ)

            # Empty values should fall back to defaults
            assert config.git_user_name == 'openhands'
            assert config.git_user_email == 'openhands@all-hands.dev'
