"""Unit tests for the Databricks User-Agent Plugin."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openhands.runtime.plugins.databricks_user_agent import (
    DatabricksUserAgentPlugin,
    DatabricksUserAgentRequirement,
    create_databricks_plugin,
)


class TestDatabricksUserAgentRequirement:
    def test_default_fields(self) -> None:
        req = DatabricksUserAgentRequirement()
        assert req.name == 'databricks_user_agent'
        assert req.patch_http_libraries is True
        assert req.configure_java is True
        assert req.enable_debug_logging is False

    def test_user_agent_property(self) -> None:
        req = DatabricksUserAgentRequirement()
        assert req.user_agent == 'OpenHandsOSS'

    def test_version_property(self) -> None:
        req = DatabricksUserAgentRequirement()
        # Version is sourced from the installed openhands-sdk package, not hardcoded.
        assert req.version, 'version should be non-empty'
        assert '/' not in req.version, 'version should not contain the product prefix'

    def test_debug_logging_flag(self) -> None:
        req = DatabricksUserAgentRequirement(enable_debug_logging=True)
        assert req.enable_debug_logging is True

    def test_disable_http_patching(self) -> None:
        req = DatabricksUserAgentRequirement(patch_http_libraries=False)
        assert req.patch_http_libraries is False

    def test_disable_java_config(self) -> None:
        req = DatabricksUserAgentRequirement(configure_java=False)
        assert req.configure_java is False


class TestDatabricksUserAgentPlugin:
    def test_default_init(self) -> None:
        plugin = DatabricksUserAgentPlugin()
        assert plugin.name == 'databricks_user_agent'
        assert plugin.user_agent == 'OpenHandsOSS'
        # Version is sourced from the installed openhands-sdk package at import time.
        assert plugin.version, 'version should be non-empty'
        assert plugin._initialized is False

    def test_init_with_requirement(self) -> None:
        req = DatabricksUserAgentRequirement(enable_debug_logging=True)
        plugin = DatabricksUserAgentPlugin(req)
        assert plugin.enable_debug is True

    def test_init_with_none_requirement(self) -> None:
        plugin = DatabricksUserAgentPlugin(None)
        assert plugin.user_agent == 'OpenHandsOSS'

    @pytest.mark.asyncio
    async def test_initialize_sets_env_vars(self) -> None:
        plugin = DatabricksUserAgentPlugin()
        fake_script_path = Path('/tmp/fake_init.py')
        # Mock filesystem-writing helpers so the test is self-contained.
        with (
            patch.object(
                plugin,
                '_create_python_init_script',
                new_callable=AsyncMock,
                return_value=fake_script_path,
            ),
            patch.object(
                plugin,
                '_configure_shell_environment',
                new_callable=AsyncMock,
            ),
            patch.object(
                plugin,
                '_create_helper_scripts',
                new_callable=AsyncMock,
            ),
        ):
            await plugin.initialize('testuser')
            assert os.environ.get('OH_DATABRICKS_INTEGRATION') == 'true'
            assert os.environ.get('DATABRICKS_SDK_UPSTREAM') == 'OpenHandsOSS'
            # Version is sourced from the SDK package, not hardcoded to '1.0'.
            assert os.environ.get('DATABRICKS_SDK_UPSTREAM_VERSION'), (
                'DATABRICKS_SDK_UPSTREAM_VERSION should be set to a non-empty version'
            )
            assert plugin._initialized is True


class TestCreateDatabricksPlugin:
    def test_default(self) -> None:
        plugin = create_databricks_plugin()
        assert isinstance(plugin, DatabricksUserAgentPlugin)
        assert plugin.user_agent == 'OpenHandsOSS'

    def test_with_debug(self) -> None:
        plugin = create_databricks_plugin(enable_debug=True)
        assert plugin.enable_debug is True

    def test_without_debug(self) -> None:
        plugin = create_databricks_plugin(enable_debug=False)
        assert plugin.enable_debug is False
