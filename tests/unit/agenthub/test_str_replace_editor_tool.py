"""Tests for str_replace_editor tool workspace path detection."""

import os
import sys
from unittest.mock import patch

import pytest


def _abspath_from_env_host_fragment(host_fragment: str) -> str:
    """Match str_replace_editor: host side of `host:/workspace` is passed through abspath."""
    return os.path.abspath(host_fragment)


# Import the functions directly to avoid dependency issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from openhands.agenthub.codeact_agent.tools.str_replace_editor import (
        _get_workspace_mount_path_from_env,
        create_str_replace_editor_tool,
    )
except ImportError:
    # If import fails due to dependencies, skip these tests
    pytest.skip(
        'Cannot import str_replace_editor due to missing dependencies',
        allow_module_level=True,
    )


class TestWorkspacePathDetection:
    """Test workspace path detection logic."""

    def test_docker_runtime_ignores_sandbox_volumes(self):
        """Test that DockerRuntime ignores SANDBOX_VOLUMES and uses /workspace."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            result = _get_workspace_mount_path_from_env(runtime_type='docker')
            assert result == '/workspace'

    def test_remote_runtime_ignores_sandbox_volumes(self):
        """Test that RemoteRuntime ignores SANDBOX_VOLUMES and uses /workspace."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            result = _get_workspace_mount_path_from_env(runtime_type='remote')
            assert result == '/workspace'

    def test_e2b_runtime_ignores_sandbox_volumes(self):
        """Test that E2BRuntime ignores SANDBOX_VOLUMES and uses /workspace."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            result = _get_workspace_mount_path_from_env(runtime_type='e2b')
            assert result == '/workspace'

    def test_local_runtime_uses_sandbox_volumes(self):
        """Test that LocalRuntime uses host path from SANDBOX_VOLUMES."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            assert result == _abspath_from_env_host_fragment('/host/app')

    def test_local_runtime_multiple_mounts(self):
        """Test LocalRuntime with multiple mounts, finds /workspace mount."""
        with patch.dict(
            os.environ,
            {
                'SANDBOX_VOLUMES': '/tmp:/tmp:rw,/home/user/project:/workspace:rw,/var:/var:rw'
            },
        ):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            assert result == _abspath_from_env_host_fragment('/home/user/project')

    def test_cli_runtime_multiple_mounts(self):
        """Test CLIRuntime with multiple mounts, finds /workspace mount."""
        with patch.dict(
            os.environ,
            {
                'SANDBOX_VOLUMES': '/tmp:/tmp:rw,/home/user/project:/workspace:rw,/var:/var:rw'
            },
        ):
            result = _get_workspace_mount_path_from_env(runtime_type='cli')
            assert result == _abspath_from_env_host_fragment('/home/user/project')

    def test_local_runtime_no_workspace_mount(self):
        """Test LocalRuntime with no /workspace mount falls back to current directory."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/tmp:/tmp:rw,/var:/var:rw'}):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            assert result == os.getcwd()

    def test_local_runtime_no_sandbox_volumes(self):
        """Test LocalRuntime with no SANDBOX_VOLUMES falls back to current directory."""
        with patch.dict(os.environ, {}, clear=True):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            assert result == os.getcwd()

    def test_local_runtime_empty_sandbox_volumes(self):
        """Test LocalRuntime with empty SANDBOX_VOLUMES falls back to current directory."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': ''}):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            assert result == os.getcwd()

    def test_relative_path_conversion(self):
        """Test that relative paths in SANDBOX_VOLUMES are converted to absolute."""
        with patch.dict(
            os.environ, {'SANDBOX_VOLUMES': './relative/path:/workspace:rw'}
        ):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            # Should be converted to absolute path
            assert os.path.isabs(result)
            norm = result.replace('\\', '/')
            assert norm.endswith('relative/path')

    def test_unknown_runtime_type(self):
        """Test that unknown runtime types use default container path."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            result = _get_workspace_mount_path_from_env(runtime_type='unknown')
            assert result == '/workspace'

    def test_none_runtime_type(self):
        """Test that None runtime type uses default container path."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            result = _get_workspace_mount_path_from_env(runtime_type=None)
            assert result == '/workspace'


class TestToolCreation:
    """Test tool creation with dynamic workspace paths."""

    def test_tool_creation_docker_runtime(self):
        """Test tool creation in DockerRuntime shows /workspace in description."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            tool = create_str_replace_editor_tool(runtime_type='docker')
            path_description = tool['function']['parameters']['properties']['path'][
                'description'
            ]
            assert '/workspace/file.py' in path_description
            assert '/workspace`.' in path_description
            # Should not contain host paths
            assert '/host/app' not in path_description

    def test_tool_creation_local_runtime(self):
        """Test tool creation in LocalRuntime shows host path in description."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            tool = create_str_replace_editor_tool(runtime_type='local')
            path_description = tool['function']['parameters']['properties']['path'][
                'description'
            ]
            host = _abspath_from_env_host_fragment('/host/app')
            assert (
                f'{host}/file.py' in path_description
                or f'{host}\\file.py' in path_description
            )
            assert f'{host}`.' in path_description or f'{host}/`.' in path_description

    def test_tool_creation_cli_runtime(self):
        """Test tool creation in CLIRuntime shows host path in description."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            tool = create_str_replace_editor_tool(runtime_type='cli')
            path_description = tool['function']['parameters']['properties']['path'][
                'description'
            ]
            host = _abspath_from_env_host_fragment('/host/app')
            assert (
                f'{host}/file.py' in path_description
                or f'{host}\\file.py' in path_description
            )
            assert f'{host}`.' in path_description or f'{host}/`.' in path_description

    def test_tool_creation_remote_runtime(self):
        """Test tool creation in RemoteRuntime shows /workspace in description."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            tool = create_str_replace_editor_tool(runtime_type='remote')
            path_description = tool['function']['parameters']['properties']['path'][
                'description'
            ]
            assert '/workspace/file.py' in path_description
            assert '/workspace`.' in path_description
            # Should not contain host paths
            assert '/host/app' not in path_description

    def test_tool_creation_e2b_runtime(self):
        """Test tool creation in E2BRuntime shows /workspace in description."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            tool = create_str_replace_editor_tool(runtime_type='e2b')
            path_description = tool['function']['parameters']['properties']['path'][
                'description'
            ]
            assert '/workspace/file.py' in path_description
            assert '/workspace`.' in path_description
            # Should not contain host paths
            assert '/host/app' not in path_description

    def test_tool_creation_explicit_workspace_path(self):
        """Test tool creation with explicitly provided workspace path."""
        tool = create_str_replace_editor_tool(
            workspace_mount_path_in_sandbox='/custom/path'
        )
        path_description = tool['function']['parameters']['properties']['path'][
            'description'
        ]
        assert '/custom/path/file.py' in path_description
        assert '/custom/path`.' in path_description

    def test_tool_creation_short_description(self):
        """Test tool creation with short description still has correct path."""
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:/workspace:rw'}):
            tool = create_str_replace_editor_tool(
                use_short_description=True, runtime_type='local'
            )
            path_description = tool['function']['parameters']['properties']['path'][
                'description'
            ]
            host = _abspath_from_env_host_fragment('/host/app')
            assert (
                f'{host}/file.py' in path_description
                or f'{host}\\file.py' in path_description
            )
            assert f'{host}`.' in path_description or f'{host}/`.' in path_description


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_sandbox_volumes(self):
        """Test handling of malformed SANDBOX_VOLUMES."""
        # Missing colon separator - falls back to current directory for local/CLI
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app/workspace'}):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            assert result == os.getcwd()

        # Only one part - falls back to current directory for local/CLI
        with patch.dict(os.environ, {'SANDBOX_VOLUMES': '/host/app:'}):
            result = _get_workspace_mount_path_from_env(runtime_type='local')
            assert result == os.getcwd()

    def test_workspace_mount_with_different_permissions(self):
        """Test /workspace mount with different permission strings."""
        test_cases = [
            '/host/app:/workspace:ro',
            '/host/app:/workspace:rw',
            '/host/app:/workspace',  # No permissions
            '/host/app:/workspace:rw,size=100m',  # Extra options
        ]

        for sandbox_volumes in test_cases:
            with patch.dict(os.environ, {'SANDBOX_VOLUMES': sandbox_volumes}):
                result = _get_workspace_mount_path_from_env(runtime_type='local')
                assert result == _abspath_from_env_host_fragment('/host/app'), (
                    f'Failed for: {sandbox_volumes}'
                )
