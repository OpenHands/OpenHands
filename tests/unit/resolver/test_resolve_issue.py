from unittest import mock

import pytest

from openhands.core.config import OpenHandsConfig, SandboxConfig
from openhands.events.action import CmdRunAction
from openhands.resolver.issue_resolver import IssueResolver


def assert_sandbox_config(
    config: SandboxConfig,
    base_container_image=SandboxConfig.model_fields['base_container_image'].default,
    runtime_container_image='ghcr.io/openhands/runtime:mock-nikolaik',  # Default to mock version
    local_runtime_url=SandboxConfig.model_fields['local_runtime_url'].default,
    enable_auto_lint=False,
):
    """Helper function to assert the properties of the SandboxConfig object."""
    assert isinstance(config, SandboxConfig)
    assert config.base_container_image == base_container_image
    assert config.runtime_container_image == runtime_container_image
    assert config.enable_auto_lint is enable_auto_lint
    assert config.use_host_network is False
    assert config.timeout == 300
    assert config.local_runtime_url == local_runtime_url


def test_setup_sandbox_config_default():
    """Test default configuration when no images provided and not experimental"""
    with mock.patch('openhands.__version__', 'mock'):
        openhands_config = OpenHandsConfig()

        IssueResolver.update_sandbox_config(
            openhands_config=openhands_config,
            base_container_image=None,
            runtime_container_image=None,
            is_experimental=False,
        )

        assert_sandbox_config(
            openhands_config.sandbox,
            runtime_container_image='ghcr.io/openhands/runtime:mock-nikolaik',
        )


def test_setup_sandbox_config_both_images():
    """Test that providing both container images raises ValueError"""
    with pytest.raises(
        ValueError, match='Cannot provide both runtime and base container images.'
    ):
        openhands_config = OpenHandsConfig()

        IssueResolver.update_sandbox_config(
            openhands_config=openhands_config,
            base_container_image='base-image',
            runtime_container_image='runtime-image',
            is_experimental=False,
        )


def test_setup_sandbox_config_base_only():
    """Test configuration when only base_container_image is provided"""
    base_image = 'custom-base-image'
    openhands_config = OpenHandsConfig()

    IssueResolver.update_sandbox_config(
        openhands_config=openhands_config,
        base_container_image=base_image,
        runtime_container_image=None,
        is_experimental=False,
    )

    assert_sandbox_config(
        openhands_config.sandbox,
        base_container_image=base_image,
        runtime_container_image=None,
    )


def test_setup_sandbox_config_runtime_only():
    """Test configuration when only runtime_container_image is provided"""
    runtime_image = 'custom-runtime-image'
    openhands_config = OpenHandsConfig()

    IssueResolver.update_sandbox_config(
        openhands_config=openhands_config,
        base_container_image=None,
        runtime_container_image=runtime_image,
        is_experimental=False,
    )

    assert_sandbox_config(
        openhands_config.sandbox, runtime_container_image=runtime_image
    )


def test_setup_sandbox_config_experimental():
    """Test configuration when experimental mode is enabled"""
    with mock.patch('openhands.__version__', 'mock'):
        openhands_config = OpenHandsConfig()

        IssueResolver.update_sandbox_config(
            openhands_config=openhands_config,
            base_container_image=None,
            runtime_container_image=None,
            is_experimental=True,
        )

        assert_sandbox_config(openhands_config.sandbox, runtime_container_image=None)


@mock.patch('openhands.resolver.issue_resolver.os.getuid', return_value=0)
@mock.patch('openhands.resolver.issue_resolver.get_unique_uid', return_value=1001)
def test_setup_sandbox_config_gitlab_ci(mock_get_unique_uid, mock_getuid):
    """Test GitLab CI specific configuration when running as root"""
    with mock.patch('openhands.__version__', 'mock'):
        with mock.patch.object(IssueResolver, 'GITLAB_CI', True):
            openhands_config = OpenHandsConfig()

            IssueResolver.update_sandbox_config(
                openhands_config=openhands_config,
                base_container_image=None,
                runtime_container_image=None,
                is_experimental=False,
            )

            assert_sandbox_config(
                openhands_config.sandbox, local_runtime_url='http://localhost'
            )


@mock.patch('openhands.resolver.issue_resolver.os.getuid', return_value=1000)
def test_setup_sandbox_config_gitlab_ci_non_root(mock_getuid):
    """Test GitLab CI configuration when not running as root"""
    with mock.patch('openhands.__version__', 'mock'):
        with mock.patch.object(IssueResolver, 'GITLAB_CI', True):
            openhands_config = OpenHandsConfig()

            IssueResolver.update_sandbox_config(
                openhands_config=openhands_config,
                base_container_image=None,
                runtime_container_image=None,
                is_experimental=False,
            )

            assert_sandbox_config(
                openhands_config.sandbox, local_runtime_url='http://localhost'
            )


@mock.patch('openhands.events.observation.CmdOutputObservation')
@mock.patch('openhands.runtime.base.Runtime')
def test_initialize_runtime_runs_setup_script_and_git_hooks(
    mock_runtime, mock_cmd_output
):
    """Test that initialize_runtime calls maybe_run_setup_script and maybe_setup_git_hooks"""

    # Create a minimal resolver instance with just the methods we need
    class MinimalResolver:
        def initialize_runtime(self, runtime):
            # This is the method we're testing
            action = CmdRunAction(command='git config --global core.pager ""')
            runtime.run_action(action)

            # Run setup script if it exists
            runtime.maybe_run_setup_script()

            # Setup git hooks if they exist
            runtime.maybe_setup_git_hooks()

    resolver = MinimalResolver()

    # Mock the runtime's run_action method to return a successful CmdOutputObservation
    mock_cmd_output.return_value.exit_code = 0
    mock_runtime.run_action.return_value = mock_cmd_output.return_value

    # Call the method
    resolver.initialize_runtime(mock_runtime)

    # Verify that both methods were called
    mock_runtime.maybe_run_setup_script.assert_called_once()
    mock_runtime.maybe_setup_git_hooks.assert_called_once()
