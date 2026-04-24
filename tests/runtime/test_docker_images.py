"""Image-related tests for the DockerRuntime, which connects to the ActionExecutor running in the sandbox."""

import os

import pytest
from conftest import _close_test_runtime, _load_runtime

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction

# ============================================================================================================================
# Image-specific tests
# ============================================================================================================================

# Skip all tests in this file if running with CLIRuntime or LocalRuntime,
# as these tests are specific to Docker images.
pytestmark = pytest.mark.skipif(
    os.environ.get('TEST_RUNTIME') in ['cli', 'local'],
    reason='Image tests are specific to DockerRuntime and not applicable to CLIRuntime or LocalRuntime.',
)


def test_bash_python_version(temp_dir, runtime_cls, base_container_image):
    """Make sure Python is available in bash."""
    if base_container_image not in [
        'python:3.12-bookworm',
    ]:
        pytest.skip('This test is only for python-related images')

    runtime, config = _load_runtime(
        temp_dir, runtime_cls, base_container_image=base_container_image
    )

    action = CmdRunAction(command='which python')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='python --version')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    assert 'Python 3.12' in obs.content  # Check for specific version

    action = CmdRunAction(command='pip --version')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    assert 'pip' in obs.content  # Check that pip is available

    _close_test_runtime(runtime)


def test_nodejs_22_version(temp_dir, runtime_cls, base_container_image):
    """Make sure Node.js is available in bash."""
    if base_container_image not in [
        'node:22-bookworm',
    ]:
        pytest.skip('This test is only for nodejs-related images')

    runtime, config = _load_runtime(
        temp_dir, runtime_cls, base_container_image=base_container_image
    )

    action = CmdRunAction(command='node --version')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    assert 'v22' in obs.content  # Check for specific version

    _close_test_runtime(runtime)


def test_go_version(temp_dir, runtime_cls, base_container_image):
    """Make sure Go is available in bash."""
    if base_container_image not in [
        'golang:1.23-bookworm',
    ]:
        pytest.skip('This test is only for go-related images')

    runtime, config = _load_runtime(
        temp_dir, runtime_cls, base_container_image=base_container_image
    )

    action = CmdRunAction(command='go version')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    assert 'go1.23' in obs.content  # Check for specific version

    _close_test_runtime(runtime)
