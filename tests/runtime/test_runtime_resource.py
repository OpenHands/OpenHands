"""Stress tests for the DockerRuntime, which connects to the ActionExecutor running in the sandbox."""

import pytest
from conftest import _close_test_runtime, _load_runtime

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction


def test_stress_docker_runtime(temp_dir, runtime_cls, repeat=1):
    pytest.skip('This test is flaky')
    runtime, config = _load_runtime(
        temp_dir,
        runtime_cls,
        docker_runtime_kwargs={
            'cpu_period': 100000,  # 100ms
            'cpu_quota': 100000,  # Can use 100ms out of each 100ms period (1 CPU)
            'mem_limit': '4G',  # 4 GB of memory
        },
    )

    action = CmdRunAction(
        command='sudo apt-get update && sudo apt-get install -y stress-ng'
    )
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    for _ in range(repeat):
        # run stress-ng stress tests for 1 minute
        action = CmdRunAction(command='stress-ng --all 1 -t 30s')
        action.set_hard_timeout(120)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    _close_test_runtime(runtime)


# def test_stress_docker_runtime_hit_memory_limits(temp_dir, runtime_cls):
#     """Test runtime behavior under resource constraints."""
#     runtime, config = _load_runtime(
#         temp_dir,
#         runtime_cls,
#         docker_runtime_kwargs={
#             'cpu_period': 100000,  # 100ms
#             'cpu_quota': 100000,  # Can use 100ms out of each 100ms period (1 CPU)
#             'mem_limit': '4G',  # 4 GB of memory
#             'memswap_limit': '0',  # No swap
#             'mem_swappiness': 0,  # Disable swapping
#             'oom_kill_disable': False,  # Enable OOM killer
#         },
#         runtime_startup_env_vars={
#             'RUNTIME_MAX_MEMORY_GB': '3',
#         },
#     )

#     action = CmdRunAction(
#         command='sudo apt-get update && sudo apt-get install -y stress-ng'
#     )
#     logger.info(action, extra={'msg_type': 'ACTION'})
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})
#     assert obs.exit_code == 0

#     action = CmdRunAction(
#         command='stress-ng --vm 1 --vm-bytes 6G --timeout 30s --metrics'
#     )
#     action.set_hard_timeout(120)
#     logger.info(action, extra={'msg_type': 'ACTION'})
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})
#     assert 'aborted early, out of system resources' in obs.content
#     assert obs.exit_code == 3  # OOM killed!

#     _close_test_runtime(runtime)


# def test_stress_docker_runtime_within_memory_limits(temp_dir, runtime_cls):
#     """Test runtime behavior under resource constraints."""
#     runtime, config = _load_runtime(
#         temp_dir,
#         runtime_cls,
#         docker_runtime_kwargs={
#             'cpu_period': 100000,  # 100ms
#             'cpu_quota': 100000,  # Can use 100ms out of each 100ms period (1 CPU)
#             'mem_limit': '4G',  # 4 GB of memory
#             'memswap_limit': '0',  # No swap
#             'mem_swappiness': 0,  # Disable swapping
#             'oom_kill_disable': False,  # Enable OOM killer
#         },
#         runtime_startup_env_vars={
#             'RUNTIME_MAX_MEMORY_GB': '7',
#         },
#     )

#     action = CmdRunAction(
#         command='sudo apt-get update && sudo apt-get install -y stress-ng'
#     )
#     logger.info(action, extra={'msg_type': 'ACTION'})
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})
#     assert obs.exit_code == 0

#     action = CmdRunAction(
#         command='stress-ng --vm 1 --vm-bytes 6G --timeout 30s --metrics'
#     )
#     action.set_hard_timeout(120)
#     logger.info(action, extra={'msg_type': 'ACTION'})
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})
#     assert obs.exit_code == 0

#     _close_test_runtime(runtime)
