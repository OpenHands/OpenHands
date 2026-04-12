"""Tests for app-server TOML config loading in Docker deployments."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset the global config before and after each test."""
    import openhands.app_server.config as config_module

    original_config = config_module._global_config
    config_module._global_config = None
    yield
    config_module._global_config = original_config


def _base_env() -> dict[str, str]:
    """Get a base environment with essential process variables preserved."""
    env = {}
    for key in ['PATH', 'HOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'TMPDIR', 'TMP', 'TEMP']:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content.strip() + '\n', encoding='utf-8')


def test_config_from_env_reads_explicit_toml_file(tmp_path: Path):
    """`OH_CONFIG_FILE` should load TOML sandbox overrides for docker mode."""
    config_file = tmp_path / 'docker.toml'
    _write_toml(
        config_file,
        """
        [sandbox]
        use_host_network = true
        volumes = "/tmp/from-toml:/workspace:rw,/tmp/readonly:/workspace/readonly:ro"
        runtime_extra_build_args = ["-p 80:80", "--publish=8080:8080"]
        """,
    )

    env = _base_env()
    env['OH_CONFIG_FILE'] = str(config_file)

    with patch.dict(os.environ, env, clear=True):
        from openhands.app_server.config import config_from_env
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )

        config = config_from_env()

    assert isinstance(config.sandbox, DockerSandboxServiceInjector)
    assert config.sandbox.use_host_network is True

    mounts_by_target = {
        mount.container_path: (mount.host_path, mount.mode)
        for mount in config.sandbox.mounts
    }
    assert mounts_by_target['/workspace'] == ('/tmp/from-toml', 'rw')
    assert mounts_by_target['/workspace/readonly'] == ('/tmp/readonly', 'ro')

    custom_ports = {
        exposed_port.container_port
        for exposed_port in config.sandbox.exposed_ports
        if exposed_port.name.startswith('CUSTOM_')
    }
    assert custom_ports == {80, 8080}


def test_environment_overrides_toml_sandbox_values(tmp_path: Path):
    """Environment variables should take precedence over TOML defaults."""
    config_file = tmp_path / 'docker.toml'
    _write_toml(
        config_file,
        """
        [sandbox]
        use_host_network = false
        volumes = "/tmp/from-toml:/workspace:rw"
        """,
    )

    env = _base_env()
    env['OH_CONFIG_FILE'] = str(config_file)
    env['AGENT_SERVER_USE_HOST_NETWORK'] = 'true'
    env['SANDBOX_VOLUMES'] = '/tmp/from-env:/workspace:ro'

    with patch.dict(os.environ, env, clear=True):
        from openhands.app_server.config import config_from_env

        config = config_from_env()

    assert config.sandbox is not None
    assert config.sandbox.use_host_network is True
    assert len(config.sandbox.mounts) == 1
    assert config.sandbox.mounts[0].host_path == '/tmp/from-env'
    assert config.sandbox.mounts[0].container_path == '/workspace'
    assert config.sandbox.mounts[0].mode == 'ro'


def test_docker_toml_is_preferred_over_config_toml(tmp_path: Path, monkeypatch):
    """Default resolution should prefer docker.toml before config.toml."""
    _write_toml(
        tmp_path / 'docker.toml',
        """
        [sandbox]
        use_host_network = true
        """,
    )
    _write_toml(
        tmp_path / 'config.toml',
        """
        [sandbox]
        use_host_network = false
        """,
    )

    monkeypatch.chdir(tmp_path)
    env = _base_env()

    with patch.dict(os.environ, env, clear=True):
        from openhands.app_server.config import config_from_env

        config = config_from_env()

    assert config.sandbox is not None
    assert config.sandbox.use_host_network is True


def test_app_server_section_overrides_supported(tmp_path: Path):
    """Support [app_server] and [app_server.sandbox] TOML sections."""
    config_file = tmp_path / 'docker.toml'
    _write_toml(
        config_file,
        """
        [app_server]
        web_url = "https://openhands.example"
        permitted_cors_origins = ["https://frontend.example"]

        [app_server.sandbox]
        use_host_network = true
        """,
    )

    env = _base_env()
    env['OH_CONFIG_FILE'] = str(config_file)

    with patch.dict(os.environ, env, clear=True):
        from openhands.app_server.config import config_from_env

        config = config_from_env()

    assert config.web_url == 'https://openhands.example'
    assert config.permitted_cors_origins == ['https://frontend.example']
    assert config.sandbox is not None
    assert config.sandbox.use_host_network is True
