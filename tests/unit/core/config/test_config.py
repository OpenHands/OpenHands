import os

import pytest

from openhands.core.config import (
    OpenHandsConfig,
    finalize_config,
    load_from_env,
    load_from_toml,
)


@pytest.fixture
def temp_toml_file(tmp_path):
    # Fixture to create a temporary directory and TOML file for testing
    tmp_toml_file = os.path.join(tmp_path, 'config.toml')
    yield tmp_toml_file


@pytest.fixture
def default_config(monkeypatch):
    # Fixture to provide a default OpenHandsConfig instance
    yield OpenHandsConfig()


def test_load_from_old_style_env(monkeypatch, default_config):
    # Test loading configuration from old-style environment variables using monkeypatch
    monkeypatch.setenv('DEFAULT_AGENT', 'BrowsingAgent')
    # Using deprecated WORKSPACE_BASE to test backward compatibility
    monkeypatch.setenv('WORKSPACE_BASE', '/opt/files/workspace')

    load_from_env(default_config, os.environ)

    assert default_config.default_agent == 'BrowsingAgent'
    # Verify deprecated variables still work
    assert default_config.workspace_base == '/opt/files/workspace'
    assert default_config.workspace_mount_path is None  # before finalize_config
    assert default_config.workspace_mount_path_in_sandbox is not None


def test_load_from_new_style_toml(default_config, temp_toml_file):
    # Test loading configuration from a new-style TOML file
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]
default_agent = "TestAgent"
"""
        )

    load_from_toml(default_config, temp_toml_file)

    assert default_config.default_agent == 'TestAgent'


def test_security_section_in_toml_is_silently_ignored(default_config, temp_toml_file):
    """Test that a legacy [security] section in TOML is silently ignored."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]
workspace_base = "/opt/files/workspace"

[security]
confirmation_mode = false
security_analyzer = "semgrep"
"""
        )

    load_from_toml(default_config, temp_toml_file)
    # Security section is ignored; no error raised
    assert default_config.workspace_base == '/opt/files/workspace'


def test_defaults_dict_after_updates(default_config):
    # Test that `defaults_dict` retains initial values after updates.
    initial_defaults = default_config.defaults_dict
    assert initial_defaults['workspace_mount_path']['default'] is None
    assert initial_defaults['default_agent']['default'] == 'CodeActAgent'

    updated_config = OpenHandsConfig()
    updated_config.default_agent = 'BrowsingAgent'

    defaults_after_updates = updated_config.defaults_dict
    assert defaults_after_updates['default_agent']['default'] == 'CodeActAgent'
    assert defaults_after_updates['workspace_mount_path']['default'] is None
    assert defaults_after_updates == initial_defaults


def test_invalid_toml_format(monkeypatch, temp_toml_file, default_config):
    # Invalid TOML format doesn't break the configuration
    monkeypatch.setenv('WORKSPACE_MOUNT_PATH', '/home/user/project')

    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write('INVALID TOML CONTENT')

    load_from_toml(default_config, temp_toml_file)
    load_from_env(default_config, os.environ)
    default_config.jwt_secret = None  # prevent leak
    assert default_config.workspace_mount_path == '/home/user/project'


def test_load_from_toml_file_not_found(default_config):
    """Test loading configuration when the TOML file doesn't exist.

    This ensures that:
    1. The program doesn't crash when the config file is missing
    2. The config object retains its default values
    3. The application remains usable
    """
    # Try to load from a non-existent file
    load_from_toml(default_config, 'nonexistent.toml')

    # Verify that config object maintains default values
    assert default_config is not None
    assert default_config.default_agent == 'CodeActAgent'


def test_finalize_config(default_config):
    # Test finalize config
    assert default_config.workspace_mount_path is None
    default_config.workspace_base = None
    finalize_config(default_config)

    assert default_config.workspace_mount_path is None


def test_workspace_mount_path_default(default_config):
    assert default_config.workspace_mount_path is None
    default_config.workspace_base = '/home/user/project'
    finalize_config(default_config)
    assert default_config.workspace_mount_path == os.path.abspath(
        default_config.workspace_base
    )


def test_workspace_mount_rewrite(default_config, monkeypatch):
    default_config.workspace_base = '/home/user/project'
    default_config.workspace_mount_rewrite = '/home/user:/sandbox'
    monkeypatch.setattr('os.getcwd', lambda: '/current/working/directory')
    finalize_config(default_config)
    assert default_config.workspace_mount_path == '/sandbox/project'


def test_cache_dir_creation(default_config, tmpdir):
    default_config.cache_dir = str(tmpdir.join('test_cache'))
    finalize_config(default_config)
    assert os.path.exists(default_config.cache_dir)


def test_api_keys_repr_str():
    # Test OpenHandsConfig
    app_config = OpenHandsConfig(
        search_api_key='my_search_api_key',
    )

    assert 'my_search_api_key' not in repr(app_config)
    assert 'my_search_api_key' not in str(app_config)

    # Check that no other attrs in OpenHandsConfig have 'key' or 'token' in their name
    # This will fail when new attrs are added, and attract attention
    known_key_token_attrs_app = [
        'search_api_key',
    ]
    for attr_name in OpenHandsConfig.model_fields.keys():
        if (
            not attr_name.startswith('__')
            and attr_name not in known_key_token_attrs_app
        ):
            assert 'key' not in attr_name.lower(), (
                f"Unexpected attribute '{attr_name}' contains 'key' in OpenHandsConfig"
            )
            assert 'token' not in attr_name.lower() or 'tokens' in attr_name.lower(), (
                f"Unexpected attribute '{attr_name}' contains 'token' in OpenHandsConfig"
            )


def test_max_iterations_and_max_budget_per_task_from_toml(temp_toml_file):
    temp_toml = """
[core]
max_iterations = 42
max_budget_per_task = 4.7
"""

    config = OpenHandsConfig()
    with open(temp_toml_file, 'w') as f:
        f.write(temp_toml)

    load_from_toml(config, temp_toml_file)

    assert config.max_iterations == 42
    assert config.max_budget_per_task == 4.7
