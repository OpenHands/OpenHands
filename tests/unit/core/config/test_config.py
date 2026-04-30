import logging
import os
from io import StringIO

import pytest

from openhands.core.config import (
    LLMConfig,
    OpenHandsConfig,
    finalize_config,
    get_llm_config_arg,
    load_from_env,
    load_from_toml,
)
from openhands.core.logger import openhands_logger


@pytest.fixture
def setup_env():
    # Create old-style and new-style TOML files
    with open('old_style_config.toml', 'w') as f:
        f.write('[default]\nLLM_MODEL="GPT-4"\n')

    with open('new_style_config.toml', 'w') as f:
        f.write('[app]\nLLM_MODEL="GPT-3"\n')

    yield

    # Cleanup TOML files after the test
    os.remove('old_style_config.toml')
    os.remove('new_style_config.toml')


@pytest.fixture
def temp_toml_file(tmp_path):
    # Fixture to create a temporary directory and TOML file for testing
    tmp_toml_file = os.path.join(tmp_path, 'config.toml')
    yield tmp_toml_file


@pytest.fixture
def default_config(monkeypatch):
    # Fixture to provide a default OpenHandsConfig instance
    yield OpenHandsConfig()


def test_compat_env_to_config(monkeypatch, setup_env):
    # Use `monkeypatch` to set environment variables for this specific test
    monkeypatch.setenv('LLM_API_KEY', 'sk-proj-rgMV0...')
    monkeypatch.setenv('LLM_MODEL', 'gpt-4o')
    monkeypatch.setenv('DEFAULT_AGENT', 'CodeActAgent')

    config = OpenHandsConfig()
    load_from_env(config, os.environ)
    finalize_config(config)

    assert isinstance(config.get_llm_config(), LLMConfig)
    assert config.get_llm_config().api_key.get_secret_value() == 'sk-proj-rgMV0...'
    assert config.get_llm_config().model == 'gpt-4o'
    assert config.default_agent == 'CodeActAgent'


def test_load_from_old_style_env(monkeypatch, default_config):
    # Test loading configuration from old-style environment variables using monkeypatch
    monkeypatch.setenv('LLM_API_KEY', 'test-api-key')
    monkeypatch.setenv('DEFAULT_AGENT', 'BrowsingAgent')
    # Using deprecated WORKSPACE_BASE to test backward compatibility
    monkeypatch.setenv('WORKSPACE_BASE', '/opt/files/workspace')

    load_from_env(default_config, os.environ)

    assert default_config.get_llm_config().api_key.get_secret_value() == 'test-api-key'
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
[llm]
model = "test-model"
api_key = "toml-api-key"

[llm.cheap]
model = "some-cheap-model"
api_key = "cheap-model-api-key"

[core]
default_agent = "TestAgent"
"""
        )

    load_from_toml(default_config, temp_toml_file)

    # default llm config
    assert default_config.default_agent == 'TestAgent'
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config().api_key.get_secret_value() == 'toml-api-key'


def test_llm_config_native_tool_calling(default_config, temp_toml_file, monkeypatch):
    # default is None
    assert default_config.get_llm_config().native_tool_calling is None

    # set to false
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]

[llm.gpt4o-mini]
native_tool_calling = false
"""
        )
    load_from_toml(default_config, temp_toml_file)
    assert default_config.get_llm_config().native_tool_calling is None
    assert default_config.get_llm_config('gpt4o-mini').native_tool_calling is False

    # set to true using string
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]

[llm.gpt4o-mini]
native_tool_calling = true
"""
        )
    load_from_toml(default_config, temp_toml_file)
    assert default_config.get_llm_config('gpt4o-mini').native_tool_calling is True

    # override to false by env
    # see utils.set_attr_from_env
    monkeypatch.setenv('LLM_NATIVE_TOOL_CALLING', 'false')
    load_from_env(default_config, os.environ)
    assert default_config.get_llm_config().native_tool_calling is False
    assert (
        default_config.get_llm_config('gpt4o-mini').native_tool_calling is True
    )  # load_from_env didn't override the named config set in the toml file under [llm.gpt4o-mini]


def test_env_overrides_compat_toml(monkeypatch, default_config, temp_toml_file):
    # test that environment variables override TOML values using monkeypatch
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm]
model = "test-model"
api_key = "toml-api-key"

[core]
disable_color = true
""")

    monkeypatch.setenv('LLM_API_KEY', 'env-api-key')
    monkeypatch.delenv('LLM_MODEL', raising=False)

    load_from_toml(default_config, temp_toml_file)
    load_from_env(default_config, os.environ)

    assert os.environ.get('LLM_MODEL') is None
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config('llm').model == 'test-model'
    assert default_config.get_llm_config().api_key.get_secret_value() == 'env-api-key'
    assert default_config.disable_color is True


def test_security_section_in_toml_is_silently_ignored(default_config, temp_toml_file):
    """Test that a legacy [security] section in TOML is silently ignored."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]
workspace_base = "/opt/files/workspace"

[llm]
model = "test-model"

[security]
confirmation_mode = false
security_analyzer = "semgrep"
"""
        )

    load_from_toml(default_config, temp_toml_file)
    # Security section is ignored; no error raised
    assert default_config.get_llm_config().model == 'test-model'


def test_defaults_dict_after_updates(default_config):
    # Test that `defaults_dict` retains initial values after updates.
    initial_defaults = default_config.defaults_dict
    assert initial_defaults['workspace_mount_path']['default'] is None
    assert initial_defaults['default_agent']['default'] == 'CodeActAgent'

    updated_config = OpenHandsConfig()
    updated_config.get_llm_config().api_key = 'updated-api-key'
    updated_config.get_llm_config('llm').api_key = 'updated-api-key'
    updated_config.default_agent = 'BrowsingAgent'

    defaults_after_updates = updated_config.defaults_dict
    assert defaults_after_updates['default_agent']['default'] == 'CodeActAgent'
    assert defaults_after_updates['workspace_mount_path']['default'] is None
    assert defaults_after_updates == initial_defaults


def test_invalid_toml_format(monkeypatch, temp_toml_file, default_config):
    # Invalid TOML format doesn't break the configuration
    monkeypatch.setenv('LLM_MODEL', 'gpt-5-turbo-1106')
    monkeypatch.setenv('WORKSPACE_MOUNT_PATH', '/home/user/project')
    monkeypatch.delenv('LLM_API_KEY', raising=False)

    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write('INVALID TOML CONTENT')

    load_from_toml(default_config, temp_toml_file)
    load_from_env(default_config, os.environ)
    default_config.jwt_secret = None  # prevent leak
    for llm in default_config.llms.values():
        llm.api_key = None  # prevent leak
    assert default_config.get_llm_config().model == 'gpt-5-turbo-1106'
    assert default_config.get_llm_config().custom_llm_provider is None
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
    assert default_config.get_llm_config() is not None


def test_core_not_in_toml(default_config, temp_toml_file):
    """Test loading configuration when the core section is not in the TOML file.

    default values should be used for the missing sections.
    """
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm]
model = "test-model"
""")

    load_from_toml(default_config, temp_toml_file)
    assert default_config.get_llm_config().model == 'test-model'


def test_load_from_toml_partial_invalid(default_config, temp_toml_file, caplog):
    """Test loading configuration with partially invalid TOML content.

    This ensures that:
    1. Valid configuration sections are properly loaded
    2. Invalid fields in LLM section log a warning but don't crash
    3. The config object maintains correct values for valid fields
    """
    with open(temp_toml_file, 'w', encoding='utf-8') as f:
        f.write("""
[core]
debug = true

[llm]
# Not set in LLMConfig
invalid_field = "test"
model = "gpt-4"

[agent]
enable_prompt_extensions = true
""")

    # Create a string buffer to capture log output
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    openhands_logger.addHandler(handler)

    try:
        load_from_toml(default_config, temp_toml_file)

        log_content = log_output.getvalue()

        # The LLM config should log a warning but not raise an exception
        assert 'Cannot parse [llm] config from toml' in log_content

        # Verify valid configurations are loaded
        assert default_config.debug is True
    finally:
        openhands_logger.removeHandler(handler)


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
    # Test LLMConfig
    llm_config = LLMConfig(
        api_key='my_api_key',
        aws_access_key_id='my_access_key',
        aws_secret_access_key='my_secret_key',
    )

    # Check that no secret keys are emitted in representations of the config object
    assert 'my_api_key' not in repr(llm_config)
    assert 'my_api_key' not in str(llm_config)
    assert 'my_access_key' not in repr(llm_config)
    assert 'my_access_key' not in str(llm_config)
    assert 'my_secret_key' not in repr(llm_config)
    assert 'my_secret_key' not in str(llm_config)

    # Check that no other attrs in LLMConfig have 'key' or 'token' in their name
    # This will fail when new attrs are added, and attract attention
    known_key_token_attrs_llm = [
        'api_key',
        'aws_access_key_id',
        'aws_secret_access_key',
        'input_cost_per_token',
        'output_cost_per_token',
        'custom_tokenizer',
    ]
    for attr_name in LLMConfig.model_fields.keys():
        if (
            not attr_name.startswith('__')
            and attr_name not in known_key_token_attrs_llm
        ):
            assert 'key' not in attr_name.lower(), (
                f"Unexpected attribute '{attr_name}' contains 'key' in LLMConfig"
            )
            assert 'token' not in attr_name.lower() or 'tokens' in attr_name.lower(), (
                f"Unexpected attribute '{attr_name}' contains 'token' in LLMConfig"
            )

    # Test OpenHandsConfig
    app_config = OpenHandsConfig(
        llms={'llm': llm_config},
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


def test_get_llm_config_arg(temp_toml_file):
    temp_toml = """
[core]
max_iterations = 100
max_budget_per_task = 4.0

[llm.gpt3]
model="gpt-3.5-turbo"
api_key="redacted"

[llm.gpt4o]
model="gpt-4o"
api_key="redacted"
"""

    with open(temp_toml_file, 'w') as f:
        f.write(temp_toml)

    llm_config = get_llm_config_arg('gpt3', temp_toml_file)
    assert llm_config.model == 'gpt-3.5-turbo'
