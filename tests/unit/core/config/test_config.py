import logging
import os
from io import StringIO

import pytest

from openhands.core.config import (
    AgentConfig,
    LLMConfig,
    OpenHandsConfig,
    finalize_config,
    get_agent_config_arg,
    get_llm_config_arg,
    load_from_env,
    load_from_toml,
    load_openhands_config,
)
from openhands.core.config.condenser_config import (
    ConversationWindowCondenserConfig,
    LLMSummarizingCondenserConfig,
    NoOpCondenserConfig,
    RecentEventsCondenserConfig,
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
    monkeypatch.setenv('SANDBOX_VOLUMES', '/repos/openhands/workspace:/workspace:rw')
    monkeypatch.setenv('LLM_API_KEY', 'sk-proj-rgMV0...')
    monkeypatch.setenv('LLM_MODEL', 'gpt-4o')
    monkeypatch.setenv('DEFAULT_AGENT', 'CodeActAgent')
    monkeypatch.setenv('SANDBOX_TIMEOUT', '10')

    config = OpenHandsConfig()
    load_from_env(config, os.environ)
    finalize_config(config)

    assert config.sandbox.volumes == '/repos/openhands/workspace:/workspace:rw'
    # Check that the old parameters are set for backward compatibility
    assert config.workspace_base == os.path.abspath('/repos/openhands/workspace')
    assert config.workspace_mount_path == os.path.abspath('/repos/openhands/workspace')
    assert config.workspace_mount_path_in_sandbox == '/workspace'
    assert isinstance(config.get_llm_config(), LLMConfig)
    assert config.get_llm_config().api_key.get_secret_value() == 'sk-proj-rgMV0...'
    assert config.get_llm_config().model == 'gpt-4o'
    assert isinstance(config.get_agent_config(), AgentConfig)
    assert config.default_agent == 'CodeActAgent'
    assert config.sandbox.timeout == 10


def test_load_from_old_style_env(monkeypatch, default_config):
    # Test loading configuration from old-style environment variables using monkeypatch
    monkeypatch.setenv('LLM_API_KEY', 'test-api-key')
    monkeypatch.setenv('DEFAULT_AGENT', 'BrowsingAgent')
    # Using deprecated WORKSPACE_BASE to test backward compatibility
    monkeypatch.setenv('WORKSPACE_BASE', '/opt/files/workspace')
    monkeypatch.setenv('SANDBOX_BASE_CONTAINER_IMAGE', 'custom_image')

    load_from_env(default_config, os.environ)

    assert default_config.get_llm_config().api_key.get_secret_value() == 'test-api-key'
    assert default_config.default_agent == 'BrowsingAgent'
    # Verify deprecated variables still work
    assert default_config.workspace_base == '/opt/files/workspace'
    assert default_config.workspace_mount_path is None  # before finalize_config
    assert default_config.workspace_mount_path_in_sandbox is not None
    assert default_config.sandbox.base_container_image == 'custom_image'


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

[agent]
enable_prompt_extensions = true

[agent.BrowsingAgent]
llm_config = "cheap"
enable_prompt_extensions = false

[sandbox]
timeout = 1
volumes = "/opt/files2/workspace:/workspace:rw"

[core]
default_agent = "TestAgent"
"""
        )

    load_from_toml(default_config, temp_toml_file)

    # default llm & agent configs
    assert default_config.default_agent == 'TestAgent'
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config().api_key.get_secret_value() == 'toml-api-key'
    assert default_config.get_agent_config().enable_prompt_extensions is True

    # undefined agent config inherits default ones
    assert (
        default_config.get_llm_config_from_agent('CodeActAgent')
        == default_config.get_llm_config()
    )
    assert (
        default_config.get_agent_config('CodeActAgent').enable_prompt_extensions is True
    )

    # defined agent config overrides default ones
    assert default_config.get_llm_config_from_agent(
        'BrowsingAgent'
    ) == default_config.get_llm_config('cheap')
    assert (
        default_config.get_llm_config_from_agent('BrowsingAgent').model
        == 'some-cheap-model'
    )
    assert (
        default_config.get_agent_config('BrowsingAgent').enable_prompt_extensions
        is False
    )

    assert default_config.sandbox.volumes == '/opt/files2/workspace:/workspace:rw'
    assert default_config.sandbox.timeout == 1

    assert default_config.workspace_mount_path is None
    assert default_config.workspace_mount_path_in_sandbox is not None
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'

    finalize_config(default_config)

    # after finalize_config, workspace_mount_path is set based on sandbox.volumes
    assert default_config.workspace_mount_path == os.path.abspath(
        '/opt/files2/workspace'
    )
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'


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
    # uses a toml file with sandbox_vars instead of a sandbox section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm]
model = "test-model"
api_key = "toml-api-key"

[core]
disable_color = true

[sandbox]
volumes = "/opt/files3/workspace:/workspace:rw"
timeout = 500
user_id = 1001
""")

    monkeypatch.setenv('LLM_API_KEY', 'env-api-key')
    monkeypatch.setenv('SANDBOX_VOLUMES', '/tmp/test:/workspace:ro')
    monkeypatch.setenv('SANDBOX_TIMEOUT', '1000')
    monkeypatch.setenv('SANDBOX_USER_ID', '1002')
    monkeypatch.delenv('LLM_MODEL', raising=False)

    load_from_toml(default_config, temp_toml_file)

    assert default_config.workspace_mount_path is None

    load_from_env(default_config, os.environ)

    assert os.environ.get('LLM_MODEL') is None
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config('llm').model == 'test-model'
    assert default_config.get_llm_config_from_agent().model == 'test-model'
    assert default_config.get_llm_config().api_key.get_secret_value() == 'env-api-key'

    # Environment variable should override TOML value
    assert default_config.sandbox.volumes == '/tmp/test:/workspace:ro'
    assert default_config.workspace_mount_path is None

    assert default_config.disable_color is True
    assert default_config.sandbox.timeout == 1000
    assert default_config.sandbox.user_id == 1002

    finalize_config(default_config)
    # after finalize_config, workspace_mount_path is set based on the sandbox.volumes
    assert default_config.workspace_mount_path == os.path.abspath('/tmp/test')
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'


def test_env_overrides_sandbox_toml(monkeypatch, default_config, temp_toml_file):
    # test that environment variables override TOML values using monkeypatch
    # uses a toml file with a sandbox section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm]
model = "test-model"
api_key = "toml-api-key"

[core]

[sandbox]
volumes = "/opt/files3/workspace:/workspace:rw"
timeout = 500
user_id = 1001
""")

    monkeypatch.setenv('LLM_API_KEY', 'env-api-key')
    monkeypatch.setenv('SANDBOX_VOLUMES', '/tmp/test:/workspace:ro')
    monkeypatch.setenv('SANDBOX_TIMEOUT', '1000')
    monkeypatch.setenv('SANDBOX_USER_ID', '1002')
    monkeypatch.delenv('LLM_MODEL', raising=False)

    load_from_toml(default_config, temp_toml_file)

    assert default_config.workspace_mount_path is None

    # before load_from_env, values are set to the values from the toml file
    assert default_config.get_llm_config().api_key.get_secret_value() == 'toml-api-key'
    assert default_config.sandbox.volumes == '/opt/files3/workspace:/workspace:rw'
    assert default_config.sandbox.timeout == 500
    assert default_config.sandbox.user_id == 1001

    load_from_env(default_config, os.environ)

    # values from env override values from toml
    assert os.environ.get('LLM_MODEL') is None
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config().api_key.get_secret_value() == 'env-api-key'
    assert default_config.sandbox.volumes == '/tmp/test:/workspace:ro'
    assert default_config.sandbox.timeout == 1000
    assert default_config.sandbox.user_id == 1002

    finalize_config(default_config)
    # after finalize_config, workspace_mount_path is set based on sandbox.volumes
    assert default_config.workspace_mount_path == os.path.abspath('/tmp/test')
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'


def test_sandbox_config_from_toml(monkeypatch, default_config, temp_toml_file):
    # Test loading configuration from a new-style TOML file
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]

[llm]
model = "test-model"

[sandbox]
volumes = "/opt/files/workspace:/workspace:rw"
timeout = 1
base_container_image = "custom_image"
user_id = 1001
"""
        )
    monkeypatch.setattr(os, 'environ', {})
    load_from_toml(default_config, temp_toml_file)
    load_from_env(default_config, os.environ)
    finalize_config(default_config)

    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.sandbox.volumes == '/opt/files/workspace:/workspace:rw'
    assert default_config.workspace_mount_path == os.path.abspath(
        '/opt/files/workspace'
    )
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'
    assert default_config.sandbox.timeout == 1
    assert default_config.sandbox.base_container_image == 'custom_image'
    assert default_config.sandbox.user_id == 1001


def test_load_from_env_with_list(monkeypatch, default_config):
    """Test loading list values from environment variables, particularly SANDBOX_RUNTIME_EXTRA_BUILD_ARGS."""
    # Set the environment variable with a list-formatted string
    monkeypatch.setenv(
        'SANDBOX_RUNTIME_EXTRA_BUILD_ARGS',
        '['
        + '  "--add-host=host.docker.internal:host-gateway",'
        + '  "--build-arg=https_proxy=https://my-proxy:912",'
        + ']',
    )

    # Load configuration from environment
    load_from_env(default_config, os.environ)

    # Verify that the list was correctly parsed
    assert isinstance(default_config.sandbox.runtime_extra_build_args, list)
    assert len(default_config.sandbox.runtime_extra_build_args) == 2
    assert (
        '--add-host=host.docker.internal:host-gateway'
        in default_config.sandbox.runtime_extra_build_args
    )
    assert (
        '--build-arg=https_proxy=https://my-proxy:912'
        in default_config.sandbox.runtime_extra_build_args
    )


def test_security_config_from_toml(default_config, temp_toml_file):
    """Test loading security specific configurations."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]  # make sure core is loaded first
workspace_base = "/opt/files/workspace"

[llm]
model = "test-model"

[security]
confirmation_mode = false
security_analyzer = "semgrep"
"""
        )

    load_from_toml(default_config, temp_toml_file)
    assert default_config.security.confirmation_mode is False
    assert default_config.security.security_analyzer == 'semgrep'


def test_security_config_from_dict():
    """Test creating SecurityConfig instance from dictionary."""
    from openhands.core.config.security_config import SecurityConfig

    # Test with all fields
    config_dict = {
        'confirmation_mode': True,
        'security_analyzer': 'some_analyzer',
    }

    security_config = SecurityConfig(**config_dict)

    # Verify all fields are correctly set
    assert security_config.confirmation_mode is True
    assert security_config.security_analyzer == 'some_analyzer'


def test_defaults_dict_after_updates(default_config):
    # Test that `defaults_dict` retains initial values after updates.
    initial_defaults = default_config.defaults_dict
    assert initial_defaults['workspace_mount_path']['default'] is None
    assert initial_defaults['default_agent']['default'] == 'CodeActAgent'

    updated_config = OpenHandsConfig()
    updated_config.get_llm_config().api_key = 'updated-api-key'
    updated_config.get_llm_config('llm').api_key = 'updated-api-key'
    updated_config.get_llm_config_from_agent('agent').api_key = 'updated-api-key'
    updated_config.get_llm_config_from_agent(
        'BrowsingAgent'
    ).api_key = 'updated-api-key'
    updated_config.default_agent = 'BrowsingAgent'

    defaults_after_updates = updated_config.defaults_dict
    assert defaults_after_updates['default_agent']['default'] == 'CodeActAgent'
    assert defaults_after_updates['workspace_mount_path']['default'] is None
    assert defaults_after_updates['sandbox']['timeout']['default'] == 120
    assert (
        defaults_after_updates['sandbox']['base_container_image']['default']
        == 'nikolaik/python-nodejs:python3.12-nodejs22-slim'
    )
    assert defaults_after_updates == initial_defaults


def test_sandbox_volumes(monkeypatch, default_config):
    # Test SANDBOX_VOLUMES with multiple mounts (no explicit /workspace mount)
    monkeypatch.setenv(
        'SANDBOX_VOLUMES',
        '/host/path1:/container/path1,/host/path2:/container/path2:ro',
    )
    # Clear any existing workspace mount path to test default behavior
    monkeypatch.delenv('WORKSPACE_MOUNT_PATH_IN_SANDBOX', raising=False)

    load_from_env(default_config, os.environ)
    finalize_config(default_config)

    # Check that sandbox.volumes is set correctly
    assert (
        default_config.sandbox.volumes
        == '/host/path1:/container/path1,/host/path2:/container/path2:ro'
    )

    # With the new behavior, workspace_base and workspace_mount_path should be None
    # when no explicit /workspace mount is found
    assert default_config.workspace_base is None
    assert default_config.workspace_mount_path is None
    assert (
        default_config.workspace_mount_path_in_sandbox == '/workspace'
    )  # Default value


def test_sandbox_volumes_with_mode(monkeypatch, default_config):
    # Test SANDBOX_VOLUMES with read-only mode (no explicit /workspace mount)
    monkeypatch.setenv('SANDBOX_VOLUMES', '/host/path1:/container/path1:ro')
    # Clear any existing workspace mount path to test default behavior
    monkeypatch.delenv('WORKSPACE_MOUNT_PATH_IN_SANDBOX', raising=False)

    load_from_env(default_config, os.environ)
    finalize_config(default_config)

    # Check that sandbox.volumes is set correctly
    assert default_config.sandbox.volumes == '/host/path1:/container/path1:ro'

    # With the new behavior, workspace_base and workspace_mount_path should be None
    # when no explicit /workspace mount is found
    assert default_config.workspace_base is None
    assert default_config.workspace_mount_path is None
    assert (
        default_config.workspace_mount_path_in_sandbox == '/workspace'
    )  # Default value


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
    assert default_config.get_agent_config() is not None
    assert default_config.sandbox is not None


def test_core_not_in_toml(default_config, temp_toml_file):
    """Test loading configuration when the core section is not in the TOML file.

    default values should be used for the missing sections.
    """
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm]
model = "test-model"

[agent]
enable_prompt_extensions = true

[sandbox]
timeout = 1
base_container_image = "custom_image"
user_id = 1001

[security]
security_analyzer = "semgrep"
""")

    load_from_toml(default_config, temp_toml_file)
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_agent_config().enable_prompt_extensions is True
    assert default_config.sandbox.base_container_image == 'custom_image'
    assert default_config.sandbox.user_id == 1001
    assert default_config.security.security_analyzer == 'semgrep'


def test_load_from_toml_partial_invalid(default_config, temp_toml_file, caplog):
    """Test loading configuration with partially invalid TOML content.

    This ensures that:
    1. Valid configuration sections are properly loaded
    2. Invalid fields in security and sandbox sections raise ValueError
    4. The config object maintains correct values for valid fields
    """
    with open(temp_toml_file, 'w', encoding='utf-8') as f:
        f.write("""
[core]
debug = true

[llm]
# Not set in `openhands/core/schema/config.py`
invalid_field = "test"
model = "gpt-4"

[agent]
enable_prompt_extensions = true

[sandbox]
invalid_field_in_sandbox = "test"
""")

    # Create a string buffer to capture log output
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    openhands_logger.addHandler(handler)

    try:
        # Since sandbox_config.from_toml_section now raises ValueError for invalid fields,
        # we need to catch that exception
        with pytest.raises(ValueError) as excinfo:
            load_from_toml(default_config, temp_toml_file)

        # Verify the error message mentions the invalid sandbox field
        assert 'Error in [sandbox] section in config.toml' in str(excinfo.value)

        log_content = log_output.getvalue()

        # The LLM config should still log a warning but not raise an exception
        assert 'Cannot parse [llm] config from toml' in log_content

        # Verify valid configurations are loaded before the error was raised
        assert default_config.debug is True
    finally:
        openhands_logger.removeHandler(handler)


def test_load_from_toml_security_invalid(default_config, temp_toml_file):
    """Test that invalid security configuration raises ValueError."""
    with open(temp_toml_file, 'w', encoding='utf-8') as f:
        f.write("""
[core]
debug = true

[security]
invalid_security_field = "test"
""")

    with pytest.raises(ValueError) as excinfo:
        load_from_toml(default_config, temp_toml_file)

    assert 'Error in [security] section in config.toml' in str(excinfo.value)


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


def test_sandbox_volumes_with_workspace(default_config):
    """Test that sandbox.volumes with explicit /workspace mount works correctly."""
    default_config.sandbox.volumes = '/home/user/mydir:/workspace:rw,/data:/data:ro'
    finalize_config(default_config)
    assert default_config.workspace_mount_path == '/home/user/mydir'
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'
    assert default_config.workspace_base == '/home/user/mydir'


def test_sandbox_volumes_without_workspace(default_config):
    """Test that sandbox.volumes without explicit /workspace mount doesn't set workspace paths."""
    default_config.sandbox.volumes = '/data:/data:ro,/models:/models:ro'
    finalize_config(default_config)
    assert default_config.workspace_mount_path is None
    assert default_config.workspace_base is None
    assert (
        default_config.workspace_mount_path_in_sandbox == '/workspace'
    )  # Default value remains unchanged


def test_sandbox_volumes_with_workspace_not_first(default_config):
    """Test that sandbox.volumes with /workspace mount not as first entry works correctly."""
    default_config.sandbox.volumes = (
        '/data:/data:ro,/home/user/mydir:/workspace:rw,/models:/models:ro'
    )
    finalize_config(default_config)
    assert default_config.workspace_mount_path == '/home/user/mydir'
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'
    assert default_config.workspace_base == '/home/user/mydir'


def test_agent_config_condenser_with_no_enabled():
    """Test default agent condenser with enable_default_condenser=False."""
    config = OpenHandsConfig(enable_default_condenser=False)
    agent_config = config.get_agent_config()
    assert isinstance(agent_config.condenser, ConversationWindowCondenserConfig)


def test_sandbox_volumes_toml(default_config, temp_toml_file):
    """Test that volumes configuration under [sandbox] works correctly."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[sandbox]
volumes = "/home/user/mydir:/workspace:rw,/data:/data:ro"
timeout = 1
""")

    load_from_toml(default_config, temp_toml_file)
    finalize_config(default_config)

    # Check that sandbox.volumes is set correctly
    assert (
        default_config.sandbox.volumes
        == '/home/user/mydir:/workspace:rw,/data:/data:ro'
    )
    assert default_config.workspace_mount_path == '/home/user/mydir'
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'
    assert default_config.workspace_base == '/home/user/mydir'
    assert default_config.sandbox.timeout == 1


def test_condenser_config_from_toml_basic(default_config, temp_toml_file):
    """Test loading basic condenser configuration from TOML."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[condenser]
type = "recent"
keep_first = 3
max_events = 15
""")

    load_from_toml(default_config, temp_toml_file)

    # Verify that the condenser config is correctly assigned to the default agent config
    agent_config = default_config.get_agent_config()
    assert isinstance(agent_config.condenser, RecentEventsCondenserConfig)
    assert agent_config.condenser.keep_first == 3
    assert agent_config.condenser.max_events == 15

    # We can also verify the function works directly
    from openhands.core.config.condenser_config import (
        condenser_config_from_toml_section,
    )

    condenser_data = {'type': 'recent', 'keep_first': 3, 'max_events': 15}
    condenser_mapping = condenser_config_from_toml_section(condenser_data)

    assert 'condenser' in condenser_mapping
    assert isinstance(condenser_mapping['condenser'], RecentEventsCondenserConfig)
    assert condenser_mapping['condenser'].keep_first == 3
    assert condenser_mapping['condenser'].max_events == 15


def test_condenser_config_from_toml_with_llm_reference(default_config, temp_toml_file):
    """Test loading condenser configuration with LLM reference from TOML."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm.condenser_llm]
model = "gpt-4"
api_key = "test-key"

[condenser]
type = "llm"
llm_config = "condenser_llm"
keep_first = 2
max_size = 50
""")

    load_from_toml(default_config, temp_toml_file)

    # Verify that the LLM config was loaded
    assert 'condenser_llm' in default_config.llms
    assert default_config.llms['condenser_llm'].model == 'gpt-4'

    # Verify that the condenser config is correctly assigned to the default agent config
    agent_config = default_config.get_agent_config()
    assert isinstance(agent_config.condenser, LLMSummarizingCondenserConfig)
    assert agent_config.condenser.keep_first == 2
    assert agent_config.condenser.max_size == 50
    assert agent_config.condenser.llm_config.model == 'gpt-4'

    # Test the condenser config with the LLM reference
    from openhands.core.config.condenser_config import (
        condenser_config_from_toml_section,
    )

    condenser_data = {
        'type': 'llm',
        'llm_config': 'condenser_llm',
        'keep_first': 2,
        'max_size': 50,
    }
    condenser_mapping = condenser_config_from_toml_section(
        condenser_data, default_config.llms
    )

    assert 'condenser' in condenser_mapping
    assert isinstance(condenser_mapping['condenser'], LLMSummarizingCondenserConfig)
    assert condenser_mapping['condenser'].keep_first == 2
    assert condenser_mapping['condenser'].max_size == 50
    assert condenser_mapping['condenser'].llm_config.model == 'gpt-4'


def test_condenser_config_from_toml_with_missing_llm_reference(
    default_config, temp_toml_file
):
    """Test loading condenser configuration with missing LLM reference from TOML."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[condenser]
type = "llm"
llm_config = "missing_llm"
keep_first = 2
max_size = 50
""")

    load_from_toml(default_config, temp_toml_file)

    # Test the condenser config with a missing LLM reference
    from openhands.core.config.condenser_config import (
        condenser_config_from_toml_section,
    )

    condenser_data = {
        'type': 'llm',
        'llm_config': 'missing_llm',
        'keep_first': 2,
        'max_size': 50,
    }
    condenser_mapping = condenser_config_from_toml_section(
        condenser_data, default_config.llms
    )

    assert 'condenser' in condenser_mapping
    assert isinstance(condenser_mapping['condenser'], NoOpCondenserConfig)
    # Should not have a default LLMConfig when the reference is missing
    assert not hasattr(condenser_mapping['condenser'], 'llm_config')


def test_condenser_config_from_toml_with_invalid_config(default_config, temp_toml_file):
    """Test loading invalid condenser configuration from TOML."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[condenser]
type = "invalid_type"
""")

    load_from_toml(default_config, temp_toml_file)

    # Test the condenser config with an invalid type
    from openhands.core.config.condenser_config import (
        condenser_config_from_toml_section,
    )

    condenser_data = {'type': 'invalid_type'}
    condenser_mapping = condenser_config_from_toml_section(condenser_data)

    # Should default to NoOpCondenserConfig when the type is invalid
    assert 'condenser' in condenser_mapping
    assert isinstance(condenser_mapping['condenser'], NoOpCondenserConfig)


def test_condenser_config_from_toml_with_validation_error(
    default_config, temp_toml_file
):
    """Test loading condenser configuration with validation error from TOML."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[condenser]
type = "recent"
keep_first = -1  # Invalid: must be >= 0
max_events = 0   # Invalid: must be >= 1
""")

    load_from_toml(default_config, temp_toml_file)

    # Test the condenser config with validation errors
    from openhands.core.config.condenser_config import (
        condenser_config_from_toml_section,
    )

    condenser_data = {'type': 'recent', 'keep_first': -1, 'max_events': 0}
    condenser_mapping = condenser_config_from_toml_section(condenser_data)

    # Should default to NoOpCondenserConfig when validation fails
    assert 'condenser' in condenser_mapping
    assert isinstance(condenser_mapping['condenser'], NoOpCondenserConfig)


def test_default_condenser_behavior_enabled(default_config, temp_toml_file):
    """Test the default condenser behavior when enable_default_condenser is True."""
    # Create a minimal TOML file with no condenser section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[core]
# Empty core section, no condenser section
""")

    # Set enable_default_condenser to True
    default_config.enable_default_condenser = True
    load_from_toml(default_config, temp_toml_file)

    # Verify the default agent config has LLMSummarizingCondenserConfig
    agent_config = default_config.get_agent_config()
    assert isinstance(agent_config.condenser, LLMSummarizingCondenserConfig)
    assert agent_config.condenser.keep_first == 1
    assert agent_config.condenser.max_size == 100


def test_default_condenser_behavior_disabled(default_config, temp_toml_file):
    """Test the default condenser behavior when enable_default_condenser is False."""
    # Create a minimal TOML file with no condenser section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[core]
# Empty core section, no condenser section
""")

    # Set enable_default_condenser to False
    default_config.enable_default_condenser = False
    load_from_toml(default_config, temp_toml_file)

    # Verify the agent config uses ConversationWindowCondenserConfig
    agent_config = default_config.get_agent_config()
    assert isinstance(agent_config.condenser, ConversationWindowCondenserConfig)


def test_default_condenser_explicit_toml_override(default_config, temp_toml_file):
    """Test that explicit condenser in TOML takes precedence over the default."""
    # Set enable_default_condenser to True
    default_config.enable_default_condenser = True

    # Create a TOML file with an explicit condenser section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[condenser]
type = "recent"
keep_first = 3
max_events = 15
""")

    # Load the config
    load_from_toml(default_config, temp_toml_file)

    # Verify the explicit condenser from TOML takes precedence
    agent_config = default_config.get_agent_config()
    assert isinstance(agent_config.condenser, RecentEventsCondenserConfig)
    assert agent_config.condenser.keep_first == 3
    assert agent_config.condenser.max_events == 15


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

    # Test AgentConfig
    # No attrs in AgentConfig have 'key' or 'token' in their name
    agent_config = AgentConfig(enable_prompt_extensions=True, enable_browsing=False)
    for attr_name in AgentConfig.model_fields.keys():
        if not attr_name.startswith('__'):
            assert 'key' not in attr_name.lower(), (
                f"Unexpected attribute '{attr_name}' contains 'key' in AgentConfig"
            )
            assert 'token' not in attr_name.lower() or 'tokens' in attr_name.lower(), (
                f"Unexpected attribute '{attr_name}' contains 'token' in AgentConfig"
            )

    # Test OpenHandsConfig
    app_config = OpenHandsConfig(
        llms={'llm': llm_config},
        agents={'agent': agent_config},
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


def test_get_agent_configs(default_config, temp_toml_file):
    temp_toml = """
[core]
max_iterations = 100
max_budget_per_task = 4.0

[agent.CodeActAgent]
enable_prompt_extensions = true

[agent.BrowsingAgent]
enable_jupyter = false
"""

    with open(temp_toml_file, 'w') as f:
        f.write(temp_toml)

    load_from_toml(default_config, temp_toml_file)

    codeact_config = default_config.get_agent_configs().get('CodeActAgent')
    assert codeact_config.enable_prompt_extensions is True
    browsing_config = default_config.get_agent_configs().get('BrowsingAgent')
    assert browsing_config.enable_jupyter is False


def test_get_agent_config_arg(temp_toml_file):
    temp_toml = """
[core]
max_iterations = 100
max_budget_per_task = 4.0

[agent.CodeActAgent]
enable_prompt_extensions = false
enable_browsing = false

[agent.BrowsingAgent]
enable_prompt_extensions = true
enable_jupyter = false
"""

    with open(temp_toml_file, 'w') as f:
        f.write(temp_toml)

    agent_config = get_agent_config_arg('CodeActAgent', temp_toml_file)
    assert not agent_config.enable_prompt_extensions
    assert not agent_config.enable_browsing

    agent_config2 = get_agent_config_arg('BrowsingAgent', temp_toml_file)
    assert agent_config2.enable_prompt_extensions
    assert not agent_config2.enable_jupyter


def test_agent_config_custom_group_name(temp_toml_file):
    temp_toml = """
[core]
max_iterations = 99

[agent.group1]
enable_prompt_extensions = true

[agent.group2]
enable_prompt_extensions = false
"""
    with open(temp_toml_file, 'w') as f:
        f.write(temp_toml)

    # just a sanity check that load app config wouldn't fail
    app_config = load_openhands_config(config_file=temp_toml_file)
    assert app_config.max_iterations == 99

    # run_infer in evaluation can use `get_agent_config_arg` to load custom
    # agent configs with any group name (not just agent name)
    agent_config1 = get_agent_config_arg('group1', temp_toml_file)
    assert agent_config1.enable_prompt_extensions
    agent_config2 = get_agent_config_arg('group2', temp_toml_file)
    assert not agent_config2.enable_prompt_extensions


def test_agent_config_from_toml_section():
    """Test that AgentConfig.from_toml_section correctly parses agent configurations from TOML."""
    from openhands.core.config.agent_config import AgentConfig

    # Test with base config and custom configs
    agent_section = {
        'enable_prompt_extensions': True,
        'enable_browsing': True,
        'CustomAgent1': {'enable_browsing': False},
        'CustomAgent2': {'enable_prompt_extensions': False},
        'InvalidAgent': {
            'invalid_field': 'some_value'  # This should be skipped but not affect others
        },
    }

    # Parse the section
    result = AgentConfig.from_toml_section(agent_section)

    # Verify the base config was correctly parsed
    assert 'agent' in result
    assert result['agent'].enable_prompt_extensions is True
    assert result['agent'].enable_browsing is True

    # Verify custom configs were correctly parsed and inherit from base
    assert 'CustomAgent1' in result
    assert result['CustomAgent1'].enable_browsing is False  # Overridden
    assert result['CustomAgent1'].enable_prompt_extensions is True  # Inherited

    assert 'CustomAgent2' in result
    assert result['CustomAgent2'].enable_browsing is True  # Inherited
    assert result['CustomAgent2'].enable_prompt_extensions is False  # Overridden

    # Verify the invalid config was skipped
    assert 'InvalidAgent' not in result


def test_agent_config_from_toml_section_with_invalid_base():
    """Test that AgentConfig.from_toml_section handles invalid base configurations gracefully."""
    from openhands.core.config.agent_config import AgentConfig

    # Test with invalid base config but valid custom configs
    agent_section = {
        'invalid_field': 'some_value',  # This should be ignored in base config
        'enable_jupyter': 'not_a_bool',  # This should cause validation error
        'CustomAgent': {
            'enable_browsing': False,
            'enable_jupyter': True,
        },
    }

    # Parse the section
    result = AgentConfig.from_toml_section(agent_section)

    # Verify a default base config was created despite the invalid fields
    assert 'agent' in result
    assert result['agent'].enable_browsing is True  # Default value
    assert result['agent'].enable_jupyter is True  # Default value

    # Verify custom config was still processed correctly
    assert 'CustomAgent' in result
    assert result['CustomAgent'].enable_browsing is False
    assert result['CustomAgent'].enable_jupyter is True


def test_agent_config_system_prompt_filename_default():
    """Test that AgentConfig defaults to 'system_prompt.j2' for system_prompt_filename."""
    config = AgentConfig()
    assert config.system_prompt_filename == 'system_prompt.j2'


def test_agent_config_system_prompt_filename_toml_integration(
    default_config, temp_toml_file
):
    """Test that system_prompt_filename is correctly loaded from TOML configuration."""
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[agent]
enable_browsing = true
system_prompt_filename = "custom_prompt.j2"

[agent.CodeReviewAgent]
system_prompt_filename = "code_review_prompt.j2"
enable_browsing = false
"""
        )

    load_from_toml(default_config, temp_toml_file)

    # Check default agent config
    default_agent_config = default_config.get_agent_config()
    assert default_agent_config.system_prompt_filename == 'custom_prompt.j2'
    assert default_agent_config.enable_browsing is True

    # Check custom agent config
    custom_agent_config = default_config.get_agent_config('CodeReviewAgent')
    assert custom_agent_config.system_prompt_filename == 'code_review_prompt.j2'
    assert custom_agent_config.enable_browsing is False
