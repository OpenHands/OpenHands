from unittest.mock import MagicMock, patch

from openhands.core.config import (
    OH_DEFAULT_AGENT,
    OH_MAX_ITERATIONS,
    OpenHandsConfig,
    setup_config_from_args,
)


def test_default_values_applied_when_none():
    """Test that default values are applied when config values are None."""
    # Create mock args with None values for agent_cls and max_iterations
    mock_args = MagicMock()
    mock_args.config_file = None
    mock_args.agent_cls = None
    mock_args.max_iterations = None

    # Load config
    with patch(
        'openhands.core.config.utils.load_openhands_config',
        return_value=OpenHandsConfig(),
    ):
        config = setup_config_from_args(mock_args)

    # Verify they match the expected defaults
    assert config.default_agent == OH_DEFAULT_AGENT
    assert config.max_iterations == OH_MAX_ITERATIONS


def test_cli_args_override_defaults():
    """Test that CLI arguments override default values."""
    # Create mock args with custom values
    mock_args = MagicMock()
    mock_args.config_file = None
    mock_args.agent_cls = 'CustomAgent'
    mock_args.max_iterations = 50

    # Load config
    with patch(
        'openhands.core.config.utils.load_openhands_config',
        return_value=OpenHandsConfig(),
    ):
        config = setup_config_from_args(mock_args)

    # Verify custom values are used instead of defaults
    assert config.default_agent == 'CustomAgent'
    assert config.max_iterations == 50


def test_cli_args_none_uses_config_toml_values():
    """Test that when CLI args agent_cls and max_iterations are None, config.toml values are used."""
    # Create mock args with None values for agent_cls and max_iterations
    mock_args = MagicMock()
    mock_args.config_file = None
    mock_args.agent_cls = None
    mock_args.max_iterations = None

    # Create a config with specific values from config.toml
    config_from_toml = OpenHandsConfig()
    config_from_toml.default_agent = 'ConfigTomlAgent'
    config_from_toml.max_iterations = 100

    # Load config
    with patch(
        'openhands.core.config.utils.load_openhands_config',
        return_value=config_from_toml,
    ):
        config = setup_config_from_args(mock_args)

    # Verify config.toml values are preserved when CLI args are None
    assert config.default_agent == 'ConfigTomlAgent'
    assert config.max_iterations == 100
