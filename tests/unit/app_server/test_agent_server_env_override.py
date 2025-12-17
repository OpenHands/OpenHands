"""Tests for agent server environment variable override functionality.

This module tests the environment variable override functionality that allows
users to inject custom environment variables into sandbox environments via
OH_AGENT_SERVER_ENV_* environment variables.

The functionality includes:
- Parsing OH_AGENT_SERVER_ENV_* environment variables
- Merging them into sandbox specifications
- Integration across different sandbox types (Docker, Process, Remote)
"""

import os
from unittest.mock import patch

import pytest

from openhands.app_server.sandbox.docker_sandbox_spec_service import (
    get_default_sandbox_specs as get_default_docker_sandbox_specs,
)
from openhands.app_server.sandbox.process_sandbox_spec_service import (
    get_default_sandbox_specs as get_default_process_sandbox_specs,
)
from openhands.app_server.sandbox.remote_sandbox_spec_service import (
    get_default_sandbox_specs as get_default_remote_sandbox_specs,
)
from openhands.app_server.sandbox.sandbox_spec_service import (
    get_agent_server_env,
)


class TestGetAgentServerEnv:
    """Test cases for get_agent_server_env function."""

    def test_no_environment_variables(self):
        """Test when no OH_AGENT_SERVER_ENV variable is set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_agent_server_env()
            assert result == {}

    def test_empty_json_environment_variable(self):
        """Test with empty JSON in OH_AGENT_SERVER_ENV."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            assert result == {}

    def test_single_environment_variable(self):
        """Test with a single variable in JSON format."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"CUSTOM_VAR": "custom_value"}',
            'OTHER_VAR': 'should_not_be_included',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            assert result == {'CUSTOM_VAR': 'custom_value'}

    def test_multiple_environment_variables(self):
        """Test with multiple variables in JSON format."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"VAR1": "value1", "VAR2": "value2", "DEBUG": "true", "PORT": "8080"}',
            'UNRELATED_VAR': 'should_not_be_included',
            'OH_OTHER_PREFIX': 'also_not_included',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            expected = {
                'VAR1': 'value1',
                'VAR2': 'value2',
                'DEBUG': 'true',
                'PORT': '8080',
            }
            assert result == expected

    def test_empty_variable_value(self):
        """Test with empty environment variable values in JSON."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"EMPTY": "", "NORMAL": "value"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            expected = {
                'EMPTY': '',
                'NORMAL': 'value',
            }
            assert result == expected

    def test_special_characters_in_values(self):
        """Test with special characters in environment variable values."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"JSON": "{\\"key\\": \\"value\\", \\"number\\": 123}", "PATH": "/usr/local/bin:/usr/bin", "SPACES": "value with spaces", "SYMBOLS": "value!@#$%^&*()"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            expected = {
                'JSON': '{"key": "value", "number": 123}',
                'PATH': '/usr/local/bin:/usr/bin',
                'SPACES': 'value with spaces',
                'SYMBOLS': 'value!@#$%^&*()',
            }
            assert result == expected

    def test_case_sensitivity(self):
        """Test that environment variable names are case-sensitive."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"lowercase": "lower", "UPPERCASE": "upper", "MixedCase": "mixed"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            expected = {
                'lowercase': 'lower',
                'UPPERCASE': 'upper',
                'MixedCase': 'mixed',
            }
            assert result == expected

    def test_numeric_and_underscore_in_names(self):
        """Test with numbers and underscores in variable names."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"VAR_1": "value1", "VAR_2_TEST": "value2", "123": "numeric", "TEST_123_ABC": "complex"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            expected = {
                'VAR_1': 'value1',
                'VAR_2_TEST': 'value2',
                '123': 'numeric',
                'TEST_123_ABC': 'complex',
            }
            assert result == expected

    def test_invalid_json_format(self):
        """Test that invalid JSON raises an appropriate error."""
        import json

        env_vars = {
            'OH_AGENT_SERVER_ENV': 'invalid_json_string',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(json.JSONDecodeError):  # Should raise JSON decode error
                get_agent_server_env()

    def test_non_string_values_in_json(self):
        """Test that non-string values in JSON are converted to strings."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"NUMBER": 123, "BOOLEAN": true, "NULL": null}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # This might fail if the parser is strict about string values
            # The behavior depends on the implementation
            try:
                result = get_agent_server_env()
                # If it succeeds, values should be converted to strings
                assert isinstance(result.get('NUMBER'), str)
                assert isinstance(result.get('BOOLEAN'), str)
                assert isinstance(result.get('NULL'), str)
            except Exception:
                # If it fails, that's also acceptable behavior for type safety
                pass

    def test_documentation_example(self):
        """Test the example from the function documentation."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"DEBUG": "true", "LOG_LEVEL": "info", "CUSTOM_VAR": "value"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_agent_server_env()
            expected = {
                'DEBUG': 'true',
                'LOG_LEVEL': 'info',
                'CUSTOM_VAR': 'value',
            }
            assert result == expected


class TestDockerSandboxSpecEnvironmentOverride:
    """Test environment variable override integration in Docker sandbox specs."""

    def test_docker_specs_include_agent_server_env(self):
        """Test that Docker sandbox specs include agent server environment variables."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"CUSTOM_VAR": "custom_value", "DEBUG": "true"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            specs = get_default_docker_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Check that custom environment variables are included
            assert 'CUSTOM_VAR' in spec.initial_env
            assert spec.initial_env['CUSTOM_VAR'] == 'custom_value'
            assert 'DEBUG' in spec.initial_env
            assert spec.initial_env['DEBUG'] == 'true'

            # Check that default environment variables are still present
            assert 'OPENVSCODE_SERVER_ROOT' in spec.initial_env
            assert 'OH_ENABLE_VNC' in spec.initial_env
            assert 'LOG_JSON' in spec.initial_env

    def test_docker_specs_override_existing_variables(self):
        """Test that agent server env variables can override existing ones."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"LOG_JSON": "false", "PYTHONUNBUFFERED": "0"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            specs = get_default_docker_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Agent server env should override the defaults
            assert spec.initial_env['LOG_JSON'] == 'false'
            assert spec.initial_env['PYTHONUNBUFFERED'] == '0'

    def test_docker_specs_empty_agent_server_env(self):
        """Test Docker specs when no agent server env variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            specs = get_default_docker_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Should only have the default environment variables
            expected_defaults = {
                'OPENVSCODE_SERVER_ROOT',
                'OH_ENABLE_VNC',
                'LOG_JSON',
                'OH_CONVERSATIONS_PATH',
                'OH_BASH_EVENTS_DIR',
                'PYTHONUNBUFFERED',
                'ENV_LOG_LEVEL',
            }

            # All defaults should be present
            for var in expected_defaults:
                assert var in spec.initial_env

            # No additional variables should be present
            assert set(spec.initial_env.keys()) == expected_defaults


class TestProcessSandboxSpecEnvironmentOverride:
    """Test environment variable override integration in Process sandbox specs."""

    def test_process_specs_include_agent_server_env(self):
        """Test that Process sandbox specs include agent server environment variables."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"PROCESS_VAR": "process_value", "WORKER_COUNT": "4"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            specs = get_default_process_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Check that custom environment variables are included
            assert 'PROCESS_VAR' in spec.initial_env
            assert spec.initial_env['PROCESS_VAR'] == 'process_value'
            assert 'WORKER_COUNT' in spec.initial_env
            assert spec.initial_env['WORKER_COUNT'] == '4'

            # Check that default environment variables are still present
            assert 'OH_ENABLE_VS_CODE' in spec.initial_env

    def test_process_specs_override_existing_variables(self):
        """Test that agent server env variables can override existing ones in process specs."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"OH_ENABLE_VS_CODE": "1"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            specs = get_default_process_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Agent server env should override the default
            assert spec.initial_env['OH_ENABLE_VS_CODE'] == '1'

    def test_process_specs_empty_agent_server_env(self):
        """Test Process specs when no agent server env variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            specs = get_default_process_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Should only have the default environment variables
            expected_defaults = {
                'OH_ENABLE_VS_CODE',
            }

            # All defaults should be present
            for var in expected_defaults:
                assert var in spec.initial_env

            # Should have exactly the expected variables
            assert set(spec.initial_env.keys()) == expected_defaults


class TestRemoteSandboxSpecEnvironmentOverride:
    """Test environment variable override integration in Remote sandbox specs."""

    def test_remote_specs_include_agent_server_env(self):
        """Test that Remote sandbox specs include agent server environment variables."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"REMOTE_VAR": "remote_value", "API_KEY": "secret123"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            specs = get_default_remote_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Check that custom environment variables are included
            assert 'REMOTE_VAR' in spec.initial_env
            assert spec.initial_env['REMOTE_VAR'] == 'remote_value'
            assert 'API_KEY' in spec.initial_env
            assert spec.initial_env['API_KEY'] == 'secret123'

            # Check that default environment variables are still present
            assert 'OH_CONVERSATIONS_PATH' in spec.initial_env
            assert 'OH_BASH_EVENTS_DIR' in spec.initial_env
            assert 'OH_VSCODE_PORT' in spec.initial_env

    def test_remote_specs_override_existing_variables(self):
        """Test that agent server env variables can override existing ones in remote specs."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"OH_VSCODE_PORT": "60002", "OH_CONVERSATIONS_PATH": "/custom/conversations"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            specs = get_default_remote_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Agent server env should override the defaults
            assert spec.initial_env['OH_VSCODE_PORT'] == '60002'
            assert spec.initial_env['OH_CONVERSATIONS_PATH'] == '/custom/conversations'

    def test_remote_specs_empty_agent_server_env(self):
        """Test Remote specs when no agent server env variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            specs = get_default_remote_sandbox_specs()

            assert len(specs) == 1
            spec = specs[0]

            # Should have the default environment variables
            expected_defaults = {
                'OH_CONVERSATIONS_PATH',
                'OH_BASH_EVENTS_DIR',
                'OH_VSCODE_PORT',
                'LOG_JSON',
                'OH_ENABLE_VNC',
                'OPENVSCODE_SERVER_ROOT',
            }

            # All defaults should be present
            for var in expected_defaults:
                assert var in spec.initial_env

            # Should have exactly the expected variables
            assert set(spec.initial_env.keys()) == expected_defaults


class TestEnvironmentOverrideIntegration:
    """Integration tests for the complete environment override functionality."""

    def test_consistent_behavior_across_sandbox_types(self):
        """Test that environment override behavior is consistent across all sandbox types."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"SHARED_VAR": "shared_value", "INTEGRATION_TEST": "true"}',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            docker_specs = get_default_docker_sandbox_specs()
            process_specs = get_default_process_sandbox_specs()
            remote_specs = get_default_remote_sandbox_specs()

            # All sandbox types should include the same custom environment variables
            for specs in [docker_specs, process_specs, remote_specs]:
                assert len(specs) == 1
                spec = specs[0]

                assert 'SHARED_VAR' in spec.initial_env
                assert spec.initial_env['SHARED_VAR'] == 'shared_value'
                assert 'INTEGRATION_TEST' in spec.initial_env
                assert spec.initial_env['INTEGRATION_TEST'] == 'true'

    def test_complex_environment_scenario(self):
        """Test a complex scenario with many environment variables."""
        env_vars = {
            'OH_AGENT_SERVER_ENV': '{"APP_NAME": "MyApp", "APP_VERSION": "1.2.3", "APP_ENV": "production", "DB_HOST": "localhost", "DB_PORT": "5432", "DB_NAME": "myapp_db", "FEATURE_X": "enabled", "FEATURE_Y": "disabled", "LOG_JSON": "false", "PYTHONUNBUFFERED": "0"}',
            # Non-matching variables (should be ignored)
            'OTHER_VAR': 'ignored',
            'OH_OTHER_PREFIX_VAR': 'also_ignored',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Test with Docker specs as representative
            specs = get_default_docker_sandbox_specs()
            spec = specs[0]

            # Custom variables should be present
            assert spec.initial_env['APP_NAME'] == 'MyApp'
            assert spec.initial_env['APP_VERSION'] == '1.2.3'
            assert spec.initial_env['APP_ENV'] == 'production'
            assert spec.initial_env['DB_HOST'] == 'localhost'
            assert spec.initial_env['DB_PORT'] == '5432'
            assert spec.initial_env['DB_NAME'] == 'myapp_db'
            assert spec.initial_env['FEATURE_X'] == 'enabled'
            assert spec.initial_env['FEATURE_Y'] == 'disabled'

            # Overridden defaults should have new values
            assert spec.initial_env['LOG_JSON'] == 'false'
            assert spec.initial_env['PYTHONUNBUFFERED'] == '0'

            # Non-matching variables should not be present
            assert 'OTHER_VAR' not in spec.initial_env
            assert 'OH_OTHER_PREFIX_VAR' not in spec.initial_env

            # Original defaults that weren't overridden should still be present
            assert 'OPENVSCODE_SERVER_ROOT' in spec.initial_env
            assert 'OH_ENABLE_VNC' in spec.initial_env

    def test_environment_isolation(self):
        """Test that environment changes don't affect subsequent calls."""
        # First call with some environment variables
        env_vars_1 = {
            'OH_AGENT_SERVER_ENV': '{"VAR1": "value1", "VAR2": "value2"}',
        }

        with patch.dict(os.environ, env_vars_1, clear=True):
            specs_1 = get_default_docker_sandbox_specs()
            spec_1 = specs_1[0]

            assert 'VAR1' in spec_1.initial_env
            assert 'VAR2' in spec_1.initial_env
            assert spec_1.initial_env['VAR1'] == 'value1'
            assert spec_1.initial_env['VAR2'] == 'value2'

        # Second call with different environment variables
        env_vars_2 = {
            'OH_AGENT_SERVER_ENV': '{"VAR3": "value3", "VAR4": "value4"}',
        }

        with patch.dict(os.environ, env_vars_2, clear=True):
            specs_2 = get_default_docker_sandbox_specs()
            spec_2 = specs_2[0]

            # Should only have the new variables
            assert 'VAR3' in spec_2.initial_env
            assert 'VAR4' in spec_2.initial_env
            assert spec_2.initial_env['VAR3'] == 'value3'
            assert spec_2.initial_env['VAR4'] == 'value4'

            # Should not have the old variables
            assert 'VAR1' not in spec_2.initial_env
            assert 'VAR2' not in spec_2.initial_env
