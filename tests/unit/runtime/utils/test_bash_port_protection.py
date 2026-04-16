"""Tests for BashSession port protection against self-termination.

See: https://github.com/OpenHands/OpenHands/issues/13927
"""

import pytest

from openhands.runtime.utils.bash import BashSession


@pytest.fixture
def bash_session_with_port():
    """Create a BashSession instance without initializing tmux, for unit testing."""
    session = BashSession(
        work_dir='/tmp',
        server_port=8000,
    )
    return session


@pytest.fixture
def bash_session_without_port():
    """Create a BashSession instance without a server port."""
    session = BashSession(
        work_dir='/tmp',
    )
    return session


class TestPortProtection:
    """Test _is_command_dangerous_for_server detects port-killing commands."""

    def test_fuser_kill_detected(self, bash_session_with_port):
        assert bash_session_with_port._is_command_dangerous_for_server(
            'fuser -k 8000/tcp'
        ) is not None

    def test_fuser_kill_flag_variants(self, bash_session_with_port):
        assert bash_session_with_port._is_command_dangerous_for_server(
            'fuser --kill 8000/tcp'
        ) is not None
        assert bash_session_with_port._is_command_dangerous_for_server(
            'fuser -k -n tcp 8000'
        ) is not None

    def test_kill_via_lsof(self, bash_session_with_port):
        assert bash_session_with_port._is_command_dangerous_for_server(
            'kill $(lsof -t -i:8000)'
        ) is not None
        assert bash_session_with_port._is_command_dangerous_for_server(
            'kill -9 $(lsof -t -i:8000)'
        ) is not None

    def test_lsof_piped_to_kill(self, bash_session_with_port):
        assert bash_session_with_port._is_command_dangerous_for_server(
            'lsof -i:8000 | awk \'{print $2}\' | xargs kill'
        ) is not None

    def test_lsof_piped_to_xargs_kill(self, bash_session_with_port):
        assert bash_session_with_port._is_command_dangerous_for_server(
            'lsof -t -i:8000 | xargs kill -9'
        ) is not None

    def test_safe_commands_not_blocked(self, bash_session_with_port):
        """Commands that don't kill processes on the server port should pass."""
        assert bash_session_with_port._is_command_dangerous_for_server(
            'ls -la'
        ) is None
        assert bash_session_with_port._is_command_dangerous_for_server(
            'python -m http.server 8080'
        ) is None
        assert bash_session_with_port._is_command_dangerous_for_server(
            'lsof -i:8000'
        ) is None  # Just listing, not killing
        assert bash_session_with_port._is_command_dangerous_for_server(
            'fuser 8000/tcp'
        ) is None  # No -k flag

    def test_different_port_not_blocked(self, bash_session_with_port):
        """Commands targeting a different port should not be blocked."""
        assert bash_session_with_port._is_command_dangerous_for_server(
            'fuser -k 3000/tcp'
        ) is None
        assert bash_session_with_port._is_command_dangerous_for_server(
            'kill $(lsof -t -i:3000)'
        ) is None

    def test_no_server_port_allows_all(self, bash_session_without_port):
        """When no server port is set, all commands are allowed."""
        assert bash_session_without_port._is_command_dangerous_for_server(
            'fuser -k 8000/tcp'
        ) is None

    def test_error_message_contains_port(self, bash_session_with_port):
        msg = bash_session_with_port._is_command_dangerous_for_server(
            'fuser -k 8000/tcp'
        )
        assert '8000' in msg
        assert 'different port' in msg

    def test_kill_all_port_variant(self, bash_session_with_port):
        """Test kill with lsof using different syntax."""
        # Both backtick and $() variants should be caught
        assert bash_session_with_port._is_command_dangerous_for_server(
            'kill -9 `lsof -t -i:8000`'
        ) is not None
        assert bash_session_with_port._is_command_dangerous_for_server(
            'kill -9 $(lsof -t -i:8000)'
        ) is not None

    def test_ss_piped_to_kill(self, bash_session_with_port):
        assert bash_session_with_port._is_command_dangerous_for_server(
            "ss -tlnp | grep 8000 | awk '{print $6}' | xargs kill"
        ) is not None
