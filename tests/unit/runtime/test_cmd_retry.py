"""Tests for command retry logic with exponential backoff.

Tests the fix for GitHub issue #12265 where GitHub token export fails
when bash session is busy (race condition).
"""

from unittest.mock import MagicMock, patch

import pytest

from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.runtime.base import (
    CMD_RETRY_BASE_DELAY_SECONDS,
    CMD_RETRY_MAX_ATTEMPTS,
    CMD_RETRY_TIMEOUT_EXIT_CODE,
)


class TestCmdRetryHelpers:
    """Tests for the helper methods used in command retry logic."""

    @pytest.fixture
    def mock_runtime(self):
        """Create a mock runtime with the retry methods."""
        from openhands.runtime.base import Runtime

        # Create a minimal mock that has the methods we need to test
        runtime = MagicMock(spec=Runtime)

        # Bind the actual methods to our mock
        runtime._is_bash_session_timeout = Runtime._is_bash_session_timeout.__get__(
            runtime, Runtime
        )
        runtime._calculate_retry_delay = Runtime._calculate_retry_delay.__get__(
            runtime, Runtime
        )
        runtime._extract_error_content = Runtime._extract_error_content.__get__(
            runtime, Runtime
        )
        runtime._run_cmd_with_retry = Runtime._run_cmd_with_retry.__get__(
            runtime, Runtime
        )

        return runtime

    def test_is_bash_session_timeout_with_timeout_exit_code(self, mock_runtime):
        """Test that timeout exit code (-1) is correctly identified."""
        obs = CmdOutputObservation(content='', command='test', exit_code=-1)
        assert mock_runtime._is_bash_session_timeout(obs) is True

    def test_is_bash_session_timeout_with_success(self, mock_runtime):
        """Test that successful commands are not identified as timeout."""
        obs = CmdOutputObservation(content='output', command='test', exit_code=0)
        assert mock_runtime._is_bash_session_timeout(obs) is False

    def test_is_bash_session_timeout_with_other_error(self, mock_runtime):
        """Test that other error codes are not identified as timeout."""
        obs = CmdOutputObservation(content='error', command='test', exit_code=1)
        assert mock_runtime._is_bash_session_timeout(obs) is False

    def test_is_bash_session_timeout_with_error_observation(self, mock_runtime):
        """Test that ErrorObservation is not identified as timeout."""
        obs = ErrorObservation(content='some error')
        assert mock_runtime._is_bash_session_timeout(obs) is False

    def test_calculate_retry_delay_exponential(self, mock_runtime):
        """Test exponential backoff delay calculation."""
        assert mock_runtime._calculate_retry_delay(0) == CMD_RETRY_BASE_DELAY_SECONDS
        assert (
            mock_runtime._calculate_retry_delay(1) == CMD_RETRY_BASE_DELAY_SECONDS * 2
        )
        assert (
            mock_runtime._calculate_retry_delay(2) == CMD_RETRY_BASE_DELAY_SECONDS * 4
        )

    def test_extract_error_content_from_none(self, mock_runtime):
        """Test error extraction from None observation."""
        assert mock_runtime._extract_error_content(None) == 'No observation received'

    def test_extract_error_content_from_cmd_output(self, mock_runtime):
        """Test error extraction from CmdOutputObservation."""
        obs = CmdOutputObservation(
            content='command failed', command='test', exit_code=1
        )
        result = mock_runtime._extract_error_content(obs)
        assert 'command failed' in result
        assert 'exit_code=1' in result

    def test_extract_error_content_from_error_observation(self, mock_runtime):
        """Test error extraction from ErrorObservation."""
        obs = ErrorObservation(content='something went wrong')
        assert mock_runtime._extract_error_content(obs) == 'something went wrong'


class TestRunCmdWithRetry:
    """Tests for the main _run_cmd_with_retry method."""

    @pytest.fixture
    def mock_runtime(self):
        """Create a mock runtime for retry testing."""
        from openhands.runtime.base import Runtime

        runtime = MagicMock(spec=Runtime)

        # Bind actual methods
        runtime._is_bash_session_timeout = Runtime._is_bash_session_timeout.__get__(
            runtime, Runtime
        )
        runtime._calculate_retry_delay = Runtime._calculate_retry_delay.__get__(
            runtime, Runtime
        )
        runtime._extract_error_content = Runtime._extract_error_content.__get__(
            runtime, Runtime
        )
        runtime._run_cmd_with_retry = Runtime._run_cmd_with_retry.__get__(
            runtime, Runtime
        )

        return runtime

    def test_success_on_first_attempt(self, mock_runtime):
        """Test successful command execution on first try."""
        success_obs = CmdOutputObservation(
            content='success', command='echo test', exit_code=0
        )
        mock_runtime.run = MagicMock(return_value=success_obs)

        result = mock_runtime._run_cmd_with_retry('echo test', 'Test error')

        assert result == success_obs
        assert mock_runtime.run.call_count == 1

    @patch('openhands.runtime.base.time.sleep')
    def test_retry_on_timeout_then_success(self, mock_sleep, mock_runtime):
        """Test retry behavior when first attempt times out."""
        timeout_obs = CmdOutputObservation(
            content='', command='export VAR=value', exit_code=-1
        )
        success_obs = CmdOutputObservation(
            content='', command='export VAR=value', exit_code=0
        )

        mock_runtime.run = MagicMock(side_effect=[timeout_obs, success_obs])

        result = mock_runtime._run_cmd_with_retry(
            'export VAR=value', 'Failed to export'
        )

        assert result == success_obs
        assert mock_runtime.run.call_count == 2
        mock_sleep.assert_called_once()

    @patch('openhands.runtime.base.time.sleep')
    def test_retry_exhaustion_raises_error(self, mock_sleep, mock_runtime):
        """Test that RuntimeError is raised after all retries fail."""
        timeout_obs = CmdOutputObservation(
            content='timeout', command='cmd', exit_code=-1
        )
        mock_runtime.run = MagicMock(return_value=timeout_obs)

        with pytest.raises(RuntimeError) as exc_info:
            mock_runtime._run_cmd_with_retry('cmd', 'Command failed', max_retries=3)

        assert 'Command failed' in str(exc_info.value)
        assert mock_runtime.run.call_count == 3
        assert mock_sleep.call_count == 2  # Called between retries, not after last

    def test_non_timeout_error_fails_immediately(self, mock_runtime):
        """Test that non-timeout errors don't trigger retry."""
        error_obs = CmdOutputObservation(
            content='permission denied', command='cmd', exit_code=1
        )
        mock_runtime.run = MagicMock(return_value=error_obs)

        with pytest.raises(RuntimeError) as exc_info:
            mock_runtime._run_cmd_with_retry('cmd', 'Command failed')

        assert 'Command failed' in str(exc_info.value)
        assert mock_runtime.run.call_count == 1  # No retries for non-timeout

    def test_empty_command_raises_value_error(self, mock_runtime):
        """Test that empty command raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            mock_runtime._run_cmd_with_retry('', 'Error')
        assert 'empty' in str(exc_info.value).lower()

    def test_invalid_max_retries_raises_value_error(self, mock_runtime):
        """Test that max_retries < 1 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            mock_runtime._run_cmd_with_retry('cmd', 'Error', max_retries=0)
        assert 'max_retries' in str(exc_info.value).lower()

    @patch('openhands.runtime.base.time.sleep')
    def test_exponential_backoff_delays(self, mock_sleep, mock_runtime):
        """Test that delays follow exponential backoff pattern."""
        timeout_obs = CmdOutputObservation(content='', command='cmd', exit_code=-1)
        mock_runtime.run = MagicMock(return_value=timeout_obs)

        with pytest.raises(RuntimeError):
            mock_runtime._run_cmd_with_retry('cmd', 'Error', max_retries=3)

        # Verify exponential delays: 1s, 2s (not called after 3rd attempt)
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == CMD_RETRY_BASE_DELAY_SECONDS  # 1s
        assert calls[1][0][0] == CMD_RETRY_BASE_DELAY_SECONDS * 2  # 2s


class TestConstants:
    """Tests for retry configuration constants."""

    def test_constants_have_sensible_values(self):
        """Test that constants are configured reasonably."""
        assert CMD_RETRY_MAX_ATTEMPTS >= 1
        assert CMD_RETRY_BASE_DELAY_SECONDS > 0
        assert (
            CMD_RETRY_TIMEOUT_EXIT_CODE == -1
        )  # Must match NO_CHANGE_TIMEOUT behavior
