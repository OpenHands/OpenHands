"""Tests for terminal compatibility detection."""
import os
import sys
from unittest.mock import Mock, patch

import pytest
from openhands_cli.agent_chat import check_terminal_compatibility


class TestTerminalDetection:
    """Test terminal compatibility detection."""

    def test_detects_proper_terminal(self):
        """Test that proper terminal is detected."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'xterm-256color'}):
            assert check_terminal_compatibility() is True

    def test_detects_non_tty_stdin(self):
        """Test that non-TTY stdin is detected."""
        with patch('sys.stdin.isatty', return_value=False), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'xterm'}):
            assert check_terminal_compatibility() is False

    def test_detects_non_tty_stdout(self):
        """Test that non-TTY stdout is detected."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=False), \
             patch.dict(os.environ, {'TERM': 'xterm'}):
            assert check_terminal_compatibility() is False

    def test_detects_dumb_terminal(self):
        """Test that dumb terminal is detected."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'dumb'}):
            assert check_terminal_compatibility() is False

    def test_detects_empty_term(self):
        """Test that empty TERM variable is detected."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': ''}):
            assert check_terminal_compatibility() is False

    def test_detects_unknown_terminal(self):
        """Test that unknown terminal type is detected."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'unknown'}):
            assert check_terminal_compatibility() is False

    def test_detects_missing_term(self):
        """Test that missing TERM variable is detected."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True):
            # Create a clean environment without TERM
            clean_env = {k: v for k, v in os.environ.items() if k != 'TERM'}
            with patch.dict(os.environ, clean_env, clear=True):
                assert check_terminal_compatibility() is False

    def test_handles_prompt_toolkit_failure(self):
        """Test graceful handling when prompt_toolkit fails."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'xterm'}), \
             patch('openhands_cli.agent_chat.create_input', side_effect=Exception('Failed')):
            assert check_terminal_compatibility() is False

    def test_xterm_terminal_works(self):
        """Test that xterm terminal is recognized."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'xterm'}):
            assert check_terminal_compatibility() is True

    def test_screen_terminal_works(self):
        """Test that screen terminal is recognized."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'screen'}):
            assert check_terminal_compatibility() is True

    def test_tmux_terminal_works(self):
        """Test that tmux terminal is recognized."""
        with patch('sys.stdin.isatty', return_value=True), \
             patch('sys.stdout.isatty', return_value=True), \
             patch.dict(os.environ, {'TERM': 'tmux-256color'}):
            assert check_terminal_compatibility() is True

    def test_skip_env_var_bypasses_check(self):
        """Test that the skip env var bypasses compatibility checks."""
        with patch('sys.stdin.isatty', return_value=False), \
             patch('sys.stdout.isatty', return_value=False), \
             patch.dict(os.environ, {'OPENHANDS_CLI_SKIP_TTY_CHECK': '1'}, clear=False):
            assert check_terminal_compatibility() is True

    def test_ci_env_bypasses_check(self):
        """Test that CI environments bypass compatibility checks."""
        with patch('sys.stdin.isatty', return_value=False), \
             patch('sys.stdout.isatty', return_value=False), \
             patch.dict(os.environ, {'CI': 'true'}, clear=False):
            assert check_terminal_compatibility() is True



