#!/usr/bin/env python3
"""
Tests for pause listener in OpenHands CLI.
"""

import time
from unittest.mock import MagicMock

from openhands.sdk import Conversation
from prompt_toolkit.input.defaults import create_pipe_input

from openhands_cli.listeners.pause_listener import PauseListener, pause_listener


class TestPauseListener:
    """Test suite for PauseListener class."""

    def test_pause_listener_stop(self) -> None:
        """Test PauseListener stop functionality."""
        mock_callback = MagicMock()
        listener = PauseListener(on_pause=mock_callback)

        listener.start()

        # Initially not paused
        assert not listener.is_paused()
        assert listener.is_alive()

        # Stop the listener
        listener.stop()

        # Listner was shutdown not paused
        assert not listener.is_paused()
        assert listener.is_stopped()

    def test_pause_listener_context_manager(self) -> None:
        """Test pause_listener context manager."""
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation.pause = MagicMock()

        with create_pipe_input() as pipe:
            with pause_listener(mock_conversation, input_source=pipe) as listener:
                assert isinstance(listener, PauseListener)
                assert listener.on_pause == mock_conversation.pause
                # Listener should be started (daemon thread)
                assert listener.is_alive()
                assert not listener.is_paused()
                pipe.send_text('\x10')  # Ctrl-P
                time.sleep(0.1)
                assert listener.is_paused()

            assert listener.is_stopped()
            assert not listener.is_alive()

    def test_double_ctrl_c_termination(self) -> None:
        """Test double Ctrl+C termination functionality."""
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation.pause = MagicMock()
        mock_terminate_callback = MagicMock()

        with create_pipe_input() as pipe:
            with pause_listener(mock_conversation, on_terminate=mock_terminate_callback, input_source=pipe) as listener:
                assert isinstance(listener, PauseListener)
                assert not listener.is_terminated()
                
                # Send two Ctrl+C quickly to trigger termination
                pipe.send_text('\x03\x03')  # Two Ctrl-C
                time.sleep(0.2)  # Give time for processing
                assert listener.is_terminated()
                mock_terminate_callback.assert_called_once()

            assert listener.is_stopped()
            assert not listener.is_alive()

    def test_single_ctrl_c_no_termination(self) -> None:
        """Test that single Ctrl+C doesn't trigger termination."""
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation.pause = MagicMock()
        mock_terminate_callback = MagicMock()

        with create_pipe_input() as pipe:
            with pause_listener(mock_conversation, on_terminate=mock_terminate_callback, input_source=pipe) as listener:
                assert isinstance(listener, PauseListener)
                assert not listener.is_terminated()
                
                # Send single Ctrl+C
                pipe.send_text('\x03')  # Ctrl-C
                time.sleep(0.2)  # Give time for processing
                
                # Should be paused but not terminated
                assert listener.is_paused()
                assert not listener.is_terminated()
                mock_terminate_callback.assert_not_called()

            assert listener.is_stopped()
            assert not listener.is_alive()
