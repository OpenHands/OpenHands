import signal
from dataclasses import dataclass, field
from signal import Signals
from typing import Callable
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from openhands.app_server.utils import shutdown_listener
from openhands.app_server.utils.shutdown_listener import (
    add_shutdown_listener,
    remove_shutdown_listener,
    should_continue,
)


@pytest.fixture(autouse=True)
def cleanup_listeners():
    shutdown_listener._shutdown_listeners.clear()
    shutdown_listener._should_exit = False


@dataclass
class MockSignal:
    handlers: dict[Signals, Callable] = field(default_factory=dict)

    def signal(self, signalnum: Signals, handler: Callable):
        result = self.handlers.get(signalnum)
        self.handlers[signalnum] = handler
        return result

    def trigger(self, signalnum: Signals):
        handler = self.handlers.get(signalnum)
        if handler:
            handler(signalnum.value, None)


def test_add_shutdown_listener():
    mock_callable = MagicMock()
    listener_id = add_shutdown_listener(mock_callable)

    assert isinstance(listener_id, UUID)
    assert listener_id in shutdown_listener._shutdown_listeners
    assert shutdown_listener._shutdown_listeners[listener_id] == mock_callable


def test_remove_shutdown_listener():
    mock_callable = MagicMock()
    listener_id = add_shutdown_listener(mock_callable)

    # Test successful removal
    assert remove_shutdown_listener(listener_id) is True
    assert listener_id not in shutdown_listener._shutdown_listeners

    # Test removing non-existent listener
    assert remove_shutdown_listener(listener_id) is False


def test_signal_handler_calls_listeners():
    mock_signal = MockSignal()
    with patch('openhands.app_server.utils.shutdown_listener.signal', mock_signal):
        mock_callable1 = MagicMock()
        mock_callable2 = MagicMock()
        add_shutdown_listener(mock_callable1)
        add_shutdown_listener(mock_callable2)

        # Register and trigger signal handler
        shutdown_listener._register_signal_handler(signal.SIGTERM)
        mock_signal.trigger(signal.SIGTERM)

        # Verify both listeners were called
        mock_callable1.assert_called_once()
        mock_callable2.assert_called_once()

        # Verify should_continue returns False after shutdown
        assert should_continue() is False


def test_listeners_called_only_once():
    mock_signal = MockSignal()
    with patch('openhands.app_server.utils.shutdown_listener.signal', mock_signal):
        mock_callable = MagicMock()
        add_shutdown_listener(mock_callable)

        # Register and trigger signal handler multiple times
        shutdown_listener._register_signal_handler(signal.SIGTERM)
        mock_signal.trigger(signal.SIGTERM)
        mock_signal.trigger(signal.SIGTERM)

        # Verify listener was called only once
        assert mock_callable.call_count == 1


def test_remove_listener_during_shutdown():
    mock_signal = MockSignal()
    with patch('openhands.app_server.utils.shutdown_listener.signal', mock_signal):
        mock_callable1 = MagicMock()
        mock_callable2 = MagicMock()

        # Second listener removes the first listener when called
        listener1_id = add_shutdown_listener(mock_callable1)

        def remove_other_listener():
            remove_shutdown_listener(listener1_id)
            mock_callable2()

        add_shutdown_listener(remove_other_listener)

        # Register and trigger signal handler
        shutdown_listener._register_signal_handler(signal.SIGTERM)
        mock_signal.trigger(signal.SIGTERM)

        # Both listeners should still be called
        assert mock_callable1.call_count == 1
        assert mock_callable2.call_count == 1
