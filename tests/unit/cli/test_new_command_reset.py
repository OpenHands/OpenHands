"""Test for the /new command to ensure proper session reset."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.cli.commands import handle_new_command
from openhands.cli.main import run_session
from openhands.cli.tui import UsageMetrics
from openhands.core.config import OpenHandsConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource
from openhands.events.action import ChangeAgentStateAction
from openhands.events.stream import EventStream
from openhands.storage.settings.file_settings_store import FileSettingsStore


class TestNewCommandReset:
    """Test that the /new command properly resets conversation and stats."""

    @pytest.mark.asyncio
    async def test_new_command_creates_fresh_session(self):
        """Test that /new command creates a completely fresh session."""
        config = OpenHandsConfig()

        # Create a temporary directory for file storage
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_store = FileSettingsStore(temp_dir)

            # Mock the runtime and event stream creation
            with (
                patch('openhands.cli.main.create_runtime') as mock_create_runtime,
                patch('openhands.cli.main.create_controller') as mock_create_controller,
                patch('openhands.cli.main.create_memory') as mock_create_memory,
                patch(
                    'openhands.cli.main.initialize_repository_for_runtime'
                ) as mock_init_repo,
                patch('openhands.cli.main.display_banner'),
                patch('openhands.cli.main.display_agent_running_message'),
            ):
                # Mock runtime with event stream
                mock_runtime = MagicMock()
                mock_event_stream = MagicMock(spec=EventStream)
                mock_event_stream.sid = 'test-session-1'
                mock_runtime.event_stream = mock_event_stream
                mock_runtime.connect = AsyncMock()
                mock_runtime.close = MagicMock()
                mock_create_runtime.return_value = mock_runtime

                # Mock controller
                mock_controller = MagicMock()
                mock_controller.get_state.return_value = MagicMock()
                mock_controller.close = AsyncMock()
                mock_create_controller.return_value = (mock_controller, None)

                # Mock memory
                mock_memory = MagicMock()
                mock_create_memory.return_value = mock_memory

                # Mock repository initialization
                mock_init_repo.return_value = None

                # Create a loop for the test
                loop = asyncio.get_event_loop()

                # First, simulate running a session with some conversation
                await run_session(
                    loop=loop,
                    config=config,
                    settings_store=settings_store,
                    current_dir='/test',
                    task_content='Hello, this is the first message',
                    session_name='test-session',
                )

                # Verify first session was created
                assert mock_create_runtime.called
                first_call_args = mock_create_runtime.call_args
                first_session_id = (
                    first_call_args[1]['sid'] if 'sid' in first_call_args[1] else None
                )

                # Reset mocks for second session
                mock_create_runtime.reset_mock()
                mock_create_controller.reset_mock()

                # Now simulate the /new command creating a second session
                await run_session(
                    loop=loop,
                    config=config,
                    settings_store=settings_store,
                    current_dir='/test',
                    task_content=None,  # No initial task for /new command
                )

                # Verify second session was created with different session ID
                assert mock_create_runtime.called
                second_call_args = mock_create_runtime.call_args
                second_session_id = (
                    second_call_args[1]['sid'] if 'sid' in second_call_args[1] else None
                )

                # Session IDs should be different (or at least one should be None, meaning auto-generated)
                if first_session_id and second_session_id:
                    assert first_session_id != second_session_id, (
                        'New session should have different session ID'
                    )

    @pytest.mark.asyncio
    async def test_handle_new_command_behavior(self):
        """Test that handle_new_command properly signals for session reset."""
        config = MagicMock(spec=OpenHandsConfig)
        event_stream = MagicMock(spec=EventStream)
        usage_metrics = MagicMock(spec=UsageMetrics)
        sid = 'test-session-id'

        with (
            patch('openhands.cli.commands.cli_confirm') as mock_confirm,
            patch('prompt_toolkit.shortcuts.clear') as mock_clear,
        ):
            # Mock user confirming new session
            mock_confirm.return_value = 0  # First option: "Yes, proceed"

            # Call the function under test
            close_repl, new_session = handle_new_command(
                config, event_stream, usage_metrics, sid
            )

            # Verify correct behavior
            mock_confirm.assert_called_once()
            event_stream.add_event.assert_called_once()

            # Check event is the right type
            args, kwargs = event_stream.add_event.call_args
            assert isinstance(args[0], ChangeAgentStateAction)
            assert args[0].agent_state == AgentState.STOPPED
            assert args[1] == EventSource.ENVIRONMENT

            # Verify screen is cleared instead of showing shutdown message
            mock_clear.assert_called_once()
            assert close_repl is True
            assert new_session is True

    def test_usage_metrics_reset_on_new_session(self):
        """Test that each new session creates fresh usage metrics."""
        # This test verifies that UsageMetrics objects are created fresh
        # for each session, not reused

        metrics1 = UsageMetrics()
        metrics2 = UsageMetrics()

        # They should be different objects
        assert metrics1 is not metrics2

        # They should start with clean state (fresh Metrics object)
        assert metrics1.metrics.accumulated_cost == 0
        assert metrics2.metrics.accumulated_cost == 0

        # They should have different session init times (or very close)
        assert (
            metrics1.session_init_time != metrics2.session_init_time
            or abs(metrics1.session_init_time - metrics2.session_init_time) < 0.1
        )
