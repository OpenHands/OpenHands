#!/usr/bin/env python3
"""
Tests for threaded agent runner in OpenHands CLI.
"""

import time
import threading
from unittest.mock import MagicMock

from openhands.sdk import Conversation

from openhands_cli.threaded_agent import ThreadedAgentRunner


class TestThreadedAgentRunner:
    """Test suite for ThreadedAgentRunner class."""

    def test_agent_runner_basic_functionality(self) -> None:
        """Test basic agent runner functionality."""
        mock_conversation = MagicMock(spec=Conversation)
        
        # Mock run method to simulate some work
        def some_work():
            time.sleep(0.1)  # Brief work
            
        mock_conversation.run = some_work
        
        runner = ThreadedAgentRunner(mock_conversation)
        
        # Initially not running
        assert not runner.is_running()
        
        # Start the agent
        runner.run_agent()
        
        # Should be running (check quickly before it finishes)
        time.sleep(0.05)  # Give thread time to start
        assert runner.is_running()
        
        # Wait for completion
        runner.wait_for_completion()
        
        # Should no longer be running
        assert not runner.is_running()

    def test_agent_runner_termination(self) -> None:
        """Test agent runner immediate termination."""
        mock_conversation = MagicMock(spec=Conversation)
        
        # Mock run method to simulate long-running work
        def long_running_work():
            for i in range(100):  # This would take a while
                time.sleep(0.1)
                
        mock_conversation.run = long_running_work
        
        runner = ThreadedAgentRunner(mock_conversation)
        
        # Start the agent
        runner.run_agent()
        
        # Give thread time to start
        time.sleep(0.05)
        assert runner.is_running()
        
        # Let it run for a bit
        time.sleep(0.1)
        
        # Terminate immediately
        runner.terminate_immediately()
        
        # Should be marked as terminated
        assert runner.is_terminated()
        
        # Wait for completion (should exit due to termination)
        runner.wait_for_completion()

    def test_agent_runner_exception_handling(self) -> None:
        """Test agent runner handles exceptions properly."""
        mock_conversation = MagicMock(spec=Conversation)
        
        # Mock run method to raise an exception
        def failing_work():
            raise ValueError("Test exception")
            
        mock_conversation.run = failing_work
        
        runner = ThreadedAgentRunner(mock_conversation)
        
        # Start the agent
        runner.run_agent()
        
        # Wait for completion (should handle exception)
        try:
            runner.wait_for_completion()
            # Should raise the exception
            assert False, "Expected exception to be raised"
        except ValueError as e:
            assert str(e) == "Test exception"
        
        # Should no longer be running
        assert not runner.is_running()

    def test_agent_runner_multiple_starts(self) -> None:
        """Test that starting an already running agent doesn't cause issues."""
        mock_conversation = MagicMock(spec=Conversation)
        
        call_count = 0
        
        # Mock run method to simulate work and count calls
        def some_work():
            nonlocal call_count
            call_count += 1
            time.sleep(0.2)
            
        mock_conversation.run = some_work
        
        runner = ThreadedAgentRunner(mock_conversation)
        
        # Start the agent
        runner.run_agent()
        
        # Give thread time to start
        time.sleep(0.05)
        assert runner.is_running()
        
        # Try to start again (should not cause issues)
        runner.run_agent()
        assert runner.is_running()
        
        # Wait for completion
        runner.wait_for_completion()
        
        # Should have called conversation.run only once
        assert call_count == 1
        
        # Should no longer be running
        assert not runner.is_running()