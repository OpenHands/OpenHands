"""
Threaded agent execution for immediate termination support.
"""

import threading
import time
from typing import Optional

from openhands.sdk import BaseConversation, Message
from openhands.sdk.conversation.state import AgentExecutionStatus
from prompt_toolkit import HTML, print_formatted_text


class ThreadedAgentRunner:
    """Runs agent in a separate thread that can be terminated immediately."""
    
    def __init__(self, conversation: BaseConversation):
        self.conversation = conversation
        self._agent_thread: Optional[threading.Thread] = None
        self._terminate_event = threading.Event()
        self._exception: Optional[Exception] = None
        
    def run_agent(self, message: Optional[Message] = None) -> None:
        """Run the agent in a separate thread.
        
        Args:
            message: Optional message to send before running
        """
        if self._agent_thread and self._agent_thread.is_alive():
            # If agent is already running, just return
            return
            
        self._terminate_event.clear()
        self._exception = None
        
        self._agent_thread = threading.Thread(
            target=self._agent_worker,
            args=(message,),
            daemon=True
        )
        self._agent_thread.start()
        
    def _agent_worker(self, message: Optional[Message]) -> None:
        """Worker function that runs in the agent thread."""
        try:
            # Send message if provided
            if message:
                self.conversation.send_message(message)
                
            # Run the agent
            self.conversation.run()
            
        except Exception as e:
            self._exception = e
            
    def wait_for_completion(self, check_interval: float = 0.1) -> None:
        """Wait for agent to complete or be terminated.
        
        Args:
            check_interval: How often to check for termination (seconds)
        """
        if not self._agent_thread:
            return
            
        while self._agent_thread.is_alive():
            if self._terminate_event.is_set():
                # Agent was terminated, don't wait for thread to finish naturally
                break
                
            time.sleep(check_interval)
            
        # Check if there was an exception
        if self._exception:
            raise self._exception
            
    def terminate_immediately(self) -> None:
        """Terminate the agent thread immediately."""
        self._terminate_event.set()
        
        if self._agent_thread and self._agent_thread.is_alive():
            # Note: Python doesn't have a clean way to forcefully terminate threads
            # The thread will continue running but we mark it as terminated
            # The conversation state should be persistent so we can resume later
            print_formatted_text(
                HTML("<red>Agent thread marked for termination. Conversation state is preserved.</red>")
            )
            
    def is_running(self) -> bool:
        """Check if agent is currently running."""
        return self._agent_thread is not None and self._agent_thread.is_alive()
        
    def is_terminated(self) -> bool:
        """Check if agent was terminated."""
        return self._terminate_event.is_set()