"""Memory-aware agent that extends CodeActAgent with memory capabilities."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from openhands.agenthub.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, MessageAction
from openhands.events.event import Event
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import Condensation, View

from .enhanced_recall import MemoryAwareRecallHandler, MemoryPromptEnhancer
from .memory_system import MemorySystem
from .types import (
    ActionOutcome,
    TaskContext,
    TaskOutcome,
)


class MemoryAwareCodeActAgent(CodeActAgent):
    """CodeAct agent enhanced with memory capabilities."""

    memory_system: MemorySystem | None

    def __init__(
        self,
        config: AgentConfig,
        llm_registry: LLMRegistry,
        memory_system: MemorySystem | None = None,
        repo_path: str | None = None,
    ):
        super().__init__(config, llm_registry)

        # Initialize memory system
        if memory_system:
            self.memory_system = memory_system
        elif repo_path:
            self.memory_system = MemorySystem(repo_path=repo_path)
        else:
            self.memory_system = None

        # Memory components
        if self.memory_system:
            self.prompt_enhancer: MemoryPromptEnhancer | None = MemoryPromptEnhancer(
                self.memory_system
            )
            self.recall_handler: MemoryAwareRecallHandler | None = (
                MemoryAwareRecallHandler(self.memory_system)
            )
        else:
            self.prompt_enhancer = None
            self.recall_handler = None

        # Current task tracking
        self.current_task: TaskContext | None = None
        self.task_start_time: datetime | None = None
        self.task_actions: list[dict[str, Any]] = []

        logger.info('Memory-aware CodeAct agent initialized')

    def reset(self) -> None:
        """Reset agent state and complete current task if any."""
        if self.current_task and self.task_start_time:
            # Complete the current task
            asyncio.create_task(self._complete_current_task(success=False))

        super().reset()
        self.current_task = None
        self.task_start_time = None
        self.task_actions = []

    def step(self, state: State) -> Action:
        """Enhanced step method with memory integration."""
        # Check if we need to start a new task
        if not self.current_task and state.history:
            self._maybe_start_new_task(state)

        # Pre-step: Retrieve relevant memory context
        if self.current_task:
            try:
                self._update_task_context(state)
            except Exception as e:
                logger.warning(f'Failed to update task context: {e}')

        # Execute original step with enhanced prompt
        action = self._step_with_memory(state)

        # Post-step: Update working memory
        if action:
            self._record_action(action, state)

        return action

    def _step_with_memory(self, state: State) -> Action:
        """Execute step with memory-enhanced prompt."""
        # Get the original prompt that would be sent to LLM
        # Use the same condensation logic as the parent class
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                # If we get a condensation action, we can't enhance the prompt
                return condensation_action

        initial_user_message = self._get_initial_user_message(state.history)
        original_messages = self._get_messages(condensed_history, initial_user_message)

        # Enhance the last user message with memory context
        if original_messages and self.current_task and self.prompt_enhancer:
            try:
                last_message = original_messages[-1]
                if last_message.role == 'user' and last_message.content:
                    # Get the text content from the message
                    text_content = ''
                    for content_item in last_message.content:
                        if hasattr(content_item, 'text'):
                            text_content += content_item.text

                    enhanced_content = self.prompt_enhancer.enhance_prompt(
                        base_prompt=text_content,
                        task_context=self.current_task,
                        include_working_memory=True,
                    )

                    # Update the first text content item
                    for content_item in last_message.content:
                        if hasattr(content_item, 'text'):
                            content_item.text = enhanced_content
                            break
            except Exception as e:
                logger.warning(f'Failed to enhance prompt with memory: {e}')

        # Call the original step method
        return super().step(state)

    def _maybe_start_new_task(self, state: State) -> None:
        """Check if we should start tracking a new task."""
        if not state.history:
            return

        # Look for user messages that indicate a new task
        recent_events = state.history[-5:]  # Check last 5 events

        for event in recent_events:
            if (
                isinstance(event, MessageAction)
                and event.source
                and event.source.value == 'user'
            ):
                # This looks like a new task request
                self._start_new_task(event.content, state)
                break

    def _start_new_task(self, user_message: str, state: State) -> None:
        """Start tracking a new task."""
        # Complete previous task if any
        if self.current_task:
            asyncio.create_task(self._complete_current_task(success=False))

        # Extract files in scope from state
        files_in_scope: list[str] = []
        if hasattr(state, 'root') and state.root:
            # Try to get files from current working directory or recent file operations
            # This is a simplified approach - could be enhanced
            pass

        # Create new task context
        if self.memory_system:
            self.current_task = self.memory_system.create_task_context(
                intent=user_message, files_in_scope=files_in_scope
            )

        self.task_start_time = datetime.now()
        self.task_actions = []

        if self.current_task:
            logger.info(f'Started new task: {self.current_task.task_id}')

    def _update_task_context(self, state: State) -> None:
        """Update task context with current state information."""
        if not self.current_task:
            return

        # Update files in scope based on recent file operations
        # This could be enhanced to track file operations from the state
        pass

    def _record_action(self, action: Action, state: State) -> None:
        """Record an action for the current task."""
        if not self.current_task:
            return

        action_start = datetime.now()

        # Create action record
        action_data = {
            'action_type': action.__class__.__name__,
            'timestamp': action_start.isoformat(),
            'action_data': self._serialize_action(action),
        }

        self.task_actions.append(action_data)

        # Update working memory with action outcome
        # Note: We don't have the observation yet, so we'll update this later
        # This is a simplified approach - in a full implementation, we'd wait for the observation
        action_outcome = ActionOutcome(
            action_type=action.__class__.__name__,
            action_data=action_data,
            success=True,  # Assume success for now
            duration=timedelta(seconds=1),  # Placeholder
        )

        if self.memory_system:
            self.memory_system.update_working_memory(action_outcome)

    def _serialize_action(self, action: Action) -> dict[str, Any]:
        """Serialize an action for storage."""
        try:
            # Basic serialization - could be enhanced
            return {
                'type': action.__class__.__name__,
                'content': getattr(action, 'content', ''),
                'thought': getattr(action, 'thought', ''),
            }
        except Exception as e:
            logger.warning(f'Failed to serialize action: {e}')
            return {'type': action.__class__.__name__}

    async def _complete_current_task(self, success: bool = True) -> None:
        """Complete the current task and learn from it."""
        if not self.current_task or not self.task_start_time:
            return

        try:
            # Calculate task duration
            duration = datetime.now() - self.task_start_time

            # Get modified files from working memory
            modified_files = []
            if self.memory_system:
                modified_files = self.memory_system.working_memory.get_modified_files()

            # Create task outcome
            outcome = TaskOutcome(
                task_id=self.current_task.task_id,
                success=success,
                duration=duration,
                files_touched=modified_files,
                actions_taken=self.task_actions,
                success_metrics={'duration_seconds': duration.total_seconds()},
            )

            # Learn from the task
            if self.memory_system:
                await self.memory_system.learn_from_task(self.current_task, outcome)

            logger.info(
                f'Completed task {self.current_task.task_id}: '
                f'{"success" if success else "failure"} in {duration.total_seconds():.1f}s'
            )

        except Exception as e:
            logger.error(f'Failed to complete task: {e}')

        finally:
            self.current_task = None
            self.task_start_time = None
            self.task_actions = []

    def get_memory_summary(self) -> dict[str, Any]:
        """Get a summary of the agent's memory state."""
        if self.memory_system:
            return self.memory_system.get_session_summary()
        return {}

    def initialize_repository(
        self, repo_path: str, force_reindex: bool = False
    ) -> None:
        """Initialize memory system for a repository."""
        if self.memory_system:
            self.memory_system.initialize_repository(repo_path, force_reindex)
            logger.info(f'Initialized repository memory for: {repo_path}')


class MemoryAgentFactory:
    """Factory for creating memory-aware agents."""

    @staticmethod
    def create_memory_aware_agent(
        config: AgentConfig,
        llm_registry: LLMRegistry,
        repo_path: str | None = None,
        memory_dir: str | None = None,
    ) -> MemoryAwareCodeActAgent:
        """Create a memory-aware CodeAct agent."""
        memory_system = MemorySystem(
            repo_path=repo_path, memory_dir=memory_dir or '.openhands/memory'
        )

        return MemoryAwareCodeActAgent(
            config=config,
            llm_registry=llm_registry,
            memory_system=memory_system,
            repo_path=repo_path,
        )

    @staticmethod
    def create_with_existing_memory(
        config: AgentConfig, llm_registry: LLMRegistry, memory_system: MemorySystem
    ) -> MemoryAwareCodeActAgent:
        """Create a memory-aware agent with an existing memory system."""
        return MemoryAwareCodeActAgent(
            config=config, llm_registry=llm_registry, memory_system=memory_system
        )
