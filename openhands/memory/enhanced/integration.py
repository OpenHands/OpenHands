"""Integration layer for hooking memory system into OpenHands core."""

from __future__ import annotations

from typing import Any

from openhands.controller.agent_controller import AgentController
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event, EventSource
from openhands.events.stream import EventStream, EventStreamSubscriber
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.memory import Memory

from .memory_aware_agent import MemoryAgentFactory, MemoryAwareCodeActAgent
from .memory_system import MemorySystem


class MemoryEnhancedController(AgentController):
    """Agent controller enhanced with memory capabilities."""

    def __init__(
        self,
        agent: MemoryAwareCodeActAgent,
        event_stream: EventStream,
        sid: str,
        confirmation_mode: bool = False,
        headless_mode: bool = True,
        max_iterations: int = 100,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, Any] | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
    ):
        # Initialize with the memory-aware agent
        # Create required objects for AgentController
        from openhands.controller.conversation_stats import ConversationStats

        conversation_stats = ConversationStats()

        super().__init__(
            agent=agent,
            event_stream=event_stream,
            conversation_stats=conversation_stats,
            iteration_delta=max_iterations or 100,
            budget_per_task_delta=max_budget_per_task,
            agent_to_llm_config=agent_to_llm_config,
            agent_configs=agent_configs,
            sid=sid,
            confirmation_mode=confirmation_mode,
            headless_mode=headless_mode,
        )

        self.memory_agent = agent
        logger.info('Memory-enhanced controller initialized')

    async def _step(self) -> None:
        """Enhanced step method with memory integration."""
        # Call the original step method
        await super()._step()

        # Additional memory-specific processing could go here
        # For example, updating memory based on the step outcome

    async def set_agent_state_to(self, new_state) -> None:
        """Override to handle task completion."""
        # Check if we're completing a task
        if hasattr(self.state, 'agent_state') and new_state in [
            'FINISHED',
            'ERROR',
            'STOPPED',
        ]:
            # Complete current task in memory agent
            if (
                hasattr(self.memory_agent, 'current_task')
                and self.memory_agent.current_task
            ):
                success = new_state == 'FINISHED'
                await self.memory_agent._complete_current_task(success=success)

        await super().set_agent_state_to(new_state)


class MemoryEnhancedMemory(Memory):
    """Enhanced memory component that integrates with the memory system."""

    def __init__(
        self,
        event_stream: EventStream,
        sid: str,
        memory_system: MemorySystem | None = None,
        status_callback=None,
    ):
        super().__init__(event_stream, sid, status_callback)

        self.memory_system = memory_system

        if self.memory_system:
            logger.info('Memory component enhanced with memory system')

    async def _on_event(self, event: Event) -> None:
        """Enhanced event handling with memory system integration."""
        # Call the original event handler
        await super()._on_event(event)

        # Additional memory system integration
        if self.memory_system:
            try:
                await self._handle_memory_event(event)
            except Exception as e:
                logger.warning(f'Memory system event handling failed: {e}')

    async def _handle_memory_event(self, event: Event) -> None:
        """Handle events for the memory system."""
        # This could be enhanced to update memory based on specific events
        # For now, we'll just log significant events
        if hasattr(event, 'source') and event.source == EventSource.USER:
            logger.debug(f'User event for memory system: {type(event).__name__}')


class MemoryIntegration:
    """Main integration class for memory system."""

    @staticmethod
    def create_memory_enhanced_setup(
        config: AgentConfig,
        llm_registry: LLMRegistry,
        event_stream: EventStream,
        sid: str,
        repo_path: str | None = None,
        memory_dir: str | None = None,
        **controller_kwargs,
    ) -> tuple[MemoryEnhancedController, MemoryEnhancedMemory, MemorySystem]:
        """Create a complete memory-enhanced setup."""

        # Create memory system
        memory_system = MemorySystem(
            repo_path=repo_path, memory_dir=memory_dir or '.openhands/memory'
        )

        # Create memory-aware agent
        agent = MemoryAgentFactory.create_memory_aware_agent(
            config=config,
            llm_registry=llm_registry,
            repo_path=repo_path,
            memory_dir=memory_dir,
        )

        # Create enhanced controller
        controller = MemoryEnhancedController(
            agent=agent, event_stream=event_stream, sid=sid, **controller_kwargs
        )

        # Create enhanced memory component
        memory = MemoryEnhancedMemory(
            event_stream=event_stream, sid=sid, memory_system=memory_system
        )

        # Initialize repository if provided
        if repo_path:
            memory_system.initialize_repository(repo_path)

        logger.info('Memory-enhanced OpenHands setup created')

        return controller, memory, memory_system

    @staticmethod
    def enhance_existing_setup(
        controller: AgentController,
        memory: Memory,
        repo_path: str | None = None,
        memory_dir: str | None = None,
    ) -> tuple[AgentController, Memory, MemorySystem]:
        """Enhance an existing OpenHands setup with memory capabilities."""

        # Create memory system
        memory_system = MemorySystem(
            repo_path=repo_path, memory_dir=memory_dir or '.openhands/memory'
        )

        # Create memory-aware agent based on existing agent
        if hasattr(controller, 'agent'):
            config = (
                controller.agent.config if hasattr(controller.agent, 'config') else None
            )
            llm_registry = (
                controller.agent.llm_registry
                if hasattr(controller.agent, 'llm_registry')
                else None
            )

            if config and llm_registry:
                memory_agent = MemoryAgentFactory.create_with_existing_memory(
                    config=config,
                    llm_registry=llm_registry,
                    memory_system=memory_system,
                )

                # Replace the agent in the controller
                controller.agent = memory_agent

        # Enhance memory component
        if hasattr(memory, 'memory_system'):
            memory.memory_system = memory_system

        logger.info('Existing OpenHands setup enhanced with memory')

        return controller, memory, memory_system


class MemoryEventSubscriber:
    """Event subscriber for memory system integration."""

    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    def subscribe_to_events(self, event_stream: EventStream, sid: str) -> None:
        """Subscribe to relevant events for memory updates."""
        event_stream.subscribe(EventStreamSubscriber.MEMORY, self._on_event, sid)

    def _on_event(self, event: Event) -> None:
        """Handle events for memory system updates."""
        try:
            # Handle file modification events
            if hasattr(event, 'path') and hasattr(event, 'content'):
                self.memory_system.update_file_state(event.path, event.content)

            # Handle other relevant events
            # This could be expanded based on specific event types

        except Exception as e:
            logger.warning(f'Memory event subscriber error: {e}')


def create_memory_enhanced_openhands(
    config: AgentConfig,
    llm_registry: LLMRegistry,
    event_stream: EventStream,
    sid: str,
    repo_path: str | None = None,
    memory_dir: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create a complete memory-enhanced OpenHands setup.

    This is the main entry point for creating an OpenHands instance
    with full memory capabilities.
    """

    controller, memory, memory_system = MemoryIntegration.create_memory_enhanced_setup(
        config=config,
        llm_registry=llm_registry,
        event_stream=event_stream,
        sid=sid,
        repo_path=repo_path,
        memory_dir=memory_dir,
        **kwargs,
    )

    # Set up event subscriber
    event_subscriber = MemoryEventSubscriber(memory_system)
    event_subscriber.subscribe_to_events(event_stream, sid)

    return {
        'controller': controller,
        'memory': memory,
        'memory_system': memory_system,
        'event_subscriber': event_subscriber,
    }
