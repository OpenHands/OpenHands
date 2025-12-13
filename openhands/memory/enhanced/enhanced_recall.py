"""Enhanced recall actions for memory-aware context retrieval."""

from __future__ import annotations

from openhands.events.action.agent import RecallAction
from openhands.events.event import RecallType
from openhands.events.observation.agent import RecallObservation

from .memory_system import MemorySystem
from .types import TaskContext


class EnhancedRecallAction(RecallAction):
    """Enhanced recall action that includes task context and file patterns."""

    def __init__(
        self,
        query: str,
        recall_type: RecallType = RecallType.KNOWLEDGE,
        context_files: list[str] | None = None,
        task_context: TaskContext | None = None,
    ):
        super().__init__(recall_type=recall_type, query=query)
        self.context_files = context_files or []
        self.task_context = task_context


class CodeContextRecallAction(EnhancedRecallAction):
    """Specialized recall action for code context retrieval."""

    def __init__(
        self,
        query: str,
        file_patterns: list[str],
        task_context: TaskContext | None = None,
    ):
        super().__init__(
            query=query,
            recall_type=RecallType.KNOWLEDGE,
            context_files=file_patterns,
            task_context=task_context,
        )
        self.file_patterns = file_patterns


class MemoryAwareRecallHandler:
    """Handles enhanced recall actions using the memory system."""

    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    def handle_enhanced_recall(self, action: EnhancedRecallAction) -> RecallObservation:
        """Handle an enhanced recall action."""
        if isinstance(action, CodeContextRecallAction):
            return self._handle_code_context_recall(action)
        else:
            return self._handle_general_recall(action)

    def _handle_code_context_recall(
        self, action: CodeContextRecallAction
    ) -> RecallObservation:
        """Handle code context recall."""
        if not self.memory_system.semantic_memory:
            return RecallObservation(
                recall_type=action.recall_type,
                content='Semantic memory not available - repository not indexed',
            )

        try:
            # Retrieve relevant code chunks
            code_chunks = self.memory_system.semantic_memory.retrieve_relevant_code(
                query=action.query, file_patterns=action.file_patterns, max_results=5
            )

            if not code_chunks:
                return RecallObservation(
                    recall_type=action.recall_type,
                    content='No relevant code found for the query',
                )

            # Format code context
            content_parts = [f'Found {len(code_chunks)} relevant code chunks:']

            for i, chunk in enumerate(code_chunks, 1):
                relevance = chunk.metadata.get('relevance_score', 0.0)
                content_parts.append(
                    f'\n{i}. {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}, '
                    f'relevance: {relevance:.2f})\n'
                    f'```{chunk.chunk_type}\n{chunk.content[:500]}{"..." if len(chunk.content) > 500 else ""}\n```'
                )

            return RecallObservation(
                recall_type=action.recall_type, content='\n'.join(content_parts)
            )

        except Exception as e:
            return RecallObservation(
                recall_type=action.recall_type,
                content=f'Error retrieving code context: {str(e)}',
            )

    def _handle_general_recall(self, action: EnhancedRecallAction) -> RecallObservation:
        """Handle general enhanced recall."""
        if not action.task_context:
            return RecallObservation(
                recall_type=action.recall_type,
                content='No task context provided for enhanced recall',
            )

        try:
            # Retrieve full memory context
            memory_context = self.memory_system.retrieve_context(action.task_context)

            # Format the context
            formatted_context = memory_context.to_prompt_context()

            if not formatted_context.strip():
                return RecallObservation(
                    recall_type=action.recall_type,
                    content='No relevant memory context found',
                )

            return RecallObservation(
                recall_type=action.recall_type, content=formatted_context
            )

        except Exception as e:
            return RecallObservation(
                recall_type=action.recall_type,
                content=f'Error retrieving memory context: {str(e)}',
            )


class MemoryPromptEnhancer:
    """Enhances prompts with memory context."""

    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    def enhance_prompt(
        self,
        base_prompt: str,
        task_context: TaskContext | None = None,
        include_working_memory: bool = True,
    ) -> str:
        """Enhance a prompt with memory context."""
        enhanced_parts = [base_prompt]

        # Add working memory context
        if include_working_memory:
            working_context = self.memory_system.working_memory.get_context_for_llm()
            if working_context.strip():
                enhanced_parts.append(
                    f'\n## Current Session Context\n{working_context}'
                )

        # Add task-specific memory context
        if task_context:
            try:
                memory_context = self.memory_system.retrieve_context(task_context)
                formatted_context = memory_context.to_prompt_context()

                if formatted_context.strip():
                    enhanced_parts.append(
                        f'\n## Relevant Memory Context\n{formatted_context}'
                    )

            except Exception as e:
                enhanced_parts.append(f'\n## Memory Context Error\n{str(e)}')

        return '\n'.join(enhanced_parts)

    def get_context_summary(self) -> str:
        """Get a summary of available context."""
        summary_parts = []

        # Working memory summary
        stats = self.memory_system.working_memory.get_session_stats()
        summary_parts.append(
            f'Session: {stats["total_actions"]} actions, '
            f'{stats["success_rate"]:.1%} success rate, '
            f'{stats["files_accessed"]} files accessed'
        )

        # Memory availability
        if self.memory_system.semantic_memory:
            summary_parts.append('Code memory: Available')
        else:
            summary_parts.append('Code memory: Not available')

        # Recent task stats
        try:
            task_stats = self.memory_system.episodic_memory.get_task_statistics(days=7)
            summary_parts.append(
                f'Recent tasks (7 days): {task_stats["total_tasks"]} tasks, '
                f'{task_stats["success_rate"]:.1%} success rate'
            )
        except Exception:
            summary_parts.append('Recent tasks: Not available')

        return ' | '.join(summary_parts)
