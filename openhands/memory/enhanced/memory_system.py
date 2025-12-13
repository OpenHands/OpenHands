"""Main memory system that coordinates all memory components."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from openhands.core.logger import openhands_logger as logger

from .episodic_memory import EpisodicTaskMemory
from .semantic_memory import SemanticCodeMemory
from .skill_memory import SkillMemory
from .types import (
    ActionOutcome,
    SkillReference,
    TaskContext,
    TaskOutcome,
    classify_task_type,
    determine_task_scope,
    generate_task_id,
)
from .working_memory import WorkingMemory


class MemoryContext:
    """Context retrieved from memory for a task."""

    def __init__(self):
        self.code_snippets: list[str] = []
        self.similar_episodes: list[str] = []
        self.applicable_skills: list[SkillReference] = []
        self.failure_warnings: list[str] = []
        self.success_patterns: list[str] = []

    def to_prompt_context(self) -> str:
        """Convert memory context to formatted text for LLM prompt."""
        context_parts = []

        if self.code_snippets:
            context_parts.append('## Relevant Code Context')
            context_parts.extend(self.code_snippets[:3])  # Limit to top 3

        if self.similar_episodes:
            context_parts.append('## Similar Past Tasks')
            context_parts.extend(self.similar_episodes[:2])  # Limit to top 2

        if self.applicable_skills:
            context_parts.append('## Applicable Skills')
            for skill in self.applicable_skills[:2]:  # Limit to top 2
                context_parts.append(
                    f'- {skill.skill_name} (confidence: {skill.confidence:.2f}, '
                    f'success rate: {skill.success_rate:.2f})'
                )

        if self.failure_warnings:
            context_parts.append('## Potential Issues')
            context_parts.extend(self.failure_warnings[:2])  # Limit to top 2

        if self.success_patterns:
            context_parts.append('## Successful Patterns')
            context_parts.extend(self.success_patterns[:2])  # Limit to top 2

        return '\n\n'.join(context_parts)


class MemorySystem:
    """Main memory system that coordinates all memory components."""

    def __init__(
        self, repo_path: str | None = None, memory_dir: str = '.openhands/memory'
    ):
        self.repo_path = repo_path
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory components
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicTaskMemory(memory_dir)
        self.skill_memory = SkillMemory(memory_dir)

        # Initialize semantic memory only if repo path is provided
        self.semantic_memory: SemanticCodeMemory | None = None
        if repo_path:
            self.semantic_memory = SemanticCodeMemory(repo_path, memory_dir)

        logger.info(f'Memory system initialized with repo: {repo_path}')

    def initialize_repository(
        self, repo_path: str, force_reindex: bool = False
    ) -> None:
        """Initialize semantic memory for a repository."""
        self.repo_path = repo_path
        self.semantic_memory = SemanticCodeMemory(repo_path, str(self.memory_dir))

        # Index repository in background
        if force_reindex:
            logger.info('Starting repository indexing...')
            self.semantic_memory.index_repository(force_reindex=True)
        else:
            # Check if indexing is needed
            asyncio.create_task(self._background_index_repository())

    async def _background_index_repository(self) -> None:
        """Index repository in background."""
        try:
            if self.semantic_memory:
                self.semantic_memory.index_repository()
        except Exception as e:
            logger.warning(f'Background repository indexing failed: {e}')

    def create_task_context(
        self,
        intent: str,
        files_in_scope: list[str] | None = None,
        constraints: list[str] | None = None,
        success_criteria: list[str] | None = None,
    ) -> TaskContext:
        """Create a new task context."""
        task_id = generate_task_id()
        task_type = classify_task_type(intent)
        scope = determine_task_scope(files_in_scope or [])

        task = TaskContext(
            task_id=task_id,
            task_type=task_type,
            scope=scope,
            intent=intent,
            files_in_scope=files_in_scope or [],
            constraints=constraints or [],
            success_criteria=success_criteria or [],
        )

        # Set as current task in working memory
        self.working_memory.set_current_task(task)

        logger.debug(f'Created task context: {task_id} ({task_type.value})')
        return task

    def retrieve_context(
        self, task: TaskContext, max_code_chunks: int = 5, max_episodes: int = 3
    ) -> MemoryContext:
        """Retrieve relevant context from all memory components."""
        context = MemoryContext()

        # Get code context from semantic memory
        if self.semantic_memory and task.files_in_scope:
            try:
                code_chunks = self.semantic_memory.retrieve_relevant_code(
                    query=task.intent,
                    file_patterns=task.files_in_scope,
                    max_results=max_code_chunks,
                )

                for chunk in code_chunks:
                    relevance = chunk.metadata.get('relevance_score', 0.0)
                    context.code_snippets.append(
                        f'File: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}, '
                        f'relevance: {relevance:.2f})\n```\n{chunk.content[:500]}...\n```'
                    )
            except Exception as e:
                logger.warning(f'Failed to retrieve code context: {e}')

        # Get similar episodes from episodic memory
        try:
            similar_episodes = self.episodic_memory.find_similar_episodes(
                task_type=task.task_type,
                scope=task.scope,
                intent=task.intent,
                limit=max_episodes,
            )

            for episode in similar_episodes:
                success_status = '✓' if episode.outcome.success else '✗'
                duration = episode.duration.total_seconds()
                context.similar_episodes.append(
                    f'{success_status} {episode.intent} (duration: {duration:.1f}s, '
                    f'files: {len(episode.files_touched)})'
                )
        except Exception as e:
            logger.warning(f'Failed to retrieve similar episodes: {e}')

        # Get applicable skills
        try:
            skills = self.skill_memory.match_skills(task, min_confidence=0.3)

            for skill in skills[:3]:  # Top 3 skills
                stats = self.skill_memory.get_skill_statistics(skill.skill_id)
                if stats:
                    context.applicable_skills.append(
                        SkillReference(
                            skill_id=skill.skill_id,
                            skill_name=skill.name,
                            confidence=skill.confidence,
                            usage_count=stats['usage_count'],
                            success_rate=stats['success_rate'],
                        )
                    )
        except Exception as e:
            logger.warning(f'Failed to retrieve applicable skills: {e}')

        # Get failure warnings
        try:
            warnings = self.episodic_memory.get_failure_warnings(
                task_type=task.task_type, context=task.intent
            )
            context.failure_warnings.extend(warnings)
        except Exception as e:
            logger.warning(f'Failed to retrieve failure warnings: {e}')

        # Get success patterns
        try:
            patterns = self.episodic_memory.get_success_patterns(task.task_type)
            for pattern in patterns[:2]:  # Top 2 patterns
                context.success_patterns.append(
                    f'Pattern: {pattern["pattern_name"]} '
                    f'(success rate: {pattern["success_rate"]:.2f})'
                )
        except Exception as e:
            logger.warning(f'Failed to retrieve success patterns: {e}')

        return context

    def update_working_memory(self, action_outcome: ActionOutcome) -> None:
        """Update working memory with action outcome."""
        self.working_memory.add_action_outcome(action_outcome)

    def update_file_state(self, file_path: str, content: str) -> None:
        """Update file state in working memory."""
        self.working_memory.update_file_state(file_path, content)

        # Also update semantic memory if available
        if self.semantic_memory:
            try:
                self.semantic_memory.index_file(file_path)
            except Exception as e:
                logger.warning(f'Failed to update semantic memory for {file_path}: {e}')

    async def learn_from_task(self, task: TaskContext, outcome: TaskOutcome) -> None:
        """Learn from a completed task."""
        try:
            # Store episode in episodic memory
            self.episodic_memory.store_episode(task, outcome)

            # Update skill statistics if skills were used
            for skill_ref in task.applicable_skills:
                self.skill_memory.record_skill_application(
                    skill_id=skill_ref.skill_id,
                    task_id=task.task_id,
                    success=outcome.success,
                    duration=outcome.duration,
                )

            # Try to extract new skills from successful patterns
            if outcome.success:
                await self._try_extract_new_skills(task, outcome)

            logger.debug(f'Learning completed for task {task.task_id}')

        except Exception as e:
            logger.error(f'Failed to learn from task {task.task_id}: {e}')

    async def _try_extract_new_skills(
        self, task: TaskContext, outcome: TaskOutcome
    ) -> None:
        """Try to extract new skills from successful task patterns."""
        try:
            # Find similar successful episodes
            similar_episodes = self.episodic_memory.find_similar_episodes(
                task_type=task.task_type, scope=task.scope, intent=task.intent, limit=10
            )

            # Filter for successful episodes
            successful_episodes = [ep for ep in similar_episodes if ep.outcome.success]

            if len(successful_episodes) >= 3:
                # Try to extract a skill
                new_skill = self.skill_memory.extract_skill_from_episodes(
                    successful_episodes
                )

                if new_skill:
                    self.skill_memory.store_skill(new_skill)
                    logger.info(f'Extracted new skill: {new_skill.name}')

        except Exception as e:
            logger.warning(f'Failed to extract skills: {e}')

    def get_session_summary(self) -> dict[str, Any]:
        """Get a summary of the current session."""
        working_stats = self.working_memory.get_session_stats()

        # Get recent task statistics
        task_stats = self.episodic_memory.get_task_statistics(days=1)

        # Get top skills
        top_skills = self.skill_memory.get_top_skills(limit=5)

        return {
            'session': working_stats,
            'recent_tasks': task_stats,
            'top_skills': [
                {
                    'name': skill.name,
                    'confidence': skill.confidence,
                    'usage_count': skill.stats.usage_count,
                }
                for skill in top_skills
            ],
            'memory_status': {
                'semantic_memory_available': self.semantic_memory is not None,
                'repo_path': self.repo_path,
            },
        }

    def cleanup_old_data(self, days: int = 90) -> dict[str, int]:
        """Clean up old data from all memory components."""
        cleanup_stats = {}

        try:
            # Clean up old episodes
            cleanup_stats['episodes_deleted'] = (
                self.episodic_memory.cleanup_old_episodes(days)
            )
        except Exception as e:
            logger.warning(f'Failed to cleanup episodes: {e}')
            cleanup_stats['episodes_deleted'] = 0

        try:
            # Clean up unused skills
            cleanup_stats['skills_deleted'] = self.skill_memory.cleanup_unused_skills(
                days
            )
        except Exception as e:
            logger.warning(f'Failed to cleanup skills: {e}')
            cleanup_stats['skills_deleted'] = 0

        # Clear working memory
        self.working_memory.clear()
        cleanup_stats['working_memory_cleared'] = True

        logger.info(f'Memory cleanup completed: {cleanup_stats}')
        return cleanup_stats

    def export_memory_data(self, export_path: str) -> None:
        """Export memory data for backup or analysis."""
        # This would implement export functionality
        # For now, just log the request
        logger.info(f'Memory export requested to: {export_path}')
        # Implementation would copy database files and create summary reports

    def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        usage: dict[str, Any] = {
            'working_memory': {
                'active_files': len(self.working_memory.active_files),
                'recent_actions': len(self.working_memory.recent_actions),
                'current_task': self.working_memory.current_task.task_id
                if self.working_memory.current_task
                else None,
            }
        }

        # Add database sizes if available
        try:
            episodic_db_size = (
                self.episodic_memory.db_path.stat().st_size
                if self.episodic_memory.db_path.exists()
                else 0
            )
            skill_db_size = (
                self.skill_memory.db_path.stat().st_size
                if self.skill_memory.db_path.exists()
                else 0
            )

            usage['storage'] = {
                'episodic_db_size_mb': round(episodic_db_size / (1024 * 1024), 2),
                'skill_db_size_mb': round(skill_db_size / (1024 * 1024), 2),
            }

            if self.semantic_memory:
                semantic_db_size = (
                    self.semantic_memory.db_path.stat().st_size
                    if self.semantic_memory.db_path.exists()
                    else 0
                )
                usage['storage']['semantic_db_size_mb'] = round(
                    semantic_db_size / (1024 * 1024), 2
                )

        except Exception as e:
            logger.warning(f'Failed to get storage usage: {e}')

        return usage
