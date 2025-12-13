"""Working memory for session-scoped context management."""

from __future__ import annotations

import hashlib
from collections import deque
from datetime import datetime
from typing import Any

from .types import ActionOutcome, FileState, TaskContext


class WorkingMemory:
    """Manages working memory for the current session.

    Working memory stores:
    - Current task context
    - Active file states
    - Recent action outcomes
    - Session summary
    """

    def __init__(self, max_actions: int = 100, max_size_mb: int = 10):
        self.max_actions = max_actions
        self.max_size_mb = max_size_mb

        self.current_task: TaskContext | None = None
        self.active_files: dict[str, FileState] = {}
        self.recent_actions: deque[ActionOutcome] = deque(maxlen=max_actions)
        self.context_summary: str = ''
        self.session_start: datetime = datetime.now()

        # Statistics
        self.total_actions: int = 0
        self.successful_actions: int = 0
        self.failed_actions: int = 0

    def set_current_task(self, task: TaskContext) -> None:
        """Set the current task context."""
        self.current_task = task
        self._update_context_summary()

    def add_action_outcome(self, outcome: ActionOutcome) -> None:
        """Add an action outcome to recent actions."""
        self.recent_actions.append(outcome)
        self.total_actions += 1

        if outcome.success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1

        self._check_memory_size()

    def update_file_state(self, file_path: str, content: str) -> None:
        """Update the state of a file."""
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if file_path in self.active_files:
            file_state = self.active_files[file_path]
            if file_state.content_hash != content_hash:
                # File was modified
                file_state.modifications.append(
                    {
                        'timestamp': datetime.now(),
                        'old_hash': file_state.content_hash,
                        'new_hash': content_hash,
                    }
                )
                file_state.content_hash = content_hash
                file_state.last_modified = datetime.now()
            file_state.access_count += 1
        else:
            # New file
            self.active_files[file_path] = FileState(
                file_path=file_path,
                content_hash=content_hash,
                last_modified=datetime.now(),
                access_count=1,
            )

    def get_recent_actions(self, count: int = 10) -> list[ActionOutcome]:
        """Get the most recent action outcomes."""
        return list(self.recent_actions)[-count:]

    def get_modified_files(self) -> list[str]:
        """Get list of files that have been modified in this session."""
        return [
            file_path
            for file_path, file_state in self.active_files.items()
            if file_state.modifications
        ]

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        duration = datetime.now() - self.session_start
        success_rate = (
            self.successful_actions / self.total_actions
            if self.total_actions > 0
            else 0.0
        )

        return {
            'session_duration': duration,
            'total_actions': self.total_actions,
            'successful_actions': self.successful_actions,
            'failed_actions': self.failed_actions,
            'success_rate': success_rate,
            'files_accessed': len(self.active_files),
            'files_modified': len(self.get_modified_files()),
            'current_task': self.current_task.task_id if self.current_task else None,
        }

    def get_context_for_llm(self) -> str:
        """Get formatted context for LLM prompt."""
        if not self.current_task:
            return 'No active task.'

        context_parts = [
            f'Current Task: {self.current_task.intent}',
            f'Task Type: {self.current_task.task_type.value}',
            f'Files in Scope: {", ".join(self.current_task.files_in_scope)}',
        ]

        if self.recent_actions:
            recent_action = self.recent_actions[-1]
            context_parts.append(
                f'Last Action: {recent_action.action_type} '
                f'({"success" if recent_action.success else "failed"})'
            )

        modified_files = self.get_modified_files()
        if modified_files:
            context_parts.append(f'Modified Files: {", ".join(modified_files)}')

        return '\n'.join(context_parts)

    def clear(self) -> None:
        """Clear working memory."""
        self.current_task = None
        self.active_files.clear()
        self.recent_actions.clear()
        self.context_summary = ''
        self.total_actions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        self.session_start = datetime.now()

    def _update_context_summary(self) -> None:
        """Update the context summary."""
        if not self.current_task:
            self.context_summary = ''
            return

        summary_parts = [
            f'Working on: {self.current_task.intent}',
            f'Scope: {len(self.current_task.files_in_scope)} files',
        ]

        if self.recent_actions:
            success_count = sum(1 for a in self.recent_actions if a.success)
            total_count = len(self.recent_actions)
            summary_parts.append(f'Recent success rate: {success_count}/{total_count}')

        self.context_summary = ' | '.join(summary_parts)

    def _check_memory_size(self) -> None:
        """Check if memory size exceeds limits and summarize if needed."""
        # Simplified size check - in practice would calculate actual memory usage
        if len(self.recent_actions) > self.max_actions * 0.9:
            self._summarize_old_actions()

    def _summarize_old_actions(self) -> None:
        """Summarize and remove old actions to free memory."""
        # Keep only the most recent half of actions
        keep_count = self.max_actions // 2
        old_actions = list(self.recent_actions)[:-keep_count]

        # Create summary of old actions
        success_count = sum(1 for a in old_actions if a.success)
        action_types: dict[str, int] = {}
        for action in old_actions:
            action_types[action.action_type] = (
                action_types.get(action.action_type, 0) + 1
            )

        summary = f'Summarized {len(old_actions)} actions: {success_count} successful. '
        summary += f'Action types: {dict(action_types)}'

        # Update context summary to include this information
        self.context_summary += f' | {summary}'

        # Keep only recent actions
        self.recent_actions = deque(
            list(self.recent_actions)[-keep_count:], maxlen=self.max_actions
        )
