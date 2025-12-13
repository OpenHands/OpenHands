"""Tests for working memory component."""

from datetime import timedelta

import pytest

from openhands.memory.enhanced.types import (
    ActionOutcome,
    TaskContext,
    TaskScope,
    TaskType,
)
from openhands.memory.enhanced.working_memory import WorkingMemory


class TestWorkingMemory:
    """Test working memory functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.working_memory = WorkingMemory()

    def test_set_current_task(self):
        """Test setting current task."""
        task = TaskContext(
            task_id='test-123',
            task_type=TaskType.DEBUG,
            scope=TaskScope.FILE,
            intent='Fix bug',
            files_in_scope=['test.py'],
        )

        self.working_memory.set_current_task(task)
        assert self.working_memory.current_task == task

    def test_add_action_outcome(self):
        """Test adding action outcomes."""
        outcome = ActionOutcome(
            action_type='str_replace_editor',
            action_data={'file': 'test.py'},
            success=True,
            duration=timedelta(seconds=2),
        )

        self.working_memory.add_action_outcome(outcome)

        assert len(self.working_memory.recent_actions) == 1
        assert self.working_memory.total_actions == 1
        assert self.working_memory.successful_actions == 1
        assert self.working_memory.failed_actions == 0

    def test_update_file_state(self):
        """Test updating file state."""
        self.working_memory.update_file_state('test.py', "print('hello')")

        assert 'test.py' in self.working_memory.active_files
        file_state = self.working_memory.active_files['test.py']
        assert file_state.file_path == 'test.py'
        assert file_state.access_count == 1

        # Update same file
        self.working_memory.update_file_state('test.py', "print('world')")
        assert file_state.access_count == 2
        assert len(file_state.modifications) == 1

    def test_get_recent_actions(self):
        """Test getting recent actions."""
        # Add multiple actions
        for i in range(5):
            outcome = ActionOutcome(
                action_type=f'action_{i}',
                action_data={},
                success=True,
                duration=timedelta(seconds=1),
            )
            self.working_memory.add_action_outcome(outcome)

        recent = self.working_memory.get_recent_actions(3)
        assert len(recent) == 3
        assert recent[-1].action_type == 'action_4'  # Most recent

    def test_get_modified_files(self):
        """Test getting modified files."""
        # Add files, some modified
        self.working_memory.update_file_state('file1.py', 'content1')
        self.working_memory.update_file_state('file2.py', 'content2')
        self.working_memory.update_file_state('file1.py', 'modified content1')

        modified = self.working_memory.get_modified_files()
        assert 'file1.py' in modified
        assert 'file2.py' not in modified

    def test_session_stats(self):
        """Test session statistics."""
        # Add some actions
        for i in range(3):
            outcome = ActionOutcome(
                action_type='test_action',
                action_data={},
                success=i < 2,  # 2 successful, 1 failed
                duration=timedelta(seconds=1),
            )
            self.working_memory.add_action_outcome(outcome)

        stats = self.working_memory.get_session_stats()

        assert stats['total_actions'] == 3
        assert stats['successful_actions'] == 2
        assert stats['failed_actions'] == 1
        assert stats['success_rate'] == 2 / 3

    def test_context_for_llm(self):
        """Test getting context for LLM."""
        task = TaskContext(
            task_id='test-123',
            task_type=TaskType.IMPLEMENT,
            scope=TaskScope.MODULE,
            intent='Add new feature',
            files_in_scope=['feature.py', 'test_feature.py'],
        )

        self.working_memory.set_current_task(task)

        # Add an action
        outcome = ActionOutcome(
            action_type='str_replace_editor',
            action_data={},
            success=True,
            duration=timedelta(seconds=1),
        )
        self.working_memory.add_action_outcome(outcome)

        # Update a file
        self.working_memory.update_file_state('feature.py', 'new code')

        context = self.working_memory.get_context_for_llm()

        assert 'Add new feature' in context
        assert 'implement' in context
        assert 'feature.py' in context
        assert 'str_replace_editor' in context
        assert 'success' in context

    def test_clear(self):
        """Test clearing working memory."""
        # Add some data
        task = TaskContext(
            task_id='test',
            task_type=TaskType.DEBUG,
            scope=TaskScope.FILE,
            intent='test',
            files_in_scope=[],
        )
        self.working_memory.set_current_task(task)
        self.working_memory.update_file_state('test.py', 'content')

        # Clear
        self.working_memory.clear()

        assert self.working_memory.current_task is None
        assert len(self.working_memory.active_files) == 0
        assert len(self.working_memory.recent_actions) == 0
        assert self.working_memory.total_actions == 0


if __name__ == '__main__':
    pytest.main([__file__])
