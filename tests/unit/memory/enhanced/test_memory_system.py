"""Tests for the enhanced memory system."""

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from openhands.memory.enhanced.memory_system import MemorySystem
from openhands.memory.enhanced.types import TaskOutcome, TaskScope, TaskType


class TestMemorySystem:
    """Test the main memory system."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_system = MemorySystem(memory_dir=self.temp_dir)

    def test_create_task_context(self):
        """Test creating a task context."""
        task = self.memory_system.create_task_context(
            intent='Fix the login bug',
            files_in_scope=['auth/login.py'],
            constraints=["Don't break existing tests"],
            success_criteria=['All tests pass'],
        )

        assert task.intent == 'Fix the login bug'
        assert task.task_type == TaskType.DEBUG
        assert task.scope == TaskScope.FILE
        assert task.files_in_scope == ['auth/login.py']
        assert task.constraints == ["Don't break existing tests"]
        assert task.success_criteria == ['All tests pass']
        assert task.task_id is not None

    def test_working_memory_integration(self):
        """Test working memory integration."""
        task = self.memory_system.create_task_context('Test task')

        # Check that task is set in working memory
        assert self.memory_system.working_memory.current_task == task

        # Test file state updates
        self.memory_system.update_file_state('test.py', "print('hello')")
        assert 'test.py' in self.memory_system.working_memory.active_files

    def test_session_summary(self):
        """Test getting session summary."""
        summary = self.memory_system.get_session_summary()

        assert 'session' in summary
        assert 'recent_tasks' in summary
        assert 'top_skills' in summary
        assert 'memory_status' in summary

    def test_memory_usage(self):
        """Test memory usage statistics."""
        usage = self.memory_system.get_memory_usage()

        assert 'working_memory' in usage
        assert 'active_files' in usage['working_memory']
        assert 'recent_actions' in usage['working_memory']


class TestMemorySystemWithRepo:
    """Test memory system with repository."""

    def setup_method(self):
        """Set up test environment with a mock repository."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_dir = Path(self.temp_dir) / 'test_repo'
        self.repo_dir.mkdir()

        # Create some test files
        (self.repo_dir / 'main.py').write_text("""
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
""")

        (self.repo_dir / 'utils.py').write_text("""
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b
""")

        self.memory_system = MemorySystem(
            repo_path=str(self.repo_dir), memory_dir=str(Path(self.temp_dir) / 'memory')
        )

    def test_repository_initialization(self):
        """Test repository initialization."""
        assert self.memory_system.semantic_memory is not None
        assert self.memory_system.repo_path == str(self.repo_dir)

    def test_semantic_memory_available(self):
        """Test that semantic memory is available."""
        summary = self.memory_system.get_session_summary()
        assert summary['memory_status']['semantic_memory_available'] is True

    @pytest.mark.asyncio
    async def test_learn_from_task(self):
        """Test learning from a completed task."""
        # Create a task
        task = self.memory_system.create_task_context(
            intent='Add a new function', files_in_scope=['utils.py']
        )

        # Create a successful outcome
        outcome = TaskOutcome(
            task_id=task.task_id,
            success=True,
            duration=timedelta(minutes=5),
            files_touched=['utils.py'],
            actions_taken=[
                {'action_type': 'str_replace_editor', 'success': True},
                {'action_type': 'execute_bash', 'success': True},
            ],
            success_metrics={'duration_seconds': 300},
        )

        # Learn from the task
        await self.memory_system.learn_from_task(task, outcome)

        # Verify the episode was stored
        episodes = self.memory_system.episodic_memory.find_similar_episodes(
            task_type=task.task_type, scope=task.scope, limit=1
        )

        assert len(episodes) == 1
        assert episodes[0].task_id == task.task_id


if __name__ == '__main__':
    pytest.main([__file__])
