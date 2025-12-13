#!/usr/bin/env python3
"""
Demo script showing the enhanced memory system capabilities.

This script demonstrates how the memory system works and how it can be
integrated with OpenHands to provide long-term learning capabilities.
"""

import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path

from openhands.memory.enhanced import MemorySystem
from openhands.memory.enhanced.types import TaskOutcome


async def demo_memory_system():
    """Demonstrate the memory system capabilities."""
    print('🧠 OpenHands Enhanced Memory System Demo')
    print('=' * 50)

    # Create a temporary directory for the demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock repository
        repo_path = temp_path / 'demo_repo'
        repo_path.mkdir()

        # Create some demo files
        (repo_path / 'main.py').write_text("""
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")

        (repo_path / 'utils.py').write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")

        (repo_path / 'tests.py').write_text("""
import unittest
from utils import add, multiply, divide

class TestUtils(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(multiply(4, 5), 20)

    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        with self.assertRaises(ValueError):
            divide(10, 0)
""")

        # Initialize memory system
        memory_dir = temp_path / 'memory'
        memory_system = MemorySystem(
            repo_path=str(repo_path), memory_dir=str(memory_dir)
        )

        print(f'📁 Created demo repository at: {repo_path}')
        print(f'🗄️  Memory system initialized at: {memory_dir}')

        # Demo 1: Create and track a task
        print('\n1️⃣  Creating a task context...')
        task1 = memory_system.create_task_context(
            intent='Fix the divide by zero bug in utils.py',
            files_in_scope=['utils.py', 'tests.py'],
            constraints=["Don't break existing functionality"],
            success_criteria=['All tests pass', 'Handle edge cases properly'],
        )

        print(f'   Task ID: {task1.task_id}')
        print(f'   Task Type: {task1.task_type.value}')
        print(f'   Scope: {task1.scope.value}')
        print(f'   Files: {task1.files_in_scope}')

        # Demo 2: Simulate working memory updates
        print('\n2️⃣  Simulating file modifications...')
        memory_system.update_file_state(
            'utils.py',
            """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""",
        )

        working_stats = memory_system.working_memory.get_session_stats()
        print(f'   Files accessed: {working_stats["files_accessed"]}')
        print(f'   Files modified: {working_stats["files_modified"]}')

        # Demo 3: Complete the task and learn
        print('\n3️⃣  Completing task and learning...')
        outcome1 = TaskOutcome(
            task_id=task1.task_id,
            success=True,
            duration=timedelta(minutes=10),
            files_touched=['utils.py'],
            actions_taken=[
                {
                    'action_type': 'str_replace_editor',
                    'file': 'utils.py',
                    'success': True,
                },
                {
                    'action_type': 'execute_bash',
                    'command': 'python -m pytest tests.py',
                    'success': True,
                },
            ],
            success_metrics={
                'duration_seconds': 600,
                'tests_passed': 4,
                'tests_failed': 0,
            },
            learned_patterns=['error_handling_pattern', 'test_driven_fix'],
        )

        await memory_system.learn_from_task(task1, outcome1)
        print('   ✅ Task completed successfully and stored in episodic memory')

        # Demo 4: Create a similar task to show memory retrieval
        print('\n4️⃣  Creating similar task to demonstrate memory retrieval...')
        task2 = memory_system.create_task_context(
            intent='Add input validation to the multiply function',
            files_in_scope=['utils.py', 'tests.py'],
        )

        # Retrieve memory context
        memory_context = memory_system.retrieve_context(task2)

        print('   📚 Retrieved memory context:')
        if memory_context.similar_episodes:
            print(f'   - Similar episodes: {len(memory_context.similar_episodes)}')
            for episode in memory_context.similar_episodes:
                print(f'     • {episode}')

        if memory_context.applicable_skills:
            print(f'   - Applicable skills: {len(memory_context.applicable_skills)}')
            for skill in memory_context.applicable_skills:
                print(f'     • {skill.skill_name} (confidence: {skill.confidence:.2f})')

        if memory_context.success_patterns:
            print(f'   - Success patterns: {len(memory_context.success_patterns)}')
            for pattern in memory_context.success_patterns:
                print(f'     • {pattern}')

        # Demo 5: Show session summary
        print('\n5️⃣  Session summary:')
        summary = memory_system.get_session_summary()

        session_stats = summary['session']
        print(f'   - Total actions: {session_stats["total_actions"]}')
        print(f'   - Success rate: {session_stats["success_rate"]:.1%}')
        print(f'   - Files accessed: {session_stats["files_accessed"]}')

        recent_tasks = summary['recent_tasks']
        print(f'   - Recent tasks: {recent_tasks["total_tasks"]}')
        print(f'   - Recent success rate: {recent_tasks["success_rate"]:.1%}')

        # Demo 6: Memory usage statistics
        print('\n6️⃣  Memory usage:')
        usage = memory_system.get_memory_usage()

        working_mem = usage['working_memory']
        print(f'   - Active files: {working_mem["active_files"]}')
        print(f'   - Recent actions: {working_mem["recent_actions"]}')
        print(f'   - Current task: {working_mem["current_task"]}')

        if 'storage' in usage:
            storage = usage['storage']
            print(f'   - Episodic DB size: {storage["episodic_db_size_mb"]:.2f} MB')
            print(f'   - Skill DB size: {storage["skill_db_size_mb"]:.2f} MB')
            if 'semantic_db_size_mb' in storage:
                print(f'   - Semantic DB size: {storage["semantic_db_size_mb"]:.2f} MB')

        print('\n✨ Demo completed! The memory system has:')
        print('   • Tracked task contexts and outcomes')
        print('   • Stored episodic memories of completed tasks')
        print('   • Maintained working memory of the current session')
        print('   • Provided relevant context for similar future tasks')
        print('   • Demonstrated learning from successful patterns')

        print(f'\n💾 Memory data persisted in: {memory_dir}')
        print('   This data would be available for future sessions!')


if __name__ == '__main__':
    asyncio.run(demo_memory_system())
