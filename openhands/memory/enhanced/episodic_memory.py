"""Episodic task memory for learning from past experiences."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openhands.core.logger import openhands_logger as logger

from .types import TaskContext, TaskOutcome, TaskScope, TaskType


class TaskEpisode:
    """Represents a completed task episode."""

    def __init__(
        self,
        task_id: str,
        task_type: TaskType,
        scope: TaskScope,
        intent: str,
        files_touched: list[str],
        actions_taken: list[dict[str, Any]],
        outcome: TaskOutcome,
        duration: timedelta,
        success_metrics: dict[str, float],
        learned_patterns: list[str] | None = None,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.scope = scope
        self.intent = intent
        self.files_touched = files_touched
        self.actions_taken = actions_taken
        self.outcome = outcome
        self.duration = duration
        self.success_metrics = success_metrics
        self.learned_patterns = learned_patterns or []
        self.created_at = datetime.now()


class EpisodicTaskMemory:
    """Manages episodic memory of completed tasks for learning and pattern recognition."""

    def __init__(self, memory_dir: str = '.openhands/memory'):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.memory_dir / 'episodic_memory.db'
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for episodic memory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_episodes (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    files_touched TEXT,  -- JSON array
                    actions_taken TEXT,  -- JSON array
                    outcome_data TEXT,   -- JSON object
                    duration_seconds REAL NOT NULL,
                    success_metrics TEXT,  -- JSON object
                    learned_patterns TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS action_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    action_sequence TEXT,  -- JSON array
                    success_rate REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_pattern TEXT NOT NULL,
                    task_type TEXT,
                    context TEXT,  -- JSON object
                    frequency INTEGER DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_episodes_type ON task_episodes (task_type)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_episodes_success ON task_episodes (success)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_episodes_created ON task_episodes (created_at)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_patterns_type ON action_patterns (task_type)'
            )

    def store_episode(self, task: TaskContext, outcome: TaskOutcome) -> None:
        """Store a completed task episode."""
        episode = TaskEpisode(
            task_id=task.task_id,
            task_type=task.task_type,
            scope=task.scope,
            intent=task.intent,
            files_touched=outcome.files_touched,
            actions_taken=outcome.actions_taken,
            outcome=outcome,
            duration=outcome.duration,
            success_metrics=outcome.success_metrics,
            learned_patterns=outcome.learned_patterns,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_episodes
                (task_id, task_type, scope, intent, files_touched, actions_taken,
                 outcome_data, duration_seconds, success_metrics, learned_patterns,
                 created_at, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    episode.task_id,
                    episode.task_type.value,
                    episode.scope.value,
                    episode.intent,
                    json.dumps(episode.files_touched),
                    json.dumps(episode.actions_taken),
                    json.dumps(
                        {
                            'success': outcome.success,
                            'error_messages': outcome.error_messages,
                            'tests_passed': outcome.tests_passed,
                            'tests_failed': outcome.tests_failed,
                        }
                    ),
                    episode.duration.total_seconds(),
                    json.dumps(episode.success_metrics),
                    json.dumps(episode.learned_patterns),
                    episode.created_at,
                    outcome.success,
                ),
            )

        # Extract and store patterns
        if outcome.success:
            self._extract_success_patterns(episode)
        else:
            self._extract_failure_patterns(episode)

        logger.debug(f'Stored episode for task {task.task_id}')

    def find_similar_episodes(
        self,
        task_type: TaskType,
        scope: TaskScope,
        intent: str | None = None,
        limit: int = 10,
    ) -> list[TaskEpisode]:
        """Find similar past episodes."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM task_episodes
                WHERE task_type = ? AND scope = ?
            """
            params: list[Any] = [task_type.value, scope.value]

            if intent:
                # Simple text matching - could be enhanced with semantic similarity
                query += ' AND intent LIKE ?'
                params.append(f'%{intent}%')

            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            episodes = []
            for row in rows:
                episode = self._row_to_episode(row)
                episodes.append(episode)

            return episodes

    def get_success_patterns(self, task_type: TaskType) -> list[dict[str, Any]]:
        """Get successful action patterns for a task type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT pattern_name, action_sequence, success_rate, usage_count
                FROM action_patterns
                WHERE task_type = ? AND success_rate > 0.5
                ORDER BY success_rate DESC, usage_count DESC
            """,
                [task_type.value],
            )

            patterns = []
            for row in cursor.fetchall():
                patterns.append(
                    {
                        'pattern_name': row[0],
                        'action_sequence': json.loads(row[1]),
                        'success_rate': row[2],
                        'usage_count': row[3],
                    }
                )

            return patterns

    def get_failure_warnings(self, task_type: TaskType, context: str) -> list[str]:
        """Get warnings about common failure patterns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT error_pattern, frequency
                FROM failure_patterns
                WHERE task_type = ? OR task_type IS NULL
                ORDER BY frequency DESC
            """,
                [task_type.value],
            )

            warnings = []
            for row in cursor.fetchall():
                error_pattern = row[0]
                frequency = row[1]

                # Simple pattern matching - could be enhanced
                if any(
                    word in context.lower() for word in error_pattern.lower().split()
                ):
                    warnings.append(
                        f"Warning: '{error_pattern}' has failed {frequency} times in similar contexts"
                    )

            return warnings

    def get_task_statistics(self, days: int = 30) -> dict[str, Any]:
        """Get statistics about recent tasks."""
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_tasks,
                    AVG(duration_seconds) as avg_duration
                FROM task_episodes
                WHERE created_at > ?
            """,
                [cutoff_date],
            )

            total, successful, avg_duration = cursor.fetchone()
            success_rate = successful / total if total > 0 else 0.0

            # Stats by task type
            cursor = conn.execute(
                """
                SELECT
                    task_type,
                    COUNT(*) as count,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                    AVG(duration_seconds) as avg_duration
                FROM task_episodes
                WHERE created_at > ?
                GROUP BY task_type
                ORDER BY count DESC
            """,
                [cutoff_date],
            )

            task_type_stats = {}
            for row in cursor.fetchall():
                task_type, count, successful, avg_dur = row
                task_type_stats[task_type] = {
                    'count': count,
                    'success_rate': successful / count if count > 0 else 0.0,
                    'avg_duration': avg_dur or 0.0,
                }

            return {
                'total_tasks': total or 0,
                'success_rate': success_rate,
                'avg_duration_seconds': avg_duration or 0.0,
                'task_type_stats': task_type_stats,
            }

    def cleanup_old_episodes(self, days: int = 90) -> int:
        """Clean up old episodes to manage storage."""
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'DELETE FROM task_episodes WHERE created_at < ?', [cutoff_date]
            )
            deleted_count = cursor.rowcount

            logger.info(f'Cleaned up {deleted_count} old episodes')
            return deleted_count

    def _extract_success_patterns(self, episode: TaskEpisode) -> None:
        """Extract and store successful action patterns."""
        if not episode.actions_taken:
            return

        # Create a pattern name based on action sequence
        action_types = [
            action.get('action_type', 'unknown') for action in episode.actions_taken
        ]
        pattern_name = ' -> '.join(action_types[:3])  # First 3 actions

        with sqlite3.connect(self.db_path) as conn:
            # Check if pattern exists
            cursor = conn.execute(
                """
                SELECT success_rate, usage_count FROM action_patterns
                WHERE pattern_name = ? AND task_type = ?
            """,
                [pattern_name, episode.task_type.value],
            )

            existing = cursor.fetchone()

            if existing:
                # Update existing pattern
                old_success_rate, old_usage_count = existing
                new_usage_count = old_usage_count + 1
                new_success_rate = (
                    old_success_rate * old_usage_count + 1.0
                ) / new_usage_count

                conn.execute(
                    """
                    UPDATE action_patterns
                    SET success_rate = ?, usage_count = ?, last_used = CURRENT_TIMESTAMP
                    WHERE pattern_name = ? AND task_type = ?
                """,
                    [
                        new_success_rate,
                        new_usage_count,
                        pattern_name,
                        episode.task_type.value,
                    ],
                )
            else:
                # Create new pattern
                conn.execute(
                    """
                    INSERT INTO action_patterns
                    (pattern_name, task_type, action_sequence, success_rate, usage_count, last_used)
                    VALUES (?, ?, ?, 1.0, 1, CURRENT_TIMESTAMP)
                """,
                    [
                        pattern_name,
                        episode.task_type.value,
                        json.dumps(episode.actions_taken),
                    ],
                )

    def _extract_failure_patterns(self, episode: TaskEpisode) -> None:
        """Extract and store failure patterns."""
        if not episode.outcome.error_messages:
            return

        for error_msg in episode.outcome.error_messages:
            # Extract key error patterns
            error_pattern = self._extract_error_pattern(error_msg)

            with sqlite3.connect(self.db_path) as conn:
                # Check if pattern exists
                cursor = conn.execute(
                    """
                    SELECT frequency FROM failure_patterns
                    WHERE error_pattern = ? AND task_type = ?
                """,
                    [error_pattern, episode.task_type.value],
                )

                existing = cursor.fetchone()

                if existing:
                    # Update frequency
                    new_frequency = existing[0] + 1
                    conn.execute(
                        """
                        UPDATE failure_patterns
                        SET frequency = ?, last_seen = CURRENT_TIMESTAMP
                        WHERE error_pattern = ? AND task_type = ?
                    """,
                        [new_frequency, error_pattern, episode.task_type.value],
                    )
                else:
                    # Create new failure pattern
                    conn.execute(
                        """
                        INSERT INTO failure_patterns
                        (error_pattern, task_type, context, frequency)
                        VALUES (?, ?, ?, 1)
                    """,
                        [
                            error_pattern,
                            episode.task_type.value,
                            json.dumps({'intent': episode.intent}),
                        ],
                    )

    def _extract_error_pattern(self, error_msg: str) -> str:
        """Extract a generalized error pattern from an error message."""
        # Simplified pattern extraction - could be enhanced with NLP
        error_lower = error_msg.lower()

        # Common error patterns
        if 'file not found' in error_lower or 'no such file' in error_lower:
            return 'file_not_found'
        elif 'permission denied' in error_lower:
            return 'permission_denied'
        elif 'syntax error' in error_lower:
            return 'syntax_error'
        elif 'import error' in error_lower or 'module not found' in error_lower:
            return 'import_error'
        elif 'type error' in error_lower:
            return 'type_error'
        elif 'attribute error' in error_lower:
            return 'attribute_error'
        else:
            # Extract first few words as pattern
            words = error_msg.split()[:3]
            return '_'.join(word.lower().strip('.,!?') for word in words)

    def _row_to_episode(self, row) -> TaskEpisode:
        """Convert database row to TaskEpisode object."""
        outcome_data = json.loads(row[6])

        outcome = TaskOutcome(
            task_id=row[0],
            success=outcome_data['success'],
            duration=timedelta(seconds=row[7]),
            files_touched=json.loads(row[4]),
            actions_taken=json.loads(row[5]),
            error_messages=outcome_data.get('error_messages', []),
            tests_passed=outcome_data.get('tests_passed', 0),
            tests_failed=outcome_data.get('tests_failed', 0),
            success_metrics=json.loads(row[8]),
            learned_patterns=json.loads(row[9]),
        )

        return TaskEpisode(
            task_id=row[0],
            task_type=TaskType(row[1]),
            scope=TaskScope(row[2]),
            intent=row[3],
            files_touched=json.loads(row[4]),
            actions_taken=json.loads(row[5]),
            outcome=outcome,
            duration=timedelta(seconds=row[7]),
            success_metrics=json.loads(row[8]),
            learned_patterns=json.loads(row[9]),
        )
