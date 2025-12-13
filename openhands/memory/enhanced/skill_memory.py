"""Skill memory for storing and retrieving reusable task patterns."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openhands.core.logger import openhands_logger as logger

from .types import ActionTemplate, Condition, Skill, SkillStats, TaskContext


class SkillMemory:
    """Manages reusable skills extracted from successful task patterns."""

    def __init__(self, memory_dir: str = '.openhands/memory'):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.memory_dir / 'skill_memory.db'
        self._init_database()

        # Cache for frequently used skills
        self._skill_cache: dict[str, Skill] = {}

    def _init_database(self) -> None:
        """Initialize SQLite database for skill memory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    task_pattern TEXT NOT NULL,
                    preconditions TEXT,  -- JSON array of conditions
                    action_template TEXT,  -- JSON object
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    total_duration_seconds REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS skill_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    duration_seconds REAL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (skill_id) REFERENCES skills (skill_id)
                )
            """)

            # Create indexes
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_skills_pattern ON skills (task_pattern)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_skills_confidence ON skills (confidence)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_applications_skill ON skill_applications (skill_id)'
            )

    def store_skill(self, skill: Skill) -> None:
        """Store a skill in memory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skills
                (skill_id, name, description, task_pattern, preconditions, action_template,
                 confidence, usage_count, success_count, failure_count, total_duration_seconds,
                 created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    skill.skill_id,
                    skill.name,
                    skill.description,
                    skill.task_pattern,
                    json.dumps(
                        [
                            {
                                'condition_type': cond.condition_type,
                                'parameters': cond.parameters,
                            }
                            for cond in skill.preconditions
                        ]
                    ),
                    json.dumps(
                        {
                            'action_type': skill.action_template.action_type,
                            'parameters': skill.action_template.parameters,
                        }
                    ),
                    skill.confidence,
                    skill.stats.usage_count,
                    skill.stats.success_count,
                    skill.stats.failure_count,
                    skill.stats.total_duration.total_seconds(),
                    skill.created_at,
                    skill.last_updated,
                ),
            )

        # Update cache
        self._skill_cache[skill.skill_id] = skill

        logger.debug(f'Stored skill: {skill.name}')

    def get_skill(self, skill_id: str) -> Skill | None:
        """Get a skill by ID."""
        # Check cache first
        if skill_id in self._skill_cache:
            return self._skill_cache[skill_id]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM skills WHERE skill_id = ?', [skill_id])
            row = cursor.fetchone()

            if row:
                skill = self._row_to_skill(row)
                self._skill_cache[skill_id] = skill
                return skill

        return None

    def match_skills(
        self, task: TaskContext, min_confidence: float = 0.3
    ) -> list[Skill]:
        """Find skills that match the given task."""
        matching_skills = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM skills
                WHERE confidence >= ?
                ORDER BY confidence DESC, usage_count DESC
            """,
                [min_confidence],
            )

            for row in cursor.fetchall():
                skill = self._row_to_skill(row)
                match_score = skill.matches_task(task)

                if match_score >= min_confidence:
                    # Add match score to skill for sorting
                    skill.metadata = {'match_score': match_score}
                    matching_skills.append(skill)

        # Sort by match score
        matching_skills.sort(
            key=lambda s: s.metadata.get('match_score', 0), reverse=True
        )

        return matching_skills

    def record_skill_application(
        self, skill_id: str, task_id: str, success: bool, duration: timedelta
    ) -> None:
        """Record the application of a skill."""
        with sqlite3.connect(self.db_path) as conn:
            # Record application
            conn.execute(
                """
                INSERT INTO skill_applications
                (skill_id, task_id, success, duration_seconds)
                VALUES (?, ?, ?, ?)
            """,
                (skill_id, task_id, success, duration.total_seconds()),
            )

            # Update skill statistics
            if success:
                conn.execute(
                    """
                    UPDATE skills
                    SET success_count = success_count + 1,
                        usage_count = usage_count + 1,
                        total_duration_seconds = total_duration_seconds + ?,
                        last_used = CURRENT_TIMESTAMP,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE skill_id = ?
                """,
                    (duration.total_seconds(), skill_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE skills
                    SET failure_count = failure_count + 1,
                        usage_count = usage_count + 1,
                        total_duration_seconds = total_duration_seconds + ?,
                        last_used = CURRENT_TIMESTAMP,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE skill_id = ?
                """,
                    (duration.total_seconds(), skill_id),
                )

            # Update confidence based on recent performance
            self._update_skill_confidence(skill_id, conn)

        # Clear cache for this skill to force reload
        if skill_id in self._skill_cache:
            del self._skill_cache[skill_id]

        logger.debug(
            f'Recorded skill application: {skill_id} ({"success" if success else "failure"})'
        )

    def get_skill_statistics(self, skill_id: str) -> dict[str, Any] | None:
        """Get statistics for a skill."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT usage_count, success_count, failure_count, total_duration_seconds,
                       confidence, created_at, last_used
                FROM skills WHERE skill_id = ?
            """,
                [skill_id],
            )

            row = cursor.fetchone()
            if not row:
                return None

            (
                usage_count,
                success_count,
                failure_count,
                total_duration,
                confidence,
                created_at,
                last_used,
            ) = row

            return {
                'usage_count': usage_count,
                'success_count': success_count,
                'failure_count': failure_count,
                'success_rate': success_count / usage_count if usage_count > 0 else 0.0,
                'avg_duration_seconds': total_duration / usage_count
                if usage_count > 0
                else 0.0,
                'confidence': confidence,
                'created_at': created_at,
                'last_used': last_used,
            }

    def get_top_skills(self, limit: int = 10) -> list[Skill]:
        """Get the top performing skills."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM skills
                WHERE usage_count > 0
                ORDER BY confidence DESC, usage_count DESC
                LIMIT ?
            """,
                [limit],
            )

            skills = []
            for row in cursor.fetchall():
                skill = self._row_to_skill(row)
                skills.append(skill)

            return skills

    def cleanup_unused_skills(self, days: int = 90) -> int:
        """Remove skills that haven't been used recently."""
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM skills
                WHERE (last_used IS NULL OR last_used < ?)
                AND usage_count < 3
            """,
                [cutoff_date],
            )

            deleted_count = cursor.rowcount

            # Also clean up applications for deleted skills
            conn.execute("""
                DELETE FROM skill_applications
                WHERE skill_id NOT IN (SELECT skill_id FROM skills)
            """)

            logger.info(f'Cleaned up {deleted_count} unused skills')
            return deleted_count

    def extract_skill_from_episodes(self, episodes: list[Any]) -> Skill | None:
        """Extract a new skill from successful task episodes."""
        if len(episodes) < 3:
            return None  # Need at least 3 episodes to extract a pattern

        # Analyze common patterns
        common_actions = self._find_common_action_patterns(episodes)
        if not common_actions:
            return None

        # Calculate confidence based on success rate
        success_count = sum(1 for ep in episodes if ep.outcome.success)
        confidence = success_count / len(episodes)

        if confidence < 0.7:  # Require high success rate
            return None

        # Generate skill
        skill_name = self._generate_skill_name(episodes)
        task_pattern = self._extract_task_pattern(episodes)

        skill = Skill(
            skill_id=f'extracted_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            name=skill_name,
            description=f'Extracted from {len(episodes)} successful episodes',
            task_pattern=task_pattern,
            preconditions=[],  # Could be enhanced to extract preconditions
            action_template=ActionTemplate(
                action_type=common_actions[0]['action_type'],
                parameters=common_actions[0].get('parameters', {}),
            ),
            confidence=confidence,
        )

        return skill

    def _update_skill_confidence(self, skill_id: str, conn) -> None:
        """Update skill confidence based on recent performance."""
        # Get recent applications (last 10)
        cursor = conn.execute(
            """
            SELECT success FROM skill_applications
            WHERE skill_id = ?
            ORDER BY applied_at DESC
            LIMIT 10
        """,
            [skill_id],
        )

        recent_results = [row[0] for row in cursor.fetchall()]

        if recent_results:
            recent_success_rate = sum(recent_results) / len(recent_results)

            # Get current confidence
            cursor = conn.execute(
                'SELECT confidence FROM skills WHERE skill_id = ?', [skill_id]
            )
            current_confidence = cursor.fetchone()[0]

            # Update confidence with weighted average (favor recent performance)
            new_confidence = 0.7 * recent_success_rate + 0.3 * current_confidence

            conn.execute(
                """
                UPDATE skills
                SET confidence = ?
                WHERE skill_id = ?
            """,
                (new_confidence, skill_id),
            )

    def _find_common_action_patterns(self, episodes: list[Any]) -> list[dict[str, Any]]:
        """Find common action patterns across episodes."""
        # Simplified pattern extraction - could be enhanced
        action_sequences = []

        for episode in episodes:
            if episode.outcome.success and episode.actions_taken:
                action_sequences.append(episode.actions_taken)

        if not action_sequences:
            return []

        # Find most common first action
        first_actions: dict[str, int] = {}
        for sequence in action_sequences:
            if sequence:
                action_type = sequence[0].get('action_type', 'unknown')
                first_actions[action_type] = first_actions.get(action_type, 0) + 1

        if not first_actions:
            return []

        # Return most common action pattern
        most_common_action = max(first_actions, key=lambda k: first_actions[k])

        # Find a representative example of this action
        for sequence in action_sequences:
            if sequence and sequence[0].get('action_type') == most_common_action:
                return [sequence[0]]

        return []

    def _generate_skill_name(self, episodes: list[Any]) -> str:
        """Generate a name for the skill based on episodes."""
        # Extract common words from task intents
        words = []
        for episode in episodes:
            words.extend(episode.intent.lower().split())

        # Find most common words (excluding common stop words)
        stop_words = {
            'the',
            'a',
            'an',
            'and',
            'or',
            'but',
            'in',
            'on',
            'at',
            'to',
            'for',
            'of',
            'with',
            'by',
        }
        word_counts: dict[str, int] = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1

        if word_counts:
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[
                :2
            ]
            return '_'.join(word[0] for word in top_words)

        return f'skill_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    def _extract_task_pattern(self, episodes: list[Any]) -> str:
        """Extract a task pattern from episodes."""
        # Simple pattern extraction - could be enhanced with NLP
        intents = [episode.intent.lower() for episode in episodes]

        # Find common words
        all_words = []
        for intent in intents:
            all_words.extend(intent.split())

        word_counts: dict[str, int] = {}
        for word in all_words:
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1

        # Create pattern from most common words
        common_words = [
            word for word, count in word_counts.items() if count >= len(episodes) * 0.5
        ]

        if common_words:
            return '|'.join(common_words[:3])  # Use top 3 words

        return 'general_task'

    def _row_to_skill(self, row) -> Skill:
        """Convert database row to Skill object."""
        preconditions_data = json.loads(row[4])
        action_template_data = json.loads(row[5])

        preconditions = [
            Condition(
                condition_type=cond['condition_type'], parameters=cond['parameters']
            )
            for cond in preconditions_data
        ]

        action_template = ActionTemplate(
            action_type=action_template_data['action_type'],
            parameters=action_template_data['parameters'],
        )

        stats = SkillStats(
            usage_count=row[7],
            success_count=row[8],
            failure_count=row[9],
            total_duration=timedelta(seconds=row[10]),
            last_used=datetime.fromisoformat(row[13]) if row[13] else None,
        )

        return Skill(
            skill_id=row[0],
            name=row[1],
            description=row[2] or '',
            task_pattern=row[3],
            preconditions=preconditions,
            action_template=action_template,
            confidence=row[6],
            stats=stats,
            created_at=datetime.fromisoformat(row[11]),
            last_updated=datetime.fromisoformat(row[12]),
        )
