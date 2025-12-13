"""Type definitions for the enhanced memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


class TaskType(str, Enum):
    """Types of tasks the agent can perform."""

    DEBUG = 'debug'
    IMPLEMENT = 'implement'
    REFACTOR = 'refactor'
    TEST = 'test'
    DOCUMENT = 'document'
    ANALYZE = 'analyze'
    FIX = 'fix'
    OPTIMIZE = 'optimize'
    REVIEW = 'review'
    UNKNOWN = 'unknown'


class TaskScope(str, Enum):
    """Scope of task impact."""

    FILE = 'file'
    MODULE = 'module'
    PACKAGE = 'package'
    REPOSITORY = 'repository'
    SYSTEM = 'system'


@dataclass
class TaskContext:
    """Context information for a task."""

    task_id: str
    task_type: TaskType
    scope: TaskScope
    intent: str
    files_in_scope: list[str]
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    estimated_complexity: int = 1

    # Memory integration
    relevant_memories: list[MemoryReference] = field(default_factory=list)
    applicable_skills: list[SkillReference] = field(default_factory=list)
    context_embeddings: np.ndarray | None = None


@dataclass
class TaskOutcome:
    """Outcome of a completed task."""

    task_id: str
    success: bool
    duration: timedelta
    files_touched: list[str]
    actions_taken: list[dict[str, Any]]
    error_messages: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    success_metrics: dict[str, float] = field(default_factory=dict)
    learned_patterns: list[str] = field(default_factory=list)


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, module, etc.
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    dependencies: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    complexity: int = 1
    last_modified: datetime | None = None


@dataclass
class MemoryReference:
    """Reference to a memory entry."""

    memory_type: str  # semantic, episodic, skill
    memory_id: str
    relevance_score: float
    content_summary: str


@dataclass
class SkillReference:
    """Reference to a skill."""

    skill_id: str
    skill_name: str
    confidence: float
    usage_count: int
    success_rate: float


@dataclass
class ActionOutcome:
    """Outcome of a single action."""

    action_type: str
    action_data: dict[str, Any]
    success: bool
    duration: timedelta
    error_message: str | None = None
    observation: str | None = None


@dataclass
class FileState:
    """State of a file during a session."""

    file_path: str
    content_hash: str
    last_modified: datetime
    modifications: list[dict[str, Any]] = field(default_factory=list)
    access_count: int = 0


@dataclass
class Symbol:
    """A code symbol (function, class, variable, etc.)."""

    name: str
    symbol_type: str  # function, class, variable, etc.
    file_path: str
    line_number: int
    scope: str
    signature: str | None = None
    docstring: str | None = None


@dataclass
class Condition:
    """A condition for skill applicability."""

    condition_type: str  # file_contains, error_mentions, etc.
    parameters: dict[str, Any]

    def evaluate(self, context: TaskContext) -> bool:
        """Evaluate if this condition is met in the given context."""
        # Implementation would depend on condition_type
        return True  # Placeholder


@dataclass
class ActionTemplate:
    """Template for generating actions."""

    action_type: str
    parameters: dict[str, str]  # Can contain template variables like {{file_path}}

    def generate_action(self, context: TaskContext) -> dict[str, Any]:
        """Generate a concrete action from this template."""
        # Implementation would substitute template variables
        return {'action': self.action_type, 'parameters': self.parameters}


@dataclass
class SkillStats:
    """Statistics for a skill."""

    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration: timedelta = field(default_factory=lambda: timedelta(0))
    last_used: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class Skill:
    """A reusable skill extracted from successful task patterns."""

    skill_id: str
    name: str
    description: str
    task_pattern: str  # Regex or semantic pattern
    preconditions: list[Condition]
    action_template: ActionTemplate
    confidence: float
    stats: SkillStats = field(default_factory=SkillStats)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches_task(self, task: TaskContext) -> float:
        """Return confidence score (0-1) for how well this skill matches the task."""
        # Check preconditions
        for condition in self.preconditions:
            if not condition.evaluate(task):
                return 0.0

        # Check task pattern match (simplified)
        if self.task_pattern.lower() in task.intent.lower():
            return self.confidence

        return 0.0

    def generate_actions(self, task: TaskContext) -> list[dict[str, Any]]:
        """Generate parameterized actions for this task."""
        return [self.action_template.generate_action(task)]


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid.uuid4())


def classify_task_type(intent: str) -> TaskType:
    """Classify task type from intent description."""
    intent_lower = intent.lower()

    # Order matters - more specific patterns first
    if any(word in intent_lower for word in ['test', 'testing', 'verify']):
        return TaskType.TEST
    elif any(word in intent_lower for word in ['debug', 'fix', 'error', 'bug']):
        return TaskType.DEBUG
    elif any(
        word in intent_lower for word in ['refactor', 'restructure', 'reorganize']
    ):
        return TaskType.REFACTOR
    elif any(word in intent_lower for word in ['document', 'docs', 'documentation']):
        return TaskType.DOCUMENT
    elif any(word in intent_lower for word in ['analyze', 'analysis', 'understand']):
        return TaskType.ANALYZE
    elif any(word in intent_lower for word in ['optimize', 'performance', 'speed']):
        return TaskType.OPTIMIZE
    elif any(word in intent_lower for word in ['review', 'check', 'audit']):
        return TaskType.REVIEW
    elif any(word in intent_lower for word in ['implement', 'create', 'add', 'build']):
        return TaskType.IMPLEMENT
    else:
        return TaskType.UNKNOWN


def determine_task_scope(files_in_scope: list[str]) -> TaskScope:
    """Determine task scope based on files involved."""
    if not files_in_scope:
        return TaskScope.SYSTEM

    if len(files_in_scope) == 1:
        return TaskScope.FILE

    # Check if files are in same directory
    directories = set()
    for file_path in files_in_scope:
        directories.add('/'.join(file_path.split('/')[:-1]))

    if len(directories) == 1:
        return TaskScope.MODULE
    elif len(directories) <= 3:
        return TaskScope.PACKAGE
    else:
        return TaskScope.REPOSITORY
