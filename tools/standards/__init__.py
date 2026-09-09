"""Standards plugin types: violations, severity, and the plugin protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITIES = (SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR)

ACTION_WARN = "warn"
ACTION_BLOCK = "block"
ACTIONS = (ACTION_WARN, ACTION_BLOCK)

SOURCE_BUILTIN = "builtin"
SOURCE_USER = "user"
SOURCE_PROJECT = "project"
SOURCES = (SOURCE_BUILTIN, SOURCE_USER, SOURCE_PROJECT)

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"
RUN_STATUSES = (STATUS_PASSED, STATUS_FAILED, STATUS_ERROR)

STANDARDS_REQUIREMENTS_OPEN = "<STANDARDS_REQUIREMENTS>"
STANDARDS_REQUIREMENTS_CLOSE = "</STANDARDS_REQUIREMENTS>"


class ViolationSeverity(str, Enum):
    INFO = SEVERITY_INFO
    WARNING = SEVERITY_WARNING
    ERROR = SEVERITY_ERROR


@dataclass(frozen=True)
class Violation:
    plugin_name: str
    rule_id: str
    severity: str
    file: str
    message: str
    line: int | None = None
    remediation: str = ""
    action: str = ACTION_WARN
    fixable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PluginLoadError:
    path: str
    message: str
    source: str = SOURCE_BUILTIN

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StandardsRun:
    run_id: str
    started_at: str
    duration_ms: int
    status: str
    summary: dict[str, Any]
    violations: list[Violation] = field(default_factory=list)
    worktree: str = ""
    enforcement: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = [item.as_dict() for item in self.violations]
        return payload


class StandardsPlugin(ABC):
    """Deterministic file checker. ``check`` must not call an LLM."""

    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    severity_default: str = SEVERITY_WARNING

    @abstractmethod
    def check(self, file: str, content: str) -> list[Violation]:
        raise NotImplementedError

    def prompt_instructions(self) -> str:
        return ""

    def auto_fix(self, file: str, content: str) -> str | None:
        return None
