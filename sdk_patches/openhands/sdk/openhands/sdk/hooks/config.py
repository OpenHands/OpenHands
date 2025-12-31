"""Hook configuration loading and management."""

import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openhands.sdk.hooks.types import HookEventType


logger = logging.getLogger(__name__)


class HookType(str, Enum):
    """Types of hooks that can be executed."""

    COMMAND = "command"  # Shell command executed via subprocess
    PROMPT = "prompt"  # LLM-based evaluation (future)


class HookDefinition(BaseModel):
    """A single hook definition."""

    type: HookType = HookType.COMMAND
    command: str
    timeout: int = 60


class HookMatcher(BaseModel):
    """Matches events to hooks based on patterns.

    Supports exact match, wildcard (*), and regex (auto-detected or /pattern/).
    """

    matcher: str = "*"
    hooks: list[HookDefinition] = Field(default_factory=list)

    # Regex metacharacters that indicate a pattern should be treated as regex
    _REGEX_METACHARACTERS = set("|.*+?[]()^$\\")

    def matches(self, tool_name: str | None) -> bool:
        """Check if this matcher matches the given tool name."""
        # Wildcard matches everything
        if self.matcher == "*" or self.matcher == "":
            return True

        if tool_name is None:
            return self.matcher in ("*", "")

        # Check for explicit regex pattern (enclosed in /)
        is_regex = (
            self.matcher.startswith("/")
            and self.matcher.endswith("/")
            and len(self.matcher) > 2
        )
        if is_regex:
            pattern = self.matcher[1:-1]
            try:
                return bool(re.fullmatch(pattern, tool_name))
            except re.error:
                return False

        # Auto-detect regex: if matcher contains metacharacters, treat as regex
        if any(c in self.matcher for c in self._REGEX_METACHARACTERS):
            try:
                return bool(re.fullmatch(self.matcher, tool_name))
            except re.error:
                # Invalid regex, fall through to exact match
                pass

        # Exact match
        return self.matcher == tool_name


class HookConfig(BaseModel):
    """Configuration for all hooks, loaded from .openhands/hooks.json."""

    hooks: dict[str, list[HookMatcher]] = Field(default_factory=dict)

    @classmethod
    def load(
        cls, path: str | Path | None = None, working_dir: str | Path | None = None
    ) -> "HookConfig":
        """Load config from path or search .openhands/hooks.json locations.

        Args:
            path: Explicit path to hooks.json file. If provided, working_dir is ignored.
            working_dir: Project directory for discovering .openhands/hooks.json.
                Falls back to cwd if not provided.
        """
        if path is None:
            # Search for hooks.json in standard locations
            base_dir = Path(working_dir) if working_dir else Path.cwd()
            search_paths = [
                base_dir / ".openhands" / "hooks.json",
                Path.home() / ".openhands" / "hooks.json",
            ]
            for search_path in search_paths:
                if search_path.exists():
                    path = search_path
                    break

        if path is None:
            return cls()

        path = Path(path)
        if not path.exists():
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            # Log warning but don't fail - just return empty config
            logger.warning(f"Failed to load hooks from {path}: {e}")
            return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HookConfig":
        """Create HookConfig from a dictionary."""
        hooks_data = data.get("hooks", {})
        hooks: dict[str, list[HookMatcher]] = {}

        for event_type, matchers in hooks_data.items():
            if not isinstance(matchers, list):
                continue

            hooks[event_type] = []
            for matcher_data in matchers:
                if isinstance(matcher_data, dict):
                    # Parse hooks within the matcher
                    hook_defs = []
                    for hook_data in matcher_data.get("hooks", []):
                        if isinstance(hook_data, dict):
                            hook_defs.append(HookDefinition(**hook_data))

                    hooks[event_type].append(
                        HookMatcher(
                            matcher=matcher_data.get("matcher", "*"),
                            hooks=hook_defs,
                        )
                    )

        return cls(hooks=hooks)

    def get_hooks_for_event(
        self, event_type: HookEventType, tool_name: str | None = None
    ) -> list[HookDefinition]:
        """Get all hooks that should run for an event."""
        event_key = event_type.value
        matchers = self.hooks.get(event_key, [])

        result: list[HookDefinition] = []
        for matcher in matchers:
            if matcher.matches(tool_name):
                result.extend(matcher.hooks)

        return result

    def has_hooks_for_event(self, event_type: HookEventType) -> bool:
        """Check if there are any hooks configured for an event type."""
        return event_type.value in self.hooks and len(self.hooks[event_type.value]) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        hooks_dict = {k: [m.model_dump() for m in v] for k, v in self.hooks.items()}
        return {"hooks": hooks_dict}

    def save(self, path: str | Path) -> None:
        """Save hook configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
