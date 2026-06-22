import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Valid source patterns for MarketplaceRegistration
# - github:owner/repo format
_GITHUB_SOURCE_PATTERN = re.compile(r'^github:[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$')
# - Git URLs (https, git, ssh protocols)
_GIT_URL_PATTERN = re.compile(
    r'^(https?://|git@|ssh://|git://)[a-zA-Z0-9_.-]+[:/][a-zA-Z0-9_./-]+$'
)
# - Relative local paths (no absolute paths, no parent traversal)
_LOCAL_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_][a-zA-Z0-9_./-]*$')

# IMPORTANT: This model must stay in sync with the frontend type in
# frontend/src/types/settings.ts. When modifying fields, update both
# locations. Consider using OpenAPI/JSON Schema generation to automate
# this synchronization.


class MarketplaceRegistration(BaseModel):
    """Registration for a plugin marketplace.

    Represents a marketplace that can be registered for plugin resolution.
    Marketplaces can be auto-loaded (plugins loaded at conversation start)
    or registered only (available for explicit plugin references).

    Examples:
        >>> # Auto-load all plugins from a marketplace
        >>> MarketplaceRegistration(
        ...     name="public",
        ...     source="github:OpenHands/skills",
        ...     auto_load="all"
        ... )

        >>> # Register marketplace without auto-loading
        >>> MarketplaceRegistration(
        ...     name="experimental",
        ...     source="github:acme/experimental"
        ... )

        >>> # Marketplace in monorepo subdirectory
        >>> MarketplaceRegistration(
        ...     name="team",
        ...     source="github:acme/monorepo",
        ...     repo_path="marketplaces/internal",
        ...     auto_load="all"
        ... )
    """

    name: str = Field(description='Identifier for this marketplace registration')
    source: str = Field(
        description="Marketplace source: 'github:owner/repo', git URL, or local path"
    )
    ref: str | None = Field(
        default=None,
        description='Optional branch, tag, or commit (only for git sources)',
    )
    repo_path: str | None = Field(
        default=None,
        description=(
            'Subdirectory path within the git repository containing the marketplace '
            "(e.g., 'marketplaces/internal' for monorepos). "
            'Only relevant for git sources, not local paths.'
        ),
    )
    auto_load: Literal['all'] | None = Field(
        default=None,
        description=(
            'Auto-load behavior for this marketplace. '
            "'all' = load all plugins at conversation start. "
            'None = registered for resolution but not auto-loaded.'
        ),
    )

    def model_dump(self, **kwargs) -> dict:
        """Serialize, stripping None values for optional fields."""
        data = super().model_dump(**kwargs)
        return {k: v for k, v in data.items() if v is not None}

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty and contains only valid identifier characters."""
        if not v or not v.strip():
            raise ValueError('name cannot be empty')
        v = v.strip()
        # Name should be a valid identifier (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v):
            raise ValueError(
                'name must start with a letter and contain only '
                'letters, numbers, hyphens, and underscores'
            )
        return v

    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate source matches expected patterns (github:owner/repo, git URL, or local path)."""
        if not v or not v.strip():
            raise ValueError('source cannot be empty')
        v = v.strip()

        # Check for valid source patterns
        if _GITHUB_SOURCE_PATTERN.match(v):
            return v
        if _GIT_URL_PATTERN.match(v):
            return v
        # Local path: must be relative, no parent traversal
        if v.startswith('/'):
            raise ValueError('local path source must be relative, not absolute')
        if '..' in v:
            raise ValueError("source cannot contain '..' (parent directory traversal)")
        if _LOCAL_PATH_PATTERN.match(v):
            return v

        raise ValueError(
            "source must be 'github:owner/repo', a git URL "
            '(https/git/ssh), or a relative local path'
        )

    @field_validator('repo_path')
    @classmethod
    def validate_repo_path(cls, v: str | None) -> str | None:
        """Validate repo_path is a safe relative path within the repository."""
        if v is None:
            return v
        # Must be relative (no absolute paths)
        if v.startswith('/'):
            raise ValueError('repo_path must be relative, not absolute')
        # No parent directory traversal
        if '..' in v:
            raise ValueError(
                "repo_path cannot contain '..' (parent directory traversal)"
            )
        return v
