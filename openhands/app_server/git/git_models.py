"""Git-related models for V1 API pagination responses."""

from enum import StrEnum

from pydantic import BaseModel

from openhands.integrations.service_types import (
    Branch,
    Repository,
    SuggestedTask,
)


class SortOrder(StrEnum):
    """Sort order for search results combining sort field and direction.

    Values:
        STAR_DESC: Sort by stars, descending
        STAR_ASC: Sort by stars, ascending
        FORKS_DESC: Sort by forks, descending
        FORKS_ASC: Sort by forks, ascending
        UPDATED_DESC: Sort by last updated, descending
        UPDATED_ASC: Sort by last updated, ascending
    """

    STAR_DESC = 'stars-desc'
    STAR_ASC = 'stars-asc'
    FORKS_DESC = 'forks-desc'
    FORKS_ASC = 'forks-asc'
    UPDATED_DESC = 'updated-desc'
    UPDATED_ASC = 'updated-asc'


class InstallationPage(BaseModel):
    """Paginated response for installations.

    Attributes:
        items: List of installation IDs.
        next_page_id: ID for the next page, or None if there are no more pages.
    """

    items: list[str]
    next_page_id: str | None = None


class RepositoryPage(BaseModel):
    """Paginated response for repositories.

    Attributes:
        items: List of repositories in the current page.
        next_page_id: ID for the next page, or None if there are no more pages.
    """

    items: list[Repository]
    next_page_id: str | None = None


class BranchPage(BaseModel):
    """Paginated response for branch search results.

    Attributes:
        items: List of branches in the current page.
        next_page_id: ID for the next page, or None if there are no more pages.
    """

    items: list[Branch]
    next_page_id: str | None = None


class SuggestedTaskPage(BaseModel):
    """Paginated response for suggested tasks.

    Attributes:
        items: List of suggested tasks in the current page.
        next_page_id: ID for the next page, or None if there are no more pages.
    """

    items: list[SuggestedTask]
    next_page_id: str | None = None
