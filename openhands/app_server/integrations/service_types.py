from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, SecretStr

from openhands.server.types import AppMode


class TokenResponse(BaseModel):
    token: str


class ProviderType(Enum):
    GITHUB = 'github'
    GITLAB = 'gitlab'
    BITBUCKET = 'bitbucket'
    BITBUCKET_DATA_CENTER = 'bitbucket_data_center'
    FORGEJO = 'forgejo'
    AZURE_DEVOPS = 'azure_devops'
    ENTERPRISE_SSO = 'enterprise_sso'


class TaskType(str, Enum):
    MERGE_CONFLICTS = 'MERGE_CONFLICTS'
    FAILING_CHECKS = 'FAILING_CHECKS'
    UNRESOLVED_COMMENTS = 'UNRESOLVED_COMMENTS'
    OPEN_ISSUE = 'OPEN_ISSUE'
    OPEN_PR = 'OPEN_PR'


class OwnerType(str, Enum):
    USER = 'user'
    ORGANIZATION = 'organization'


class SuggestedTask(BaseModel):
    git_provider: ProviderType
    task_type: TaskType
    repo: str
    issue_number: int
    title: str

    def get_provider_terms(self) -> dict:
        if self.git_provider == ProviderType.GITHUB:
            return {
                'requestType': 'Pull Request',
                'requestTypeShort': 'PR',
                'apiName': 'GitHub API',
                'tokenEnvVar': 'GITHUB_TOKEN',
                'ciSystem': 'GitHub Actions',
                'ciProvider': 'GitHub',
                'requestVerb': 'pull request',
            }
        elif self.git_provider == ProviderType.GITLAB:
            return {
                'requestType': 'Merge Request',
                'requestTypeShort': 'MR',
                'apiName': 'GitLab API',
                'tokenEnvVar': 'GITLAB_TOKEN',
                'ciSystem': 'CI pipelines',
                'ciProvider': 'GitLab',
                'requestVerb': 'merge request',
            }
        elif self.git_provider == ProviderType.BITBUCKET:
            return {
                'requestType': 'Pull Request',
                'requestTypeShort': 'PR',
                'apiName': 'Bitbucket API',
                'tokenEnvVar': 'BITBUCKET_TOKEN',
                'ciSystem': 'Bitbucket Pipelines',
                'ciProvider': 'Bitbucket',
                'requestVerb': 'pull request',
            }
        elif self.git_provider == ProviderType.BITBUCKET_DATA_CENTER:
            return {
                'requestType': 'Pull Request',
                'requestTypeShort': 'PR',
                'apiName': 'Bitbucket Data Center API',
                'tokenEnvVar': 'BITBUCKET_DATA_CENTER_TOKEN',
                'ciSystem': 'Bitbucket Pipelines',
                'ciProvider': 'Bitbucket Data Center',
                'requestVerb': 'pull request',
            }

        raise ValueError(f'Provider {self.git_provider} for suggested task prompts')

    def get_prompt_for_task(
        self,
    ) -> str:
        task_type = self.task_type
        issue_number = self.issue_number
        repo = self.repo

        env = Environment(
            loader=FileSystemLoader(
                'openhands/app_server/integrations/templates/suggested_task'
            )
        )

        template = None
        if task_type == TaskType.MERGE_CONFLICTS:
            template = env.get_template('merge_conflict_prompt.j2')
        elif task_type == TaskType.FAILING_CHECKS:
            template = env.get_template('failing_checks_prompt.j2')
        elif task_type == TaskType.UNRESOLVED_COMMENTS:
            template = env.get_template('unresolved_comments_prompt.j2')
        elif task_type == TaskType.OPEN_ISSUE:
            template = env.get_template('open_issue_prompt.j2')
        else:
            raise ValueError(f'Unsupported task type: {task_type}')

        terms = self.get_provider_terms()

        return template.render(issue_number=issue_number, repo=repo, **terms)


class UserGitInfo(BaseModel):
    id: str
    login: str
    avatar_url: str
    company: str | None = None
    name: str | None = None
    email: str | None = None


# Keep old name of UserGitInfo for now for backwards compatibility
User = UserGitInfo


class Branch(BaseModel):
    name: str
    commit_sha: str
    protected: bool
    last_push_date: str | None = None  # ISO 8601 format date string


class PaginatedBranchesResponse(BaseModel):
    branches: list[Branch]
    has_next_page: bool
    current_page: int
    per_page: int
    total_count: int | None = None  # Some APIs don't provide total count


class Repository(BaseModel):
    id: str
    full_name: str
    git_provider: ProviderType
    is_public: bool
    stargazers_count: int | None = None
    link_header: str | None = None
    pushed_at: str | None = None  # ISO 8601 format date string
    owner_type: OwnerType | None = (
        None  # Whether the repository is owned by a user or organization
    )
    main_branch: str | None = None  # The main/default branch of the repository


class Comment(BaseModel):
    id: str
    body: str
    author: str
    created_at: datetime
    updated_at: datetime
    system: bool = False  # Whether this is a system-generated comment


class AuthenticationError(ValueError):
    """Raised when there is an issue with GitHub authentication."""

    pass


class UnknownException(ValueError):
    """Raised when there is an issue with GitHub communcation."""

    pass


class RateLimitError(ValueError):
    """Raised when the git provider's API rate limits are exceeded."""

    pass


class ProviderTimeoutError(ValueError):
    """Raised when a request to a git provider times out."""

    pass


class ResourceNotFoundError(ValueError):
    """Raised when a requested resource (file, directory, etc.) is not found."""

    pass


class RequestMethod(Enum):
    POST = 'post'
    GET = 'get'


class BaseGitService(ABC):
    @property
    def provider(self) -> str:
        raise NotImplementedError('Subclasses must implement the provider property')

    # Method used to satisfy mypy for abstract class definition
    @abstractmethod
    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ) -> tuple[Any, dict]: ...

    def _truncate_comment(
        self, comment_body: str, max_comment_length: int = 500
    ) -> str:
        """Truncate comment body to a maximum length."""
        if len(comment_body) > max_comment_length:
            return comment_body[:max_comment_length] + '...'
        return comment_body


class InstallationsService(Protocol):
    async def get_installations(self) -> list[str]:
        """Get installations for the service; repos live underneath these installations"""
        ...


class GitService(Protocol):
    """Protocol defining the interface for Git service providers"""

    def __init__(
        self,
        user_id: str | None = None,
        token: SecretStr | None = None,
        external_auth_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
    ) -> None:
        """Initialize the service with authentication details"""
        ...

    async def get_latest_token(self) -> SecretStr | None:
        """Get latest working token of the user"""
        ...

    async def get_user(self) -> User:
        """Get the authenticated user's information"""
        ...

    async def search_repositories(
        self,
        query: str,
        per_page: int,
        sort: str,
        order: str,
        public: bool,
        app_mode: AppMode,
    ) -> list[Repository]:
        """Search for public repositories"""
        ...

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode
    ) -> list[Repository]:
        """Get repositories for the authenticated user"""
        ...

    async def get_paginated_repos(
        self,
        page: int,
        per_page: int,
        sort: str,
        installation_id: str | None,
        query: str | None = None,
    ) -> list[Repository]:
        """Get a page of repositories for the authenticated user"""
        ...

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        """Get suggested tasks for the authenticated user across all repositories"""
        ...

    async def get_repository_details_from_repo_name(
        self, repository: str
    ) -> Repository:
        """Gets all repository details from repository name"""

    async def get_branches(self, repository: str) -> list[Branch]:
        """Get branches for a repository"""

    async def get_paginated_branches(
        self, repository: str, page: int = 1, per_page: int = 30
    ) -> PaginatedBranchesResponse:
        """Get branches for a repository with pagination"""

    async def search_branches(
        self, repository: str, query: str, per_page: int = 30
    ) -> list[Branch]:
        """Search for branches within a repository"""

    async def get_pr_details(self, repository: str, pr_number: int) -> dict:
        """Get detailed information about a specific pull request/merge request

        Args:
            repository: Repository name in format specific to the provider
            pr_number: The pull request/merge request number

        Returns:
            Raw API response from the git provider
        """
        ...

    async def is_pr_open(self, repository: str, pr_number: int) -> bool:
        """Check if a PR is still active (not closed/merged).

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The PR number to check

        Returns:
            True if PR is active (open), False if closed/merged
        """
        ...
