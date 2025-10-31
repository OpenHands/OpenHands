from __future__ import annotations

import os
from types import MappingProxyType
from typing import Annotated, Any, Coroutine, Literal, cast, overload
from urllib.parse import quote

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    WithJsonSchema,
)

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.events.action.commands import CmdRunAction
from openhands.events.stream import EventStream
from openhands.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl,
)
from openhands.integrations.bitbucket.bitbucket_service import BitBucketServiceImpl
from openhands.integrations.github.github_service import GithubServiceImpl
from openhands.integrations.gitlab.gitlab_service import GitLabServiceImpl
from openhands.integrations.service_types import (
    AuthenticationError,
    Branch,
    GitService,
    InstallationsService,
    MicroagentParseError,
    PaginatedBranchesResponse,
    ProviderType,
    Repository,
    ResourceNotFoundError,
    SuggestedTask,
    TokenResponse,
    User,
)
from openhands.microagent.types import MicroagentContentResponse, MicroagentResponse
from openhands.server.types import AppMode
from openhands.utils.http_session import httpx_verify_option


class ProviderToken(BaseModel):
    token: SecretStr | None = Field(default=None)
    user_id: str | None = Field(default=None)
    host: str | None = Field(default=None)

    model_config = ConfigDict(
        frozen=True,  # Makes the entire model immutable
        validate_assignment=True,
    )

    @classmethod
    def from_value(cls, token_value: ProviderToken | dict[str, str]) -> ProviderToken:
        """Factory method to create a ProviderToken from various input types"""
        if isinstance(token_value, cls):
            return token_value
        elif isinstance(token_value, dict):
            token_str = token_value.get('token', '')
            # Override with emtpy string if it was set to None
            # Cannot pass None to SecretStr
            if token_str is None:
                token_str = ''  # type: ignore[unreachable]
            user_id = token_value.get('user_id')
            host = token_value.get('host')
            return cls(token=SecretStr(token_str), user_id=user_id, host=host)

        else:
            raise ValueError('Unsupported Provider token type')


class CustomSecret(BaseModel):
    secret: SecretStr = Field(default_factory=lambda: SecretStr(''))
    description: str = Field(default='')

    model_config = ConfigDict(
        frozen=True,  # Makes the entire model immutable
        validate_assignment=True,
    )

    @classmethod
    def from_value(cls, secret_value: CustomSecret | dict[str, str]) -> CustomSecret:
        """Factory method to create a ProviderToken from various input types"""
        if isinstance(secret_value, CustomSecret):
            return secret_value
        elif isinstance(secret_value, dict):
            secret = secret_value.get('secret', '')
            description = secret_value.get('description', '')
            return cls(secret=SecretStr(secret), description=description)

        else:
            raise ValueError('Unsupport Provider token type')


PROVIDER_TOKEN_TYPE = MappingProxyType[ProviderType, ProviderToken]
CUSTOM_SECRETS_TYPE = MappingProxyType[str, CustomSecret]
PROVIDER_TOKEN_TYPE_WITH_JSON_SCHEMA = Annotated[
    PROVIDER_TOKEN_TYPE,
    WithJsonSchema({'type': 'object', 'additionalProperties': {'type': 'string'}}),
]
CUSTOM_SECRETS_TYPE_WITH_JSON_SCHEMA = Annotated[
    CUSTOM_SECRETS_TYPE,
    WithJsonSchema({'type': 'object', 'additionalProperties': {'type': 'string'}}),
]


class ProviderHandler:
    # Class variable for provider domains
    PROVIDER_DOMAINS: dict[ProviderType, str] = {
        ProviderType.GITHUB: 'github.com',
        ProviderType.GITLAB: 'gitlab.com',
        ProviderType.BITBUCKET: 'bitbucket.org',
        ProviderType.AZURE_DEVOPS: 'dev.azure.com',
    }

    def __init__(
        self,
        provider_tokens: PROVIDER_TOKEN_TYPE,
        external_auth_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        external_token_manager: bool = False,
        session_api_key: str | None = None,
        sid: str | None = None,
    ):
        if not isinstance(provider_tokens, MappingProxyType):
            raise TypeError(
                f'provider_tokens must be a MappingProxyType, got {type(provider_tokens).__name__}'
            )

        self.service_class_map: dict[ProviderType, type[GitService]] = {
            ProviderType.GITHUB: GithubServiceImpl,
            ProviderType.GITLAB: GitLabServiceImpl,
            ProviderType.BITBUCKET: BitBucketServiceImpl,
            ProviderType.AZURE_DEVOPS: AzureDevOpsServiceImpl,
        }

        self.external_auth_id = external_auth_id
        self.external_auth_token = external_auth_token
        self.external_token_manager = external_token_manager
        self.session_api_key = session_api_key
        self.sid = sid
        self._provider_tokens = provider_tokens
        WEB_HOST = os.getenv('WEB_HOST', '').strip()
        self.REFRESH_TOKEN_URL = (
            f'https://{WEB_HOST}/api/refresh-tokens' if WEB_HOST else None
        )

    @property
    def provider_tokens(self) -> PROVIDER_TOKEN_TYPE:
        """Read-only access to provider tokens."""
        return self._provider_tokens

    def get_service(self, provider: ProviderType) -> GitService:
        """Helper method to instantiate a service for a given provider"""
        token = self.provider_tokens[provider]
        service_class = self.service_class_map[provider]
        return service_class(
            user_id=token.user_id,
            external_auth_id=self.external_auth_id,
            external_auth_token=self.external_auth_token,
            token=token.token,
            external_token_manager=self.external_token_manager,
            base_domain=token.host,
        )

    async def get_user(self) -> User:
        """Get user information from the first available provider"""
        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                return await service.get_user()
            except Exception:
                continue
        raise AuthenticationError('Need valid provider token')

    async def _get_latest_provider_token(
        self, provider: ProviderType
    ) -> SecretStr | None:
        """Get latest token from service"""
        try:
            async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
                resp = await client.get(
                    self.REFRESH_TOKEN_URL,
                    headers={
                        'X-Session-API-Key': self.session_api_key,
                    },
                    params={'provider': provider.value, 'sid': self.sid},
                )

            resp.raise_for_status()
            data = TokenResponse.model_validate_json(resp.text)
            return SecretStr(data.token)

        except Exception as e:
            logger.error(
                f'Failed to fetch latest token for provider {provider}: {e}',
                exc_info=True,
            )

        return None

    async def get_github_installations(self) -> list[str]:
        service = cast(InstallationsService, self.get_service(ProviderType.GITHUB))
        try:
            return await service.get_installations()
        except Exception as e:
            logger.warning(f'Failed to get github installations {e}')

        return []

    async def get_bitbucket_workspaces(self) -> list[str]:
        service = cast(InstallationsService, self.get_service(ProviderType.BITBUCKET))
        try:
            return await service.get_installations()
        except Exception as e:
            logger.warning(f'Failed to get bitbucket workspaces {e}')

        return []

    async def get_azure_devops_organizations(self) -> list[str]:
        service = cast(
            InstallationsService, self.get_service(ProviderType.AZURE_DEVOPS)
        )
        try:
            return await service.get_installations()
        except Exception as e:
            logger.warning(f'Failed to get azure devops organizations {e}')

        return []

    async def get_repositories(
        self,
        sort: str,
        app_mode: AppMode,
        selected_provider: ProviderType | None,
        page: int | None,
        per_page: int | None,
        installation_id: str | None,
    ) -> list[Repository]:
        """Get repositories from providers"""
        """
        Get repositories from providers
        """

        if selected_provider:
            if not page or not per_page:
                raise ValueError('Failed to provider params for paginating repos')

            service = self.get_service(selected_provider)
            return await service.get_paginated_repos(
                page, per_page, sort, installation_id
            )

        all_repos: list[Repository] = []
        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                service_repos = await service.get_all_repositories(sort, app_mode)
                all_repos.extend(service_repos)
            except Exception as e:
                logger.warning(f'Error fetching repos from {provider}: {e}')

        return all_repos

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        """Get suggested tasks from providers"""
        tasks: list[SuggestedTask] = []
        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                service_repos = await service.get_suggested_tasks()
                tasks.extend(service_repos)
            except Exception as e:
                logger.warning(f'Error fetching repos from {provider}: {e}')

        return tasks

    async def search_branches(
        self,
        selected_provider: ProviderType | None,
        repository: str,
        query: str,
        per_page: int = 30,
    ) -> list[Branch]:
        """Search for branches within a repository using the appropriate provider service."""
        if selected_provider:
            service = self.get_service(selected_provider)
            try:
                return await service.search_branches(repository, query, per_page)
            except Exception as e:
                logger.warning(
                    f'Error searching branches from selected provider {selected_provider}: {e}'
                )
                return []

        # If provider not specified, determine provider by verifying repository access
        try:
            repo_details = await self.verify_repo_provider(repository)
            service = self.get_service(repo_details.git_provider)
            return await service.search_branches(repository, query, per_page)
        except Exception as e:
            logger.warning(f'Error searching branches for {repository}: {e}')
            return []

    async def search_repositories(
        self,
        selected_provider: ProviderType | None,
        query: str,
        per_page: int,
        sort: str,
        order: str,
        app_mode: AppMode,
    ) -> list[Repository]:
        if selected_provider:
            service = self.get_service(selected_provider)
            public = self._is_repository_url(query, selected_provider)
            user_repos = await service.search_repositories(
                query, per_page, sort, order, public, app_mode
            )
            return self._deduplicate_repositories(user_repos)

        all_repos: list[Repository] = []
        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                public = self._is_repository_url(query, provider)
                service_repos = await service.search_repositories(
                    query, per_page, sort, order, public, app_mode
                )
                all_repos.extend(service_repos)
            except Exception as e:
                logger.warning(f'Error searching repos from {provider}: {e}')
                continue

        return all_repos

    def _is_repository_url(self, query: str, provider: ProviderType) -> bool:
        """Check if the query is a repository URL."""
        custom_host = self.provider_tokens[provider].host
        custom_host_exists = custom_host and custom_host in query
        default_host_exists = self.PROVIDER_DOMAINS[provider] in query

        return query.startswith(('http://', 'https://')) and (
            custom_host_exists or default_host_exists
        )

    def _deduplicate_repositories(self, repos: list[Repository]) -> list[Repository]:
        """Remove duplicate repositories based on full_name."""
        seen = set()
        unique_repos = []
        for repo in repos:
            if repo.full_name not in seen:
                seen.add(repo.id)
                unique_repos.append(repo)
        return unique_repos

    async def set_event_stream_secrets(
        self,
        event_stream: EventStream,
        env_vars: dict[ProviderType, SecretStr] | None = None,
    ) -> None:
        """This ensures that the latest provider tokens are masked from the event stream
        It is called when the provider tokens are first initialized in the runtime or when tokens are re-exported with the latest working ones

        Args:
            event_stream: Agent session's event stream
            env_vars: Dict of providers and their tokens that require updating
        """
        if env_vars:
            exposed_env_vars = self.expose_env_vars(env_vars)
        else:
            exposed_env_vars = await self.get_env_vars(expose_secrets=True)
        event_stream.set_secrets(exposed_env_vars)

    def expose_env_vars(
        self, env_secrets: dict[ProviderType, SecretStr]
    ) -> dict[str, str]:
        """Return string values instead of typed values for environment secrets
        Called just before exporting secrets to runtime, or setting secrets in the event stream
        """
        exposed_envs = {}
        for provider, token in env_secrets.items():
            env_key = ProviderHandler.get_provider_env_key(provider)
            exposed_envs[env_key] = token.get_secret_value()

        return exposed_envs

    @overload
    def get_env_vars(
        self,
        expose_secrets: Literal[True],
        providers: list[ProviderType] | None = ...,
        get_latest: bool = False,
    ) -> Coroutine[Any, Any, dict[str, str]]: ...

    @overload
    def get_env_vars(
        self,
        expose_secrets: Literal[False],
        providers: list[ProviderType] | None = ...,
        get_latest: bool = False,
    ) -> Coroutine[Any, Any, dict[ProviderType, SecretStr]]: ...

    async def get_env_vars(
        self,
        expose_secrets: bool = False,
        providers: list[ProviderType] | None = None,
        get_latest: bool = False,
    ) -> dict[ProviderType, SecretStr] | dict[str, str]:
        """Retrieves the provider tokens from ProviderHandler object
        This is used when initializing/exporting new provider tokens in the runtime

        Args:
            expose_secrets: Flag which returns strings instead of secrets
            providers: Return provider tokens for the list passed in, otherwise return all available providers
            get_latest: Get the latest working token for the providers if True, otherwise get the existing ones
        """
        if not self.provider_tokens:
            return {}

        env_vars: dict[ProviderType, SecretStr] = {}
        all_providers = [provider for provider in ProviderType]
        provider_list = providers if providers else all_providers

        for provider in provider_list:
            if provider in self.provider_tokens:
                token = (
                    self.provider_tokens[provider].token
                    if self.provider_tokens
                    else SecretStr('')
                )

                if get_latest and self.REFRESH_TOKEN_URL and self.sid:
                    token = await self._get_latest_provider_token(provider)

                if token:
                    env_vars[provider] = token

        if not expose_secrets:
            return env_vars

        return self.expose_env_vars(env_vars)

    @classmethod
    def check_cmd_action_for_provider_token_ref(
        cls, event: Action
    ) -> list[ProviderType]:
        """Detect if agent run action is using a provider token (e.g $GITHUB_TOKEN)
        Returns a list of providers which are called by the agent
        """
        if not isinstance(event, CmdRunAction):
            return []

        called_providers = []
        for provider in ProviderType:
            if ProviderHandler.get_provider_env_key(provider) in event.command.lower():
                called_providers.append(provider)

        return called_providers

    @classmethod
    def get_provider_env_key(cls, provider: ProviderType) -> str:
        """Map ProviderType value to the environment variable name in the runtime"""
        return f'{provider.value}_token'.lower()

    async def verify_repo_provider(
        self,
        repository: str,
        specified_provider: ProviderType | None = None,
        is_optional: bool = False,
    ) -> Repository:
        errors = []

        if specified_provider:
            try:
                service = self.get_service(specified_provider)
                return await service.get_repository_details_from_repo_name(repository)
            except Exception as e:
                errors.append(f'{specified_provider.value}: {str(e)}')

        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                return await service.get_repository_details_from_repo_name(repository)
            except Exception as e:
                errors.append(f'{provider.value}: {str(e)}')

        # Log detailed error based on whether we had tokens or not
        # For optional repositories (like org-level microagents), use debug level
        log_fn = logger.debug if is_optional else logger.error

        if not self.provider_tokens:
            log_fn(
                f'Failed to access repository {repository}: No provider tokens available. '
                f'provider_tokens dict is empty.'
            )
        elif errors:
            log_fn(
                f'Failed to access repository {repository} with all available providers. '
                f'Tried providers: {list(self.provider_tokens.keys())}. '
                f'Errors: {"; ".join(errors)}'
            )
        else:
            log_fn(
                f'Failed to access repository {repository}: Unknown error (no providers tried, no errors recorded)'
            )
        raise AuthenticationError(f'Unable to access repo {repository}')

    async def get_branches(
        self,
        repository: str,
        specified_provider: ProviderType | None = None,
        page: int = 1,
        per_page: int = 30,
    ) -> PaginatedBranchesResponse:
        """Get branches for a repository

        Args:
            repository: The repository name
            specified_provider: Optional provider type to use
            page: Page number for pagination (default: 1)
            per_page: Number of branches per page (default: 30)

        Returns:
            A paginated response with branches for the repository
        """
        if specified_provider:
            try:
                service = self.get_service(specified_provider)
                return await service.get_paginated_branches(repository, page, per_page)
            except Exception as e:
                logger.warning(
                    f'Error fetching branches from {specified_provider}: {e}'
                )

        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                return await service.get_paginated_branches(repository, page, per_page)
            except Exception as e:
                logger.warning(f'Error fetching branches from {provider}: {e}')

        # Return empty response if no provider worked
        return PaginatedBranchesResponse(
            branches=[],
            has_next_page=False,
            current_page=page,
            per_page=per_page,
            total_count=0,
        )

    async def get_microagents(self, repository: str) -> list[MicroagentResponse]:
        """Get microagents from a repository using the appropriate service.

        Args:
            repository: Repository name in the format 'owner/repo'

        Returns:
            List of microagents found in the repository

        Raises:
            AuthenticationError: If authentication fails
        """
        # Try all available providers in order
        errors = []
        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                result = await service.get_microagents(repository)
                # Only return early if we got a non-empty result
                if result:
                    return result
                # If we got an empty array, continue checking other providers
                logger.debug(
                    f'No microagents found on {provider} for {repository}, trying other providers'
                )
            except Exception as e:
                errors.append(f'{provider.value}: {str(e)}')
                logger.warning(
                    f'Error fetching microagents from {provider} for {repository}: {e}'
                )

        # If all providers failed or returned empty results, return empty array
        if errors:
            logger.error(
                f'Failed to fetch microagents for {repository} with all available providers. Errors: {"; ".join(errors)}'
            )
            raise AuthenticationError(f'Unable to fetch microagents for {repository}')

        # All providers returned empty arrays
        return []

    async def get_microagent_content(
        self, repository: str, file_path: str
    ) -> MicroagentContentResponse:
        """Get content of a specific microagent file from a repository.

        Args:
            repository: Repository name in the format 'owner/repo'
            file_path: Path to the microagent file within the repository

        Returns:
            MicroagentContentResponse with parsed content and triggers

        Raises:
            AuthenticationError: If authentication fails
        """
        # Try all available providers in order
        errors = []
        for provider in self.provider_tokens:
            try:
                service = self.get_service(provider)
                result = await service.get_microagent_content(repository, file_path)
                # If we got content, return it immediately
                if result:
                    return result
                # If we got empty content, continue checking other providers
                logger.debug(
                    f'No content found on {provider} for {repository}/{file_path}, trying other providers'
                )
            except ResourceNotFoundError:
                logger.debug(
                    f'File not found on {provider} for {repository}/{file_path}, trying other providers'
                )
                continue
            except MicroagentParseError as e:
                # Parsing errors are specific to the provider, add to errors list
                errors.append(f'{provider.value}: {str(e)}')
                logger.warning(
                    f'Error parsing microagent content from {provider} for {repository}: {e}'
                )
            except Exception as e:
                # For other errors (auth, rate limit, etc.), add to errors list
                errors.append(f'{provider.value}: {str(e)}')
                logger.warning(
                    f'Error fetching microagent content from {provider} for {repository}: {e}'
                )

        # If all providers failed or returned empty results, raise an error
        if errors:
            logger.error(
                f'Failed to fetch microagent content for {repository} with all available providers. Errors: {"; ".join(errors)}'
            )

        # All providers returned empty content or file not found
        raise AuthenticationError(
            f'Microagent file {file_path} not found in {repository}'
        )

    async def get_authenticated_git_url(
        self, repo_name: str, is_optional: bool = False
    ) -> str:
        """Get an authenticated git URL for a repository.

        Args:
            repo_name: Repository name (owner/repo)
            is_optional: If True, logs at debug level instead of error level when repo not found

        Returns:
            Authenticated git URL if credentials are available, otherwise regular HTTPS URL
        """
        try:
            repository = await self.verify_repo_provider(
                repo_name, is_optional=is_optional
            )
        except AuthenticationError:
            raise Exception('Git provider authentication issue when getting remote URL')

        provider = repository.git_provider
        repo_name = repository.full_name

        domain = self.PROVIDER_DOMAINS[provider]

        # If provider tokens are provided, use the host from the token if available
        # Note: For Azure DevOps, don't use the host field as it may contain org/project path
        if self.provider_tokens and provider in self.provider_tokens:
            if provider != ProviderType.AZURE_DEVOPS:
                domain = self.provider_tokens[provider].host or domain

        # Try to use token if available, otherwise use public URL
        if self.provider_tokens and provider in self.provider_tokens:
            git_token = self.provider_tokens[provider].token
            if git_token:
                token_value = git_token.get_secret_value()
                if provider == ProviderType.GITLAB:
                    remote_url = (
                        f'https://oauth2:{token_value}@{domain}/{repo_name}.git'
                    )
                elif provider == ProviderType.BITBUCKET:
                    # For Bitbucket, handle username:app_password format
                    if ':' in token_value:
                        # App token format: username:app_password
                        remote_url = f'https://{token_value}@{domain}/{repo_name}.git'
                    else:
                        # Access token format: use x-token-auth
                        remote_url = f'https://x-token-auth:{token_value}@{domain}/{repo_name}.git'
                elif provider == ProviderType.AZURE_DEVOPS:
                    # Azure DevOps uses PAT with Basic auth
                    # Format: https://{anything}:{PAT}@dev.azure.com/{org}/{project}/_git/{repo}
                    # The username can be anything (it's ignored), but cannot be empty
                    # We use the org name as the username for clarity
                    # repo_name is in format: org/project/repo
                    logger.info(
                        f'[Azure DevOps] Constructing authenticated git URL for repository: {repo_name}'
                    )
                    logger.debug(f'[Azure DevOps] Original domain: {domain}')
                    logger.debug(
                        f'[Azure DevOps] Token available: {bool(token_value)}, '
                        f'Token length: {len(token_value) if token_value else 0}'
                    )

                    # Remove domain prefix if it exists in domain variable
                    clean_domain = domain.replace('https://', '').replace('http://', '')
                    logger.debug(f'[Azure DevOps] Cleaned domain: {clean_domain}')

                    parts = repo_name.split('/')
                    logger.debug(
                        f'[Azure DevOps] Repository parts: {parts} (length: {len(parts)})'
                    )

                    if len(parts) >= 3:
                        org, project, repo = parts[0], parts[1], parts[2]
                        logger.info(
                            f'[Azure DevOps] Parsed repository - org: {org}, project: {project}, repo: {repo}'
                        )
                        # URL-encode org, project, and repo to handle spaces and special characters
                        org_encoded = quote(org, safe='')
                        project_encoded = quote(project, safe='')
                        repo_encoded = quote(repo, safe='')
                        logger.debug(
                            f'[Azure DevOps] URL-encoded parts - org: {org_encoded}, project: {project_encoded}, repo: {repo_encoded}'
                        )
                        # Use org name as username (it's ignored by Azure DevOps but required for git)
                        remote_url = f'https://{org}:***@{clean_domain}/{org_encoded}/{project_encoded}/_git/{repo_encoded}'
                        logger.info(
                            f'[Azure DevOps] Constructed git URL (token masked): {remote_url}'
                        )
                        # Set the actual URL with token
                        remote_url = f'https://{org}:{token_value}@{clean_domain}/{org_encoded}/{project_encoded}/_git/{repo_encoded}'
                    else:
                        # Fallback if format is unexpected
                        logger.warning(
                            f'[Azure DevOps] Unexpected repository format: {repo_name}. '
                            f'Expected org/project/repo (3 parts), got {len(parts)} parts. '
                            'Using fallback URL format.'
                        )
                        remote_url = (
                            f'https://user:{token_value}@{clean_domain}/{repo_name}.git'
                        )
                        logger.warning(
                            f'[Azure DevOps] Fallback URL constructed (token masked): '
                            f'https://user:***@{clean_domain}/{repo_name}.git'
                        )
                else:
                    # GitHub
                    remote_url = f'https://{token_value}@{domain}/{repo_name}.git'
            else:
                remote_url = f'https://{domain}/{repo_name}.git'
        else:
            remote_url = f'https://{domain}/{repo_name}.git'

        return remote_url

    async def is_pr_open(
        self, repository: str, pr_number: int, git_provider: ProviderType
    ) -> bool:
        """Check if a PR is still active (not closed/merged).

        This method checks the PR status using the provider's service method.

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The PR number to check
            git_provider: The Git provider type for this repository

        Returns:
            True if PR is active (open), False if closed/merged, True if can't determine
        """
        try:
            service = self.get_service(git_provider)
            return await service.is_pr_open(repository, pr_number)

        except Exception as e:
            logger.warning(
                f'Could not determine PR status for {repository}#{pr_number}: {e}. '
                f'Including conversation to be safe.'
            )
            # If we can't determine the PR status, include the conversation to be safe
            return True
