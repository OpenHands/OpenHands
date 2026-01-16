import asyncio

from integrations.store_repo_utils import store_repositories_in_db
from pydantic import SecretStr
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.github_service import GitHubService
from openhands.integrations.service_types import ProviderType, Repository
from openhands.server.types import AppMode


class SaaSGitHubService(GitHubService):
    def __init__(
        self,
        user_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        external_auth_id: str | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
    ):
        logger.debug(
            f'SaaSGitHubService created with user_id {user_id}, external_auth_id {external_auth_id}, external_auth_token {'set' if external_auth_token else 'None'}, github_token {'set' if token else 'None'}, external_token_manager {external_token_manager}'
        )
        super().__init__(
            user_id=user_id,
            external_auth_token=external_auth_token,
            external_auth_id=external_auth_id,
            token=token,
            external_token_manager=external_token_manager,
            base_domain=base_domain,
        )

        self.external_auth_token = external_auth_token
        self.external_auth_id = external_auth_id
        self.token_manager = TokenManager(external=external_token_manager)

    async def get_latest_token(self) -> SecretStr | None:
        github_token = None
        if self.external_auth_token:
            github_token = SecretStr(
                await self.token_manager.get_idp_token(
                    self.external_auth_token.get_secret_value(), ProviderType.GITHUB
                )
            )
            logger.debug(
                f'Got GitHub token {github_token} from access token: {self.external_auth_token}'
            )
        elif self.external_auth_id:
            offline_token = await self.token_manager.load_offline_token(
                self.external_auth_id
            )
            github_token = SecretStr(
                await self.token_manager.get_idp_token_from_offline_token(
                    offline_token, ProviderType.GITHUB
                )
            )
            logger.debug(
                f'Got GitHub token {github_token} from external auth user ID: {self.external_auth_id}'
            )
        elif self.user_id:
            github_token = SecretStr(
                await self.token_manager.get_idp_token_from_idp_user_id(
                    self.user_id, ProviderType.GITHUB
                )
            )
            logger.debug(
                f'Got GitHub token {github_token} from user ID: {self.user_id}'
            )
        else:
            logger.warning('external_auth_token and user_id not set!')
        return github_token

    async def get_pr_patches(
        self, owner: str, repo: str, pr_number: int, per_page: int = 30, page: int = 1
    ):
        """Get patches for files changed in a PR with pagination support.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            per_page: Number of files per page (default: 30, max: 100)
            page: Page number to fetch (default: 1)
        """
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
        params = {'per_page': min(per_page, 100), 'page': page}  # GitHub max is 100
        response, headers = await self._make_request(url, params)

        # Parse pagination info from headers
        has_next_page = 'next' in headers.get('link', '')
        total_count = int(headers.get('total', 0))

        return {
            'files': response,
            'pagination': {
                'has_next_page': has_next_page,
                'total_count': total_count,
                'current_page': page,
                'per_page': per_page,
            },
        }

    async def get_repository_node_id(self, repo_id: str) -> str:
        """
        Get the new GitHub GraphQL node ID for a repository using REST API.

        Args:
            repo_id: Numeric repository ID as string (e.g., "123456789")

        Returns:
            New format node ID for GraphQL queries (e.g., "R_kgDOLfkiww")

        Raises:
            Exception: If the API request fails or node_id is not found
        """
        url = f'https://api.github.com/repositories/{repo_id}'
        response, _ = await self._make_request(url)
        node_id = response.get('node_id')
        if not node_id:
            raise Exception(f'No node_id found for repository {repo_id}')
        return node_id

    async def get_paginated_repos(self, page, per_page, sort, installation_id):
        repositories = await super().get_paginated_repos(
            page, per_page, sort, installation_id
        )
        asyncio.create_task(
            store_repositories_in_db(repositories, self.external_auth_id)
        )
        return repositories

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode
    ) -> list[Repository]:
        repositories = await super().get_all_repositories(sort, app_mode)
        # Schedule the background task without awaiting it
        asyncio.create_task(
            store_repositories_in_db(repositories, self.external_auth_id)
        )
        # Return repositories immediately
        return repositories
