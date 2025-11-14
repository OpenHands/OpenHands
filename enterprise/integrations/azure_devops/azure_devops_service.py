import asyncio

import httpx
from integrations.utils import store_repositories_in_db
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from storage.azure_devops_webhook import AzureDevOpsWebhook
from storage.azure_devops_webhook_store import AzureDevOpsWebhookStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl,
)
from openhands.integrations.service_types import (
    ProviderType,
    Repository,
    SuggestedTask,
    User,
)
from openhands.server.types import AppMode


class SaaSAzureDevOpsService(AzureDevOpsServiceImpl):
    def __init__(
        self,
        user_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        external_auth_id: str | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
    ):
        logger.info(
            f"SaaSAzureDevOpsService created with user_id {user_id}, "
            f"external_auth_id {external_auth_id}, "
            f"external_auth_token {'set' if external_auth_token else 'None'}, "
            f"azure_devops_token {'set' if token else 'None'}, "
            f"external_token_manager {external_token_manager}"
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
        azure_devops_token = None

        # If token is already set (e.g., service principal token), return it
        if self.token and self.token.get_secret_value():
            logger.debug(
                'Using pre-configured token (service principal or direct token)'
            )
            return self.token

        if self.external_auth_token:
            azure_devops_token = SecretStr(
                await self.token_manager.get_idp_token(
                    self.external_auth_token.get_secret_value(),
                    idp=ProviderType.AZURE_DEVOPS,
                )
            )
            logger.debug(
                f'Got Azure DevOps token from access token: {self.external_auth_token}'
            )
        elif self.external_auth_id:
            offline_token = await self.token_manager.load_offline_token(
                self.external_auth_id
            )
            azure_devops_token = SecretStr(
                await self.token_manager.get_idp_token_from_offline_token(
                    offline_token, ProviderType.AZURE_DEVOPS
                )
            )
            logger.info(
                f'Got Azure DevOps token from external auth user ID: {self.external_auth_id}'
            )
        elif self.user_id:
            azure_devops_token = SecretStr(
                await self.token_manager.get_idp_token_from_idp_user_id(
                    self.user_id, ProviderType.AZURE_DEVOPS
                )
            )
            logger.debug(f'Got Azure DevOps token from user ID: {self.user_id}')
        else:
            logger.warning('external_auth_token and user_id not set!')

        return azure_devops_token

    async def get_user(self) -> User:
        # Ensure organization is discovered before constructing any URLs
        await self._ensure_organization_set()

        # Call parent implementation
        return await super().get_user()

    async def _discover_user_organizations(self) -> list[dict]:
        try:
            # Get Azure DevOps token
            if not self.token:
                latest_token = await self.get_latest_token()
                if latest_token:
                    self.token = latest_token

            if not self.token:
                raise ValueError(
                    'No Azure DevOps token available for organization discovery'
                )

            # Use the base class method to get proper auth headers
            # This automatically handles both OAuth (Bearer) and PAT (Basic) authentication
            headers = await self._get_azure_devops_headers()

            async with httpx.AsyncClient() as client:
                # Get user profile to get memberId
                profile_url = 'https://app.vssps.visualstudio.com/_apis/profile/profiles/me?api-version=7.1-preview.3'
                profile_response = await client.get(profile_url, headers=headers)
                profile_response.raise_for_status()
                profile_data = profile_response.json()
                member_id = profile_data.get('id')

                if not member_id:
                    raise ValueError('Failed to get user memberId from profile API')

                logger.info(f'Discovered Azure DevOps user memberId: {member_id}')

                # Get organizations using Accounts API
                accounts_url = f'https://app.vssps.visualstudio.com/_apis/accounts?memberId={member_id}&api-version=7.1'
                accounts_response = await client.get(accounts_url, headers=headers)
                accounts_response.raise_for_status()
                accounts_data = accounts_response.json()

                organizations = accounts_data.get('value', [])
                logger.info(
                    f'Discovered {len(organizations)} Azure DevOps organizations for user'
                )

                return organizations

        except Exception as e:
            logger.warning(
                f'Failed to discover Azure DevOps organizations: {e}', exc_info=True
            )
            raise

    async def _ensure_organization_set(self) -> None:
        """Ensure organization is set for legacy single-org methods.

        NOTE: This is for backward compatibility with methods that expect a single organization.
        New code should use get_all_repositories() which queries all organizations.
        """
        if self.organization:
            # Organization already set, nothing to do
            return

        logger.info(
            '[AzureDevOps] Organization not set for legacy method, using first discovered org'
        )

        try:
            organizations = await self._discover_user_organizations()

            if not organizations:
                raise ValueError(
                    'No Azure DevOps organizations found for user. '
                    'User must be a member of at least one organization.'
                )

            # Use the first organization for legacy single-org methods
            first_org = organizations[0]
            org_name = first_org.get('accountName')

            if not org_name:
                raise ValueError('Organization name not found in discovery response')

            self.organization = org_name
            logger.info(
                f'[AzureDevOps] Set organization to first discovered org: {org_name} '
                f'(found {len(organizations)} total organizations)'
            )

            # Log other available organizations for visibility
            if len(organizations) > 1:
                other_orgs = [org.get('accountName') for org in organizations[1:]]
                logger.info(
                    f'[AzureDevOps] Other available organizations: {other_orgs}'
                )

        except Exception as e:
            logger.error(f'[AzureDevOps] Failed to auto-discover organization: {e}')
            raise ValueError(
                f'Unable to determine Azure DevOps organization. Error: {e}'
            )

    async def add_owned_projects_and_repos_to_db(
        self, projects_and_repos: list[dict]
    ) -> None:
        webhooks = []

        # Get user's Azure DevOps ID for webhook mapping
        # Note: User may have different IDs in different organizations
        # We'll use the user's email to create a consistent mapping
        azure_devops_user_id = None
        try:
            # Get user info from Azure DevOps to obtain their Azure DevOps ID
            user = await self.get_user()
            azure_devops_user_id = user.id  # This is the Azure DevOps originId
            logger.info(
                f'Got Azure DevOps user ID: {azure_devops_user_id} for Keycloak user: {self.external_auth_id}'
            )
        except Exception as e:
            logger.error(f'Failed to get Azure DevOps user ID: {e}', exc_info=True)
            # Fallback: use Keycloak user ID (not ideal but allows system to continue)
            azure_devops_user_id = self.external_auth_id
            logger.warning(
                f'Using Keycloak user ID as fallback: {azure_devops_user_id}'
            )

        for item in projects_and_repos:
            webhook = AzureDevOpsWebhook(
                organization=item[
                    'organization'
                ],  # Now from item, not self.organization
                project_id=item['project_id'],
                repository_id=item.get('repo_id'),  # None for project-level
                user_id=azure_devops_user_id,  # Azure DevOps ID
                webhook_exists=False,
            )
            webhooks.append(webhook)

        # Store webhooks in the database
        if webhooks:
            try:
                webhook_store = AzureDevOpsWebhookStore()
                await webhook_store.store_webhooks(webhooks)
                logger.info(
                    f'Added {len(webhooks)} Azure DevOps webhooks to db for Azure DevOps user {azure_devops_user_id} (Keycloak: {self.external_auth_id})'
                )
            except Exception:
                logger.warning(
                    'Failed to add Azure DevOps webhooks to db', exc_info=True
                )

    async def store_repository_data(
        self, projects_and_repos: list[dict], repositories: list[Repository]
    ) -> None:
        try:
            # Add projects and repos to webhook tracking
            await self.add_owned_projects_and_repos_to_db(projects_and_repos)

            # Store repositories in the database
            await store_repositories_in_db(repositories, self.external_auth_id)

            logger.info(
                f'Successfully stored repository data for user {self.external_auth_id}'
            )
        except Exception:
            logger.warning('Error storing repository data', exc_info=True)

    async def get_repositories(self, sort: str, app_mode: AppMode) -> list[Repository]:
        # Ensure organization is discovered before constructing any URLs
        await self._ensure_organization_set()

        # Call parent implementation
        return await super().get_repositories(sort, app_mode)

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode, store_in_background: bool = True
    ) -> list[Repository]:
        """Get repositories from ALL Azure DevOps organizations the user has access to."""
        MAX_REPOS = 1000

        # Discover all organizations the user has access to
        logger.info(
            '[AzureDevOps] Discovering user organizations for multi-org repository query'
        )
        try:
            organizations = await self._discover_user_organizations()
        except Exception as e:
            logger.error(
                f'[AzureDevOps] Failed to discover organizations: {e}', exc_info=True
            )
            return []

        if not organizations:
            logger.warning('[AzureDevOps] No organizations found for user')
            return []

        org_names = [
            org['accountName'] for org in organizations if 'accountName' in org
        ]
        logger.info(
            f'[AzureDevOps] Querying repositories from {len(org_names)} organizations: {org_names}'
        )

        all_repos = []
        projects_and_repos = []

        # Query each organization
        for org_data in organizations:
            org_name = org_data.get('accountName')
            if not org_name:
                continue

            try:
                # Build base URL for this organization
                org_base_url = (
                    f'https://dev.azure.com/{self._encode_url_component(org_name)}'
                )

                # Get all projects in this organization
                projects_url = f'{org_base_url}/_apis/projects?api-version=7.1'
                projects_response, _ = await self._make_request(projects_url)
                projects = projects_response.get('value', [])

                logger.info(
                    f'[AzureDevOps] Found {len(projects)} projects in organization {org_name}'
                )

                # For each project, get its repositories
                for project in projects:
                    project_name = project.get('name')
                    project_id = project.get('id')

                    # Track project for webhooks
                    projects_and_repos.append(
                        {
                            'organization': org_name,
                            'project_id': project_id,
                            'project_name': project_name,
                            'repo_id': None,  # Project-level webhook
                            'repo_name': None,
                        }
                    )

                    # URL-encode project name to handle spaces and special characters
                    project_enc = self._encode_url_component(project_name)

                    repos_url = f'{org_base_url}/{project_enc}/_apis/git/repositories?api-version=7.1'
                    repos_response, _ = await self._make_request(repos_url)
                    repos = repos_response.get('value', [])

                    for repo in repos:
                        repo_id = repo.get('id')
                        repo_name = repo.get('name')

                        all_repos.append(
                            {
                                'id': repo_id,
                                'name': repo_name,
                                'organization': org_name,
                                'project_id': project_id,
                                'project_name': project_name,
                                'updated_date': repo.get('lastUpdateTime'),
                            }
                        )

                        # Track repository for webhooks
                        projects_and_repos.append(
                            {
                                'organization': org_name,
                                'project_id': project_id,
                                'project_name': project_name,
                                'repo_id': repo_id,
                                'repo_name': repo_name,
                            }
                        )

                        if len(all_repos) >= MAX_REPOS:
                            break

                    if len(all_repos) >= MAX_REPOS:
                        break

            except Exception as e:
                logger.error(
                    f'[AzureDevOps] Error querying organization {org_name}: {e}',
                    exc_info=True,
                )
                continue

            if len(all_repos) >= MAX_REPOS:
                break

        # Sort repositories based on the sort parameter
        if sort == 'updated' or sort == 'pushed':
            all_repos.sort(key=lambda r: r.get('updated_date', ''), reverse=True)
        elif sort == 'name':
            all_repos.sort(key=lambda r: r.get('name', '').lower())

        # Convert to Repository objects
        repositories = [
            Repository(
                id=str(repo.get('id')),
                full_name=f"{repo.get('organization')}/{repo.get('project_name')}/{repo.get('name')}",
                git_provider=ProviderType.AZURE_DEVOPS,
                is_public=False,  # Azure DevOps repos are private by default
            )
            for repo in all_repos[:MAX_REPOS]
        ]

        logger.info(
            f'[AzureDevOps] Found {len(repositories)} total repositories across all organizations'
        )

        # Store webhook and repository info
        if store_in_background:
            asyncio.create_task(
                self.store_repository_data(projects_and_repos, repositories)
            )
        else:
            await self.store_repository_data(projects_and_repos, repositories)

        return repositories

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        # Ensure organization is discovered before constructing any URLs
        await self._ensure_organization_set()

        # Call parent implementation
        return await super().get_suggested_tasks()

    async def search_repositories(
        self,
        query: str,
        per_page: int = 30,
        sort: str = 'updated',
        order: str = 'desc',
        public: bool = False,
        app_mode: AppMode = AppMode.OSS,
    ) -> list[Repository]:
        # Ensure organization is discovered before constructing any URLs
        await self._ensure_organization_set()

        # Call parent implementation
        return await super().search_repositories(
            query, per_page, sort, order, public, app_mode
        )

    async def get_installations(self) -> list[str]:
        try:
            organizations = await self._discover_user_organizations()
            org_names = [
                org['accountName']
                for org in organizations
                if 'accountName' in org and org['accountName']
            ]

            logger.info(
                f'Retrieved {len(org_names)} Azure DevOps organizations for user'
            )

            return org_names
        except Exception as e:
            logger.warning(
                f'Failed to get Azure DevOps organizations: {e}', exc_info=True
            )
            return []
