import asyncio

from integrations.store_repo_utils import store_repositories_in_db
from integrations.types import GitLabResourceType
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from storage.gitlab_webhook import GitlabWebhook, WebhookStatus
from storage.gitlab_webhook_store import GitlabWebhookStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.gitlab.gitlab_service import GitLabService
from openhands.integrations.service_types import (
    ProviderType,
    RateLimitError,
    Repository,
    RequestMethod,
)
from openhands.server.types import AppMode


class SaaSGitLabService(GitLabService):
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
            f'SaaSGitLabService created with user_id {user_id}, external_auth_id {external_auth_id}, external_auth_token {'set' if external_auth_token else 'None'}, gitlab_token {'set' if token else 'None'}, external_token_manager {external_token_manager}'
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
        gitlab_token = None
        if self.external_auth_token:
            gitlab_token = SecretStr(
                await self.token_manager.get_idp_token(
                    self.external_auth_token.get_secret_value(), idp=ProviderType.GITLAB
                )
            )
            logger.debug(
                f'Got GitLab token {gitlab_token} from access token: {self.external_auth_token}'
            )
        elif self.external_auth_id:
            offline_token = await self.token_manager.load_offline_token(
                self.external_auth_id
            )
            gitlab_token = SecretStr(
                await self.token_manager.get_idp_token_from_offline_token(
                    offline_token, ProviderType.GITLAB
                )
            )
            logger.info(
                f'Got GitLab token {gitlab_token.get_secret_value()} from external auth user ID: {self.external_auth_id}'
            )
        elif self.user_id:
            gitlab_token = SecretStr(
                await self.token_manager.get_idp_token_from_idp_user_id(
                    self.user_id, ProviderType.GITLAB
                )
            )
            logger.debug(
                f'Got Gitlab token {gitlab_token} from user ID: {self.user_id}'
            )
        else:
            logger.warning('external_auth_token and user_id not set!')
        return gitlab_token

    async def get_owned_groups(self, min_access_level: int = 40) -> list[dict]:
        """
        Get all top-level groups where the current user has admin access.

        This method supports pagination and fetches all groups where the user has
        at least the specified access level.

        Args:
            min_access_level: Minimum access level required (default: 40 for Maintainer or Owner)
                             - 40: Maintainer or Owner
                             - 50: Owner only

        Returns:
            list[dict]: A list of groups where user has the specified access level or higher.
        """
        groups_with_admin_access = []
        page = 1
        per_page = 100

        while True:
            try:
                url = f'{self.BASE_URL}/groups'
                params = {
                    'page': str(page),
                    'per_page': str(per_page),
                    'min_access_level': min_access_level,
                    'top_level_only': 'true',
                }
                response, headers = await self._make_request(url, params)

                if not response:
                    break

                groups_with_admin_access.extend(response)
                page += 1

                # Check if we've reached the last page
                link_header = headers.get('Link', '')
                if 'rel="next"' not in link_header:
                    break

            except Exception:
                logger.warning(f'Error fetching groups on page {page}', exc_info=True)
                break

        return groups_with_admin_access

    async def add_owned_projects_and_groups_to_db(self, owned_personal_projects):
        """
        Add owned projects and groups to the database for webhook tracking.

        Args:
            owned_personal_projects: List of personal projects owned by the user
        """
        owned_groups = await self.get_owned_groups()
        webhooks = []

        def build_group_webhook_entries(groups):
            return [
                GitlabWebhook(
                    group_id=str(group['id']),
                    project_id=None,
                    user_id=self.external_auth_id,
                    webhook_exists=False,
                )
                for group in groups
            ]

        def build_project_webhook_entries(projects):
            return [
                GitlabWebhook(
                    group_id=None,
                    project_id=str(project['id']),
                    user_id=self.external_auth_id,
                    webhook_exists=False,
                )
                for project in projects
            ]

        # Collect all webhook entries
        webhooks.extend(build_group_webhook_entries(owned_groups))
        webhooks.extend(build_project_webhook_entries(owned_personal_projects))

        # Store webhooks in the database
        if webhooks:
            try:
                webhook_store = GitlabWebhookStore()
                await webhook_store.store_webhooks(webhooks)
                logger.info(
                    f'Added GitLab webhooks to db for user {self.external_auth_id}'
                )
            except Exception:
                logger.warning('Failed to add Gitlab webhooks to db', exc_info=True)

    async def store_repository_data(
        self, users_personal_projects: list[dict], repositories: list[Repository]
    ) -> None:
        """
        Store repository data in the database.
        This function combines the functionality of add_owned_projects_and_groups_to_db and store_repositories_in_db.

        Args:
            users_personal_projects: List of personal projects owned by the user
            repositories: List of Repository objects to store
        """
        try:
            # First, add owned projects and groups to the database
            await self.add_owned_projects_and_groups_to_db(users_personal_projects)

            # Then, store repositories in the database
            await store_repositories_in_db(repositories, self.external_auth_id)

            logger.info(
                f'Successfully stored repository data for user {self.external_auth_id}'
            )
        except Exception:
            logger.warning('Error storing repository data', exc_info=True)

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode, store_in_background: bool = True
    ) -> list[Repository]:
        """
        Get repositories for the authenticated user, including information about the kind of project.
        Also collects repositories where the kind is "user" and the user is the owner.

        Args:
            sort: The field to sort repositories by
            app_mode: The application mode (OSS or SAAS)

        Returns:
            List[Repository]: A list of repositories for the authenticated user
        """
        MAX_REPOS = 1000
        PER_PAGE = 100  # Maximum allowed by GitLab API
        all_repos: list[dict] = []
        users_personal_projects: list[dict] = []
        page = 1

        url = f'{self.BASE_URL}/projects'
        # Map GitHub's sort values to GitLab's order_by values
        order_by = {
            'pushed': 'last_activity_at',
            'updated': 'last_activity_at',
            'created': 'created_at',
            'full_name': 'name',
        }.get(sort, 'last_activity_at')

        user_id = None
        try:
            user_info = await self.get_user()
            user_id = user_info.id
        except Exception as e:
            logger.warning(f'Could not fetch user id: {e}')

        while len(all_repos) < MAX_REPOS:
            params = {
                'page': str(page),
                'per_page': str(PER_PAGE),
                'order_by': order_by,
                'sort': 'desc',  # GitLab uses sort for direction (asc/desc)
                'membership': 1,  # Use 1 instead of True
            }

            try:
                response, headers = await self._make_request(url, params)

                if not response:  # No more repositories
                    break

                # Process each repository to identify user-owned ones
                for repo in response:
                    namespace = repo.get('namespace', {})
                    kind = namespace.get('kind')
                    owner_id = repo.get('owner', {}).get('id')

                    # Collect user owned personal projects
                    if kind == 'user' and str(owner_id) == str(user_id):
                        users_personal_projects.append(repo)

                    # Add to all repos regardless
                    all_repos.append(repo)

                page += 1

                # Check if we've reached the last page
                link_header = headers.get('Link', '')
                if 'rel="next"' not in link_header:
                    break

            except Exception:
                logger.warning(
                    f'Error fetching repositories on page {page}', exc_info=True
                )
                break

        # Trim to MAX_REPOS if needed and convert to Repository objects
        all_repos = all_repos[:MAX_REPOS]
        repositories = [
            Repository(
                id=str(repo.get('id')),
                full_name=str(repo.get('path_with_namespace')),
                stargazers_count=repo.get('star_count'),
                git_provider=ProviderType.GITLAB,
                is_public=repo.get('visibility') == 'public',
            )
            for repo in all_repos
        ]

        # Store webhook and repository info
        if store_in_background:
            asyncio.create_task(
                self.store_repository_data(users_personal_projects, repositories)
            )
        else:
            await self.store_repository_data(users_personal_projects, repositories)
        return repositories

    async def check_resource_exists(
        self, resource_type: GitLabResourceType, resource_id: str
    ) -> tuple[bool, WebhookStatus | None]:
        """
        Check if resource exists and the user has access to it.

        Args:
            resource_type: The type of resource
            resource_id: The ID of resource to check

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if the resource exists and the user has access to it, False otherwise
                - str: A reason message explaining the result
        """

        if resource_type == GitLabResourceType.GROUP:
            url = f'{self.BASE_URL}/groups/{resource_id}'
        else:
            url = f'{self.BASE_URL}/projects/{resource_id}'

        try:
            response, _ = await self._make_request(url)
            # If we get a response, the resource exists and the user has access to it
            return bool(response and 'id' in response), None
        except RateLimitError:
            return False, WebhookStatus.RATE_LIMITED
        except Exception:
            logger.warning('Resource existence check failed', exc_info=True)
            return False, WebhookStatus.INVALID

    async def check_webhook_exists_on_resource(
        self, resource_type: GitLabResourceType, resource_id: str, webhook_url: str
    ) -> tuple[bool, WebhookStatus | None]:
        """
        Check if a webhook already exists for resource with a specific URL.

        Args:
            resource_type: The type of resource
            resource_id: The ID of the resource to check
            webhook_url: The URL of the webhook to check for

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if the webhook exists, False otherwise
                - str: A reason message explaining the result
        """

        # Construct the URL based on the resource type
        if resource_type == GitLabResourceType.GROUP:
            url = f'{self.BASE_URL}/groups/{resource_id}/hooks'
        else:
            url = f'{self.BASE_URL}/projects/{resource_id}/hooks'

        try:
            # Get all webhooks for the resource
            response, _ = await self._make_request(url)

            # Check if any webhook has the specified URL
            exists = False
            if response:
                for webhook in response:
                    if webhook.get('url') == webhook_url:
                        exists = True

            return exists, None

        except RateLimitError:
            return False, WebhookStatus.RATE_LIMITED
        except Exception:
            logger.warning('Webhook existence check failed', exc_info=True)
            return False, WebhookStatus.INVALID

    async def check_user_has_admin_access_to_resource(
        self, resource_type: GitLabResourceType, resource_id: str
    ) -> tuple[bool, WebhookStatus | None]:
        """
        Check if the user has admin access to resource (is either an owner or maintainer)

        Args:
            resource_type: The type of resource
            resource_id: The ID of the resource to check

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if the user has admin access to the resource (owner or maintainer), False otherwise
                - str: A reason message explaining the result
        """

        # For groups, we need to check if the user is an owner or maintainer
        if resource_type == GitLabResourceType.GROUP:
            url = f'{self.BASE_URL}/groups/{resource_id}/members/all'
            try:
                response, _ = await self._make_request(url)
                # Check if the current user is in the members list with access level >= 40 (Maintainer or Owner)

                exists = False
                if response:
                    current_user = await self.get_user()
                    user_id = current_user.id
                    for member in response:
                        if (
                            str(member.get('id')) == str(user_id)
                            and member.get('access_level', 0) >= 40
                        ):
                            exists = True
                return exists, None
            except RateLimitError:
                return False, WebhookStatus.RATE_LIMITED
            except Exception:
                return False, WebhookStatus.INVALID

        # For projects, we need to check if the user has maintainer or owner access
        else:
            url = f'{self.BASE_URL}/projects/{resource_id}/members/all'
            try:
                response, _ = await self._make_request(url)
                exists = False
                # Check if the current user is in the members list with access level >= 40 (Maintainer)
                if response:
                    current_user = await self.get_user()
                    user_id = current_user.id
                    for member in response:
                        if (
                            str(member.get('id')) == str(user_id)
                            and member.get('access_level', 0) >= 40
                        ):
                            exists = True
                return exists, None
            except RateLimitError:
                return False, WebhookStatus.RATE_LIMITED
            except Exception:
                logger.warning('Admin access check failed', exc_info=True)
                return False, WebhookStatus.INVALID

    async def install_webhook(
        self,
        resource_type: GitLabResourceType,
        resource_id: str,
        webhook_name: str,
        webhook_url: str,
        webhook_secret: str,
        webhook_uuid: str,
        scopes: list[str],
    ) -> tuple[str | None, WebhookStatus | None]:
        """
        Install webhook for user's group or project

        Args:
            resource_type: The type of resource
            resource_id: The ID of the resource to check
            webhook_secret: Webhook secret that is used to verify payload
            webhook_name: Name of webhook
            webhook_url: Webhook URL
            scopes: activity webhook listens for

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if installation was successful, False otherwise
                - str: A reason message explaining the result
        """

        description = 'Cloud OpenHands Resolver'

        # Set up webhook parameters
        webhook_data = {
            'url': webhook_url,
            'name': webhook_name,
            'enable_ssl_verification': True,
            'token': webhook_secret,
            'description': description,
        }

        for scope in scopes:
            webhook_data[scope] = True

        # Add custom headers with user id
        if self.external_auth_id:
            webhook_data['custom_headers'] = [
                {'key': 'X-OpenHands-User-ID', 'value': self.external_auth_id},
                {'key': 'X-OpenHands-Webhook-ID', 'value': webhook_uuid},
            ]

        if resource_type == GitLabResourceType.GROUP:
            url = f'{self.BASE_URL}/groups/{resource_id}/hooks'
        else:
            url = f'{self.BASE_URL}/projects/{resource_id}/hooks'

        try:
            # Make the API request
            response, _ = await self._make_request(
                url=url, params=webhook_data, method=RequestMethod.POST
            )

            if response and 'id' in response:
                return str(response['id']), None

            # Check if the webhook was created successfully
            return None, None

        except RateLimitError:
            return None, WebhookStatus.RATE_LIMITED
        except Exception:
            logger.warning('Webhook installation failed', exc_info=True)
            return None, WebhookStatus.INVALID

    async def user_has_write_access(self, project_id: str) -> bool:
        url = f'{self.BASE_URL}/projects/{project_id}'
        try:
            response, _ = await self._make_request(url)
            # Check if the current user is in the members list with access level >= 30 (Developer)

            if 'permissions' not in response:
                logger.info('permissions not found', extra={'response': response})
                return False

            permissions = response['permissions']
            if permissions['project_access']:
                logger.info('[GitLab]: Checking project access')
                return permissions['project_access']['access_level'] >= 30

            if permissions['group_access']:
                logger.info('[GitLab]: Checking group access')
                return permissions['group_access']['access_level'] >= 30

            return False
        except Exception:
            logger.warning('Access check failed', exc_info=True)
            return False

    async def reply_to_issue(
        self, project_id: str, issue_number: str, discussion_id: str | None, body: str
    ):
        """
        Either create new comment thread, or reply to comment thread (depending on discussion_id param)
        """
        try:
            if discussion_id:
                url = f'{self.BASE_URL}/projects/{project_id}/issues/{issue_number}/discussions/{discussion_id}/notes'
            else:
                url = f'{self.BASE_URL}/projects/{project_id}/issues/{issue_number}/discussions'
            params = {'body': body}

            await self._make_request(url=url, params=params, method=RequestMethod.POST)
        except Exception as e:
            logger.exception(f'[GitLab]: Reply to issue failed {e}')

    async def reply_to_mr(
        self, project_id: str, merge_request_iid: str, discussion_id: str, body: str
    ):
        """
        Reply to comment thread on MR
        """
        try:
            url = f'{self.BASE_URL}/projects/{project_id}/merge_requests/{merge_request_iid}/discussions/{discussion_id}/notes'
            params = {'body': body}

            await self._make_request(url=url, params=params, method=RequestMethod.POST)
        except Exception as e:
            logger.exception(f'[GitLab]: Reply to MR failed {e}')

    async def get_user_resources_with_admin_access(
        self,
    ) -> tuple[list[dict], list[dict]]:
        """
        Get all projects and groups where the current user has admin access (maintainer or owner).

        Returns:
            tuple[list[dict], list[dict]]: A tuple containing:
                - list of projects where user has admin access
                - list of groups where user has admin access
        """
        projects_with_admin_access = []
        groups_with_admin_access = []

        # Fetch all projects the user is a member of
        page = 1
        per_page = 100
        while True:
            try:
                url = f'{self.BASE_URL}/projects'
                params = {
                    'page': str(page),
                    'per_page': str(per_page),
                    'membership': 1,
                    'min_access_level': 40,  # Maintainer or Owner
                }
                response, headers = await self._make_request(url, params)

                if not response:
                    break

                projects_with_admin_access.extend(response)
                page += 1

                # Check if we've reached the last page
                link_header = headers.get('Link', '')
                if 'rel="next"' not in link_header:
                    break

            except Exception:
                logger.warning(f'Error fetching projects on page {page}', exc_info=True)
                break

        # Fetch all groups where user is owner or maintainer
        groups_with_admin_access = await self.get_owned_groups(min_access_level=40)

        logger.info(
            f'Found {len(projects_with_admin_access)} projects and {len(groups_with_admin_access)} groups with admin access'
        )

        return projects_with_admin_access, groups_with_admin_access
