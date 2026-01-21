from types import MappingProxyType

from integrations.azure_devops.azure_devops_solvability import (
    summarize_azure_devops_solvability,
)
from integrations.azure_devops.azure_devops_types import AzureDevOpsEventType
from integrations.azure_devops.azure_devops_view import (
    AzureDevOpsFactory,
    AzureDevOpsView,
)
from integrations.azure_devops.azure_devops_view_classes import (
    AzureDevOpsInlinePRComment,
    AzureDevOpsPRComment,
    AzureDevOpsViewType,
    AzureDevOpsWorkItem,
)
from integrations.azure_devops.data_collector import AzureDevOpsDataCollector
from integrations.models import Message, SourceType
from integrations.utils import (
    CONVERSATION_URL,
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
)
from integrations.v1_utils import get_saas_user_auth
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.utils.conversation_callback_utils import register_callback_processor
from storage.azure_devops_webhook_store import AzureDevOpsWebhookStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsService,
)
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.storage.data_models.secrets import Secrets
from openhands.utils.async_utils import call_sync_from_async


class AzureDevOpsManager:
    """Manager for Azure DevOps webhook events and interactions."""

    async def _resolve_keycloak_user_id(
        self, author_azure_id: str, organization: str
    ) -> str | None:
        """Resolve Azure DevOps user ID to Keycloak user ID.

        Uses a multi-plan approach:
        - Plan A: Webhook table lookup
        - Plan B: Direct Keycloak attribute lookup
        - Plan C: Email-based lookup with opportunistic enrichment

        Args:
            author_azure_id: Azure DevOps user ID (GUID)
            organization: Azure DevOps organization name

        Returns:
            Keycloak user ID or None if not found
        """
        keycloak_user_id = None

        # Plan A: Try webhook table lookup (Azure DevOps ID -> Keycloak user)
        try:
            from sqlalchemy import select
            from storage.azure_devops_webhook import AzureDevOpsWebhook
            from storage.database import a_session_maker

            async with a_session_maker() as session:
                stmt = (
                    select(AzureDevOpsWebhook)
                    .where(
                        AzureDevOpsWebhook.user_id == str(author_azure_id),
                        AzureDevOpsWebhook.organization == organization,
                    )
                    .limit(1)
                )
                result = await session.execute(stmt)
                webhook = result.scalar_one_or_none()

                if webhook:
                    keycloak_user_id = (
                        await self.token_manager.get_user_id_from_idp_user_id(
                            str(author_azure_id), ProviderType.AZURE_DEVOPS
                        )
                    )
        except Exception as e:
            logger.debug(f'[Azure DevOps] Plan A (webhook lookup) failed: {e}')

        if keycloak_user_id:
            return keycloak_user_id

        # Plan B: Try direct lookup by azure_devops_id attribute in Keycloak
        keycloak_user_id = await self.token_manager.get_user_id_from_idp_user_id(
            str(author_azure_id), ProviderType.AZURE_DEVOPS
        )

        if keycloak_user_id:
            return keycloak_user_id

        # Plan C: Fallback to email-based lookup
        try:
            sp_token_for_graph = (
                await self.token_manager.get_azure_devops_service_principal_token()
            )

            if sp_token_for_graph:
                from .azure_devops_id_resolver import AzureDevOpsIdResolver

                resolver = AzureDevOpsIdResolver(SecretStr(sp_token_for_graph))
                author_email = await resolver.get_user_email_from_id(
                    str(author_azure_id), organization
                )

                if author_email:
                    from server.auth.keycloak_manager import get_keycloak_admin

                    keycloak_admin = get_keycloak_admin(external=True)
                    users = await keycloak_admin.a_get_users({'email': author_email})

                    if users and len(users) > 0:
                        keycloak_user_id = users[0]['id']

                        # Opportunistic enrichment: Store azure_devops_id for future use
                        try:
                            await keycloak_admin.a_update_user(
                                keycloak_user_id,
                                {
                                    'attributes': {
                                        'azure_devops_id': [str(author_azure_id)]
                                    }
                                },
                            )
                        except Exception as e:
                            logger.debug(
                                f'[Azure DevOps] Failed to enrich Keycloak user with Azure ID: {e}'
                            )
        except Exception as e:
            logger.debug(f'[Azure DevOps] Plan C (email lookup) failed: {e}')

        return keycloak_user_id

    def __init__(
        self, token_manager: TokenManager, data_collector: AzureDevOpsDataCollector
    ):
        """Initialize Azure DevOps manager.

        Args:
            token_manager: Token manager for retrieving user tokens dynamically
            data_collector: Data collector for tracking interactions
        """
        self.token_manager = token_manager
        self.webhook_store = AzureDevOpsWebhookStore()
        self.view = AzureDevOpsView()
        self.data_collector = data_collector
        self._service_principal_id: str | None = (
            None  # Cached Azure DevOps identity GUID
        )

        from pathlib import Path

        template_dir = Path(OPENHANDS_RESOLVER_TEMPLATES_DIR) / 'azure_devops'
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

    @staticmethod
    def _strip_html_tags(html_text: str) -> str:
        """Strip HTML tags from text and convert to plain text.

        Azure DevOps stores descriptions in HTML format. This function:
        1. Removes all HTML tags using BeautifulSoup for robust parsing
        2. Decodes HTML entities (e.g., &amp; -> &)
        3. Normalizes whitespace

        Args:
            html_text: Text that may contain HTML tags

        Returns:
            Plain text with HTML tags removed
        """
        if not html_text:
            return ''

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_text, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logger.warning(f'[Azure DevOps] Failed to parse HTML: {e}')
            return html_text

    @staticmethod
    def _extract_field_value(field_value):
        """Extract the actual value from a field that might be in newValue/oldValue format.

        Azure DevOps webhook payloads for workitem.updated events send changed
        fields as dictionaries with 'newValue' and 'oldValue' keys.

        Args:
            field_value: The field value which can be:
                - A string (direct value)
                - A dict with 'newValue' key (changed field in webhook)
                - None

        Returns:
            The extracted string value or empty string if None
        """
        if field_value is None:
            return ''

        if isinstance(field_value, dict):
            # Handle webhook format: {'newValue': '...', 'oldValue': '...'}
            return field_value.get('newValue', '')

        # Direct string value
        return field_value

    async def _get_service_principal_id(self, organization: str) -> str:
        """Get the service principal's Azure DevOps identity GUID.

        This ID is used to detect mentions in comments. When users mention
        the service principal in Azure DevOps UI (e.g., @OpenHands), it appears
        as @<GUID> in the webhook payload.

        Azure DevOps uses different IDs in different contexts:
        - memberId: Used in work item mentions
        - subjectId: Used in PR mentions and assignments

        Both are extracted from X-VSS-UserData header.

        Args:
            organization: Azure DevOps organization name (required for API calls)

        Returns:
            The service principal's identity GUID (uppercase, no braces)
        """
        if self._service_principal_id:
            return self._service_principal_id

        try:
            # Get service principal token
            sp_token = (
                await self.token_manager.get_azure_devops_service_principal_token()
            )

            # Create service instance with organization for correct base_url
            sp_service = AzureDevOpsService(
                token=SecretStr(sp_token),
                base_domain=f'https://dev.azure.com/{organization}',
            )
            user = await sp_service.get_user()

            # Cache the ID (uppercase to match webhook format)
            self._service_principal_id = user.id.upper()

            # Also extract and cache the memberId for work item mentions
            # The get_user() returns subjectId, but we also need memberId
            # We can get it by calling the API again and parsing X-VSS-UserData
            url = (
                f'{sp_service.base_url}/_apis/connectionData?api-version=7.1-preview.1'
            )
            _, headers = await sp_service._make_request(url)
            x_vss_user_data = headers.get('X-VSS-UserData', '')

            # Format: memberId:tenantId\\subjectId or memberId:subjectId
            if ':' in x_vss_user_data:
                member_id = x_vss_user_data.split(':')[0].strip().upper()
                # Store both IDs as a comma-separated string for checking
                self._service_principal_id = f'{member_id},{self._service_principal_id}'
                logger.info(
                    f'[Azure DevOps] Service principal IDs: memberId={member_id}, subjectId={user.id.upper()}'
                )
            else:
                logger.info(
                    f'[Azure DevOps] Service principal identity ID: {self._service_principal_id}'
                )

            return self._service_principal_id
        except Exception as e:
            logger.error(
                f'[Azure DevOps] Failed to get service principal identity: {e}'
            )
            raise

    def _confirm_incoming_source_type(self, message: Message):
        """Validate that the message is from Azure DevOps.

        Args:
            message: The incoming message

        Raises:
            ValueError: If message is not from Azure DevOps
        """
        if message.source != SourceType.AZURE_DEVOPS:
            raise ValueError(f'Expected Azure DevOps message, got {message.source}')

    @staticmethod
    def _extract_organization_from_payload(payload: dict) -> str:
        """Extract organization name from webhook payload's resourceContainers.

        Args:
            payload: Azure DevOps webhook payload

        Returns:
            Organization name or empty string if not found
        """
        try:
            # resourceContainers.collection.baseUrl contains the org URL
            # e.g., "https://dev.azure.com/AIngineer/"
            base_url = (
                payload.get('resourceContainers', {})
                .get('collection', {})
                .get('baseUrl', '')
            )
            if base_url:
                # Extract org from URL: https://dev.azure.com/OrgName/
                return AzureDevOpsManager._extract_organization_from_url(base_url)
        except Exception:
            pass
        return ''

    async def is_job_requested(self, message: Message) -> bool:
        """Check if a job is requested via service principal mention or assignment.

        Supports:
        - Work item assigned to OpenHands service principal
        - Work item commented with service principal mention
        - PR commented with service principal mention (regular or inline code review)

        Args:
            message: The incoming webhook message

        Returns:
            True if job is requested, False otherwise
        """
        self._confirm_incoming_source_type(message)

        # Extract payload and organization for API calls
        payload = message.message if isinstance(message.message, dict) else {}
        organization = self._extract_organization_from_payload(payload)

        # Get service principal ID for mention detection
        try:
            if not organization:
                logger.warning(
                    '[Azure DevOps] No organization found in payload, cannot verify service principal'
                )
                sp_id = None
            else:
                sp_id = await self._get_service_principal_id(organization)
        except Exception as e:
            logger.warning(f'[Azure DevOps] Failed to get service principal ID: {e}')
            sp_id = None

        # Use factory methods to check for different trigger types
        is_assigned = AzureDevOpsFactory.is_assigned_work_item(message, sp_id)
        is_work_item_comment = AzureDevOpsFactory.is_work_item_comment(message, sp_id)
        is_pr_comment = AzureDevOpsFactory.is_pr_comment(
            message, service_principal_id=sp_id
        )
        is_inline_comment = AzureDevOpsFactory.is_inline_pr_comment(message, sp_id)

        if not (
            is_assigned or is_work_item_comment or is_pr_comment or is_inline_comment
        ):
            return False

        return True

    async def receive_message(self, message: Message):
        """Process incoming Azure DevOps webhook message.

        Args:
            message: The incoming webhook message
        """
        self._confirm_incoming_source_type(message)

        # Process payload for data collection
        try:
            await call_sync_from_async(self.data_collector.process_payload, message)
        except Exception as e:
            logger.warning(
                f'[Azure DevOps]: Error processing payload for interaction data: {e}',
                exc_info=True,
            )

        # Extract payload
        payload = message.message if isinstance(message.message, dict) else {}

        # Check if this is a mention/tag that requires action
        is_job_requested = await self.is_job_requested(message)

        if is_job_requested:
            event_type = self.view.get_event_type(payload)

            # Handle different trigger types
            if event_type in (
                AzureDevOpsEventType.WORKITEM_UPDATED,
                AzureDevOpsEventType.WORKITEM_CREATED,
            ):
                await self._handle_work_item_trigger(payload, event_type)
            elif event_type == AzureDevOpsEventType.PR_COMMENTED:
                await self._handle_pr_trigger(payload)
            else:
                logger.warning(f'[Azure DevOps] Unhandled event type: {event_type}')

    async def _handle_work_item_trigger(
        self, payload: dict, event_type: AzureDevOpsEventType
    ):
        """Handle @openhands mention or assignment in a work item.

        Supports:
        - Work item commented with @openhands (service principal mention)
        - Work item assigned to OpenHands service principal

        Args:
            payload: The webhook payload
            event_type: The type of event that triggered this
        """
        # Extract work item info from payload
        resource = payload.get('resource', {})
        fields = resource.get('fields', {})
        url = resource.get('url', '')

        # Payload structure differs between created and updated events:
        # - workitem.created: resource.fields contains all fields directly, resource.id is work item ID
        # - workitem.updated: resource.fields contains only CHANGED fields (as change objects),
        #                     resource.revision.fields contains full state, URL contains work item ID
        is_created_event = event_type == AzureDevOpsEventType.WORKITEM_CREATED

        if is_created_event:
            # For created events, resource.id IS the work item ID
            work_item_id = resource.get('id')
            # Fields contain direct values
            full_fields = fields
        else:
            # For updated events, extract work item ID from URL
            # URL format: https://dev.azure.com/org/project/_apis/wit/workItems/{workItemId}/updates/{updateId}
            work_item_id = self._extract_work_item_id_from_url(url)
            # Full field values are in revision.fields
            revision = resource.get('revision', {})
            full_fields = revision.get('fields', {})

        if not work_item_id:
            logger.error('[Azure DevOps] No work item ID in payload - aborting')
            return

        # Extract organization and project from URL for API calls
        organization = self._extract_organization_from_url(url)
        if not organization:
            logger.error(
                f'[Azure DevOps] Could not extract organization from URL: {url} - aborting'
            )
            return

        # Extract project ID from URL (for API calls that need it)
        project_id = self._extract_project_from_url(url)
        if not project_id:
            logger.error(
                f'[Azure DevOps] Could not extract project from URL: {url} - aborting'
            )
            return

        # Get project name - for created events it's in fields, for updated it's in revision.fields
        project_name = full_fields.get('System.TeamProject', '')
        if not project_name:
            # Fallback: use project_id if name not available
            project_name = project_id
            logger.warning(
                f'[Azure DevOps] Could not get project name from fields, using ID: {project_id}'
            )

        # Get service principal token for API operations (needed to fetch work item details)
        try:
            sp_token = (
                await self.token_manager.get_azure_devops_service_principal_token()
            )
        except Exception as e:
            logger.error(
                f'[Azure DevOps] Failed to get service principal token: {e}',
                exc_info=True,
            )
            return

        # Extract title and description from full fields
        # For workitem.created: full_fields contains all fields directly
        # For workitem.updated: full_fields is revision.fields (complete state)
        title = full_fields.get('System.Title') or ''
        description = (
            self._strip_html_tags(full_fields.get('System.Description', '')) or ''
        )
        work_item_type = full_fields.get('System.WorkItemType') or ''

        # Fetch repository and branch from work item's development section
        repository, branch = await self._get_work_item_repository(
            organization, project_id, work_item_id, sp_token
        )

        repository_linked = bool(repository)
        if not repository:
            # No repository linked - follow Jira pattern: infer from description or ask user
            logger.info(
                f'[Azure DevOps] No repository linked to work item {work_item_id} - attempting to infer'
            )

            # Get all repositories in the project
            project_repos = await self._get_project_repositories(
                organization, project_id, project_name, sp_token
            )

            if not project_repos:
                # No repositories in project - ask user to create one
                await self._send_repo_selection_comment(
                    organization, project_name, work_item_id, sp_token, []
                )
                logger.warning(
                    f'[Azure DevOps] No repositories in project {project_name} - asked user to create one'
                )
                return

            # Try to infer repository from work item title and description
            from integrations.utils import filter_potential_repos_by_user_msg

            from openhands.integrations.service_types import Repository

            # Convert to Repository objects for the utility function
            repo_objects = [
                Repository(
                    id=0,
                    full_name=full_name,
                    git_url='',
                    is_public=False,
                )
                for full_name, _ in project_repos
            ]

            # Combine title and description for inference
            search_text = f'{title}\n{description}'
            match_found, matched_repos = filter_potential_repos_by_user_msg(
                search_text, repo_objects
            )

            if match_found and len(matched_repos) == 1:
                # Found exact match - use it
                repository = matched_repos[0].full_name
                logger.info(f'[Azure DevOps] Inferred repository: {repository}')
            else:
                # No clear match - ask user to specify
                await self._send_repo_selection_comment(
                    organization, project_name, work_item_id, sp_token, project_repos
                )
                logger.info(
                    '[Azure DevOps] Could not infer repository - asked user to specify'
                )
                return

        # Extract comment/change author for conversation ownership
        # Service principal will be used for API authentication
        # For workitem.created: author is in System.CreatedBy field (email format)
        # For workitem.updated: author is in resource.revisedBy (has Azure ID)
        keycloak_user_id = None
        if is_created_event:
            # Extract author from System.CreatedBy field (format: "Name <email>")
            created_by = full_fields.get('System.CreatedBy', '')
            username = created_by.split(' <')[0] if created_by else 'User'
            # Extract email from angle brackets
            author_email = None
            if '<' in created_by and '>' in created_by:
                potential_email = created_by.split('<')[1].split('>')[0]
                if '@' in potential_email:
                    author_email = potential_email

            # Look up keycloak user by email (like Plan C in _resolve_keycloak_user_id)
            if author_email:
                logger.info(
                    f'[Azure DevOps] Looking up keycloak user by email: {author_email}'
                )
                try:
                    from server.auth.keycloak_manager import get_keycloak_admin

                    keycloak_admin = get_keycloak_admin(external=True)
                    users = await keycloak_admin.a_get_users({'email': author_email})
                    logger.info(
                        f'[Azure DevOps] Keycloak lookup result: found {len(users) if users else 0} users'
                    )
                    if users and len(users) > 0:
                        keycloak_user_id = users[0]['id']
                        logger.info(
                            f'[Azure DevOps] Found keycloak user: {keycloak_user_id}'
                        )
                except Exception as e:
                    logger.warning(
                        f'[Azure DevOps] Failed to lookup user by email {author_email}: {e}'
                    )
        else:
            # For workitem.updated: author is in resource.revisedBy
            revised_by = resource.get('revisedBy', {})
            author_azure_id = (
                revised_by.get('id') if isinstance(revised_by, dict) else None
            )
            username = (
                revised_by.get('displayName', 'User')
                if isinstance(revised_by, dict)
                else 'User'
            )

            # Try to look up author's keycloak user (for conversation ownership)
            if author_azure_id:
                keycloak_user_id = await self._resolve_keycloak_user_id(
                    author_azure_id, organization
                )

        # Check if keycloak user was found - required for conversation ownership
        if not keycloak_user_id:
            logger.warning(
                '[Azure DevOps] Could not find keycloak user for author. '
                'User must be logged in to OpenHands to trigger jobs.'
            )
            return

        # Create UserData
        # keycloak_user_id is for conversation ownership
        # Service principal token will be used for API operations
        from integrations.types import UserData

        user_info = UserData(
            user_id=0,  # Not used for service principal
            username=username,  # Display name
            keycloak_user_id=keycloak_user_id,  # For conversation ownership
        )

        # Create service principal service for API operations
        from integrations.azure_devops.azure_devops_service import (
            SaaSAzureDevOpsService,
        )

        sp_service: SaaSAzureDevOpsService = AzureDevOpsService(
            token=SecretStr(sp_token)
        )

        # Extract work item details
        # Title, description, work_item_type already extracted earlier for repository inference
        # Now fetch full data from API if any are missing

        # If title/description not in webhook payload, fetch from API
        if not title or not description or not work_item_type:
            try:
                work_item_data = await sp_service.get_work_item(
                    repository, work_item_id
                )
                work_item_fields = work_item_data.get('fields', {})

                if not title:
                    title = work_item_fields.get('System.Title', 'Untitled')
                if not description:
                    description = self._strip_html_tags(
                        work_item_fields.get('System.Description', '')
                    )
                if not work_item_type:
                    work_item_type = work_item_fields.get(
                        'System.WorkItemType', 'WorkItem'
                    )

            except Exception as e:
                logger.warning(
                    f'[Azure DevOps] Failed to fetch full work item details: {e}',
                    exc_info=True,
                )
                # Use defaults if API call fails
                if not title:
                    title = 'Untitled'
                if not description:
                    description = ''
                if not work_item_type:
                    work_item_type = 'WorkItem'

        # Get previous comments (if any) using service principal
        from openhands.integrations.service_types import Comment

        previous_comments: list[Comment] = []
        try:
            previous_comments = await sp_service.get_work_item_comments(
                repository, work_item_id
            )
        except Exception as e:
            logger.warning(
                f'[Azure DevOps] Failed to get work item comments: {e}',
                exc_info=True,
            )

        # Extract comment body if this was triggered by a comment (not assignment)
        comment_body = None
        history_change = fields.get('System.History')
        if history_change and isinstance(history_change, dict):
            comment_text = history_change.get('newValue', '')
            if comment_text:
                # Strip HTML tags and extract plain text from comment
                comment_body = self._strip_html_tags(comment_text)

        # Create view instance
        from integrations.models import Message

        azure_devops_view = AzureDevOpsWorkItem(
            # Base class fields
            organization=organization,
            project_name=project_name,
            full_repo_name=repository,
            is_public_repo=False,  # Azure DevOps work items are typically private
            user_info=user_info,
            raw_payload=Message(source=SourceType.AZURE_DEVOPS, message=payload),
            conversation_id='',  # Will be set by initialize_new_conversation
            uuid=None,
            should_extract=True,
            send_summary_instruction=True,
            title=title,
            description=description,
            previous_comments=previous_comments,
            v1_enabled=False,  # Will be set by initialize_new_conversation
            # ResolverViewInterface required fields
            installation_id=0,  # Azure DevOps doesn't use installation_id
            issue_number=work_item_id,  # Map work_item_id to issue_number for interface compatibility
            # Work item specific fields
            work_item_id=work_item_id,
            work_item_type=work_item_type,
            selected_branch=branch,  # Branch from work item development section
            comment_body=comment_body,  # Comment text when triggered by @mention
            repository_linked=repository_linked,  # Whether repository was found in development section
        )

        # Start job execution
        await self.start_job(azure_devops_view)

    async def _handle_pr_trigger(self, payload: dict):
        """Handle @openhands mention in a PR comment (regular or inline).

        Supports:
        - PR commented with @openhands (general discussion)
        - PR inline comment with @openhands (code review comment)

        Args:
            payload: The webhook payload
        """
        # Extract PR info from payload
        resource = payload.get('resource', {})

        # For PR comment events, PR info is nested under resource.pullRequest
        # For PR created/updated events, PR info is directly in resource
        pr_data = resource.get('pullRequest', resource)

        pr_id = pr_data.get('pullRequestId')
        repository = pr_data.get('repository', {})
        repo_name = repository.get('name', '')
        project_name = repository.get('project', {}).get('name', '')
        url = pr_data.get('url', '')

        # Check if inline comment
        comment_data = resource.get('comment', {})
        thread_context = comment_data.get('threadContext')
        is_inline = thread_context is not None

        # Extract thread_id from comment's _links for replying in the same thread
        # URL format: .../_apis/git/repositories/.../pullRequests/{pr_id}/threads/{thread_id}
        thread_id = None
        try:
            threads_href = (
                comment_data.get('_links', {}).get('threads', {}).get('href', '')
            )
            if threads_href and '/threads/' in threads_href:
                # Extract thread ID from URL
                thread_id_str = threads_href.split('/threads/')[-1].split('/')[0]
                if thread_id_str.isdigit():
                    thread_id = int(thread_id_str)
                    logger.info(f'[Azure DevOps] Extracted thread_id: {thread_id}')
        except Exception as e:
            logger.warning(f'[Azure DevOps] Failed to extract thread_id: {e}')

        if not pr_id:
            logger.error('[Azure DevOps] No PR ID in payload')
            return

        # Extract organization from URL
        organization = self._extract_organization_from_url(url)
        if not organization:
            logger.error(
                f'[Azure DevOps] Could not extract organization from URL: {url}'
            )
            return

        # Create repository string in format "org/project/repo"
        repository_str = f'{organization}/{project_name}/{repo_name}'

        # Extract comment author for conversation ownership
        # Service principal will be used for API authentication
        comment_author = comment_data.get('author', {})
        author_azure_id = (
            comment_author.get('id') if isinstance(comment_author, dict) else None
        )
        username = (
            comment_author.get('displayName', 'User')
            if isinstance(comment_author, dict)
            else 'User'
        )

        # Try to look up author's keycloak user (for conversation ownership)
        # Following GitHub pattern: lookup can return None, handled gracefully in start_job
        keycloak_user_id = None
        if author_azure_id:
            keycloak_user_id = await self._resolve_keycloak_user_id(
                author_azure_id, organization
            )

        # Get service principal token for API operations
        # Service principal acts as the application, not on behalf of any user
        try:
            sp_token = (
                await self.token_manager.get_azure_devops_service_principal_token()
            )
        except Exception as e:
            logger.error(f'[Azure DevOps] Failed to get service principal token: {e}')
            return

        # Create UserData
        # keycloak_user_id is for conversation ownership
        # Service principal token will be used for API operations
        from integrations.types import UserData

        user_info = UserData(
            user_id=0,  # Not used for service principal
            username=username,  # Display name
            keycloak_user_id=keycloak_user_id,  # For conversation ownership
        )

        # Get PR details
        pullrequest = resource.get('pullRequest', {})
        title = pullrequest.get('title', 'Untitled PR')
        description = self._strip_html_tags(pullrequest.get('description', ''))

        # Extract source branch name from PR
        # sourceRefName format: "refs/heads/branch-name"
        source_ref = pullrequest.get('sourceRefName', '')
        branch_name = None
        if source_ref.startswith('refs/heads/'):
            branch_name = source_ref.replace('refs/heads/', '')

        # Get previous comments (if any) using service principal
        from integrations.azure_devops.azure_devops_service import (
            SaaSAzureDevOpsService,
        )

        from openhands.integrations.service_types import Comment

        sp_service: SaaSAzureDevOpsService = AzureDevOpsService(
            token=SecretStr(sp_token)
        )

        previous_comments: list[Comment] = []
        try:
            previous_comments = await sp_service.get_pr_comments(repository_str, pr_id)
        except Exception as e:
            logger.warning(f'[Azure DevOps] Failed to get PR comments: {e}')

        # Create view instance
        from integrations.models import Message

        # Use the appropriate class based on whether this is an inline comment or general PR comment
        if is_inline:
            azure_devops_view = AzureDevOpsInlinePRComment(
                # Base class fields
                organization=organization,
                project_name=project_name,
                full_repo_name=repository_str,
                is_public_repo=False,  # Azure DevOps PRs are typically private
                user_info=user_info,
                raw_payload=Message(source=SourceType.AZURE_DEVOPS, message=payload),
                conversation_id='',  # Will be set by initialize_new_conversation
                uuid=None,
                should_extract=True,
                send_summary_instruction=True,
                title=title,
                description=description,
                previous_comments=previous_comments,
                v1_enabled=False,  # Will be set by initialize_new_conversation
                # ResolverViewInterface required fields
                installation_id=0,  # Azure DevOps doesn't use installation_id
                issue_number=pr_id,  # Map pr_id to issue_number for interface compatibility
                # PR comment fields
                pr_id=pr_id,
                repository_name=repo_name,
                branch_name=branch_name,
                thread_id=thread_id,  # For replying in the original thread
                # Inline-specific fields
                thread_context=thread_context,
            )
        else:
            azure_devops_view = AzureDevOpsPRComment(
                # Base class fields
                organization=organization,
                project_name=project_name,
                full_repo_name=repository_str,
                is_public_repo=False,  # Azure DevOps PRs are typically private
                user_info=user_info,
                raw_payload=Message(source=SourceType.AZURE_DEVOPS, message=payload),
                conversation_id='',  # Will be set by initialize_new_conversation
                uuid=None,
                should_extract=True,
                send_summary_instruction=True,
                title=title,
                description=description,
                previous_comments=previous_comments,
                v1_enabled=False,  # Will be set by initialize_new_conversation
                # ResolverViewInterface required fields
                installation_id=0,  # Azure DevOps doesn't use installation_id
                issue_number=pr_id,  # Map pr_id to issue_number for interface compatibility
                # PR comment fields
                pr_id=pr_id,
                repository_name=repo_name,
                branch_name=branch_name,
                thread_id=thread_id,  # For replying in the original thread
            )

        # Start job execution
        await self.start_job(azure_devops_view)

    @staticmethod
    def _parse_azure_devops_url(url: str) -> tuple[str, str]:
        """Parse Azure DevOps URL to extract organization and project.

        Args:
            url: Azure DevOps URL (e.g., https://dev.azure.com/org/project/_apis/...)

        Returns:
            Tuple of (organization, project) or ('', '') if parse fails
        """
        try:
            if 'dev.azure.com/' not in url:
                return '', ''
            path = url.split('dev.azure.com/')[1]
            parts = path.split('/')
            org = parts[0] if len(parts) > 0 else ''
            project = parts[1] if len(parts) > 1 else ''
            return org, project
        except Exception:
            return '', ''

    @staticmethod
    def _extract_organization_from_url(url: str) -> str:
        """Extract organization name from Azure DevOps URL.

        Args:
            url: Azure DevOps URL (e.g., https://dev.azure.com/org/_apis/...)

        Returns:
            Organization name or empty string if not found
        """
        org, _ = AzureDevOpsManager._parse_azure_devops_url(url)
        return org

    @staticmethod
    def _extract_project_from_url(url: str) -> str:
        """Extract project name/GUID from Azure DevOps URL.

        Args:
            url: Azure DevOps URL (e.g., https://dev.azure.com/org/project/_apis/...)

        Returns:
            Project name/GUID or empty string if not found
        """
        _, project = AzureDevOpsManager._parse_azure_devops_url(url)
        return project

    @staticmethod
    def _extract_work_item_id_from_url(url: str) -> int | None:
        """Extract work item ID from Azure DevOps URL.

        Args:
            url: Azure DevOps URL (e.g., https://dev.azure.com/org/project/_apis/wit/workItems/1254/updates/49)

        Returns:
            Work item ID as integer or None if not found
        """
        try:
            # URL format for updates: https://dev.azure.com/{org}/{project}/_apis/wit/workItems/{id}/updates/{updateId}
            # We want the {id} part, not the {updateId}
            if '/workItems/' in url:
                # Split by /workItems/ and take the part after it
                parts = url.split('/workItems/')
                if len(parts) > 1:
                    # The work item ID is between /workItems/ and the next /
                    # e.g., "1254/updates/49" -> extract "1254"
                    id_part = parts[1].split('/')[0]
                    # Try to parse as integer
                    return int(id_part)
            return None
        except Exception:
            return None

    async def _get_work_item_repository(
        self, organization: str, project: str, work_item_id: int, token: str
    ) -> tuple[str | None, str | None]:
        """Fetch repository and branch from work item's development section.

        Args:
            organization: Azure DevOps organization name
            project: Project name or GUID
            work_item_id: Work item ID
            token: Service principal access token

        Returns:
            Tuple of (repository, branch) where repository is in "org/project/repo" format
            Returns (None, None) if no repository is linked
        """
        try:
            import httpx

            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            }

            # Get work item with relations expanded
            url = f'https://dev.azure.com/{organization}/{project}/_apis/wit/workItems/{work_item_id}?$expand=relations&api-version=7.1'

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                work_item = response.json()
                relations = work_item.get('relations', [])

                # Look for Git repository relations
                # Development links have rel type like "ArtifactLink" and url contains "vstfs:///Git/"
                for relation in relations:
                    rel_type = relation.get('rel', '')
                    relation_url = relation.get('url', '')
                    attributes = relation.get('attributes', {})

                    if 'ArtifactLink' in rel_type and 'vstfs:///Git/' in relation_url:
                        # Handle branch refs: vstfs:///Git/Ref/{project_id}%2F{repo_id}%2FGB{branch_name}
                        if '/Git/Ref/' in relation_url:
                            # URL format: vstfs:///Git/Ref/0030a99b-0f29-44df-87b5-678bd217470a%2F44c0000e-bb70-41e7-956d-7d4c50facb89%2FGBdoc-updates
                            # Extract the encoded part after /Git/Ref/
                            import urllib.parse

                            ref_part = relation_url.split('/Git/Ref/')[1]
                            # Decode URL encoding (%2F -> /)
                            decoded = urllib.parse.unquote(ref_part)
                            # Format: {project_id}/{repo_id}/GB{branch_name}
                            parts = decoded.split('/')

                            if len(parts) >= 3:
                                project_id = parts[0]
                                repo_id = parts[1]
                                # Branch name may contain slashes (e.g., feature/chat-tq-guru-v2)
                                # Join all parts from index 2 onwards
                                branch_part = '/'.join(parts[2:])

                                # Extract branch name (remove GB prefix if present)
                                # GB prefix is only at the start of the branch part
                                branch = (
                                    branch_part[2:]
                                    if branch_part.startswith('GB')
                                    else branch_part
                                )

                                # Query project details to get project name
                                project_url = f'https://dev.azure.com/{organization}/_apis/projects/{project_id}?api-version=7.1'
                                project_response = await client.get(
                                    project_url, headers=headers
                                )
                                project_response.raise_for_status()

                                project_data = project_response.json()
                                project_name = project_data.get('name', '')

                                # Query repository details to get repository name
                                repo_url = f'https://dev.azure.com/{organization}/{project_id}/_apis/git/repositories/{repo_id}?api-version=7.1'
                                repo_response = await client.get(
                                    repo_url, headers=headers
                                )
                                repo_response.raise_for_status()

                                repo_data = repo_response.json()
                                repo_name = repo_data.get('name', '')

                                if repo_name and project_name:
                                    # Format: org/project_name/repo
                                    repository = (
                                        f'{organization}/{project_name}/{repo_name}'
                                    )
                                    return repository, branch

                        # Handle commit/PR refs: vstfs:///Git/Commit/{project}/{repo_id}/{commit_id}
                        elif (
                            '/Git/Commit/' in relation_url
                            or '/Git/PullRequest/' in relation_url
                        ):
                            # Original logic for commits and PRs
                            branch_ref = attributes.get(
                                'name', ''
                            )  # e.g., "refs/heads/doc-updates"

                            # Extract branch name from ref
                            branch = None
                            if branch_ref.startswith('refs/heads/'):
                                branch = branch_ref.replace('refs/heads/', '')

                            # Parse the vstfs URL to get repository ID
                            parts = relation_url.split('/')
                            if len(parts) >= 7:
                                repo_id = parts[6]  # Repository GUID

                                # Query repository details to get repository name
                                repo_url = f'https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo_id}?api-version=7.1'
                                repo_response = await client.get(
                                    repo_url, headers=headers
                                )
                                repo_response.raise_for_status()

                                repo_data = repo_response.json()
                                repo_name = repo_data.get('name', '')

                                if repo_name:
                                    # Format: org/project/repo
                                    repository = f'{organization}/{project}/{repo_name}'
                                    return repository, branch

                return None, None

        except httpx.HTTPStatusError as e:
            logger.error(
                f'[Azure DevOps] HTTP error fetching work item repository: '
                f'{e.response.status_code} - {e.response.text}'
            )
            return None, None
        except Exception as e:
            logger.error(
                f'[Azure DevOps] Error fetching work item repository: {e}',
                exc_info=True,
            )
            return None, None

    async def _get_project_repositories(
        self, organization: str, project: str, project_name: str, token: str
    ) -> list[tuple[str, str]]:
        """Fetch all repositories from a project.

        Args:
            organization: Azure DevOps organization name
            project: Project ID or name for API calls
            project_name: Project name for repository string format
            token: Service principal access token

        Returns:
            List of tuples (full_repo_name, repo_name) for all repositories in the project
        """
        try:
            import httpx

            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            }

            # List all repositories in the project
            url = f'https://dev.azure.com/{organization}/{project}/_apis/git/repositories?api-version=7.1'

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                repos_data = response.json()
                repos = repos_data.get('value', [])

                result = []
                for repo in repos:
                    repo_name = repo.get('name', '')
                    if repo_name:
                        full_repo_name = f'{organization}/{project_name}/{repo_name}'
                        result.append((full_repo_name, repo_name))

                return result

        except httpx.HTTPStatusError as e:
            logger.error(
                f'[Azure DevOps] HTTP error fetching project repositories: '
                f'{e.response.status_code} - {e.response.text}'
            )
            return []
        except Exception as e:
            logger.error(
                f'[Azure DevOps] Error fetching project repositories: {e}',
                exc_info=True,
            )
            return []

    async def _send_repo_selection_comment(
        self,
        organization: str,
        project_name: str,
        work_item_id: int,
        token: str,
        repos: list[tuple[str, str]],
    ):
        """Send a comment asking the user to specify which repository to use.

        Args:
            organization: Azure DevOps organization name
            project_name: Project name
            work_item_id: Work item ID
            token: Service principal access token
            repos: List of (full_repo_name, repo_name) tuples
        """
        try:
            import httpx

            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            }

            # Build the comment message
            if repos:
                repo_list = '\n'.join([f'- {repo_name}' for _, repo_name in repos[:10]])
                comment_text = (
                    'I need to know which repository to work with. '
                    f'Available repositories in this project:\n\n{repo_list}\n\n'
                    'To proceed, either:\n'
                    '1. Link a repository to this work item in the Development section and mention @openhands in a comment, OR\n'
                    '2. Reply to this comment with @openhands and the repository name'
                )
            else:
                comment_text = (
                    'I need to know which repository to work with, but no repositories '
                    'were found in this project. Please create a repository and then '
                    'mention @openhands in a comment to start the job.'
                )

            # Add comment to work item using the update API
            url = f'https://dev.azure.com/{organization}/{project_name}/_apis/wit/workitems/{work_item_id}?api-version=7.1'

            # Work item update uses JSON Patch format
            update_payload = [
                {
                    'op': 'add',
                    'path': '/fields/System.History',
                    'value': comment_text,
                }
            ]

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.patch(
                    url,
                    headers={**headers, 'Content-Type': 'application/json-patch+json'},
                    json=update_payload,
                )
                response.raise_for_status()

            logger.info(
                f'[Azure DevOps] Sent repository selection comment to work item {work_item_id}'
            )

        except Exception as e:
            logger.error(
                f'[Azure DevOps] Error sending repo selection comment: {e}',
                exc_info=True,
            )

    def create_outgoing_message(self, msg: str | dict, ephemeral: bool = False):
        """Create an outgoing message."""
        return Message(source=SourceType.OPENHANDS, message=msg, ephemeral=ephemeral)

    async def send_message(
        self, message: Message, azure_devops_view: AzureDevOpsViewType
    ):
        """Send a message to Azure DevOps work item or PR.

        Args:
            message: The message to send
            azure_devops_view: The view containing context about where to send the message
        """
        # Get service principal token
        try:
            sp_token = (
                await self.token_manager.get_azure_devops_service_principal_token()
            )

            from integrations.azure_devops.azure_devops_service import (
                SaaSAzureDevOpsService,
            )

            service: SaaSAzureDevOpsService = AzureDevOpsService(
                token=SecretStr(sp_token)
            )

            outgoing_message = message.message

            if isinstance(azure_devops_view, AzureDevOpsWorkItem):
                # Send comment to work item
                await service.add_work_item_comment(
                    azure_devops_view.full_repo_name,
                    azure_devops_view.work_item_id,
                    outgoing_message,
                )

            elif isinstance(azure_devops_view, AzureDevOpsPRComment):
                # Send comment to PR
                if azure_devops_view.thread_id:
                    # Reply in the original thread
                    await service.add_pr_comment_to_thread(
                        azure_devops_view.full_repo_name,
                        azure_devops_view.pr_id,
                        azure_devops_view.thread_id,
                        outgoing_message,
                    )
                else:
                    # Create new thread for general discussion
                    await service.add_pr_thread(
                        azure_devops_view.full_repo_name,
                        azure_devops_view.pr_id,
                        outgoing_message,
                    )

            else:
                logger.warning(
                    '[Azure DevOps] Unsupported view type for sending message'
                )
                return

        except Exception as e:
            logger.exception(f'[Azure DevOps] Failed to send message: {e}')

    async def start_job(self, azure_devops_view: AzureDevOpsViewType):
        """Kick off a job with openhands agent.

        Args:
            azure_devops_view: The view containing context about the Azure DevOps trigger
        """
        from server.conversation_callback_processor.azure_devops_callback_processor import (
            AzureDevOpsCallbackProcessor,
        )

        try:
            msg_info = None

            try:
                user_info = azure_devops_view.user_info
                logger.info(
                    f'[Azure DevOps] Starting job for user {user_info.username} (id={user_info.keycloak_user_id})'
                )

                # Get service principal token for API operations
                # Service principal acts as the application, not on behalf of any user
                try:
                    sp_token = await self.token_manager.get_azure_devops_service_principal_token()
                except Exception as e:
                    logger.error(
                        f'[Azure DevOps] Failed to get service principal token: {e}'
                    )
                    raise MissingSettingsError('Service principal token not configured')

                # Create secret store with service principal token
                # Agent will use this token for all API operations
                secret_store = Secrets(
                    provider_tokens=MappingProxyType(
                        {
                            ProviderType.AZURE_DEVOPS: ProviderToken(
                                token=SecretStr(sp_token),
                                user_id='service_principal',  # Not a real user ID
                            )
                        }
                    )
                )

                # Initialize conversation and generate solvability report
                convo_metadata = await azure_devops_view.initialize_new_conversation()
                solvability_summary = None
                try:
                    solvability_summary = await summarize_azure_devops_solvability(
                        azure_devops_view, sp_token
                    )
                except Exception as e:
                    logger.warning(
                        f'[Azure DevOps]: Error summarizing solvability: {str(e)}'
                    )

                # Get saas_user_auth for V1 conversations
                saas_user_auth = await get_saas_user_auth(
                    azure_devops_view.user_info.keycloak_user_id, self.token_manager
                )

                await azure_devops_view.create_new_conversation(
                    self.jinja_env,
                    secret_store.provider_tokens,
                    convo_metadata,
                    saas_user_auth,
                )

                conversation_id = azure_devops_view.conversation_id

                # Only register V0 callback processor if not using V1
                # V1 callback processor is registered during conversation creation
                if not azure_devops_view.v1_enabled:
                    processor = AzureDevOpsCallbackProcessor(
                        azure_devops_view=azure_devops_view,
                        send_summary_instruction=True,
                    )
                    register_callback_processor(conversation_id, processor)

                # Send message with conversation link
                conversation_link = CONVERSATION_URL.format(conversation_id)
                base_msg = f"I'm on it! {user_info.username} can [track my progress at all-hands.dev]({conversation_link})"

                # Combine messages: include solvability report with "I'm on it!" if successful
                if solvability_summary:
                    msg_info = f'{base_msg}\n\n{solvability_summary}'
                else:
                    msg_info = base_msg

            except MissingSettingsError as e:
                logger.warning(f'[Azure DevOps] Missing settings error: {str(e)}')

                msg_info = f'@{user_info.username} Failed to start job: Service principal not configured. Please contact your administrator.'

            except LLMAuthenticationError as e:
                logger.warning(
                    f'[Azure DevOps] LLM authentication error for user {user_info.username}: {str(e)}'
                )

                msg_info = f'@{user_info.username} please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

            msg = self.create_outgoing_message(msg_info)
            await self.send_message(msg, azure_devops_view)

        except Exception:
            logger.exception('[Azure DevOps]: Error starting job')
            msg = self.create_outgoing_message(
                msg='Uh oh! There was an unexpected error starting the job :('
            )
            await self.send_message(msg, azure_devops_view)

        try:
            await self.data_collector.save_data(azure_devops_view)
        except Exception:
            logger.warning(
                '[Azure DevOps]: Error saving interaction data', exc_info=True
            )
