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
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.utils.conversation_callback_utils import register_callback_processor
from storage.azure_devops_webhook_store import AzureDevOpsWebhookStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl,
)
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.storage.data_models.secrets import Secrets
from openhands.utils.async_utils import call_sync_from_async


class AzureDevOpsManager:
    """Manager for Azure DevOps webhook events and interactions."""

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

        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'azure_devops')
        )

    @staticmethod
    def _strip_html_tags(html_text: str) -> str:
        """Strip HTML tags from text and convert to plain text.

        Azure DevOps stores descriptions in HTML format. This function:
        1. Removes all HTML tags
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
            import html
            import re

            # Decode HTML entities
            text = html.unescape(html_text)

            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)

            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)

            return text.strip()
        except Exception:
            return html_text

    async def _get_service_principal_id(self) -> str:
        """Get the service principal's Azure DevOps identity GUID.

        This ID is used to detect mentions in comments. When users mention
        the service principal in Azure DevOps UI (e.g., @OpenHands), it appears
        as @<GUID> in the webhook payload.

        Azure DevOps uses different IDs in different contexts:
        - memberId: Used in work item mentions
        - subjectId: Used in PR mentions and assignments

        Both are extracted from X-VSS-UserData header.

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

            # Create service instance and call get_user() to get identity
            sp_service = AzureDevOpsServiceImpl(token=SecretStr(sp_token))
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

    async def _lookup_keycloak_user_id(
        self, author_azure_id: str | None, organization: str
    ) -> str | None:
        """Look up Keycloak user ID from Azure DevOps author ID.

        Uses a 2-tier fallback strategy:
        1. Direct Keycloak attribute lookup (azure_devops_ids map)
        2. Email-based lookup via Identities API

        Args:
            author_azure_id: The Azure DevOps user ID (GUID)
            organization: The Azure DevOps organization name

        Returns:
            Keycloak user ID if found, None otherwise
        """
        keycloak_user_id = None

        if not author_azure_id:
            return None

        # Plan A: Try organization-specific lookup using azure_devops_ids map
        # This is the primary lookup mechanism after enrichment
        keycloak_user_id = await self.token_manager.get_user_id_from_azure_devops_id(
            str(author_azure_id), organization
        )

        if not keycloak_user_id:
            # Plan B: Fallback to email-based lookup via Identities API
            try:
                # Get service principal token for Identities API access
                sp_token_for_identities = (
                    await self.token_manager.get_azure_devops_service_principal_token()
                )

                if sp_token_for_identities:
                    # Resolve Azure DevOps ID -> email via Identities API
                    from .azure_devops_id_resolver import AzureDevOpsIdResolver

                    resolver = AzureDevOpsIdResolver(SecretStr(sp_token_for_identities))
                    author_email = await resolver.get_user_email_from_id(
                        str(author_azure_id), organization
                    )

                    if author_email:
                        # Lookup Keycloak user by email
                        from server.auth.keycloak_manager import get_keycloak_admin

                        keycloak_admin = get_keycloak_admin(external=True)
                        users = await keycloak_admin.a_get_users(
                            {'email': author_email}
                        )

                        if users and len(users) > 0:
                            keycloak_user_id = users[0]['id']

                            # Opportunistic enrichment: Update azure_devops_ids map
                            # Add this org -> ID mapping to the user's profile
                            try:
                                import json

                                user = users[0]
                                attributes = user.get('attributes', {})

                                # Get existing map or create new one
                                existing_ids_map = {}
                                if 'azure_devops_ids' in attributes:
                                    ids_json_list = attributes['azure_devops_ids']
                                    if ids_json_list and len(ids_json_list) > 0:
                                        try:
                                            existing_ids_map = json.loads(
                                                ids_json_list[0]
                                            )
                                        except json.JSONDecodeError:
                                            pass

                                # Add this organization's ID
                                existing_ids_map[organization] = str(author_azure_id)

                                # Update Keycloak
                                await keycloak_admin.a_update_user(
                                    keycloak_user_id,
                                    {
                                        'attributes': {
                                            'azure_devops_ids': [
                                                json.dumps(existing_ids_map)
                                            ]
                                        },
                                        'email': author_email,
                                    },
                                )
                                logger.info(
                                    f'[AzureDevOpsManager] Opportunistically enriched user {keycloak_user_id} '
                                    f'with Azure DevOps ID for org {organization}'
                                )
                            except Exception as e:
                                logger.warning(
                                    f'[AzureDevOpsManager] Failed opportunistic enrichment: {e}'
                                )
            except Exception as e:
                logger.warning(
                    f'[AzureDevOpsManager] Email-based lookup failed: {e}',
                    exc_info=True,
                )

        return keycloak_user_id

    def _confirm_incoming_source_type(self, message: Message):
        """Validate that the message is from Azure DevOps.

        Args:
            message: The incoming message

        Raises:
            ValueError: If message is not from Azure DevOps
        """
        if message.source != SourceType.AZURE_DEVOPS:
            raise ValueError(f'Expected Azure DevOps message, got {message.source}')

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

        # Get service principal ID for mention detection
        try:
            sp_id = await self._get_service_principal_id()
        except Exception:
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
                f'[Azure DevOps Manager]: Error processing payload for interaction data: {e}',
                exc_info=True,
            )

        # Extract payload
        payload = message.message if isinstance(message.message, dict) else {}

        # Check if this is a mention/tag that requires action
        is_job_requested = await self.is_job_requested(message)

        if is_job_requested:
            event_type = self.view.get_event_type(payload)

            # Handle different trigger types
            if event_type == AzureDevOpsEventType.WORKITEM_UPDATED:
                await self._handle_work_item_trigger(payload, event_type)
            elif event_type == AzureDevOpsEventType.PR_COMMENTED:
                await self._handle_pr_trigger(payload)
            else:
                logger.warning(
                    f'[Azure DevOps Manager] Unhandled event type: {event_type}'
                )

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

        # IMPORTANT: resource.id is the UPDATE ID, not the work item ID!
        # For workitem.updated webhooks, the URL is:
        # https://dev.azure.com/org/project/_apis/wit/workItems/{workItemId}/updates/{updateId}
        # resource.id = updateId (e.g., 49)
        # We need to extract workItemId from the URL (e.g., 1254)
        work_item_id = self._extract_work_item_id_from_url(url)

        if not work_item_id:
            logger.error(
                '[Azure DevOps Work Item Handler] No work item ID in payload - aborting'
            )
            return

        # Extract organization and project from URL for API calls
        organization = self._extract_organization_from_url(url)
        if not organization:
            logger.error(
                f'[Azure DevOps Work Item Handler] Could not extract organization from URL: {url} - aborting'
            )
            return

        # Extract project from URL (webhook fields only contain changed fields, not System.TeamProject)
        project = self._extract_project_from_url(url)
        if not project:
            logger.error(
                f'[Azure DevOps Work Item Handler] Could not extract project from URL: {url} - aborting'
            )
            return

        # Get service principal token for API operations (needed to fetch work item details)
        try:
            sp_token = (
                await self.token_manager.get_azure_devops_service_principal_token()
            )
        except Exception as e:
            logger.error(
                f'[Azure DevOps Work Item Handler] Failed to get service principal token: {e}',
                exc_info=True,
            )
            return

        # Fetch repository and branch from work item's development section
        repository, branch = await self._get_work_item_repository(
            organization, project, work_item_id, sp_token
        )

        repository_linked = bool(repository)
        if not repository:
            # No repository linked - follow GitHub pattern and let agent create new branch
            # Work item APIs only need org/project, so we use a placeholder for repo name
            repository = f'{organization}/{project}'

        # Extract comment/change author for conversation ownership
        # Service principal will be used for API authentication
        # For workitem.updated webhooks, the user is in resource.revisedBy
        revised_by = resource.get('revisedBy', {})
        author_azure_id = revised_by.get('id') if isinstance(revised_by, dict) else None
        username = (
            revised_by.get('displayName', 'User')
            if isinstance(revised_by, dict)
            else 'User'
        )

        # Try to look up author's keycloak user (for conversation ownership)
        # Following GitHub pattern: lookup can return None, handled gracefully in start_job
        keycloak_user_id = await self._lookup_keycloak_user_id(
            author_azure_id, organization
        )

        # Create UserData
        # keycloak_user_id is for conversation ownership
        # Service principal token will be used for API operations
        from integrations.types import UserData

        user_info = UserData(
            user_id=0,  # Not used for service principal
            username=username,  # Display name
            keycloak_user_id=keycloak_user_id,  # For conversation ownership
        )

        # Validate that user has an OpenHands account
        # Without a keycloak_user_id, the job will fail due to missing LLM settings
        if not keycloak_user_id:
            logger.warning(
                f'[Azure DevOps Work Item Handler] User {username} (Azure ID: {author_azure_id}) does not have an OpenHands account'
            )
            # Send error message to work item
            from integrations.models import Message

            temp_view = AzureDevOpsWorkItem(
                work_item_id=work_item_id,
                project_name=project,
                organization=organization,
                full_repo_name=repository,
                is_public_repo=False,
                user_info=user_info,
                raw_payload=Message(source=SourceType.AZURE_DEVOPS, message=payload),
                conversation_id='',
                uuid=None,
                should_extract=False,
                send_summary_instruction=False,
                title='',
                description='',
                previous_comments=[],
                work_item_type='',
                selected_branch=None,
                comment_body=None,
                repository_linked=False,
                installation_id=0,
                issue_number=work_item_id,
            )
            error_msg = self.create_outgoing_message(
                f'@{username} You need an OpenHands account to start a job. Please sign up at [{HOST_URL}]({HOST_URL}) and connect your Azure DevOps account.'
            )
            await self.send_message(error_msg, temp_view)
            return

        # Extract work item details
        # NOTE: For workitem.updated webhooks, fields only contains CHANGED fields
        # We need to fetch the full work item to get title, description, etc.
        title = fields.get('System.Title')
        description = self._strip_html_tags(fields.get('System.Description', ''))
        work_item_type = fields.get('System.WorkItemType')

        # If title/description not in webhook payload, fetch from API
        if not title or not description or not work_item_type:
            try:
                from integrations.azure_devops.azure_devops_service import (
                    SaaSAzureDevOpsService,
                )

                sp_service: SaaSAzureDevOpsService = AzureDevOpsServiceImpl(
                    token=SecretStr(sp_token)
                )

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
                    f'[Azure DevOps Work Item Handler] Failed to fetch full work item details: {e}',
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
            comments_data = await sp_service.get_work_item_comments(
                repository, work_item_id
            )
            comment_list = comments_data.get('comments', [])

            for comment in comment_list:
                from datetime import datetime

                previous_comments.append(
                    Comment(
                        id=str(comment.get('id', 0)),
                        author=comment.get('createdBy', {}).get(
                            'displayName', 'Unknown'
                        ),
                        body=self._strip_html_tags(comment.get('text', '')),
                        created_at=datetime.fromisoformat(
                            comment.get(
                                'createdDate', datetime.now().isoformat()
                            ).replace('Z', '+00:00')
                        ),
                        updated_at=datetime.fromisoformat(
                            comment.get(
                                'modifiedDate',
                                comment.get('createdDate', datetime.now().isoformat()),
                            ).replace('Z', '+00:00')
                        ),
                    )
                )
        except Exception as e:
            logger.warning(
                f'[Azure DevOps Work Item Handler] Failed to get work item comments: {e}',
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
            work_item_id=work_item_id,
            project_name=project,
            organization=organization,
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
            work_item_type=work_item_type,
            selected_branch=branch,  # Branch from work item development section
            comment_body=comment_body,  # Comment text when triggered by @mention
            repository_linked=repository_linked,  # Whether repository was found in development section
            # GitHub-specific fields required by base class (not applicable to Azure DevOps)
            installation_id=0,
            issue_number=work_item_id,
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
        keycloak_user_id = await self._lookup_keycloak_user_id(
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

        # Add reaction/acknowledgment using service principal
        try:
            # Get service instance with service principal token
            from integrations.azure_devops.azure_devops_service import (
                SaaSAzureDevOpsService,
            )

            sp_service: SaaSAzureDevOpsService = AzureDevOpsServiceImpl(
                token=SecretStr(sp_token)
            )

            await sp_service.add_pr_reaction(repository_str, pr_id)
        except Exception:
            pass  # Non-critical: reaction is just acknowledgment

        # Create UserData
        # keycloak_user_id is for conversation ownership
        # Service principal token will be used for API operations
        from integrations.types import UserData

        user_info = UserData(
            user_id=0,  # Not used for service principal
            username=username,  # Display name
            keycloak_user_id=keycloak_user_id,  # For conversation ownership
        )

        # Validate that user has an OpenHands account
        # Without a keycloak_user_id, the job will fail due to missing LLM settings
        if not keycloak_user_id:
            logger.warning(
                f'[Azure DevOps PR Handler] User {username} (Azure ID: {author_azure_id}) does not have an OpenHands account'
            )
            # Send error message to PR
            from integrations.models import Message

            temp_view = AzureDevOpsPRComment(
                pr_id=pr_id,
                project_name=project_name,
                organization=organization,
                repository_name=repo_name,
                full_repo_name=repository_str,
                is_public_repo=False,
                user_info=user_info,
                raw_payload=Message(source=SourceType.AZURE_DEVOPS, message=payload),
                conversation_id='',
                uuid=None,
                should_extract=False,
                send_summary_instruction=False,
                title='',
                description='',
                previous_comments=[],
                comment_body='',
                is_inline=False,
                thread_context=None,
                branch_name=None,
                installation_id=0,
                issue_number=pr_id,
            )
            error_msg = self.create_outgoing_message(
                f'@{username} You need an OpenHands account to start a job. Please sign up at [{HOST_URL}]({HOST_URL}) and connect your Azure DevOps account.'
            )
            await self.send_message(error_msg, temp_view)
            return

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

        # Extract the triggering comment text
        # Azure DevOps uses 'content' field for comment text
        comment_body = self._strip_html_tags(
            comment_data.get('content') or comment_data.get('text', '')
        )

        # Get previous comments (if any) using service principal
        from openhands.integrations.service_types import Comment

        previous_comments: list[Comment] = []
        try:
            comments_data = await sp_service.get_pr_comments(repository_str, pr_id)
            for thread in comments_data:
                for comment in thread.get('comments', []):
                    from datetime import datetime

                    previous_comments.append(
                        Comment(
                            id=str(comment.get('id', 0)),
                            author=comment.get('author', {}).get(
                                'displayName', 'Unknown'
                            ),
                            body=self._strip_html_tags(comment.get('content', '')),
                            created_at=datetime.fromisoformat(
                                comment.get(
                                    'publishedDate', datetime.now().isoformat()
                                ).replace('Z', '+00:00')
                            ),
                            updated_at=datetime.fromisoformat(
                                comment.get(
                                    'lastUpdatedDate',
                                    comment.get(
                                        'publishedDate', datetime.now().isoformat()
                                    ),
                                ).replace('Z', '+00:00')
                            ),
                        )
                    )
        except Exception as e:
            logger.warning(f'[Azure DevOps] Failed to get PR comments: {e}')

        # Create view instance
        from integrations.models import Message

        azure_devops_view = AzureDevOpsPRComment(
            pr_id=pr_id,
            project_name=project_name,
            organization=organization,
            repository_name=repo_name,
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
            comment_body=comment_body,  # The triggering comment text
            is_inline=is_inline,
            thread_context=thread_context,
            branch_name=branch_name,
            # GitHub-specific fields required by base class (not applicable to Azure DevOps)
            installation_id=0,
            issue_number=pr_id,
        )

        # Start job execution
        await self.start_job(azure_devops_view)

    @staticmethod
    def _extract_organization_from_url(url: str) -> str:
        """Extract organization name from Azure DevOps URL.

        Args:
            url: Azure DevOps URL (e.g., https://dev.azure.com/org/_apis/...)

        Returns:
            Organization name or empty string if not found
        """
        try:
            # URL format: https://dev.azure.com/{organization}/...
            if 'dev.azure.com/' in url:
                parts = url.split('dev.azure.com/')
                if len(parts) > 1:
                    org_parts = parts[1].split('/')
                    if org_parts:
                        return org_parts[0]
            return ''
        except Exception:
            return ''

    @staticmethod
    def _extract_project_from_url(url: str) -> str:
        """Extract project name/GUID from Azure DevOps URL.

        Args:
            url: Azure DevOps URL (e.g., https://dev.azure.com/org/project/_apis/...)

        Returns:
            Project name/GUID or empty string if not found
        """
        try:
            # URL format: https://dev.azure.com/{organization}/{project}/_apis/...
            if 'dev.azure.com/' in url:
                parts = url.split('dev.azure.com/')
                if len(parts) > 1:
                    path_parts = parts[1].split('/')
                    # path_parts[0] is organization, path_parts[1] is project
                    if len(path_parts) >= 2:
                        return path_parts[1]
            return ''
        except Exception:
            return ''

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
                                branch_part = parts[2]

                                # Extract branch name (remove GB prefix if present)
                                branch = (
                                    branch_part.replace('GB', '')
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

            service: SaaSAzureDevOpsService = AzureDevOpsServiceImpl(
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
                # Send comment to PR - create new thread for general discussion
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

                await azure_devops_view.create_new_conversation(
                    self.jinja_env, secret_store.provider_tokens, convo_metadata
                )

                conversation_id = azure_devops_view.conversation_id

                # Create and register callback processor
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
