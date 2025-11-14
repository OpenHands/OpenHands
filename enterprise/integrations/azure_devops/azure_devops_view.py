"""Parse and validate Azure DevOps webhook payloads."""

import re
from typing import Any, Optional

from integrations.azure_devops.azure_devops_types import (
    AzureDevOpsEventType,
    PullRequestCommentedPayload,
)
from integrations.models import Message

from openhands.core.logger import openhands_logger as logger

# Azure DevOps mention pattern
# When mentioning service principal in Azure DevOps UI, it appears as @<GUID> in webhook payload
# Pattern: @<XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX>
# Any GUID mention sent to our webhook endpoint is assumed to be the OpenHands service principal
AZURE_MENTION_PATTERN = re.compile(
    r'@<[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}>'
)


class AzureDevOpsView:
    """View model for Azure DevOps webhook payloads."""

    @staticmethod
    def parse_pr_commented(payload: dict) -> Optional[PullRequestCommentedPayload]:
        """Parse git.pullrequest.commented webhook payload.

        Args:
            payload: Raw webhook payload dictionary

        Returns:
            Typed payload or None if invalid
        """
        try:
            event_type = payload.get('eventType')
            if event_type != AzureDevOpsEventType.PR_COMMENTED:
                logger.warning(
                    f'Expected git.pullrequest.commented event, got {event_type}'
                )
                return None

            # Validate required fields
            resource = payload.get('resource', {})
            if not resource.get('pullRequestId'):
                logger.warning('Missing pull request ID in payload')
                return None

            return payload  # type: ignore

        except Exception as e:
            logger.exception(f'Error parsing PR commented payload: {str(e)}')
            return None

    @staticmethod
    def extract_pr_info(payload: PullRequestCommentedPayload) -> dict[str, Any]:
        """Extract key information from PR commented payload.

        Args:
            payload: Parsed PR commented payload

        Returns:
            Dictionary with extracted information
        """
        resource = payload['resource']
        containers = payload['resourceContainers']
        repository = resource.get('repository', {})

        return {
            'pull_request_id': resource['pullRequestId'],
            'repository_id': repository.get('id', ''),
            'repository_name': repository.get('name', ''),
            'project': repository.get('project', {}).get('name', ''),
            'title': resource.get('title', ''),
            'description': resource.get('description', ''),
            'account_id': containers.get('account', {}).get('id', ''),
            'url': resource.get('url', ''),
        }

    @staticmethod
    def get_event_type(payload: dict) -> Optional[AzureDevOpsEventType]:
        """Get the event type from a webhook payload.

        Args:
            payload: Raw webhook payload dictionary

        Returns:
            Event type enum or None if invalid
        """
        event_type_str = payload.get('eventType')
        try:
            return AzureDevOpsEventType(event_type_str)
        except ValueError:
            logger.warning(f'Unknown Azure DevOps event type: {event_type_str}')
            return None


class AzureDevOpsFactory:
    """Factory methods for detecting different Azure DevOps webhook event types."""

    @staticmethod
    def is_assigned_work_item(
        message: Message, service_principal_id: str | None = None
    ) -> bool:
        """Check if a work item is assigned to the OpenHands service principal.

        When a user assigns a work item to OpenHands in the Azure DevOps UI
        (via the "Assigned To" dropdown), the webhook payload includes the
        assigned identity in System.AssignedTo field.

        Args:
            message: The webhook message
            service_principal_id: Optional service principal GUID to check for assignment

        Returns:
            True if work item is assigned to service principal, False otherwise
        """
        payload = message.message if isinstance(message.message, dict) else {}
        event_type = payload.get('eventType')

        if event_type != AzureDevOpsEventType.WORKITEM_UPDATED:
            return False

        # Check if work item is assigned to service principal
        resource = payload.get('resource', {})
        fields = resource.get('fields', {})

        # Log what fields are actually in the webhook payload
        logger.debug(
            f'[is_assigned_work_item] Webhook fields keys: {list(fields.keys())}'
        )

        # System.AssignedTo in fields is a change object with oldValue and newValue
        # Format: {'oldValue': 'User <guid>', 'newValue': 'OpenHands <guid>'}
        assigned_to_change = fields.get('System.AssignedTo')

        if not assigned_to_change:
            logger.debug(
                '[is_assigned_work_item] No System.AssignedTo field in webhook, returning False'
            )
            return False

        logger.debug(
            f'[is_assigned_work_item] System.AssignedTo field: {assigned_to_change}'
        )

        # Check if this is an assignment change (has newValue)
        new_value = assigned_to_change.get('newValue')
        if not new_value:
            logger.debug(
                '[is_assigned_work_item] No newValue in System.AssignedTo, returning False'
            )
            return False

        logger.debug(
            f'[is_assigned_work_item] Trying to extract GUID from newValue: {new_value}'
        )

        # Extract GUID from newValue string format: "DisplayName <GUID>"
        # Use regex to extract GUID from angle brackets
        import re

        guid_match = re.search(r'<([a-fA-F0-9\-]+)>', new_value)

        if not guid_match:
            logger.warning(
                f'[is_assigned_work_item] Could not extract GUID from newValue: {new_value}'
            )
            return False

        assigned_id = guid_match.group(1).upper()

        # If service principal ID is provided, check if assignment matches
        if service_principal_id:
            # Handle comma-separated IDs (memberId,subjectId)
            sp_ids = [id.strip().upper() for id in service_principal_id.split(',')]
            if assigned_id in sp_ids:
                logger.info(
                    f'[is_assigned_work_item] Work item assigned to service principal (ID: {assigned_id})'
                )
                return True
            return False

        # Fallback: if no service principal ID provided, accept any assignment
        # (This case should not happen in production)
        logger.warning(
            '[Azure DevOps] No service principal ID provided, cannot verify assignment'
        )
        return False

    @staticmethod
    def is_pr_comment(
        message: Message, inline: bool = False, service_principal_id: str | None = None
    ) -> bool:
        """Check if a PR was commented on with service principal mention.

        Args:
            message: The webhook message
            inline: If True, check for inline (code review) comments.
                   If False, check for general PR comments.
            service_principal_id: Optional service principal GUID to check for mentions

        Returns:
            True if PR comment contains service principal mention, False otherwise
        """
        payload = message.message if isinstance(message.message, dict) else {}
        event_type = payload.get('eventType')

        if event_type != AzureDevOpsEventType.PR_COMMENTED:
            return False

        # Get comment from payload
        resource = payload.get('resource', {})
        comment_data = resource.get('comment', {})

        # Check if this is an inline comment based on threadContext
        thread_context = comment_data.get('threadContext')
        is_inline_comment = thread_context is not None

        # Filter based on inline parameter
        if inline and not is_inline_comment:
            return False
        if not inline and is_inline_comment:
            return False

        # Check for service principal mention
        # Azure DevOps uses "content" for the comment text
        comment_text = comment_data.get('content') or comment_data.get('text', '')

        # If service principal ID is provided, do direct string matching
        if service_principal_id:
            # Handle comma-separated IDs (memberId,subjectId)
            sp_ids = [id.strip().upper() for id in service_principal_id.split(',')]

            # Check if any of the IDs are mentioned as @<ID>
            for sp_id in sp_ids:
                mention_pattern = f'@<{sp_id}>'
                if mention_pattern.upper() in comment_text.upper():
                    return True

            return False

        # Fallback: if no service principal ID provided, check for any GUID mention using regex
        guid_matches = AZURE_MENTION_PATTERN.findall(comment_text)
        has_mention = bool(guid_matches)
        return has_mention

    @staticmethod
    def is_inline_pr_comment(
        message: Message, service_principal_id: str | None = None
    ) -> bool:
        """Check if an inline PR comment (code review comment) contains service principal mention.

        Inline comments are comments on specific lines of code in a PR.
        They include threadContext with file path and line position.

        Args:
            message: The webhook message
            service_principal_id: Optional service principal GUID to check for mentions

        Returns:
            True if inline PR comment contains service principal mention, False otherwise
        """
        return AzureDevOpsFactory.is_pr_comment(
            message, inline=True, service_principal_id=service_principal_id
        )

    @staticmethod
    def is_work_item_comment(
        message: Message, service_principal_id: str | None = None
    ) -> bool:
        """Check if a work item comment contains service principal mention.

        When a user adds a comment to a work item in Azure DevOps UI,
        the webhook payload includes the comment text in System.History field.

        Work item mentions use HTML format:
        <a href="#" data-vss-mention="version:2.0,{guid}">@DisplayName</a>

        The GUID in mentions is the memberId, not the subjectId.

        Args:
            message: The webhook message
            service_principal_id: Service principal IDs (can be comma-separated: "memberId,subjectId")

        Returns:
            True if work item comment contains service principal mention, False otherwise
        """
        payload = message.message if isinstance(message.message, dict) else {}
        event_type = payload.get('eventType')

        if event_type != AzureDevOpsEventType.WORKITEM_UPDATED:
            return False

        # Service principal ID is required
        if not service_principal_id:
            logger.warning(
                '[is_work_item_comment] No service principal ID provided, cannot verify mention'
            )
            return False

        # Check if this update includes a comment (System.History field)
        resource = payload.get('resource', {})
        fields = resource.get('fields', {})

        # System.History contains the comment text when a comment is added
        history_change = fields.get('System.History')

        if not history_change:
            logger.debug(
                '[is_work_item_comment] No System.History field, returning False'
            )
            return False

        # System.History is a change object with newValue containing the comment
        comment_text = history_change.get('newValue', '')

        if not comment_text:
            logger.debug(
                '[is_work_item_comment] No newValue in System.History, returning False'
            )
            return False

        logger.debug(f'[is_work_item_comment] Comment text: {comment_text[:100]}...')

        # Parse HTML to extract GUIDs from data-vss-mention attributes
        # Format: <a href="#" data-vss-mention="version:2.0,{guid}">@DisplayName</a>
        import re

        mention_pattern = re.compile(r'data-vss-mention="[^"]*,([a-fA-F0-9\-]+)"')
        guid_matches = mention_pattern.findall(comment_text)

        if not guid_matches:
            logger.debug(
                '[is_work_item_comment] No data-vss-mention GUIDs found in comment'
            )
            return False

        logger.debug(
            f'[is_work_item_comment] Found {len(guid_matches)} mention(s) in comment: {guid_matches}'
        )

        # Parse service principal IDs (can be comma-separated: "memberId,subjectId")
        sp_ids = [id.strip().upper() for id in service_principal_id.split(',')]
        logger.debug(
            f'[is_work_item_comment] Checking against service principal IDs: {sp_ids}'
        )

        # Check if any service principal ID is mentioned
        for guid in guid_matches:
            guid_upper = guid.upper()
            if guid_upper in sp_ids:
                logger.info(
                    f'[is_work_item_comment] Found service principal mention (matched {guid_upper})'
                )
                return True

        logger.debug(
            f'[is_work_item_comment] Service principal IDs {sp_ids} not found in mentions'
        )
        return False
