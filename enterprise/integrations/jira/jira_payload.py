"""Centralized payload parsing for Jira webhooks.

This module provides a single source of truth for parsing and validating
Jira webhook payloads, replacing scattered parsing logic throughout the codebase.
"""

from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

from openhands.core.logger import openhands_logger as logger


class JiraEventType(Enum):
    """Types of Jira events we handle."""

    LABELED_TICKET = 'labeled_ticket'
    COMMENT_MENTION = 'comment_mention'


@dataclass(frozen=True)
class JiraWebhookPayload:
    """Normalized, validated representation of a Jira webhook payload.

    This immutable dataclass replaces JobContext and provides a single
    source of truth for all webhook data. All parsing happens in
    JiraPayloadParser, ensuring consistent validation.
    """

    event_type: JiraEventType
    raw_event: str  # Original webhookEvent value

    # Issue data
    issue_id: str
    issue_key: str

    # User data
    user_email: str
    display_name: str
    account_id: str

    # Workspace data (derived from issue self URL)
    workspace_name: str
    base_api_url: str

    # Event-specific data
    comment_body: str = ''  # For comment events

    @property
    def user_msg(self) -> str:
        """Alias for comment_body for backward compatibility."""
        return self.comment_body


class JiraPayloadParseError(Exception):
    """Raised when payload parsing fails."""

    def __init__(self, reason: str, event_type: str | None = None):
        self.reason = reason
        self.event_type = event_type
        super().__init__(reason)


@dataclass(frozen=True)
class JiraPayloadSuccess:
    """Result when parsing succeeds."""

    payload: JiraWebhookPayload


@dataclass(frozen=True)
class JiraPayloadSkipped:
    """Result when event is intentionally skipped."""

    skip_reason: str


@dataclass(frozen=True)
class JiraPayloadError:
    """Result when parsing fails due to invalid data."""

    error: str


JiraPayloadParseResult = JiraPayloadSuccess | JiraPayloadSkipped | JiraPayloadError


class JiraPayloadParser:
    """Centralized parser for Jira webhook payloads.

    This class provides a single entry point for parsing webhooks,
    determining event types, and extracting all necessary fields.
    Replaces scattered parsing in JiraFactory and JiraManager.
    """

    def __init__(self, oh_label: str, inline_oh_label: str):
        """Initialize parser with OpenHands label configuration.

        Args:
            oh_label: Label that triggers OpenHands (e.g., 'openhands')
            inline_oh_label: Mention that triggers OpenHands (e.g., '@openhands')
        """
        self.oh_label = oh_label
        self.inline_oh_label = inline_oh_label

    def parse(self, raw_payload: dict) -> JiraPayloadParseResult:
        """Parse a raw webhook payload into a normalized JiraWebhookPayload.

        Args:
            raw_payload: The raw webhook payload dict from Jira

        Returns:
            One of:
            - JiraPayloadSuccess: Valid, actionable event with payload
            - JiraPayloadSkipped: Event we intentionally don't process
            - JiraPayloadError: Malformed payload we expected to process
        """
        webhook_event = raw_payload.get('webhookEvent', '')

        logger.debug(
            '[Jira] Parsing webhook payload', extra={'webhook_event': webhook_event}
        )

        if webhook_event == 'jira:issue_updated':
            return self._parse_label_event(raw_payload, webhook_event)
        elif webhook_event == 'comment_created':
            return self._parse_comment_event(raw_payload, webhook_event)
        else:
            return JiraPayloadSkipped(f'Unhandled webhook event type: {webhook_event}')

    def _parse_label_event(
        self, payload: dict, webhook_event: str
    ) -> JiraPayloadParseResult:
        """Parse an issue_updated event for label changes."""
        changelog = payload.get('changelog', {})
        items = changelog.get('items', [])

        # Extract labels that were added
        labels = [
            item.get('toString', '')
            for item in items
            if item.get('field') == 'labels' and 'toString' in item
        ]

        if self.oh_label not in labels:
            return JiraPayloadSkipped(
                f"Label event does not contain '{self.oh_label}' label"
            )

        # For label events, user data comes from 'user' field
        user_data = payload.get('user', {})
        return self._extract_and_validate(
            payload=payload,
            user_data=user_data,
            event_type=JiraEventType.LABELED_TICKET,
            webhook_event=webhook_event,
            comment_body='',
        )

    def _parse_comment_event(
        self, payload: dict, webhook_event: str
    ) -> JiraPayloadParseResult:
        """Parse a comment_created event."""
        comment_data = payload.get('comment', {})
        comment_body = comment_data.get('body', '')

        if not self._has_mention(comment_body):
            return JiraPayloadSkipped(
                f"Comment does not mention '{self.inline_oh_label}'"
            )

        # For comment events, user data comes from 'comment.author'
        user_data = comment_data.get('author', {})
        return self._extract_and_validate(
            payload=payload,
            user_data=user_data,
            event_type=JiraEventType.COMMENT_MENTION,
            webhook_event=webhook_event,
            comment_body=comment_body,
        )

    def _has_mention(self, text: str) -> bool:
        """Check if text contains an exact mention of OpenHands."""
        from integrations.utils import has_exact_mention

        return has_exact_mention(text, self.inline_oh_label)

    def _extract_and_validate(
        self,
        payload: dict,
        user_data: dict,
        event_type: JiraEventType,
        webhook_event: str,
        comment_body: str,
    ) -> JiraPayloadParseResult:
        """Extract common fields and validate required data is present."""
        issue_data = payload.get('issue', {})

        # Extract all fields with empty string defaults (makes them str type)
        issue_id = issue_data.get('id', '')
        issue_key = issue_data.get('key', '')
        user_email = user_data.get('emailAddress', '')
        display_name = user_data.get('displayName', '')
        account_id = user_data.get('accountId', '')
        base_api_url, workspace_name = self._extract_workspace_from_url(
            issue_data.get('self', '')
        )

        # Validate required fields
        missing: list[str] = []
        if not issue_id:
            missing.append('issue.id')
        if not issue_key:
            missing.append('issue.key')
        if not user_email:
            missing.append('user.emailAddress')
        if not display_name:
            missing.append('user.displayName')
        if not account_id:
            missing.append('user.accountId')
        if not workspace_name:
            missing.append('workspace_name (derived from issue.self)')
        if not base_api_url:
            missing.append('base_api_url (derived from issue.self)')

        if missing:
            return JiraPayloadError(f"Missing required fields: {', '.join(missing)}")

        return JiraPayloadSuccess(
            JiraWebhookPayload(
                event_type=event_type,
                raw_event=webhook_event,
                issue_id=issue_id,
                issue_key=issue_key,
                user_email=user_email,
                display_name=display_name,
                account_id=account_id,
                workspace_name=workspace_name,
                base_api_url=base_api_url,
                comment_body=comment_body,
            )
        )

    def _extract_workspace_from_url(self, self_url: str) -> tuple[str, str]:
        """Extract base API URL and workspace name from issue self URL.

        Args:
            self_url: The 'self' URL from the issue data

        Returns:
            Tuple of (base_api_url, workspace_name)
        """
        if not self_url:
            return '', ''

        # Extract base URL (everything before /rest/)
        if '/rest/' in self_url:
            base_api_url = self_url.split('/rest/')[0]
        else:
            parsed = urlparse(self_url)
            base_api_url = f'{parsed.scheme}://{parsed.netloc}'

        # Extract workspace name (hostname)
        parsed = urlparse(base_api_url)
        workspace_name = parsed.hostname or ''

        return base_api_url, workspace_name
