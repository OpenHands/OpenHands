"""
Tests for JiraPayloadParser.

These tests verify the parsing behavior of Jira webhook payloads,
including the handling of optional fields like user_email which
may not be present in webhook payloads from Jira.
"""

import pytest
from integrations.jira.jira_payload import (
    JiraEventType,
    JiraPayloadError,
    JiraPayloadParser,
    JiraPayloadSkipped,
    JiraPayloadSuccess,
)


@pytest.fixture
def parser():
    """Create a JiraPayloadParser with standard OpenHands labels."""
    return JiraPayloadParser(oh_label='openhands', inline_oh_label='@openhands')


@pytest.fixture
def valid_label_payload():
    """Create a valid jira:issue_updated payload with OpenHands label."""
    return {
        'webhookEvent': 'jira:issue_updated',
        'issue': {
            'id': '12345',
            'key': 'TEST-123',
            'self': 'https://test.atlassian.net/rest/api/2/issue/12345',
        },
        'user': {
            'displayName': 'Test User',
            'accountId': 'account-123',
            'emailAddress': 'test@example.com',
        },
        'changelog': {
            'items': [
                {
                    'field': 'labels',
                    'toString': 'openhands',
                }
            ]
        },
    }


@pytest.fixture
def valid_comment_payload():
    """Create a valid comment_created payload with OpenHands mention."""
    return {
        'webhookEvent': 'comment_created',
        'issue': {
            'id': '12345',
            'key': 'TEST-123',
            'self': 'https://test.atlassian.net/rest/api/2/issue/12345',
        },
        'comment': {
            'body': '@openhands please fix this bug',
            'author': {
                'displayName': 'Test User',
                'accountId': 'account-123',
                'emailAddress': 'test@example.com',
            },
        },
    }


class TestUserEmailOptional:
    """Tests verifying user_email is optional in webhook payloads.

    Jira webhooks may not include emailAddress in the user data.
    The parser should accept payloads without this field.
    """

    def test_label_event_succeeds_without_email_address(
        self, parser, valid_label_payload
    ):
        """Verify label event parsing succeeds when emailAddress is missing."""
        # Arrange - remove emailAddress from user data
        del valid_label_payload['user']['emailAddress']

        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.user_email == ''
        assert result.payload.display_name == 'Test User'
        assert result.payload.account_id == 'account-123'

    def test_comment_event_succeeds_without_email_address(
        self, parser, valid_comment_payload
    ):
        """Verify comment event parsing succeeds when emailAddress is missing."""
        # Arrange - remove emailAddress from author data
        del valid_comment_payload['comment']['author']['emailAddress']

        # Act
        result = parser.parse(valid_comment_payload)

        # Assert
        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.user_email == ''
        assert result.payload.display_name == 'Test User'
        assert result.payload.account_id == 'account-123'

    def test_user_email_preserved_when_present(self, parser, valid_label_payload):
        """Verify user_email is captured when emailAddress is present."""
        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.user_email == 'test@example.com'


class TestRequiredFieldValidation:
    """Tests verifying required fields are still validated."""

    def test_missing_issue_id_returns_error(self, parser, valid_label_payload):
        """Verify parsing fails when issue.id is missing."""
        # Arrange
        del valid_label_payload['issue']['id']

        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadError)
        assert 'issue.id' in result.error

    def test_missing_issue_key_returns_error(self, parser, valid_label_payload):
        """Verify parsing fails when issue.key is missing."""
        # Arrange
        del valid_label_payload['issue']['key']

        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadError)
        assert 'issue.key' in result.error

    def test_missing_display_name_returns_error(self, parser, valid_label_payload):
        """Verify parsing fails when user.displayName is missing."""
        # Arrange
        del valid_label_payload['user']['displayName']

        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadError)
        assert 'displayName' in result.error

    def test_missing_account_id_returns_error(self, parser, valid_label_payload):
        """Verify parsing fails when user.accountId is missing."""
        # Arrange
        del valid_label_payload['user']['accountId']

        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadError)
        assert 'accountId' in result.error

    def test_missing_issue_self_url_returns_error(self, parser, valid_label_payload):
        """Verify parsing fails when issue.self URL is missing."""
        # Arrange
        del valid_label_payload['issue']['self']

        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadError)
        assert 'workspace_name' in result.error or 'base_api_url' in result.error


class TestEventTypeDetection:
    """Tests for webhook event type detection."""

    def test_issue_updated_with_label_returns_labeled_ticket(
        self, parser, valid_label_payload
    ):
        """Verify jira:issue_updated with label is detected as LABELED_TICKET."""
        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.event_type == JiraEventType.LABELED_TICKET

    def test_comment_created_with_mention_returns_comment_mention(
        self, parser, valid_comment_payload
    ):
        """Verify comment_created with mention is detected as COMMENT_MENTION."""
        # Act
        result = parser.parse(valid_comment_payload)

        # Assert
        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.event_type == JiraEventType.COMMENT_MENTION

    def test_unhandled_event_type_returns_skipped(self, parser):
        """Verify unknown event types are skipped."""
        # Arrange
        payload = {'webhookEvent': 'jira:issue_deleted'}

        # Act
        result = parser.parse(payload)

        # Assert
        assert isinstance(result, JiraPayloadSkipped)
        assert 'Unhandled' in result.skip_reason


class TestLabelFiltering:
    """Tests for OpenHands label filtering."""

    def test_label_event_without_openhands_label_skipped(
        self, parser, valid_label_payload
    ):
        """Verify label events without OpenHands label are skipped."""
        # Arrange - change label to something else
        valid_label_payload['changelog']['items'][0]['toString'] = 'other-label'

        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadSkipped)
        assert 'openhands' in result.skip_reason


class TestCommentFiltering:
    """Tests for OpenHands comment mention filtering."""

    def test_comment_without_mention_skipped(self, parser, valid_comment_payload):
        """Verify comments without OpenHands mention are skipped."""
        # Arrange - remove mention from comment body
        valid_comment_payload['comment']['body'] = 'Please fix this bug'

        # Act
        result = parser.parse(valid_comment_payload)

        # Assert
        assert isinstance(result, JiraPayloadSkipped)
        assert '@openhands' in result.skip_reason


class TestWorkspaceExtraction:
    """Tests for workspace name extraction from issue URL."""

    def test_workspace_name_extracted_from_self_url(self, parser, valid_label_payload):
        """Verify workspace name is extracted from issue self URL."""
        # Act
        result = parser.parse(valid_label_payload)

        # Assert
        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.workspace_name == 'test.atlassian.net'
        assert result.payload.base_api_url == 'https://test.atlassian.net'
