"""
Tests for Jira view classes and factory.
"""

from unittest.mock import AsyncMock, patch

import pytest
from integrations.jira.jira_types import StartingConvoException
from integrations.jira.jira_view import (
    JiraFactory,
    JiraNewConversationView,
)
from integrations.models import Message, SourceType


class TestJiraNewConversationView:
    """Tests for JiraNewConversationView"""

    def test_get_instructions(self, new_conversation_view, mock_jinja_env):
        """Test _get_instructions method"""
        instructions, user_msg = new_conversation_view._get_instructions(mock_jinja_env)

        assert instructions == 'Test Jira instructions template'
        assert 'TEST-123' in user_msg
        assert 'Test Issue' in user_msg
        assert 'Fix this bug @openhands' in user_msg

    @patch('integrations.jira.jira_view.create_new_conversation')
    @patch('integrations.jira.jira_view.integration_store')
    async def test_create_or_update_conversation_success(
        self,
        mock_store,
        mock_create_conversation,
        new_conversation_view,
        mock_jinja_env,
        mock_agent_loop_info,
    ):
        """Test successful conversation creation"""
        mock_create_conversation.return_value = mock_agent_loop_info
        mock_store.create_conversation = AsyncMock()

        result = await new_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result == 'conv-123'
        mock_create_conversation.assert_called_once()
        mock_store.create_conversation.assert_called_once()

    async def test_create_or_update_conversation_no_repo(
        self, new_conversation_view, mock_jinja_env
    ):
        """Test conversation creation without selected repo"""
        new_conversation_view.selected_repo = None

        with pytest.raises(StartingConvoException, match='No repository selected'):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    @patch('integrations.jira.jira_view.create_new_conversation')
    async def test_create_or_update_conversation_failure(
        self, mock_create_conversation, new_conversation_view, mock_jinja_env
    ):
        """Test conversation creation failure"""
        mock_create_conversation.side_effect = Exception('Creation failed')

        with pytest.raises(
            StartingConvoException, match='Failed to create conversation'
        ):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    def test_get_response_msg(self, new_conversation_view):
        """Test get_response_msg method"""
        response = new_conversation_view.get_response_msg()

        assert "I'm on it!" in response
        assert 'Test User' in response
        assert 'track my progress here' in response
        assert 'conv-123' in response


class TestJiraFactory:
    """Tests for JiraFactory"""

    @patch('integrations.jira.jira_view.integration_store')
    async def test_create_jira_view_from_payload_new_conversation(
        self,
        mock_store,
        sample_job_context,
        sample_user_auth,
        sample_jira_user,
        sample_jira_workspace,
    ):
        """Test factory creating new conversation view"""
        mock_store.get_user_conversations_by_issue_id = AsyncMock(return_value=None)

        view = await JiraFactory.create_jira_view_from_payload(
            sample_job_context,
            sample_user_auth,
            sample_jira_user,
            sample_jira_workspace,
        )

        assert isinstance(view, JiraNewConversationView)
        assert view.conversation_id == ''

    async def test_create_jira_view_from_payload_no_user(
        self, sample_job_context, sample_user_auth, sample_jira_workspace
    ):
        """Test factory with no Jira user"""
        with pytest.raises(StartingConvoException, match='User not authenticated'):
            await JiraFactory.create_jira_view_from_payload(
                sample_job_context,
                sample_user_auth,
                None,
                sample_jira_workspace,  # type: ignore
            )

    async def test_create_jira_view_from_payload_no_auth(
        self, sample_job_context, sample_jira_user, sample_jira_workspace
    ):
        """Test factory with no SaaS auth"""
        with pytest.raises(StartingConvoException, match='User not authenticated'):
            await JiraFactory.create_jira_view_from_payload(
                sample_job_context,
                None,
                sample_jira_user,
                sample_jira_workspace,  # type: ignore
            )

    async def test_create_jira_view_from_payload_no_workspace(
        self, sample_job_context, sample_user_auth, sample_jira_user
    ):
        """Test factory with no workspace"""
        with pytest.raises(StartingConvoException, match='User not authenticated'):
            await JiraFactory.create_jira_view_from_payload(
                sample_job_context,
                sample_user_auth,
                sample_jira_user,
                None,  # type: ignore
            )


class TestJiraViewEdgeCases:
    """Tests for edge cases and error scenarios"""

    @patch('integrations.jira.jira_view.create_new_conversation')
    @patch('integrations.jira.jira_view.integration_store')
    async def test_conversation_creation_with_no_user_secrets(
        self,
        mock_store,
        mock_create_conversation,
        new_conversation_view,
        mock_jinja_env,
        mock_agent_loop_info,
    ):
        """Test conversation creation when user has no secrets"""
        new_conversation_view.saas_user_auth.get_secrets.return_value = None
        mock_create_conversation.return_value = mock_agent_loop_info
        mock_store.create_conversation = AsyncMock()

        result = await new_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result == 'conv-123'
        # Verify create_new_conversation was called with custom_secrets=None
        call_kwargs = mock_create_conversation.call_args[1]
        assert call_kwargs['custom_secrets'] is None

    @patch('integrations.jira.jira_view.create_new_conversation')
    @patch('integrations.jira.jira_view.integration_store')
    async def test_conversation_creation_store_failure(
        self,
        mock_store,
        mock_create_conversation,
        new_conversation_view,
        mock_jinja_env,
        mock_agent_loop_info,
    ):
        """Test conversation creation when store creation fails"""
        mock_create_conversation.return_value = mock_agent_loop_info
        mock_store.create_conversation = AsyncMock(side_effect=Exception('Store error'))

        with pytest.raises(
            StartingConvoException, match='Failed to create conversation'
        ):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    def test_new_conversation_view_attributes(self, new_conversation_view):
        """Test new conversation view attribute access"""
        assert new_conversation_view.job_context.issue_key == 'TEST-123'
        assert new_conversation_view.selected_repo == 'test/repo1'
        assert new_conversation_view.conversation_id == 'conv-123'


class TestJiraFactoryIsLabeledTicket:
    """Parameterized tests for JiraFactory.is_labeled_ticket method."""

    @pytest.mark.parametrize(
        'payload,expected',
        [
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [{'field': 'labels', 'toString': 'openhands'}]
                    },
                },
                True,
                id='issue_updated_with_openhands_label',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [
                            {'field': 'labels', 'toString': 'bug'},
                            {'field': 'labels', 'toString': 'openhands'},
                        ]
                    },
                },
                True,
                id='issue_updated_with_multiple_labels_including_openhands',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [{'field': 'labels', 'toString': 'bug,urgent'}]
                    },
                },
                False,
                id='issue_updated_without_openhands_label',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {'items': []},
                },
                False,
                id='issue_updated_with_empty_changelog_items',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {},
                },
                False,
                id='issue_updated_with_empty_changelog',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                },
                False,
                id='issue_updated_without_changelog',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'changelog': {
                        'items': [{'field': 'labels', 'toString': 'openhands'}]
                    },
                },
                False,
                id='comment_created_event_with_label',
            ),
            pytest.param(
                {
                    'webhookEvent': 'issue_deleted',
                    'changelog': {
                        'items': [{'field': 'labels', 'toString': 'openhands'}]
                    },
                },
                False,
                id='unsupported_event_type',
            ),
            pytest.param(
                {},
                False,
                id='empty_payload',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [{'field': 'status', 'toString': 'In Progress'}]
                    },
                },
                False,
                id='issue_updated_with_non_label_field',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [{'field': 'labels', 'fromString': 'openhands'}]
                    },
                },
                False,
                id='issue_updated_with_fromString_instead_of_toString',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [
                            {'field': 'labels', 'toString': 'not-openhands'},
                            {'field': 'priority', 'toString': 'High'},
                        ]
                    },
                },
                False,
                id='issue_updated_with_mixed_fields_no_openhands',
            ),
        ],
    )
    def test_is_labeled_ticket(self, payload, expected):
        """Test is_labeled_ticket with various payloads."""
        with patch('integrations.jira.jira_view.OH_LABEL', 'openhands'):
            message = Message(source=SourceType.JIRA, message={'payload': payload})
            result = JiraFactory.is_labeled_ticket(message)
            assert result == expected

    @pytest.mark.parametrize(
        'payload,expected',
        [
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [{'field': 'labels', 'toString': 'openhands-exp'}]
                    },
                },
                True,
                id='issue_updated_with_staging_label',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'changelog': {
                        'items': [{'field': 'labels', 'toString': 'openhands'}]
                    },
                },
                False,
                id='issue_updated_with_prod_label_in_staging_env',
            ),
        ],
    )
    def test_is_labeled_ticket_staging_labels(self, payload, expected):
        """Test is_labeled_ticket with staging environment labels."""
        with patch('integrations.jira.jira_view.OH_LABEL', 'openhands-exp'):
            message = Message(source=SourceType.JIRA, message={'payload': payload})
            result = JiraFactory.is_labeled_ticket(message)
            assert result == expected


class TestJiraFactoryIsTicketComment:
    """Parameterized tests for JiraFactory.is_ticket_comment method."""

    @pytest.mark.parametrize(
        'payload,expected',
        [
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Please fix this @openhands'},
                },
                True,
                id='comment_with_openhands_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': '@openhands please help'},
                },
                True,
                id='comment_starting_with_openhands_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Hello @openhands!'},
                },
                True,
                id='comment_with_openhands_mention_and_punctuation',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': '(@openhands)'},
                },
                True,
                id='comment_with_openhands_in_parentheses',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Hey @OpenHands can you help?'},
                },
                True,
                id='comment_with_case_insensitive_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Hey @OPENHANDS!'},
                },
                True,
                id='comment_with_uppercase_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Regular comment without mention'},
                },
                False,
                id='comment_without_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Hello @openhands-agent!'},
                },
                False,
                id='comment_with_openhands_as_prefix',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'user@openhands.com'},
                },
                False,
                id='comment_with_openhands_in_email',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': ''},
                },
                False,
                id='comment_with_empty_body',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {},
                },
                False,
                id='comment_without_body',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                },
                False,
                id='comment_created_without_comment_data',
            ),
            pytest.param(
                {
                    'webhookEvent': 'jira:issue_updated',
                    'comment': {'body': 'Please fix this @openhands'},
                },
                False,
                id='issue_updated_event_with_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'issue_deleted',
                    'comment': {'body': '@openhands'},
                },
                False,
                id='unsupported_event_type_with_mention',
            ),
            pytest.param(
                {},
                False,
                id='empty_payload',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Multiple @openhands @openhands mentions'},
                },
                True,
                id='comment_with_multiple_mentions',
            ),
        ],
    )
    def test_is_ticket_comment(self, payload, expected):
        """Test is_ticket_comment with various payloads."""
        with patch('integrations.jira.jira_view.INLINE_OH_LABEL', '@openhands'), patch(
            'integrations.jira.jira_view.has_exact_mention'
        ) as mock_has_exact_mention:
            from integrations.utils import has_exact_mention

            mock_has_exact_mention.side_effect = (
                lambda text, mention: has_exact_mention(text, mention)
            )

            message = Message(source=SourceType.JIRA, message={'payload': payload})
            result = JiraFactory.is_ticket_comment(message)
            assert result == expected

    @pytest.mark.parametrize(
        'payload,expected',
        [
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Please fix this @openhands-exp'},
                },
                True,
                id='comment_with_staging_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': '@openhands-exp please help'},
                },
                True,
                id='comment_starting_with_staging_mention',
            ),
            pytest.param(
                {
                    'webhookEvent': 'comment_created',
                    'comment': {'body': 'Please fix this @openhands'},
                },
                False,
                id='comment_with_prod_mention_in_staging_env',
            ),
        ],
    )
    def test_is_ticket_comment_staging_labels(self, payload, expected):
        """Test is_ticket_comment with staging environment labels."""
        with patch(
            'integrations.jira.jira_view.INLINE_OH_LABEL', '@openhands-exp'
        ), patch(
            'integrations.jira.jira_view.has_exact_mention'
        ) as mock_has_exact_mention:
            from integrations.utils import has_exact_mention

            mock_has_exact_mention.side_effect = (
                lambda text, mention: has_exact_mention(text, mention)
            )

            message = Message(source=SourceType.JIRA, message={'payload': payload})
            result = JiraFactory.is_ticket_comment(message)
            assert result == expected
