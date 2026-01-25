"""
Shared fixtures for Jira integration tests.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.jira.jira_manager import JiraManager
from integrations.jira.jira_payload import (
    JiraEventType,
    JiraWebhookPayload,
)
from integrations.jira.jira_view import (
    JiraNewConversationView,
)
from jinja2 import DictLoader, Environment
from storage.jira_conversation import JiraConversation
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace

from openhands.integrations.service_types import ProviderType, Repository
from openhands.server.user_auth.user_auth import UserAuth


@pytest.fixture
def mock_token_manager():
    """Create a mock TokenManager for testing."""
    token_manager = MagicMock()
    token_manager.get_user_id_from_user_email = AsyncMock()
    token_manager.decrypt_text = MagicMock(return_value='decrypted_key')
    return token_manager


@pytest.fixture
def jira_manager(mock_token_manager):
    """Create a JiraManager instance for testing."""
    with patch(
        'integrations.jira.jira_manager.JiraIntegrationStore.get_instance'
    ) as mock_store_class:
        mock_store = MagicMock()
        mock_store.get_active_user = AsyncMock()
        mock_store.get_workspace_by_name = AsyncMock()
        mock_store_class.return_value = mock_store
        manager = JiraManager(mock_token_manager)
        return manager


@pytest.fixture
def sample_jira_user():
    """Create a sample JiraUser for testing."""
    user = MagicMock(spec=JiraUser)
    user.id = 1
    user.keycloak_user_id = 'test_keycloak_id'
    user.jira_workspace_id = 1
    user.status = 'active'
    return user


@pytest.fixture
def sample_jira_workspace():
    """Create a sample JiraWorkspace for testing."""
    workspace = MagicMock(spec=JiraWorkspace)
    workspace.id = 1
    workspace.name = 'test.atlassian.net'
    workspace.jira_cloud_id = 'cloud-123'
    workspace.admin_user_id = 'admin_id'
    workspace.webhook_secret = 'encrypted_secret'
    workspace.svc_acc_email = 'service@example.com'
    workspace.svc_acc_api_key = 'encrypted_api_key'
    workspace.status = 'active'
    return workspace


@pytest.fixture
def sample_user_auth():
    """Create a mock UserAuth for testing."""
    user_auth = MagicMock(spec=UserAuth)
    user_auth.get_provider_tokens = AsyncMock(return_value={})
    user_auth.get_access_token = AsyncMock(return_value='test_token')
    user_auth.get_user_id = AsyncMock(return_value='test_user_id')
    user_auth.get_secrets = AsyncMock(return_value=None)
    return user_auth


@pytest.fixture
def sample_webhook_payload():
    """Create a sample JiraWebhookPayload for testing."""
    return JiraWebhookPayload(
        event_type=JiraEventType.COMMENT_MENTION,
        raw_event='comment_created',
        issue_id='12345',
        issue_key='TEST-123',
        user_email='user@test.com',
        display_name='Test User',
        account_id='user123',
        workspace_name='test.atlassian.net',
        base_api_url='https://test.atlassian.net',
        comment_body='Fix this bug @openhands',
    )


@pytest.fixture
def sample_label_webhook_payload():
    """Create a sample labeled ticket JiraWebhookPayload for testing."""
    return JiraWebhookPayload(
        event_type=JiraEventType.LABELED_TICKET,
        raw_event='jira:issue_updated',
        issue_id='12345',
        issue_key='PROJ-123',
        user_email='user@company.com',
        display_name='Test User',
        account_id='user456',
        workspace_name='jira.company.com',
        base_api_url='https://jira.company.com',
        comment_body='',
    )


@pytest.fixture
def sample_comment_webhook_payload():
    """Create a sample comment webhook payload for testing."""
    return {
        'webhookEvent': 'comment_created',
        'comment': {
            'body': 'Please fix this @openhands',
            'author': {
                'emailAddress': 'user@test.com',
                'displayName': 'Test User',
                'accountId': 'user123',
                'self': 'https://test.atlassian.net/rest/api/2/user?accountId=123',
            },
        },
        'issue': {
            'id': '12345',
            'key': 'TEST-123',
            'self': 'https://test.atlassian.net/rest/api/2/issue/12345',
        },
    }


@pytest.fixture
def sample_issue_update_webhook_payload():
    """Sample issue update webhook payload."""
    return {
        'webhookEvent': 'jira:issue_updated',
        'changelog': {'items': [{'field': 'labels', 'toString': 'openhands'}]},
        'issue': {
            'id': '12345',
            'key': 'PROJ-123',
            'self': 'https://jira.company.com/rest/api/2/issue/12345',
        },
        'user': {
            'emailAddress': 'user@company.com',
            'displayName': 'Test User',
            'accountId': 'user456',
            'self': 'https://jira.company.com/rest/api/2/user?username=testuser',
        },
    }


@pytest.fixture
def sample_repositories():
    """Create sample repositories for testing."""
    return [
        Repository(
            id='1',
            full_name='test/repo1',
            stargazers_count=10,
            git_provider=ProviderType.GITHUB,
            is_public=True,
        ),
        Repository(
            id='2',
            full_name='test/repo2',
            stargazers_count=5,
            git_provider=ProviderType.GITHUB,
            is_public=False,
        ),
    ]


@pytest.fixture
def mock_jinja_env():
    """Mock Jinja2 environment with templates"""
    templates = {
        'jira_instructions.j2': 'Test Jira instructions template',
        'jira_new_conversation.j2': 'New Jira conversation: {{issue_key}} - {{issue_title}}\n{{issue_description}}\nUser: {{user_message}}',
        'jira_existing_conversation.j2': 'Existing Jira conversation: {{issue_key}} - {{issue_title}}\n{{issue_description}}\nUser: {{user_message}}',
    }
    return Environment(loader=DictLoader(templates))


@pytest.fixture
def jira_conversation():
    """Sample Jira conversation for testing"""
    return JiraConversation(
        conversation_id='conv-123',
        issue_id='PROJ-123',
        issue_key='PROJ-123',
        jira_user_id='jira-user-123',
    )


@pytest.fixture
def new_conversation_view(
    sample_webhook_payload, sample_user_auth, sample_jira_user, sample_jira_workspace
):
    """JiraNewConversationView instance for testing"""
    return JiraNewConversationView(
        payload=sample_webhook_payload,
        saas_user_auth=sample_user_auth,
        jira_user=sample_jira_user,
        jira_workspace=sample_jira_workspace,
        selected_repo='test/repo1',
        conversation_id='conv-123',
        _decrypted_api_key='decrypted_key',
    )


@pytest.fixture
def mock_agent_loop_info():
    """Mock agent loop info"""
    mock_info = MagicMock()
    mock_info.conversation_id = 'conv-123'
    mock_info.event_store = []
    return mock_info


@pytest.fixture
def mock_conversation_metadata():
    """Mock conversation metadata"""
    metadata = MagicMock()
    metadata.conversation_id = 'conv-123'
    return metadata


@pytest.fixture
def mock_conversation_store():
    """Mock conversation store"""
    store = AsyncMock()
    store.get_metadata.return_value = MagicMock()
    return store


@pytest.fixture
def mock_conversation_init_data():
    """Mock conversation initialization data"""
    return MagicMock()
