"""Shared fixtures for Jira DC integration tests.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.jira_dc.jira_dc_manager import JiraDcManager
from integrations.jira_dc.jira_dc_view import (
    JiraDcExistingConversationView,
    JiraDcNewConversationView,
)
from integrations.models import JobContext
from jinja2 import DictLoader, Environment
from storage.jira_dc_conversation import JiraDcConversation
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.integrations.service_types import ProviderType, Repository
from openhands.server.user_auth.user_auth import UserAuth


@pytest.fixture
def mock_token_manager():
    """Create a mock TokenManager for testing."""
    token_manager = MagicMock()
    token_manager.get_user_id_from_user_email = AsyncMock()
    token_manager.decrypt_text = MagicMock()
    return token_manager


@pytest.fixture
def jira_dc_manager(mock_token_manager):
    """Create a JiraDcManager instance for testing."""
    with patch(
        'integrations.jira_dc.jira_dc_manager.JiraDcIntegrationStore.get_instance'
    ) as mock_store_class:
        mock_store = MagicMock()
        mock_store.get_active_user = AsyncMock()
        mock_store.get_workspace_by_name = AsyncMock()
        mock_store_class.return_value = mock_store
        manager = JiraDcManager(mock_token_manager)
        return manager


@pytest.fixture
def sample_jira_dc_user():
    """Create a sample JiraDcUser for testing."""
    user = MagicMock(spec=JiraDcUser)
    user.id = 1
    user.keycloak_user_id = 'test_keycloak_id'
    user.jira_dc_workspace_id = 1
    user.status = 'active'
    return user


@pytest.fixture
def sample_jira_dc_workspace():
    """Create a sample JiraDcWorkspace for testing."""
    workspace = MagicMock(spec=JiraDcWorkspace)
    workspace.id = 1
    workspace.name = 'jira.company.com'
    workspace.admin_user_id = 'admin_id'
    workspace.webhook_secret = 'encrypted_secret'
    workspace.svc_acc_email = 'service@company.com'
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
    return user_auth


@pytest.fixture
def sample_job_context():
    """Create a sample JobContext for testing."""
    return JobContext(
        issue_id='12345',
        issue_key='PROJ-123',
        user_msg='Fix this bug @openhands',
        user_email='user@company.com',
        display_name='Test User',
        platform_user_id='testuser',
        workspace_name='jira.company.com',
        base_api_url='https://jira.company.com',
        issue_title='Test Issue',
        issue_description='This is a test issue',
    )


@pytest.fixture
def sample_comment_webhook_payload():
    """Create a sample comment webhook payload for testing."""
    return {
        'webhookEvent': 'comment_created',
        'comment': {
            'body': 'Please fix this @openhands',
            'author': {
                'emailAddress': 'user@company.com',
                'displayName': 'Test User',
                'key': 'testuser',
                'accountId': 'user123',
                'self': 'https://jira.company.com/rest/api/2/user?username=testuser',
            },
        },
        'issue': {
            'id': '12345',
            'key': 'PROJ-123',
            'self': 'https://jira.company.com/rest/api/2/issue/12345',
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
            'key': 'testuser',
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
            full_name='company/repo1',
            stargazers_count=10,
            git_provider=ProviderType.GITHUB,
            is_public=True,
        ),
        Repository(
            id='2',
            full_name='company/repo2',
            stargazers_count=5,
            git_provider=ProviderType.GITHUB,
            is_public=False,
        ),
    ]


@pytest.fixture
def mock_jinja_env():
    """Mock Jinja2 environment with templates"""
    templates = {
        'jira_dc_instructions.j2': 'Test Jira DC instructions template',
        'jira_dc_new_conversation.j2': 'New Jira DC conversation: {{issue_key}} - {{issue_title}}\n{{issue_description}}\nUser: {{user_message}}',
        'jira_dc_existing_conversation.j2': 'Existing Jira DC conversation: {{issue_key}} - {{issue_title}}\n{{issue_description}}\nUser: {{user_message}}',
    }
    return Environment(loader=DictLoader(templates))


@pytest.fixture
def jira_dc_conversation():
    """Sample Jira DC conversation for testing"""
    return JiraDcConversation(
        conversation_id='conv-123',
        issue_id='12345',
        issue_key='PROJ-123',
        jira_dc_user_id='jira-dc-user-123',
    )


@pytest.fixture
def new_conversation_view(
    sample_job_context, sample_user_auth, sample_jira_dc_user, sample_jira_dc_workspace
):
    """JiraDcNewConversationView instance for testing"""
    return JiraDcNewConversationView(
        job_context=sample_job_context,
        saas_user_auth=sample_user_auth,
        jira_dc_user=sample_jira_dc_user,
        jira_dc_workspace=sample_jira_dc_workspace,
        selected_repo='company/repo1',
        conversation_id='conv-123',
    )


@pytest.fixture
def existing_conversation_view(
    sample_job_context, sample_user_auth, sample_jira_dc_user, sample_jira_dc_workspace
):
    """JiraDcExistingConversationView instance for testing"""
    return JiraDcExistingConversationView(
        job_context=sample_job_context,
        saas_user_auth=sample_user_auth,
        jira_dc_user=sample_jira_dc_user,
        jira_dc_workspace=sample_jira_dc_workspace,
        selected_repo='company/repo1',
        conversation_id='conv-123',
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
