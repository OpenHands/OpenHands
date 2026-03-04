"""Shared fixtures for Linear integration tests.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.linear.linear_manager import LinearManager
from integrations.linear.linear_view import (
    LinearExistingConversationView,
    LinearNewConversationView,
)
from integrations.models import JobContext
from jinja2 import DictLoader, Environment
from storage.linear_conversation import LinearConversation
from storage.linear_user import LinearUser
from storage.linear_workspace import LinearWorkspace

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
def linear_manager(mock_token_manager):
    """Create a LinearManager instance for testing."""
    with patch(
        'integrations.linear.linear_manager.LinearIntegrationStore.get_instance'
    ) as mock_store_class:
        mock_store = MagicMock()
        mock_store.get_active_user = AsyncMock()
        mock_store.get_workspace_by_name = AsyncMock()
        mock_store_class.return_value = mock_store
        manager = LinearManager(mock_token_manager)
        return manager


@pytest.fixture
def sample_linear_user():
    """Create a sample LinearUser for testing."""
    user = MagicMock(spec=LinearUser)
    user.id = 1
    user.keycloak_user_id = 'test_keycloak_id'
    user.linear_workspace_id = 1
    user.status = 'active'
    return user


@pytest.fixture
def sample_linear_workspace():
    """Create a sample LinearWorkspace for testing."""
    workspace = MagicMock(spec=LinearWorkspace)
    workspace.id = 1
    workspace.name = 'test-workspace'
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
    return user_auth


@pytest.fixture
def sample_job_context():
    """Create a sample JobContext for testing."""
    return JobContext(
        issue_id='test_issue_id',
        issue_key='TEST-123',
        user_msg='Fix this bug @openhands',
        user_email='user@test.com',
        display_name='Test User',
        workspace_name='test-workspace',
        issue_title='Test Issue',
        issue_description='This is a test issue',
    )


@pytest.fixture
def sample_webhook_payload():
    """Create a sample webhook payload for testing."""
    return {
        'action': 'create',
        'type': 'Comment',
        'data': {
            'body': 'Please fix this @openhands',
            'issue': {
                'id': 'test_issue_id',
                'identifier': 'TEST-123',
            },
        },
        'actor': {
            'id': 'user123',
            'name': 'Test User',
            'email': 'user@test.com',
            'url': 'https://linear.app/test-workspace/profiles/user123',
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
        'linear_instructions.j2': 'Test instructions template',
        'linear_new_conversation.j2': 'New conversation: {{issue_key}} - {{issue_title}}\n{{issue_description}}\nUser: {{user_message}}',
        'linear_existing_conversation.j2': 'Existing conversation: {{issue_key}} - {{issue_title}}\n{{issue_description}}\nUser: {{user_message}}',
    }
    return Environment(loader=DictLoader(templates))


@pytest.fixture
def linear_conversation():
    """Sample Linear conversation for testing"""
    return LinearConversation(
        conversation_id='conv-123',
        issue_id='test_issue_id',
        issue_key='TEST-123',
        linear_user_id='linear-user-123',
    )


@pytest.fixture
def new_conversation_view(
    sample_job_context, sample_user_auth, sample_linear_user, sample_linear_workspace
):
    """LinearNewConversationView instance for testing"""
    return LinearNewConversationView(
        job_context=sample_job_context,
        saas_user_auth=sample_user_auth,
        linear_user=sample_linear_user,
        linear_workspace=sample_linear_workspace,
        selected_repo='test/repo1',
        conversation_id='conv-123',
    )


@pytest.fixture
def existing_conversation_view(
    sample_job_context, sample_user_auth, sample_linear_user, sample_linear_workspace
):
    """LinearExistingConversationView instance for testing"""
    return LinearExistingConversationView(
        job_context=sample_job_context,
        saas_user_auth=sample_user_auth,
        linear_user=sample_linear_user,
        linear_workspace=sample_linear_workspace,
        selected_repo='test/repo1',
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
