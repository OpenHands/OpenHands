import json
from contextlib import contextmanager
from datetime import datetime, timezone
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationPage,
)
from openhands.integrations.service_types import (
    AuthenticationError,
    CreateMicroagent,
    ProviderType,
    SuggestedTask,
    TaskType,
)
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.data_models.conversation_info import ConversationInfo
from openhands.server.data_models.conversation_info_result_set import (
    ConversationInfoResultSet,
)
from openhands.server.routes.manage_conversations import (
    ConversationResponse,
    InitSessionRequest,
    delete_conversation,
    get_conversation,
    new_conversation,
    search_conversations,
)
from openhands.server.routes.manage_conversations import app as conversation_app
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.server.user_auth.user_auth import AuthType
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.storage.data_models.conversation_status import ConversationStatus
from openhands.storage.locations import get_conversation_metadata_filename
from openhands.storage.memory import InMemoryFileStore


@contextmanager
def _patch_store():
    file_store = InMemoryFileStore()
    file_store.write(
        get_conversation_metadata_filename('some_conversation_id'),
        json.dumps(
            {
                'title': 'Some ServerConversation',
                'selected_repository': 'foobar',
                'conversation_id': 'some_conversation_id',
                'user_id': '12345',
                'created_at': '2025-01-01T00:00:00+00:00',
                'last_updated_at': '2025-01-01T00:01:00+00:00',
            }
        ),
    )
    with patch(
        'openhands.storage.conversation.file_conversation_store.get_file_store',
        MagicMock(return_value=file_store),
    ):
        with patch(
            'openhands.server.routes.manage_conversations.conversation_manager.file_store',
            file_store,
        ):
            yield


@pytest.fixture
def test_client():
    """Create a test client for the settings API."""
    app = FastAPI()
    app.include_router(conversation_app)
    return TestClient(app)


def create_new_test_conversation(
    test_request: InitSessionRequest, auth_type: AuthType | None = None
):
    # Create a mock Secrets object with the required custom_secrets attribute
    mock_user_secrets = MagicMock()
    mock_user_secrets.custom_secrets = MappingProxyType({})

    return new_conversation(
        data=test_request,
        user_id='test_user',
        provider_tokens=MappingProxyType({'github': 'token123'}),
        user_secrets=mock_user_secrets,
        auth_type=auth_type,
    )


@pytest.fixture
def provider_handler_mock():
    with patch(
        'openhands.server.routes.manage_conversations.ProviderHandler'
    ) as mock_cls:
        mock_instance = MagicMock()
        mock_instance.verify_repo_provider = AsyncMock(return_value=ProviderType.GITHUB)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_search_conversations():
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='some_conversation_id',
                                    title='Some ServerConversation',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='foobar',
                                    user_id='12345',
                                )
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository=None,
                        conversation_trigger=None,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    expected = ConversationInfoResultSet(
                        results=[
                            ConversationInfo(
                                conversation_id='some_conversation_id',
                                title='Some ServerConversation',
                                created_at=datetime.fromisoformat(
                                    '2025-01-01T00:00:00+00:00'
                                ),
                                last_updated_at=datetime.fromisoformat(
                                    '2025-01-01T00:01:00+00:00'
                                ),
                                status=ConversationStatus.STOPPED,
                                selected_repository='foobar',
                                num_connections=0,
                                url=None,
                                pr_number=[],  # Default empty list for pr_number
                            )
                        ]
                    )
                    assert result_set == expected


@pytest.mark.asyncio
async def test_search_conversations_with_repository_filter():
    """Test searching conversations with repository filter."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_1',
                                    title='Conversation 1',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                )
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository='test/repo',
                        conversation_trigger=None,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify that search was called with only pagination parameters (filtering is done at API level)
                    mock_store.search.assert_called_once_with(None, 20)

                    # Verify the result contains only conversations from the specified repository
                    assert len(result_set.results) == 1
                    assert result_set.results[0].selected_repository == 'test/repo'


@pytest.mark.asyncio
async def test_search_conversations_with_trigger_filter():
    """Test searching conversations with conversation trigger filter."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_1',
                                    title='Conversation 1',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    trigger=ConversationTrigger.GUI,
                                    user_id='12345',
                                )
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository=None,
                        conversation_trigger=ConversationTrigger.GUI,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify that search was called with only pagination parameters (filtering is done at API level)
                    mock_store.search.assert_called_once_with(None, 20)

                    # Verify the result contains only conversations with the specified trigger
                    assert len(result_set.results) == 1
                    assert result_set.results[0].trigger == ConversationTrigger.GUI


@pytest.mark.asyncio
async def test_search_conversations_with_both_filters():
    """Test searching conversations with both repository and trigger filters."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_1',
                                    title='Conversation 1',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    trigger=ConversationTrigger.SUGGESTED_TASK,
                                    user_id='12345',
                                )
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository='test/repo',
                        conversation_trigger=ConversationTrigger.SUGGESTED_TASK,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify that search was called with only pagination parameters (filtering is done at API level)
                    mock_store.search.assert_called_once_with(None, 20)

                    # Verify the result contains only conversations matching both filters
                    assert len(result_set.results) == 1
                    result = result_set.results[0]
                    assert result.selected_repository == 'test/repo'
                    assert result.trigger == ConversationTrigger.SUGGESTED_TASK


@pytest.mark.asyncio
async def test_search_conversations_with_pagination():
    """Test searching conversations with pagination."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_1',
                                    title='Conversation 1',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                )
                            ],
                            next_page_id='next_page_123',
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id='eyJ2MCI6ICJwYWdlXzEyMyIsICJ2MSI6IG51bGx9',
                        limit=10,
                        selected_repository=None,
                        conversation_trigger=None,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify that search was called with pagination parameters (filtering is done at API level)
                    mock_store.search.assert_called_once_with('page_123', 10)

                    # Verify the result includes pagination info
                    assert (
                        result_set.next_page_id
                        == 'eyJ2MCI6ICJuZXh0X3BhZ2VfMTIzIiwgInYxIjogbnVsbH0='
                    )


@pytest.mark.asyncio
async def test_search_conversations_with_filters_and_pagination():
    """Test searching conversations with filters and pagination."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_1',
                                    title='Conversation 1',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    trigger=ConversationTrigger.GUI,
                                    user_id='12345',
                                )
                            ],
                            next_page_id='next_page_456',
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id='eyJ2MCI6ICJwYWdlXzQ1NiIsICJ2MSI6IG51bGx9',
                        limit=5,
                        selected_repository='test/repo',
                        conversation_trigger=ConversationTrigger.GUI,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify that search was called with only pagination parameters (filtering is done at API level)
                    mock_store.search.assert_called_once_with('page_456', 5)

                    # Verify the result includes pagination info
                    assert (
                        result_set.next_page_id
                        == 'eyJ2MCI6ICJuZXh0X3BhZ2VfNDU2IiwgInYxIjogbnVsbH0='
                    )
                    assert len(result_set.results) == 1
                    result = result_set.results[0]
                    assert result.selected_repository == 'test/repo'
                    assert result.trigger == ConversationTrigger.GUI


@pytest.mark.asyncio
async def test_search_conversations_empty_results():
    """Test searching conversations that returns empty results."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[], next_page_id=None
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository='nonexistent/repo',
                        conversation_trigger=ConversationTrigger.GUI,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify that search was called with only pagination parameters (filtering is done at API level)
                    mock_store.search.assert_called_once_with(None, 20)

                    # Verify the result is empty
                    assert len(result_set.results) == 0
                    assert result_set.next_page_id is None


@pytest.mark.asyncio
async def test_get_conversation():
    with _patch_store():
        # Mock the conversation store
        mock_store = MagicMock()
        mock_store.get_metadata = AsyncMock(
            return_value=ConversationMetadata(
                conversation_id='some_conversation_id',
                title='Some ServerConversation',
                created_at=datetime.fromisoformat('2025-01-01T00:00:00+00:00'),
                last_updated_at=datetime.fromisoformat('2025-01-01T00:01:00+00:00'),
                selected_repository='foobar',
                user_id='12345',
            )
        )

        # Mock the conversation manager
        with patch(
            'openhands.server.routes.manage_conversations.conversation_manager'
        ) as mock_manager:
            mock_manager.is_agent_loop_running = AsyncMock(return_value=False)
            mock_manager.get_connections = AsyncMock(return_value={})
            mock_manager.get_agent_loop_info = AsyncMock(return_value=[])

            conversation = await get_conversation(
                'some_conversation_id', conversation_store=mock_store
            )

            expected = ConversationInfo(
                conversation_id='some_conversation_id',
                title='Some ServerConversation',
                created_at=datetime.fromisoformat('2025-01-01T00:00:00+00:00'),
                last_updated_at=datetime.fromisoformat('2025-01-01T00:01:00+00:00'),
                status=ConversationStatus.STOPPED,
                selected_repository='foobar',
                num_connections=0,
                url=None,
                pr_number=[],  # Default empty list for pr_number
            )
            assert conversation == expected


@pytest.mark.asyncio
async def test_get_missing_conversation():
    with _patch_store():
        # Mock the conversation store
        mock_store = MagicMock()
        mock_store.get_metadata = AsyncMock(side_effect=FileNotFoundError)

        assert (
            await get_conversation(
                'no_such_conversation', conversation_store=mock_store
            )
            is None
        )


@pytest.mark.asyncio
async def test_new_conversation_success(provider_handler_mock):
    """Test successful creation of a new conversation."""
    with _patch_store():
        # Mock the create_new_conversation function directly
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = MagicMock(
                conversation_id='test_conversation_id',
                url='https://my-conversation.com',
                session_api_key=None,
                status=ConversationStatus.RUNNING,
            )

            test_request = InitSessionRequest(
                repository='test/repo',
                selected_branch='main',
                initial_user_msg='Hello, agent!',
                image_urls=['https://example.com/image.jpg'],
            )

            # Call new_conversation
            response = await create_new_test_conversation(test_request)

            # Verify the response
            assert isinstance(response, ConversationResponse)
            assert response.status == 'ok'
            # Don't check the exact conversation_id as it's now generated dynamically
            assert response.conversation_id is not None
            assert isinstance(response.conversation_id, str)

            # Verify that create_new_conversation was called with the correct arguments
            mock_create_conversation.assert_called_once()
            call_args = mock_create_conversation.call_args[1]
            assert call_args['user_id'] == 'test_user'
            assert call_args['selected_repository'] == 'test/repo'
            assert call_args['selected_branch'] == 'main'
            assert call_args['initial_user_msg'] == 'Hello, agent!'
            assert call_args['image_urls'] == ['https://example.com/image.jpg']
            assert call_args['conversation_trigger'] == ConversationTrigger.GUI


@pytest.mark.asyncio
async def test_new_conversation_with_suggested_task(provider_handler_mock):
    """Test creating a new conversation with a suggested task."""
    with _patch_store():
        # Mock the create_new_conversation function directly
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = MagicMock(
                conversation_id='test_conversation_id',
                url='https://my-conversation.com',
                session_api_key=None,
                status=ConversationStatus.RUNNING,
            )

            # Mock SuggestedTask.get_prompt_for_task
            with patch(
                'openhands.integrations.service_types.SuggestedTask.get_prompt_for_task'
            ) as mock_get_prompt:
                mock_get_prompt.return_value = (
                    'Please fix the failing checks in PR #123'
                )

                test_task = SuggestedTask(
                    git_provider=ProviderType.GITHUB,
                    task_type=TaskType.FAILING_CHECKS,
                    repo='test/repo',
                    issue_number=123,
                    title='Fix failing checks',
                )

                test_request = InitSessionRequest(
                    repository='test/repo',
                    selected_branch='main',
                    suggested_task=test_task,
                )

                # Call new_conversation
                response = await create_new_test_conversation(test_request)

                # Verify the response
                assert isinstance(response, ConversationResponse)
                assert response.status == 'ok'
                # Don't check the exact conversation_id as it's now generated dynamically
                assert response.conversation_id is not None
                assert isinstance(response.conversation_id, str)

                # Verify that create_new_conversation was called with the correct arguments
                mock_create_conversation.assert_called_once()
                call_args = mock_create_conversation.call_args[1]
                assert call_args['user_id'] == 'test_user'
                assert call_args['selected_repository'] == 'test/repo'
                assert call_args['selected_branch'] == 'main'
                assert (
                    call_args['initial_user_msg']
                    == 'Please fix the failing checks in PR #123'
                )
                assert (
                    call_args['conversation_trigger']
                    == ConversationTrigger.SUGGESTED_TASK
                )

                # Verify that get_prompt_for_task was called
                mock_get_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_new_conversation_missing_settings(provider_handler_mock):
    """Test creating a new conversation when settings are missing."""
    with _patch_store():
        # Mock the create_new_conversation function to raise MissingSettingsError
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to raise MissingSettingsError
            mock_create_conversation.side_effect = MissingSettingsError(
                'Settings not found'
            )

            test_request = InitSessionRequest(
                repository='test/repo',
                selected_branch='main',
                initial_user_msg='Hello, agent!',
            )

            # Call new_conversation
            response = await create_new_test_conversation(test_request)

            # Verify the response
            assert isinstance(response, JSONResponse)
            assert response.status_code == 400
            assert 'Settings not found' in response.body.decode('utf-8')
            assert 'CONFIGURATION$SETTINGS_NOT_FOUND' in response.body.decode('utf-8')


@pytest.mark.asyncio
async def test_new_conversation_invalid_session_api_key(provider_handler_mock):
    """Test creating a new conversation with an invalid API key."""
    with _patch_store():
        # Mock the create_new_conversation function to raise LLMAuthenticationError
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to raise LLMAuthenticationError
            mock_create_conversation.side_effect = LLMAuthenticationError(
                'Error authenticating with the LLM provider. Please check your API key'
            )

            test_request = InitSessionRequest(
                repository='test/repo',
                selected_branch='main',
                initial_user_msg='Hello, agent!',
            )

            # Call new_conversation
            response = await create_new_test_conversation(test_request)

            # Verify the response
            assert isinstance(response, JSONResponse)
            assert response.status_code == 400
            assert 'Error authenticating with the LLM provider' in response.body.decode(
                'utf-8'
            )
            assert RuntimeStatus.ERROR_LLM_AUTHENTICATION.value in response.body.decode(
                'utf-8'
            )


@pytest.mark.asyncio
async def test_delete_conversation():
    with _patch_store():
        # Mock the ConversationStoreImpl.get_instance
        with patch(
            'openhands.server.routes.manage_conversations.ConversationStoreImpl.get_instance'
        ) as mock_get_instance:
            # Create a mock conversation store
            mock_store = MagicMock()

            # Set up the mock to return metadata and then delete it
            mock_store.get_metadata = AsyncMock(
                return_value=ConversationMetadata(
                    conversation_id='some_conversation_id',
                    title='Some ServerConversation',
                    created_at=datetime.fromisoformat('2025-01-01T00:00:00+00:00'),
                    last_updated_at=datetime.fromisoformat('2025-01-01T00:01:00+00:00'),
                    selected_repository='foobar',
                    user_id='12345',
                )
            )
            mock_store.delete_metadata = AsyncMock()

            # Return the mock store from get_instance
            mock_get_instance.return_value = mock_store

            # Create a mock app conversation service
            mock_app_conversation_service = MagicMock()

            # Create a mock app conversation info service
            mock_app_conversation_info_service = MagicMock()
            mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
                return_value=None
            )

            # Create a mock sandbox service
            mock_sandbox_service = MagicMock()

            # Mock the conversation manager
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:
                mock_manager.is_agent_loop_running = AsyncMock(return_value=False)
                mock_manager.get_connections = AsyncMock(return_value={})

                # Mock the runtime class
                with patch(
                    'openhands.server.routes.manage_conversations.get_runtime_cls'
                ) as mock_get_runtime_cls:
                    mock_runtime_cls = MagicMock()
                    mock_runtime_cls.delete = AsyncMock()
                    mock_get_runtime_cls.return_value = mock_runtime_cls

                    # Call delete_conversation
                    result = await delete_conversation(
                        request=MagicMock(),
                        conversation_id='some_conversation_id',
                        user_id='12345',
                        app_conversation_service=mock_app_conversation_service,
                        app_conversation_info_service=mock_app_conversation_info_service,
                        sandbox_service=mock_sandbox_service,
                    )

                    # Verify the result
                    assert result is True

                    # Verify that delete_metadata was called
                    mock_store.delete_metadata.assert_called_once_with(
                        'some_conversation_id'
                    )

                    # Verify that runtime.delete was called
                    mock_runtime_cls.delete.assert_called_once_with(
                        'some_conversation_id'
                    )


@pytest.mark.asyncio
async def test_delete_v1_conversation_success():
    """Test successful deletion of a V1 conversation."""
    from uuid import uuid4

    from openhands.app_server.app_conversation.app_conversation_models import (
        AppConversation,
    )
    from openhands.app_server.sandbox.sandbox_models import SandboxStatus
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)

    # Mock the app conversation service
    with patch(
        'openhands.server.routes.manage_conversations.app_conversation_service_dependency'
    ) as mock_service_dep:
        mock_service = MagicMock()
        mock_service_dep.return_value = mock_service

        # Mock the app conversation info service
        with patch(
            'openhands.server.routes.manage_conversations.app_conversation_info_service_dependency'
        ) as mock_info_service_dep:
            mock_info_service = MagicMock()
            mock_info_service_dep.return_value = mock_info_service

            # Mock the sandbox service
            with patch(
                'openhands.server.routes.manage_conversations.sandbox_service_dependency'
            ) as mock_sandbox_service_dep:
                mock_sandbox_service = MagicMock()
                mock_sandbox_service_dep.return_value = mock_sandbox_service

                # Mock the conversation info exists
                mock_app_conversation_info = AppConversation(
                    id=conversation_uuid,
                    created_by_user_id='test_user',
                    sandbox_id='test-sandbox-id',
                    title='Test V1 Conversation',
                    sandbox_status=SandboxStatus.RUNNING,
                    execution_status=ConversationExecutionStatus.RUNNING,
                    session_api_key='test-api-key',
                    selected_repository='test/repo',
                    selected_branch='main',
                    git_provider=ProviderType.GITHUB,
                    trigger=ConversationTrigger.GUI,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                mock_info_service.get_app_conversation_info = AsyncMock(
                    return_value=mock_app_conversation_info
                )
                mock_service.delete_app_conversation = AsyncMock(return_value=True)

                # Call delete_conversation with V1 conversation ID
                result = await delete_conversation(
                    request=MagicMock(),
                    conversation_id=conversation_id,
                    user_id='test_user',
                    app_conversation_service=mock_service,
                    app_conversation_info_service=mock_info_service,
                    sandbox_service=mock_sandbox_service,
                )

                # Verify the result
                assert result is True

                # Verify that get_app_conversation_info was called
                mock_info_service.get_app_conversation_info.assert_called_once_with(
                    conversation_uuid
                )

                # Verify that delete_app_conversation was called with the conversation ID
                mock_service.delete_app_conversation.assert_called_once_with(
                    conversation_uuid
                )


@pytest.mark.asyncio
async def test_delete_v1_conversation_not_found():
    """Test deletion of a V1 conversation that doesn't exist."""
    from uuid import uuid4

    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)

    # Mock the app conversation service
    with patch(
        'openhands.server.routes.manage_conversations.app_conversation_service_dependency'
    ) as mock_service_dep:
        mock_service = MagicMock()
        mock_service_dep.return_value = mock_service

        # Mock the app conversation info service
        with patch(
            'openhands.server.routes.manage_conversations.app_conversation_info_service_dependency'
        ) as mock_info_service_dep:
            mock_info_service = MagicMock()
            mock_info_service_dep.return_value = mock_info_service

            # Mock the sandbox service
            with patch(
                'openhands.server.routes.manage_conversations.sandbox_service_dependency'
            ) as mock_sandbox_service_dep:
                mock_sandbox_service = MagicMock()
                mock_sandbox_service_dep.return_value = mock_sandbox_service

                # Mock the conversation doesn't exist
                mock_info_service.get_app_conversation_info = AsyncMock(
                    return_value=None
                )
                mock_service.delete_app_conversation = AsyncMock(return_value=False)

                # Call delete_conversation with V1 conversation ID
                result = await delete_conversation(
                    request=MagicMock(),
                    conversation_id=conversation_id,
                    user_id='test_user',
                    app_conversation_service=mock_service,
                    app_conversation_info_service=mock_info_service,
                    sandbox_service=mock_sandbox_service,
                )

                # Verify the result
                assert result is False

                # Verify that get_app_conversation_info was called
                mock_info_service.get_app_conversation_info.assert_called_once_with(
                    conversation_uuid
                )

                # Verify that delete_app_conversation was NOT called
                mock_service.delete_app_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_delete_v1_conversation_invalid_uuid():
    """Test deletion with invalid UUID falls back to V0 logic."""
    conversation_id = 'invalid-uuid-format'

    # Mock the app conversation service
    with patch(
        'openhands.server.routes.manage_conversations.app_conversation_service_dependency'
    ) as mock_service_dep:
        mock_service = MagicMock()
        mock_service_dep.return_value = mock_service

        # Mock V0 conversation logic
        with patch(
            'openhands.server.routes.manage_conversations.ConversationStoreImpl.get_instance'
        ) as mock_get_instance:
            mock_store = MagicMock()
            mock_store.get_metadata = AsyncMock(
                return_value=ConversationMetadata(
                    conversation_id=conversation_id,
                    title='Test V0 Conversation',
                    created_at=datetime.fromisoformat('2025-01-01T00:00:00+00:00'),
                    last_updated_at=datetime.fromisoformat('2025-01-01T00:01:00+00:00'),
                    selected_repository='test/repo',
                    user_id='test_user',
                )
            )
            mock_store.delete_metadata = AsyncMock()
            mock_get_instance.return_value = mock_store

            # Mock conversation manager
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:
                mock_manager.is_agent_loop_running = AsyncMock(return_value=False)
                mock_manager.get_connections = AsyncMock(return_value={})

                # Mock runtime
                with patch(
                    'openhands.server.routes.manage_conversations.get_runtime_cls'
                ) as mock_get_runtime_cls:
                    mock_runtime_cls = MagicMock()
                    mock_runtime_cls.delete = AsyncMock()
                    mock_get_runtime_cls.return_value = mock_runtime_cls

                    # Mock the app conversation info service
                    with patch(
                        'openhands.server.routes.manage_conversations.app_conversation_info_service_dependency'
                    ) as mock_info_service_dep:
                        mock_info_service = MagicMock()
                        mock_info_service_dep.return_value = mock_info_service

                        # Mock the sandbox service
                        with patch(
                            'openhands.server.routes.manage_conversations.sandbox_service_dependency'
                        ) as mock_sandbox_service_dep:
                            mock_sandbox_service = MagicMock()
                            mock_sandbox_service_dep.return_value = mock_sandbox_service

                            # Call delete_conversation
                            result = await delete_conversation(
                                request=MagicMock(),
                                conversation_id=conversation_id,
                                user_id='test_user',
                                app_conversation_service=mock_service,
                                app_conversation_info_service=mock_info_service,
                                sandbox_service=mock_sandbox_service,
                            )

                            # Verify the result
                            assert result is True

                            # Verify V0 logic was used
                            mock_store.delete_metadata.assert_called_once_with(
                                conversation_id
                            )
                            mock_runtime_cls.delete.assert_called_once_with(
                                conversation_id
                            )


@pytest.mark.asyncio
async def test_delete_v1_conversation_service_error():
    """Test deletion when app conversation service raises an error."""
    from uuid import uuid4

    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)

    # Mock the app conversation service
    with patch(
        'openhands.server.routes.manage_conversations.app_conversation_service_dependency'
    ) as mock_service_dep:
        mock_service = MagicMock()
        mock_service_dep.return_value = mock_service

        # Mock the app conversation info service
        with patch(
            'openhands.server.routes.manage_conversations.app_conversation_info_service_dependency'
        ) as mock_info_service_dep:
            mock_info_service = MagicMock()
            mock_info_service_dep.return_value = mock_info_service

            # Mock the sandbox service
            with patch(
                'openhands.server.routes.manage_conversations.sandbox_service_dependency'
            ) as mock_sandbox_service_dep:
                mock_sandbox_service = MagicMock()
                mock_sandbox_service_dep.return_value = mock_sandbox_service

                # Mock service error
                mock_info_service.get_app_conversation_info = AsyncMock(
                    side_effect=Exception('Service error')
                )

                # Mock V0 conversation logic as fallback
                with patch(
                    'openhands.server.routes.manage_conversations.ConversationStoreImpl.get_instance'
                ) as mock_get_instance:
                    mock_store = MagicMock()
                    mock_store.get_metadata = AsyncMock(
                        return_value=ConversationMetadata(
                            conversation_id=conversation_id,
                            title='Test V0 Conversation',
                            created_at=datetime.fromisoformat(
                                '2025-01-01T00:00:00+00:00'
                            ),
                            last_updated_at=datetime.fromisoformat(
                                '2025-01-01T00:01:00+00:00'
                            ),
                            selected_repository='test/repo',
                            user_id='test_user',
                        )
                    )
                    mock_store.delete_metadata = AsyncMock()
                    mock_get_instance.return_value = mock_store

                    # Mock conversation manager
                    with patch(
                        'openhands.server.routes.manage_conversations.conversation_manager'
                    ) as mock_manager:
                        mock_manager.is_agent_loop_running = AsyncMock(
                            return_value=False
                        )
                        mock_manager.get_connections = AsyncMock(return_value={})

                        # Mock runtime
                        with patch(
                            'openhands.server.routes.manage_conversations.get_runtime_cls'
                        ) as mock_get_runtime_cls:
                            mock_runtime_cls = MagicMock()
                            mock_runtime_cls.delete = AsyncMock()
                            mock_get_runtime_cls.return_value = mock_runtime_cls

                            # Call delete_conversation
                            result = await delete_conversation(
                                request=MagicMock(),
                                conversation_id=conversation_id,
                                user_id='test_user',
                                app_conversation_service=mock_service,
                                app_conversation_info_service=mock_info_service,
                                sandbox_service=mock_sandbox_service,
                            )

                            # Verify the result (should fallback to V0)
                            assert result is True

                            # Verify V0 logic was used
                            mock_store.delete_metadata.assert_called_once_with(
                                conversation_id
                            )
                            mock_runtime_cls.delete.assert_called_once_with(
                                conversation_id
                            )


@pytest.mark.asyncio
async def test_delete_v1_conversation_with_agent_server():
    """Test V1 conversation deletion with agent server integration."""
    from uuid import uuid4

    from openhands.app_server.app_conversation.app_conversation_models import (
        AppConversation,
    )
    from openhands.app_server.sandbox.sandbox_models import SandboxStatus
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)

    # Mock the app conversation service
    with patch(
        'openhands.server.routes.manage_conversations.app_conversation_service_dependency'
    ) as mock_service_dep:
        mock_service = MagicMock()
        mock_service_dep.return_value = mock_service

        # Mock the app conversation info service
        with patch(
            'openhands.server.routes.manage_conversations.app_conversation_info_service_dependency'
        ) as mock_info_service_dep:
            mock_info_service = MagicMock()
            mock_info_service_dep.return_value = mock_info_service

            # Mock the sandbox service
            with patch(
                'openhands.server.routes.manage_conversations.sandbox_service_dependency'
            ) as mock_sandbox_service_dep:
                mock_sandbox_service = MagicMock()
                mock_sandbox_service_dep.return_value = mock_sandbox_service

                # Mock the conversation exists with running sandbox
                mock_app_conversation_info = AppConversation(
                    id=conversation_uuid,
                    created_by_user_id='test_user',
                    sandbox_id='test-sandbox-id',
                    title='Test V1 Conversation',
                    sandbox_status=SandboxStatus.RUNNING,
                    execution_status=ConversationExecutionStatus.RUNNING,
                    session_api_key='test-api-key',
                    selected_repository='test/repo',
                    selected_branch='main',
                    git_provider=ProviderType.GITHUB,
                    trigger=ConversationTrigger.GUI,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                mock_info_service.get_app_conversation_info = AsyncMock(
                    return_value=mock_app_conversation_info
                )
                mock_service.delete_app_conversation = AsyncMock(return_value=True)

                # Call delete_conversation with V1 conversation ID
                result = await delete_conversation(
                    request=MagicMock(),
                    conversation_id=conversation_id,
                    user_id='test_user',
                    app_conversation_service=mock_service,
                    app_conversation_info_service=mock_info_service,
                    sandbox_service=mock_sandbox_service,
                )

                # Verify the result
                assert result is True

                # Verify that get_app_conversation_info was called
                mock_info_service.get_app_conversation_info.assert_called_once_with(
                    conversation_uuid
                )

                # Verify that delete_app_conversation was called with the conversation ID
                mock_service.delete_app_conversation.assert_called_once_with(
                    conversation_uuid
                )


@pytest.mark.asyncio
async def test_new_conversation_with_bearer_auth(provider_handler_mock):
    """Test creating a new conversation with bearer authentication."""
    with _patch_store():
        # Mock the create_new_conversation function
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = MagicMock(
                conversation_id='test_conversation_id',
                url='https://my-conversation.com',
                session_api_key=None,
                status=ConversationStatus.RUNNING,
            )

            # Create the request object
            test_request = InitSessionRequest(
                repository='test/repo',
                selected_branch='main',
                initial_user_msg='Hello, agent!',
            )

            # Call new_conversation with auth_type=BEARER
            response = await create_new_test_conversation(test_request, AuthType.BEARER)

            # Verify the response
            assert isinstance(response, ConversationResponse)
            assert response.status == 'ok'

            # Verify that create_new_conversation was called with REMOTE_API_KEY trigger
            mock_create_conversation.assert_called_once()
            call_args = mock_create_conversation.call_args[1]
            assert (
                call_args['conversation_trigger'] == ConversationTrigger.REMOTE_API_KEY
            )


@pytest.mark.asyncio
async def test_new_conversation_with_null_repository():
    """Test creating a new conversation with null repository."""
    with _patch_store():
        # Mock the create_new_conversation function
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = MagicMock(
                conversation_id='test_conversation_id',
                url='https://my-conversation.com',
                session_api_key=None,
                status=ConversationStatus.RUNNING,
            )

            # Create the request object with null repository
            test_request = InitSessionRequest(
                repository=None,  # Explicitly set to None
                selected_branch=None,
                initial_user_msg='Hello, agent!',
            )

            # Call new_conversation
            response = await create_new_test_conversation(test_request)

            # Verify the response
            assert isinstance(response, ConversationResponse)
            assert response.status == 'ok'

            # Verify that create_new_conversation was called with None repository
            mock_create_conversation.assert_called_once()
            call_args = mock_create_conversation.call_args[1]
            assert call_args['selected_repository'] is None


@pytest.mark.asyncio
async def test_new_conversation_with_provider_authentication_error(
    provider_handler_mock,
):
    provider_handler_mock.verify_repo_provider = AsyncMock(
        side_effect=AuthenticationError('auth error')
    )

    """Test creating a new conversation when provider authentication fails."""
    with _patch_store():
        # Mock the create_new_conversation function
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = 'test_conversation_id'

            # Create the request object
            test_request = InitSessionRequest(
                repository='test/repo',
                selected_branch='main',
                initial_user_msg='Hello, agent!',
            )

            # Call new_conversation
            with pytest.raises(AuthenticationError):
                await create_new_test_conversation(test_request)

            # Verify that verify_repo_provider was called with the repository
            provider_handler_mock.verify_repo_provider.assert_called_once_with(
                'test/repo', None
            )

            # Verify that create_new_conversation was not called
            mock_create_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_new_conversation_with_unsupported_params():
    """Test that unsupported parameters are rejected."""
    # Create a test request with an unsupported parameter
    with _patch_store():
        # Create a direct instance of InitSessionRequest to test validation
        with pytest.raises(Exception) as excinfo:
            # This should raise a validation error because of the extra parameter
            InitSessionRequest(
                repository='test/repo',
                selected_branch='main',
                initial_user_msg='Hello, agent!',
                unsupported_param='unsupported param',  # This should cause validation to fail
            )

        # Verify that the error message mentions the unsupported parameter
        assert 'Extra inputs are not permitted' in str(excinfo.value)
        assert 'unsupported_param' in str(excinfo.value)


@pytest.mark.asyncio
async def test_new_conversation_with_create_microagent(provider_handler_mock):
    """Test creating a new conversation with a CreateMicroagent object."""
    with _patch_store():
        # Mock the create_new_conversation function directly
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = MagicMock(
                conversation_id='test_conversation_id',
                url='https://my-conversation.com',
                session_api_key=None,
                status=ConversationStatus.RUNNING,
            )

            # Create the CreateMicroagent object
            create_microagent = CreateMicroagent(
                repo='test/repo',
                git_provider=ProviderType.GITHUB,
                title='Create a new microagent',
            )

            test_request = InitSessionRequest(
                repository=None,  # Not set in request, should be set from create_microagent
                selected_branch='main',
                initial_user_msg='Hello, agent!',
                create_microagent=create_microagent,
            )

            # Call new_conversation
            response = await create_new_test_conversation(test_request)

            # Verify the response
            assert isinstance(response, ConversationResponse)
            assert response.status == 'ok'
            assert response.conversation_id is not None
            assert isinstance(response.conversation_id, str)

            # Verify that create_new_conversation was called with the correct arguments
            mock_create_conversation.assert_called_once()
            call_args = mock_create_conversation.call_args[1]
            assert call_args['user_id'] == 'test_user'
            assert (
                call_args['selected_repository'] == 'test/repo'
            )  # Should be set from create_microagent
            assert call_args['selected_branch'] == 'main'
            assert call_args['initial_user_msg'] == 'Hello, agent!'
            assert (
                call_args['conversation_trigger']
                == ConversationTrigger.MICROAGENT_MANAGEMENT
            )
            assert (
                call_args['git_provider'] == ProviderType.GITHUB
            )  # Should be set from create_microagent


@pytest.mark.asyncio
async def test_new_conversation_with_create_microagent_repository_override(
    provider_handler_mock,
):
    """Test creating a new conversation with CreateMicroagent when repository is already set."""
    with _patch_store():
        # Mock the create_new_conversation function directly
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = MagicMock(
                conversation_id='test_conversation_id',
                url='https://my-conversation.com',
                session_api_key=None,
                status=ConversationStatus.RUNNING,
            )

            # Create the CreateMicroagent object
            create_microagent = CreateMicroagent(
                repo='microagent/repo',
                git_provider=ProviderType.GITLAB,
                title='Create a new microagent',
            )

            test_request = InitSessionRequest(
                repository='existing/repo',  # Already set in request
                selected_branch='main',
                initial_user_msg='Hello, agent!',
                create_microagent=create_microagent,
            )

            # Call new_conversation
            response = await create_new_test_conversation(test_request)

            # Verify the response
            assert isinstance(response, ConversationResponse)
            assert response.status == 'ok'
            assert response.conversation_id is not None
            assert isinstance(response.conversation_id, str)

            # Verify that create_new_conversation was called with the correct arguments
            mock_create_conversation.assert_called_once()
            call_args = mock_create_conversation.call_args[1]
            assert call_args['user_id'] == 'test_user'
            assert (
                call_args['selected_repository'] == 'existing/repo'
            )  # Should keep existing value
            assert call_args['selected_branch'] == 'main'
            assert call_args['initial_user_msg'] == 'Hello, agent!'
            assert (
                call_args['conversation_trigger']
                == ConversationTrigger.MICROAGENT_MANAGEMENT
            )
            assert (
                call_args['git_provider'] == ProviderType.GITLAB
            )  # Should be set from create_microagent


@pytest.mark.asyncio
async def test_new_conversation_with_create_microagent_minimal(provider_handler_mock):
    """Test creating a new conversation with minimal CreateMicroagent object (only repo field)."""
    with _patch_store():
        # Mock the create_new_conversation function directly
        with patch(
            'openhands.server.routes.manage_conversations.create_new_conversation'
        ) as mock_create_conversation:
            # Set up the mock to return a conversation ID
            mock_create_conversation.return_value = MagicMock(
                conversation_id='test_conversation_id',
                url='https://my-conversation.com',
                session_api_key=None,
                status=ConversationStatus.RUNNING,
            )

            # Create the CreateMicroagent object with only required field
            create_microagent = CreateMicroagent(
                repo='minimal/repo',
            )

            test_request = InitSessionRequest(
                repository=None,
                selected_branch='main',
                initial_user_msg='Hello, agent!',
                create_microagent=create_microagent,
            )

            # Call new_conversation
            response = await create_new_test_conversation(test_request)

            # Verify the response
            assert isinstance(response, ConversationResponse)
            assert response.status == 'ok'
            assert response.conversation_id is not None
            assert isinstance(response.conversation_id, str)

            # Verify that create_new_conversation was called with the correct arguments
            mock_create_conversation.assert_called_once()
            call_args = mock_create_conversation.call_args[1]
            assert call_args['user_id'] == 'test_user'
            assert (
                call_args['selected_repository'] == 'minimal/repo'
            )  # Should be set from create_microagent
            assert call_args['selected_branch'] == 'main'
            assert call_args['initial_user_msg'] == 'Hello, agent!'
            assert (
                call_args['conversation_trigger']
                == ConversationTrigger.MICROAGENT_MANAGEMENT
            )
            assert (
                call_args['git_provider'] is None
            )  # Should remain None since not set in create_microagent


@pytest.mark.asyncio
async def test_search_conversations_with_pr_number():
    """Test searching conversations includes pr_number field in response."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_with_pr',
                                    title='Conversation with PR',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                    pr_number=[123, 456],  # Multiple PR numbers
                                )
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository=None,
                        conversation_trigger=None,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify the result includes pr_number field
                    assert len(result_set.results) == 1
                    conversation_info = result_set.results[0]
                    assert conversation_info.pr_number == [123, 456]
                    assert conversation_info.conversation_id == 'conversation_with_pr'
                    assert conversation_info.title == 'Conversation with PR'


@pytest.mark.asyncio
async def test_search_conversations_with_empty_pr_number():
    """Test searching conversations with empty pr_number field."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_no_pr',
                                    title='Conversation without PR',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                    pr_number=[],  # Empty PR numbers list
                                )
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository=None,
                        conversation_trigger=None,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify the result includes empty pr_number field
                    assert len(result_set.results) == 1
                    conversation_info = result_set.results[0]
                    assert conversation_info.pr_number == []
                    assert conversation_info.conversation_id == 'conversation_no_pr'
                    assert conversation_info.title == 'Conversation without PR'


@pytest.mark.asyncio
async def test_search_conversations_with_single_pr_number():
    """Test searching conversations with single PR number."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_single_pr',
                                    title='Conversation with Single PR',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                    pr_number=[789],  # Single PR number
                                )
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository=None,
                        conversation_trigger=None,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify the result includes single pr_number
                    assert len(result_set.results) == 1
                    conversation_info = result_set.results[0]
                    assert conversation_info.pr_number == [789]
                    assert conversation_info.conversation_id == 'conversation_single_pr'
                    assert conversation_info.title == 'Conversation with Single PR'


@pytest.mark.asyncio
async def test_get_conversation_with_pr_number():
    """Test getting a single conversation includes pr_number field."""
    with _patch_store():
        # Mock the conversation store
        mock_store = MagicMock()
        mock_store.get_metadata = AsyncMock(
            return_value=ConversationMetadata(
                conversation_id='conversation_with_pr',
                title='Conversation with PR',
                created_at=datetime.fromisoformat('2025-01-01T00:00:00+00:00'),
                last_updated_at=datetime.fromisoformat('2025-01-01T00:01:00+00:00'),
                selected_repository='test/repo',
                user_id='12345',
                pr_number=[123, 456, 789],  # Multiple PR numbers
            )
        )

        # Mock the conversation manager
        with patch(
            'openhands.server.routes.manage_conversations.conversation_manager'
        ) as mock_manager:
            mock_manager.is_agent_loop_running = AsyncMock(return_value=False)
            mock_manager.get_connections = AsyncMock(return_value={})
            mock_manager.get_agent_loop_info = AsyncMock(return_value=[])

            conversation = await get_conversation(
                'conversation_with_pr', conversation_store=mock_store
            )

            expected = ConversationInfo(
                conversation_id='conversation_with_pr',
                title='Conversation with PR',
                created_at=datetime.fromisoformat('2025-01-01T00:00:00+00:00'),
                last_updated_at=datetime.fromisoformat('2025-01-01T00:01:00+00:00'),
                status=ConversationStatus.STOPPED,
                selected_repository='test/repo',
                num_connections=0,
                url=None,
                pr_number=[123, 456, 789],  # Should include PR numbers
            )
            assert conversation == expected


@pytest.mark.asyncio
async def test_search_conversations_multiple_with_pr_numbers():
    """Test searching conversations with multiple conversations having different PR numbers."""
    with _patch_store():
        with patch(
            'openhands.server.routes.manage_conversations.config'
        ) as mock_config:
            mock_config.conversation_max_age_seconds = 864000  # 10 days
            with patch(
                'openhands.server.routes.manage_conversations.conversation_manager'
            ) as mock_manager:

                async def mock_get_running_agent_loops(*args, **kwargs):
                    return set()

                async def mock_get_connections(*args, **kwargs):
                    return {}

                async def get_agent_loop_info(*args, **kwargs):
                    return []

                mock_manager.get_running_agent_loops = mock_get_running_agent_loops
                mock_manager.get_connections = mock_get_connections
                mock_manager.get_agent_loop_info = get_agent_loop_info
                with patch(
                    'openhands.server.routes.manage_conversations.datetime'
                ) as mock_datetime:
                    mock_datetime.now.return_value = datetime.fromisoformat(
                        '2025-01-01T00:00:00+00:00'
                    )
                    mock_datetime.fromisoformat = datetime.fromisoformat
                    mock_datetime.timezone = timezone

                    # Mock the conversation store
                    mock_store = MagicMock()
                    mock_store.search = AsyncMock(
                        return_value=ConversationInfoResultSet(
                            results=[
                                ConversationMetadata(
                                    conversation_id='conversation_1',
                                    title='Conversation 1',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                    pr_number=[100, 200],  # Multiple PR numbers
                                ),
                                ConversationMetadata(
                                    conversation_id='conversation_2',
                                    title='Conversation 2',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                    pr_number=[],  # Empty PR numbers
                                ),
                                ConversationMetadata(
                                    conversation_id='conversation_3',
                                    title='Conversation 3',
                                    created_at=datetime.fromisoformat(
                                        '2025-01-01T00:00:00+00:00'
                                    ),
                                    last_updated_at=datetime.fromisoformat(
                                        '2025-01-01T00:01:00+00:00'
                                    ),
                                    selected_repository='test/repo',
                                    user_id='12345',
                                    pr_number=[300],  # Single PR number
                                ),
                            ]
                        )
                    )

                    mock_app_conversation_service = AsyncMock()
                    mock_app_conversation_service.search_app_conversations.return_value = AppConversationPage(
                        items=[]
                    )

                    result_set = await search_conversations(
                        page_id=None,
                        limit=20,
                        selected_repository=None,
                        conversation_trigger=None,
                        conversation_store=mock_store,
                        app_conversation_service=mock_app_conversation_service,
                    )

                    # Verify all results include pr_number field
                    assert len(result_set.results) == 3

                    # Check first conversation
                    assert result_set.results[0].conversation_id == 'conversation_1'
                    assert result_set.results[0].pr_number == [100, 200]

                    # Check second conversation
                    assert result_set.results[1].conversation_id == 'conversation_2'
                    assert result_set.results[1].pr_number == []

                    # Check third conversation
                    assert result_set.results[2].conversation_id == 'conversation_3'
                    assert result_set.results[2].pr_number == [300]
