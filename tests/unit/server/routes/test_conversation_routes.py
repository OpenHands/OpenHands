import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import status
from fastapi.responses import JSONResponse

from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AgentType,
    AppConversation,
    AppConversationInfo,
    AppConversationPage,
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.microagent.microagent import KnowledgeMicroagent, RepoMicroagent
from openhands.microagent.types import MicroagentMetadata, MicroagentType
from openhands.server.data_models.conversation_info import ConversationStatus
from openhands.server.data_models.conversation_info_result_set import (
    ConversationInfoResultSet,
)
from openhands.server.routes.conversation import (
    AddMessageRequest,
    add_message,
    get_microagents,
)
from openhands.server.routes.manage_conversations import (
    _RESUME_GRACE_PERIOD,
    UpdateConversationRequest,
    get_conversation,
    search_conversations,
    update_conversation,
)
from openhands.server.session.conversation import ServerConversation
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)


@pytest.mark.asyncio
async def test_get_microagents():
    """Test the get_microagents function directly."""
    # Create mock microagents
    from openhands.core.config.mcp_config import MCPConfig, MCPStdioServerConfig

    repo_microagent = RepoMicroagent(
        name='test_repo',
        content='This is a test repo microagent',
        metadata=MicroagentMetadata(
            name='test_repo',
            type=MicroagentType.REPO_KNOWLEDGE,
            inputs=[],  # Empty inputs to match the expected behavior
            mcp_tools=MCPConfig(
                stdio_servers=[
                    MCPStdioServerConfig(name='git', command='git'),
                    MCPStdioServerConfig(name='file_editor', command='editor'),
                ]
            ),
        ),
        source='test_source',
        type=MicroagentType.REPO_KNOWLEDGE,
    )

    knowledge_microagent = KnowledgeMicroagent(
        name='test_knowledge',
        content='This is a test knowledge microagent',
        metadata=MicroagentMetadata(
            name='test_knowledge',
            type=MicroagentType.KNOWLEDGE,
            triggers=['test', 'knowledge'],
            inputs=[],  # Empty inputs to match the expected behavior
            mcp_tools=MCPConfig(
                stdio_servers=[
                    MCPStdioServerConfig(name='search', command='search'),
                    MCPStdioServerConfig(name='fetch', command='fetch'),
                ]
            ),
        ),
        source='test_source',
        type=MicroagentType.KNOWLEDGE,
    )

    # Mock the agent session and memory
    mock_memory = MagicMock()
    mock_memory.repo_microagents = {'test_repo': repo_microagent}
    mock_memory.knowledge_microagents = {'test_knowledge': knowledge_microagent}

    mock_agent_session = MagicMock()
    mock_agent_session.memory = mock_memory

    # Create a mock ServerConversation
    mock_conversation = MagicMock(spec=ServerConversation)
    mock_conversation.sid = 'test_sid'

    # Mock the conversation manager
    with patch(
        'openhands.server.routes.conversation.conversation_manager'
    ) as mock_manager:
        # Set up the mocks
        mock_manager.get_agent_session.return_value = mock_agent_session

        # Call the function directly
        response = await get_microagents(conversation=mock_conversation)

        # Verify the response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200

        # Parse the JSON content
        content = json.loads(response.body)
        assert 'microagents' in content
        assert len(content['microagents']) == 2

        # Check repo microagent
        repo_agent = next(m for m in content['microagents'] if m['name'] == 'test_repo')
        assert repo_agent['type'] == 'repo'
        assert repo_agent['content'] == 'This is a test repo microagent'
        assert repo_agent['triggers'] == []
        assert repo_agent['inputs'] == []  # Expect empty inputs
        assert repo_agent['tools'] == ['git', 'file_editor']

        # Check knowledge microagent
        knowledge_agent = next(
            m for m in content['microagents'] if m['name'] == 'test_knowledge'
        )
        assert knowledge_agent['type'] == 'knowledge'
        assert knowledge_agent['content'] == 'This is a test knowledge microagent'
        assert knowledge_agent['triggers'] == ['test', 'knowledge']
        assert knowledge_agent['inputs'] == []  # Expect empty inputs
        assert knowledge_agent['tools'] == ['search', 'fetch']


@pytest.mark.asyncio
async def test_get_microagents_no_agent_session():
    """Test the get_microagents function when no agent session is found."""
    # Create a mock ServerConversation
    mock_conversation = MagicMock(spec=ServerConversation)
    mock_conversation.sid = 'test_sid'

    # Mock the conversation manager
    with patch(
        'openhands.server.routes.conversation.conversation_manager'
    ) as mock_manager:
        # Set up the mocks
        mock_manager.get_agent_session.return_value = None

        # Call the function directly
        response = await get_microagents(conversation=mock_conversation)

        # Verify the response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 404

        # Parse the JSON content
        content = json.loads(response.body)
        assert 'error' in content
        assert 'Agent session not found' in content['error']


@pytest.mark.asyncio
async def test_get_microagents_exception():
    """Test the get_microagents function when an exception occurs."""
    # Create a mock ServerConversation
    mock_conversation = MagicMock(spec=ServerConversation)
    mock_conversation.sid = 'test_sid'

    # Mock the conversation manager
    with patch(
        'openhands.server.routes.conversation.conversation_manager'
    ) as mock_manager:
        # Set up the mocks to raise an exception
        mock_manager.get_agent_session.side_effect = Exception('Test exception')

        # Call the function directly
        response = await get_microagents(conversation=mock_conversation)

        # Verify the response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Parse the JSON content
        content = json.loads(response.body)
        assert 'error' in content
        assert 'Test exception' in content['error']


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_success():
    """Test successful conversation update."""
    # Mock data
    conversation_id = 'test_conversation_123'
    user_id = 'test_user_456'
    original_title = 'Original Title'
    new_title = 'Updated Title'

    # Create mock metadata
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=user_id,
        title=original_title,
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)
    mock_conversation_store.save_metadata = AsyncMock()

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Mock the conversation manager socket
    mock_sio = AsyncMock()

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
        )

        # Verify the result
        assert result is True

        # Verify metadata was fetched
        mock_conversation_store.get_metadata.assert_called_once_with(conversation_id)

        # Verify metadata was updated and saved
        mock_conversation_store.save_metadata.assert_called_once()
        saved_metadata = mock_conversation_store.save_metadata.call_args[0][0]
        assert saved_metadata.title == new_title.strip()
        assert saved_metadata.last_updated_at is not None

        # Verify socket emission
        mock_sio.emit.assert_called_once()
        emit_call = mock_sio.emit.call_args
        assert emit_call[0][0] == 'oh_event'
        assert emit_call[0][1]['conversation_title'] == new_title
        assert emit_call[1]['to'] == f'room:{conversation_id}'


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_not_found():
    """Test conversation update when conversation doesn't exist."""
    conversation_id = 'nonexistent_conversation'
    user_id = 'test_user_456'

    # Create mock conversation store that raises FileNotFoundError
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(side_effect=FileNotFoundError())

    # Create update request
    update_request = UpdateConversationRequest(title='New Title')

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
    )

    # Verify the result is a 404 error response
    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_404_NOT_FOUND

    # Parse the JSON content
    content = json.loads(result.body)
    assert content['status'] == 'error'
    assert content['message'] == 'Conversation not found'
    assert content['msg_id'] == 'CONVERSATION$NOT_FOUND'


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_permission_denied():
    """Test conversation update when user doesn't own the conversation."""
    conversation_id = 'test_conversation_123'
    user_id = 'test_user_456'
    owner_id = 'different_user_789'

    # Create mock metadata owned by different user
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=owner_id,
        title='Original Title',
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)

    # Create update request
    update_request = UpdateConversationRequest(title='New Title')

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
    )

    # Verify the result is a 403 error response
    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_403_FORBIDDEN

    # Parse the JSON content
    content = json.loads(result.body)
    assert content['status'] == 'error'
    assert (
        content['message']
        == 'Permission denied: You can only update your own conversations'
    )
    assert content['msg_id'] == 'AUTHORIZATION$PERMISSION_DENIED'


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_permission_denied_no_user_id():
    """Test conversation update when user_id is None and metadata has user_id."""
    conversation_id = 'test_conversation_123'
    user_id = None
    owner_id = 'some_user_789'

    # Create mock metadata owned by a user
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=owner_id,
        title='Original Title',
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)

    # Create update request
    update_request = UpdateConversationRequest(title='New Title')

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
    )

    # Verify the result is successful (current logic allows this)
    assert result is True


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_socket_emission_error():
    """Test conversation update when socket emission fails."""
    conversation_id = 'test_conversation_123'
    user_id = 'test_user_456'
    new_title = 'Updated Title'

    # Create mock metadata
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=user_id,
        title='Original Title',
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)
    mock_conversation_store.save_metadata = AsyncMock()

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Mock the conversation manager socket to raise an exception
    mock_sio = AsyncMock()
    mock_sio.emit.side_effect = Exception('Socket error')

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function (should still succeed despite socket error)
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
        )

        # Verify the result is still successful
        assert result is True

        # Verify metadata was still saved
        mock_conversation_store.save_metadata.assert_called_once()


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_general_exception():
    """Test conversation update when an unexpected exception occurs."""
    conversation_id = 'test_conversation_123'
    user_id = 'test_user_456'

    # Create mock conversation store that raises a general exception
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(
        side_effect=Exception('Database error')
    )

    # Create update request
    update_request = UpdateConversationRequest(title='New Title')

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
    )

    # Verify the result is a 500 error response
    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # Parse the JSON content
    content = json.loads(result.body)
    assert content['status'] == 'error'
    assert 'Failed to update conversation' in content['message']
    assert content['msg_id'] == 'CONVERSATION$UPDATE_ERROR'


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_title_whitespace_trimming():
    """Test that conversation title is properly trimmed of whitespace."""
    conversation_id = 'test_conversation_123'
    user_id = 'test_user_456'
    title_with_whitespace = '  Trimmed Title  '
    expected_title = 'Trimmed Title'

    # Create mock metadata
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=user_id,
        title='Original Title',
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)
    mock_conversation_store.save_metadata = AsyncMock()

    # Create update request with whitespace
    update_request = UpdateConversationRequest(title=title_with_whitespace)

    # Mock the conversation manager socket
    mock_sio = AsyncMock()

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
        )

        # Verify the result
        assert result is True

        # Verify metadata was updated with trimmed title
        mock_conversation_store.save_metadata.assert_called_once()
        saved_metadata = mock_conversation_store.save_metadata.call_args[0][0]
        assert saved_metadata.title == expected_title


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_user_owns_conversation():
    """Test successful update when user owns the conversation."""
    conversation_id = 'test_conversation_123'
    user_id = 'test_user_456'
    new_title = 'Updated Title'

    # Create mock metadata owned by the same user
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=user_id,  # Same user
        title='Original Title',
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)
    mock_conversation_store.save_metadata = AsyncMock()

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Mock the conversation manager socket
    mock_sio = AsyncMock()

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
        )

        # Verify success
        assert result is True
        mock_conversation_store.save_metadata.assert_called_once()


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_last_updated_at_set():
    """Test that last_updated_at is properly set when updating."""
    conversation_id = 'test_conversation_123'
    user_id = 'test_user_456'
    new_title = 'Updated Title'

    # Create mock metadata
    original_timestamp = datetime.now(timezone.utc)
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=user_id,
        title='Original Title',
        selected_repository=None,
        last_updated_at=original_timestamp,
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)
    mock_conversation_store.save_metadata = AsyncMock()

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Mock the conversation manager socket
    mock_sio = AsyncMock()

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
        )

        # Verify success
        assert result is True

        # Verify last_updated_at was updated
        mock_conversation_store.save_metadata.assert_called_once()
        saved_metadata = mock_conversation_store.save_metadata.call_args[0][0]
        assert saved_metadata.last_updated_at > original_timestamp


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_conversation_no_user_id_no_metadata_user_id():
    """Test successful update when both user_id and metadata.user_id are None."""
    conversation_id = 'test_conversation_123'
    user_id = None
    new_title = 'Updated Title'

    # Create mock metadata with no user_id
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=None,
        title='Original Title',
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)
    mock_conversation_store.save_metadata = AsyncMock()

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Mock the conversation manager socket
    mock_sio = AsyncMock()

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
        )

        # Verify success (should work when both are None)
        assert result is True
        mock_conversation_store.save_metadata.assert_called_once()


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_v1_conversation_success():
    """Test successful V1 conversation update."""
    # Mock data
    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)
    user_id = 'test_user_456'
    original_title = 'Original V1 Title'
    new_title = 'Updated V1 Title'

    # Create mock V1 conversation info
    mock_app_conversation_info = AppConversationInfo(
        id=conversation_uuid,
        created_by_user_id=user_id,
        sandbox_id='test_sandbox_123',
        title=original_title,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create mock app conversation info service
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)
    mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )
    mock_app_conversation_info_service.save_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )

    # Create mock conversation store (won't be used for V1)
    mock_conversation_store = MagicMock(spec=ConversationStore)

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
        app_conversation_info_service=mock_app_conversation_info_service,
    )

    # Verify the result
    assert result is True

    # Verify V1 service was called
    mock_app_conversation_info_service.get_app_conversation_info.assert_called_once_with(
        conversation_uuid
    )
    mock_app_conversation_info_service.save_app_conversation_info.assert_called_once()

    # Verify the conversation store was NOT called (V1 doesn't use it)
    mock_conversation_store.get_metadata.assert_not_called()

    # Verify the saved info has updated title
    saved_info = (
        mock_app_conversation_info_service.save_app_conversation_info.call_args[0][0]
    )
    assert saved_info.title == new_title.strip()
    assert saved_info.updated_at is not None


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_v1_conversation_not_found():
    """Test V1 conversation update when conversation doesn't exist."""
    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)
    user_id = 'test_user_456'

    # Create mock app conversation info service that returns None
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)
    mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
        return_value=None
    )

    # Create mock conversation store that also raises FileNotFoundError
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(side_effect=FileNotFoundError())

    # Create update request
    update_request = UpdateConversationRequest(title='New Title')

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
        app_conversation_info_service=mock_app_conversation_info_service,
    )

    # Verify the result is a 404 error response
    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_404_NOT_FOUND

    # Parse the JSON content
    content = json.loads(result.body)
    assert content['status'] == 'error'
    assert content['message'] == 'Conversation not found'
    assert content['msg_id'] == 'CONVERSATION$NOT_FOUND'


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_v1_conversation_permission_denied():
    """Test V1 conversation update when user doesn't own the conversation."""
    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)
    user_id = 'test_user_456'
    owner_id = 'different_user_789'

    # Create mock V1 conversation info owned by different user
    mock_app_conversation_info = AppConversationInfo(
        id=conversation_uuid,
        created_by_user_id=owner_id,
        sandbox_id='test_sandbox_123',
        title='Original Title',
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create mock app conversation info service
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)
    mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )

    # Create mock conversation store (won't be used)
    mock_conversation_store = MagicMock(spec=ConversationStore)

    # Create update request
    update_request = UpdateConversationRequest(title='New Title')

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
        app_conversation_info_service=mock_app_conversation_info_service,
    )

    # Verify the result is a 403 error response
    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_403_FORBIDDEN

    # Parse the JSON content
    content = json.loads(result.body)
    assert content['status'] == 'error'
    assert (
        content['message']
        == 'Permission denied: You can only update your own conversations'
    )
    assert content['msg_id'] == 'AUTHORIZATION$PERMISSION_DENIED'

    # Verify save was NOT called
    mock_app_conversation_info_service.save_app_conversation_info.assert_not_called()


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_v1_conversation_save_assertion_error():
    """Test V1 conversation update when save raises AssertionError (permission check)."""
    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)
    user_id = 'test_user_456'

    # Create mock V1 conversation info
    mock_app_conversation_info = AppConversationInfo(
        id=conversation_uuid,
        created_by_user_id=user_id,
        sandbox_id='test_sandbox_123',
        title='Original Title',
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create mock app conversation info service
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)
    mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )
    # Simulate AssertionError on save (permission check in service)
    mock_app_conversation_info_service.save_app_conversation_info = AsyncMock(
        side_effect=AssertionError('User does not own conversation')
    )

    # Create mock conversation store (won't be used)
    mock_conversation_store = MagicMock(spec=ConversationStore)

    # Create update request
    update_request = UpdateConversationRequest(title='New Title')

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
        app_conversation_info_service=mock_app_conversation_info_service,
    )

    # Verify the result is a 403 error response
    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_403_FORBIDDEN

    # Parse the JSON content
    content = json.loads(result.body)
    assert content['status'] == 'error'
    assert (
        content['message']
        == 'Permission denied: You can only update your own conversations'
    )
    assert content['msg_id'] == 'AUTHORIZATION$PERMISSION_DENIED'


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_v1_conversation_title_whitespace_trimming():
    """Test that V1 conversation title is properly trimmed of whitespace."""
    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)
    user_id = 'test_user_456'
    title_with_whitespace = '  Trimmed V1 Title  '
    expected_title = 'Trimmed V1 Title'

    # Create mock V1 conversation info
    mock_app_conversation_info = AppConversationInfo(
        id=conversation_uuid,
        created_by_user_id=user_id,
        sandbox_id='test_sandbox_123',
        title='Original Title',
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create mock app conversation info service
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)
    mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )
    mock_app_conversation_info_service.save_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )

    # Create mock conversation store (won't be used)
    mock_conversation_store = MagicMock(spec=ConversationStore)

    # Create update request with whitespace
    update_request = UpdateConversationRequest(title=title_with_whitespace)

    # Call the function
    result = await update_conversation(
        conversation_id=conversation_id,
        data=update_request,
        user_id=user_id,
        conversation_store=mock_conversation_store,
        app_conversation_info_service=mock_app_conversation_info_service,
    )

    # Verify the result
    assert result is True

    # Verify the saved info has trimmed title
    saved_info = (
        mock_app_conversation_info_service.save_app_conversation_info.call_args[0][0]
    )
    assert saved_info.title == expected_title


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_v1_conversation_invalid_uuid_falls_back_to_v0():
    """Test that invalid UUID conversation_id falls back to V0 logic."""
    conversation_id = 'not_a_valid_uuid_123'
    user_id = 'test_user_456'
    new_title = 'Updated Title'

    # Create mock V0 metadata
    mock_metadata = ConversationMetadata(
        conversation_id=conversation_id,
        user_id=user_id,
        title='Original Title',
        selected_repository=None,
        last_updated_at=datetime.now(timezone.utc),
    )

    # Create mock conversation store for V0
    mock_conversation_store = MagicMock(spec=ConversationStore)
    mock_conversation_store.get_metadata = AsyncMock(return_value=mock_metadata)
    mock_conversation_store.save_metadata = AsyncMock()

    # Create mock app conversation info service (won't be called)
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Mock the conversation manager socket
    mock_sio = AsyncMock()

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
            app_conversation_info_service=mock_app_conversation_info_service,
        )

        # Verify the result is successful
        assert result is True

        # Verify V0 store was used, not V1 service
        mock_conversation_store.get_metadata.assert_called_once_with(conversation_id)
        mock_conversation_store.save_metadata.assert_called_once()
        mock_app_conversation_info_service.get_app_conversation_info.assert_not_called()


@pytest.mark.update_conversation
@pytest.mark.asyncio
async def test_update_v1_conversation_no_socket_emission():
    """Test that V1 conversation update does NOT emit socket.io events."""
    conversation_uuid = uuid4()
    conversation_id = str(conversation_uuid)
    user_id = 'test_user_456'
    new_title = 'Updated V1 Title'

    # Create mock V1 conversation info
    mock_app_conversation_info = AppConversationInfo(
        id=conversation_uuid,
        created_by_user_id=user_id,
        sandbox_id='test_sandbox_123',
        title='Original Title',
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create mock app conversation info service
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)
    mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )
    mock_app_conversation_info_service.save_app_conversation_info = AsyncMock(
        return_value=mock_app_conversation_info
    )

    # Create mock conversation store (won't be used)
    mock_conversation_store = MagicMock(spec=ConversationStore)

    # Create update request
    update_request = UpdateConversationRequest(title=new_title)

    # Mock the conversation manager socket
    mock_sio = AsyncMock()

    with patch(
        'openhands.server.routes.manage_conversations.conversation_manager'
    ) as mock_manager:
        mock_manager.sio = mock_sio

        # Call the function
        result = await update_conversation(
            conversation_id=conversation_id,
            data=update_request,
            user_id=user_id,
            conversation_store=mock_conversation_store,
            app_conversation_info_service=mock_app_conversation_info_service,
        )

        # Verify the result is successful
        assert result is True

        # Verify socket.io was NOT called for V1 conversation
        mock_sio.emit.assert_not_called()


@pytest.mark.asyncio
async def test_add_message_success():
    """Test successful message addition to conversation."""
    # Create a mock ServerConversation
    mock_conversation = MagicMock(spec=ServerConversation)
    mock_conversation.sid = 'test_conversation_123'

    # Create message request
    message_request = AddMessageRequest(message='Hello, this is a test message!')

    # Mock the conversation manager
    with patch(
        'openhands.server.routes.conversation.conversation_manager'
    ) as mock_manager:
        mock_manager.send_event_to_conversation = AsyncMock()

        # Call the function directly
        response = await add_message(
            data=message_request, conversation=mock_conversation
        )

        # Verify the response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200

        # Parse the JSON content
        content = json.loads(response.body)
        assert content['success'] is True

        # Verify that send_event_to_conversation was called
        mock_manager.send_event_to_conversation.assert_called_once()
        call_args = mock_manager.send_event_to_conversation.call_args
        assert call_args[0][0] == 'test_conversation_123'  # conversation ID

        # Verify the message data structure
        message_data = call_args[0][1]
        assert message_data['action'] == 'message'
        assert message_data['args']['content'] == 'Hello, this is a test message!'


@pytest.mark.asyncio
async def test_add_message_conversation_manager_error():
    """Test add_message when conversation manager raises an exception."""
    # Create a mock ServerConversation
    mock_conversation = MagicMock(spec=ServerConversation)
    mock_conversation.sid = 'test_conversation_123'

    # Create message request
    message_request = AddMessageRequest(message='Test message')

    # Mock the conversation manager to raise an exception
    with patch(
        'openhands.server.routes.conversation.conversation_manager'
    ) as mock_manager:
        mock_manager.send_event_to_conversation = AsyncMock(
            side_effect=Exception('Conversation manager error')
        )

        # Call the function directly
        response = await add_message(
            data=message_request, conversation=mock_conversation
        )

        # Verify the response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Parse the JSON content
        content = json.loads(response.body)
        assert content['success'] is False
        assert 'Conversation manager error' in content['error']


@pytest.mark.asyncio
async def test_add_message_empty_message():
    """Test add_message with an empty message."""
    # Create a mock ServerConversation
    mock_conversation = MagicMock(spec=ServerConversation)
    mock_conversation.sid = 'test_conversation_123'

    # Create message request with empty message
    message_request = AddMessageRequest(message='')

    # Mock the conversation manager
    with patch(
        'openhands.server.routes.conversation.conversation_manager'
    ) as mock_manager:
        mock_manager.send_event_to_conversation = AsyncMock()

        # Call the function directly
        response = await add_message(
            data=message_request, conversation=mock_conversation
        )

        # Verify the response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200

        # Parse the JSON content
        content = json.loads(response.body)
        assert content['success'] is True

        # Verify that send_event_to_conversation was called with empty content
        mock_manager.send_event_to_conversation.assert_called_once()
        call_args = mock_manager.send_event_to_conversation.call_args
        message_data = call_args[0][1]
        assert message_data['args']['content'] == ''


@pytest.mark.sub_conversation
@pytest.mark.asyncio
async def test_create_sub_conversation_with_planning_agent():
    """Test creating a sub-conversation from a parent conversation with planning agent."""
    from uuid import uuid4

    parent_conversation_id = uuid4()
    user_id = 'test_user_456'
    sandbox_id = 'test_sandbox_123'

    # Create mock parent conversation info
    parent_info = AppConversationInfo(
        id=parent_conversation_id,
        created_by_user_id=user_id,
        sandbox_id=sandbox_id,
        selected_repository='test/repo',
        selected_branch='main',
        git_provider=None,
        title='Parent Conversation',
        llm_model='anthropic/claude-3-5-sonnet-20241022',
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create sub-conversation request with planning agent
    sub_conversation_request = AppConversationStartRequest(
        parent_conversation_id=parent_conversation_id,
        agent_type=AgentType.PLAN,
        initial_message=None,
    )

    # Create mock app conversation service
    mock_app_conversation_service = MagicMock(spec=AppConversationService)
    mock_app_conversation_info_service = MagicMock(spec=AppConversationInfoService)

    # Mock the service to return parent info
    mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
        return_value=parent_info
    )

    # Mock the start_app_conversation method to return a task
    async def mock_start_generator(request):
        task = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id=user_id,
            status=AppConversationStartTaskStatus.READY,
            app_conversation_id=uuid4(),
            sandbox_id=sandbox_id,
            agent_server_url='http://agent-server:8000',
            request=request,
        )
        yield task

    mock_app_conversation_service.start_app_conversation = mock_start_generator

    # Test the service method directly
    async for task in mock_app_conversation_service.start_app_conversation(
        sub_conversation_request
    ):
        # Verify the task was created with planning agent
        assert task is not None
        assert task.status == AppConversationStartTaskStatus.READY
        assert task.request.agent_type == AgentType.PLAN
        assert task.request.parent_conversation_id == parent_conversation_id
        assert task.sandbox_id == sandbox_id
        break


@pytest.mark.asyncio
async def test_search_conversations_include_sub_conversations_default_false():
    """Test that include_sub_conversations defaults to False when not provided."""
    with patch('openhands.server.routes.manage_conversations.config') as mock_config:
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
                    return_value=ConversationInfoResultSet(results=[])
                )

                # Create a mock app conversation service
                mock_app_conversation_service = AsyncMock()
                mock_app_conversation_service.search_app_conversations.return_value = (
                    AppConversationPage(items=[])
                )

                # Call search_conversations without include_sub_conversations parameter
                await search_conversations(
                    page_id=None,
                    limit=20,
                    selected_repository=None,
                    conversation_trigger=None,
                    conversation_store=mock_store,
                    app_conversation_service=mock_app_conversation_service,
                )

                # Verify that search_app_conversations was called with include_sub_conversations=False (default)
                mock_app_conversation_service.search_app_conversations.assert_called_once()
                call_kwargs = (
                    mock_app_conversation_service.search_app_conversations.call_args[1]
                )
                assert call_kwargs.get('include_sub_conversations') is False


@pytest.mark.asyncio
async def test_search_conversations_include_sub_conversations_explicit_false():
    """Test that include_sub_conversations=False is properly passed through."""
    with patch('openhands.server.routes.manage_conversations.config') as mock_config:
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
                    return_value=ConversationInfoResultSet(results=[])
                )

                # Create a mock app conversation service
                mock_app_conversation_service = AsyncMock()
                mock_app_conversation_service.search_app_conversations.return_value = (
                    AppConversationPage(items=[])
                )

                # Call search_conversations with include_sub_conversations=False
                await search_conversations(
                    page_id=None,
                    limit=20,
                    selected_repository=None,
                    conversation_trigger=None,
                    include_sub_conversations=False,
                    conversation_store=mock_store,
                    app_conversation_service=mock_app_conversation_service,
                )

                # Verify that search_app_conversations was called with include_sub_conversations=False
                mock_app_conversation_service.search_app_conversations.assert_called_once()
                call_kwargs = (
                    mock_app_conversation_service.search_app_conversations.call_args[1]
                )
                assert call_kwargs.get('include_sub_conversations') is False


@pytest.mark.asyncio
async def test_search_conversations_include_sub_conversations_explicit_true():
    """Test that include_sub_conversations=True is properly passed through."""
    with patch('openhands.server.routes.manage_conversations.config') as mock_config:
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
                    return_value=ConversationInfoResultSet(results=[])
                )

                # Create a mock app conversation service
                mock_app_conversation_service = AsyncMock()
                mock_app_conversation_service.search_app_conversations.return_value = (
                    AppConversationPage(items=[])
                )

                # Call search_conversations with include_sub_conversations=True
                await search_conversations(
                    page_id=None,
                    limit=20,
                    selected_repository=None,
                    conversation_trigger=None,
                    include_sub_conversations=True,
                    conversation_store=mock_store,
                    app_conversation_service=mock_app_conversation_service,
                )

                # Verify that search_app_conversations was called with include_sub_conversations=True
                mock_app_conversation_service.search_app_conversations.assert_called_once()
                call_kwargs = (
                    mock_app_conversation_service.search_app_conversations.call_args[1]
                )
                assert call_kwargs.get('include_sub_conversations') is True


@pytest.mark.asyncio
async def test_search_conversations_include_sub_conversations_with_other_filters():
    """Test that include_sub_conversations works correctly with other filters."""
    with patch('openhands.server.routes.manage_conversations.config') as mock_config:
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
                    return_value=ConversationInfoResultSet(results=[])
                )

                # Create a mock app conversation service
                mock_app_conversation_service = AsyncMock()
                mock_app_conversation_service.search_app_conversations.return_value = (
                    AppConversationPage(items=[])
                )

                # Create a valid base64-encoded page_id for testing
                import base64

                page_id_data = json.dumps({'v0': None, 'v1': 'test_v1_page_id'})
                encoded_page_id = base64.b64encode(page_id_data.encode()).decode()

                # Call search_conversations with include_sub_conversations and other filters
                await search_conversations(
                    page_id=encoded_page_id,
                    limit=50,
                    selected_repository='test/repo',
                    conversation_trigger=ConversationTrigger.GUI,
                    include_sub_conversations=True,
                    conversation_store=mock_store,
                    app_conversation_service=mock_app_conversation_service,
                )

                # Verify that search_app_conversations was called with all parameters including include_sub_conversations=True
                mock_app_conversation_service.search_app_conversations.assert_called_once()
                call_kwargs = (
                    mock_app_conversation_service.search_app_conversations.call_args[1]
                )
                assert call_kwargs.get('include_sub_conversations') is True
                assert call_kwargs.get('page_id') == 'test_v1_page_id'
                assert call_kwargs.get('limit') == 50


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'sandbox_status,execution_status,server_uptime,server_error,expected_status,should_call_server',
    [
        # RUNNING sandbox, no execution_status, server responds within grace period -> STARTING
        (SandboxStatus.RUNNING, None, 30, None, ConversationStatus.STARTING, True),
        # RUNNING sandbox, no execution_status, server responds past grace period -> RUNNING
        (
            SandboxStatus.RUNNING,
            None,
            _RESUME_GRACE_PERIOD + 10,
            None,
            ConversationStatus.RUNNING,
            True,
        ),
        # RUNNING sandbox, no execution_status, server unresponsive -> STARTING
        (
            SandboxStatus.RUNNING,
            None,
            None,
            httpx.ConnectError('Connection refused'),
            ConversationStatus.STARTING,
            True,
        ),
        # RUNNING sandbox, WITH execution_status -> RUNNING, skip server check
        (SandboxStatus.RUNNING, 'IDLE', None, None, ConversationStatus.RUNNING, False),
        # Non-RUNNING sandbox -> STARTING, skip server check
        (SandboxStatus.STARTING, None, None, None, ConversationStatus.STARTING, False),
    ],
    ids=[
        'running_no_exec_status_within_grace_period',
        'running_no_exec_status_past_grace_period',
        'running_no_exec_status_server_unresponsive',
        'running_with_exec_status_skips_check',
        'non_running_skips_check',
    ],
)
async def test_get_conversation_resume_status_handling(
    sandbox_status,
    execution_status,
    server_uptime,
    server_error,
    expected_status,
    should_call_server,
):
    """Test get_conversation handles resume status correctly for various scenarios."""
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    conversation_id = uuid4()

    # Convert string execution_status to enum if provided
    exec_status = None
    if execution_status == 'IDLE':
        exec_status = ConversationExecutionStatus.IDLE

    mock_app_conversation = AppConversation(
        id=conversation_id,
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        sandbox_status=sandbox_status,
        execution_status=exec_status,
        conversation_url='https://sandbox.example.com/conversation',
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    mock_app_conversation_service = AsyncMock(spec=AppConversationService)
    mock_app_conversation_service.get_app_conversation.return_value = (
        mock_app_conversation
    )
    mock_conversation_store = AsyncMock(spec=ConversationStore)
    mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)

    if server_error:
        mock_httpx_client.get.side_effect = server_error
    elif server_uptime is not None:
        mock_response = MagicMock()
        mock_response.json.return_value = {'uptime': server_uptime}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

    result = await get_conversation(
        conversation_id=str(conversation_id),
        conversation_store=mock_conversation_store,
        app_conversation_service=mock_app_conversation_service,
        httpx_client=mock_httpx_client,
    )

    assert result is not None
    assert result.status == expected_status

    if should_call_server:
        mock_httpx_client.get.assert_called_once_with(
            'https://sandbox.example.com/server_info'
        )
    else:
        mock_httpx_client.get.assert_not_called()
