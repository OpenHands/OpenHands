"""Unit tests for the app_conversation_router endpoints.

This module tests the batch_get_app_conversations endpoint,
focusing on UUID string parsing, validation, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
    AppConversationInfo,
    AppConversationPage,
    SwitchProfileRequest,
)
from openhands.app_server.app_conversation.app_conversation_router import (
    AgentServerContext,
    batch_get_app_conversations,
    count_app_conversations,
    search_app_conversations,
    switch_conversation_profile,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.app_server.settings.llm_profiles import LLMProfiles
from openhands.app_server.settings.settings_models import Settings
from openhands.sdk.llm import LLM


def _make_mock_app_conversation(
    conversation_id=None, user_id='test-user', sandbox_id=None
):
    """Create a mock AppConversation for testing."""
    if conversation_id is None:
        conversation_id = uuid4()
    if sandbox_id is None:
        sandbox_id = str(uuid4())
    return AppConversation(
        id=conversation_id,
        created_by_user_id=user_id,
        sandbox_id=sandbox_id,
        sandbox_status=SandboxStatus.RUNNING,
    )


def _make_mock_service(
    get_conversation_return=None,
    batch_get_return=None,
    search_return=None,
    count_return=0,
):
    """Create a mock AppConversationService for testing."""
    service = MagicMock()
    service.get_app_conversation = AsyncMock(return_value=get_conversation_return)
    service.batch_get_app_conversations = AsyncMock(return_value=batch_get_return or [])
    service.search_app_conversations = AsyncMock(
        return_value=search_return or AppConversationPage(items=[])
    )
    service.count_app_conversations = AsyncMock(return_value=count_return)
    return service


@pytest.mark.asyncio
class TestBatchGetAppConversations:
    """Test suite for batch_get_app_conversations endpoint."""

    async def test_accepts_uuids_with_dashes(self):
        """Test that standard UUIDs with dashes are accepted.

        Arrange: Create UUIDs with dashes and mock service
        Act: Call batch_get_app_conversations
        Assert: Service is called with parsed UUIDs
        """
        # Arrange
        uuid1 = uuid4()
        uuid2 = uuid4()
        ids = [str(uuid1), str(uuid2)]

        mock_conversations = [
            _make_mock_app_conversation(uuid1),
            _make_mock_app_conversation(uuid2),
        ]
        mock_service = _make_mock_service(batch_get_return=mock_conversations)

        # Act
        result = await batch_get_app_conversations(
            ids=ids,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.batch_get_app_conversations.assert_called_once()
        call_args = mock_service.batch_get_app_conversations.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == uuid1
        assert call_args[1] == uuid2
        assert result == mock_conversations

    async def test_accepts_uuids_without_dashes(self):
        """Test that UUIDs without dashes are accepted and correctly parsed.

        Arrange: Create UUIDs without dashes
        Act: Call batch_get_app_conversations
        Assert: Service is called with correctly parsed UUIDs
        """
        # Arrange
        uuid1 = uuid4()
        uuid2 = uuid4()
        # Remove dashes from UUID strings
        ids = [str(uuid1).replace('-', ''), str(uuid2).replace('-', '')]

        mock_conversations = [
            _make_mock_app_conversation(uuid1),
            _make_mock_app_conversation(uuid2),
        ]
        mock_service = _make_mock_service(batch_get_return=mock_conversations)

        # Act
        result = await batch_get_app_conversations(
            ids=ids,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.batch_get_app_conversations.assert_called_once()
        call_args = mock_service.batch_get_app_conversations.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == uuid1
        assert call_args[1] == uuid2
        assert result == mock_conversations

    async def test_returns_400_for_invalid_uuid_strings(self):
        """Test that invalid UUID strings return 400 Bad Request.

        Arrange: Create list with invalid UUID strings
        Act: Call batch_get_app_conversations
        Assert: HTTPException is raised with 400 status and details about invalid IDs
        """
        # Arrange
        valid_uuid = str(uuid4())
        invalid_ids = ['not-a-uuid', 'also-invalid', '12345']
        ids = [valid_uuid] + invalid_ids

        mock_service = _make_mock_service()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await batch_get_app_conversations(
                ids=ids,
                app_conversation_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid UUID format' in exc_info.value.detail
        # All invalid IDs should be mentioned in the error
        for invalid_id in invalid_ids:
            assert invalid_id in exc_info.value.detail

    async def test_returns_400_for_too_many_ids(self):
        """Test that requesting too many IDs returns 400 Bad Request.

        Arrange: Create list with 100+ IDs
        Act: Call batch_get_app_conversations
        Assert: HTTPException is raised with 400 status
        """
        # Arrange
        ids = [str(uuid4()) for _ in range(100)]
        mock_service = _make_mock_service()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await batch_get_app_conversations(
                ids=ids,
                app_conversation_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Too many ids' in exc_info.value.detail

    async def test_returns_empty_list_for_empty_input(self):
        """Test that empty input returns empty list.

        Arrange: Create empty list of IDs
        Act: Call batch_get_app_conversations
        Assert: Empty list is returned
        """
        # Arrange
        mock_service = _make_mock_service(batch_get_return=[])

        # Act
        result = await batch_get_app_conversations(
            ids=[],
            app_conversation_service=mock_service,
        )

        # Assert
        assert result == []
        mock_service.batch_get_app_conversations.assert_called_once_with([])

    async def test_returns_none_for_missing_conversations(self):
        """Test that None is returned for conversations that don't exist.

        Arrange: Request IDs where some don't exist
        Act: Call batch_get_app_conversations
        Assert: Result contains None for missing conversations
        """
        # Arrange
        uuid1 = uuid4()
        uuid2 = uuid4()
        ids = [str(uuid1), str(uuid2)]

        # Only first conversation exists
        mock_service = _make_mock_service(
            batch_get_return=[_make_mock_app_conversation(uuid1), None]
        )

        # Act
        result = await batch_get_app_conversations(
            ids=ids,
            app_conversation_service=mock_service,
        )

        # Assert
        assert len(result) == 2
        assert result[0] is not None
        assert result[0].id == uuid1
        assert result[1] is None


@pytest.mark.asyncio
class TestSearchAppConversations:
    """Test suite for search_app_conversations endpoint."""

    async def test_search_with_sandbox_id_filter(self):
        """Test that sandbox_id__eq filter is passed to the service.

        Arrange: Create mock service and specific sandbox_id
        Act: Call search_app_conversations with sandbox_id__eq
        Assert: Service is called with the sandbox_id__eq parameter
        """
        # Arrange
        sandbox_id = 'test-sandbox-123'
        mock_conversation = _make_mock_app_conversation(sandbox_id=sandbox_id)
        mock_service = _make_mock_service(
            search_return=AppConversationPage(items=[mock_conversation])
        )

        # Act
        result = await search_app_conversations(
            sandbox_id__eq=sandbox_id,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.search_app_conversations.assert_called_once()
        call_kwargs = mock_service.search_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert len(result.items) == 1
        assert result.items[0].sandbox_id == sandbox_id

    async def test_search_without_sandbox_id_filter(self):
        """Test that sandbox_id__eq defaults to None when not provided.

        Arrange: Create mock service
        Act: Call search_app_conversations without sandbox_id__eq
        Assert: Service is called with sandbox_id__eq=None
        """
        # Arrange
        mock_service = _make_mock_service()

        # Act
        await search_app_conversations(
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.search_app_conversations.assert_called_once()
        call_kwargs = mock_service.search_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') is None

    async def test_search_with_sandbox_id_and_other_filters(self):
        """Test that sandbox_id__eq works correctly with other filters.

        Arrange: Create mock service
        Act: Call search_app_conversations with sandbox_id__eq and other filters
        Assert: Service is called with all parameters correctly
        """
        # Arrange
        sandbox_id = 'test-sandbox-456'
        mock_service = _make_mock_service()

        # Act
        await search_app_conversations(
            title__contains='test',
            sandbox_id__eq=sandbox_id,
            limit=50,
            include_sub_conversations=True,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.search_app_conversations.assert_called_once()
        call_kwargs = mock_service.search_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert call_kwargs.get('title__contains') == 'test'
        assert call_kwargs.get('limit') == 50
        assert call_kwargs.get('include_sub_conversations') is True


@pytest.mark.asyncio
class TestCountAppConversations:
    """Test suite for count_app_conversations endpoint."""

    async def test_count_with_sandbox_id_filter(self):
        """Test that sandbox_id__eq filter is passed to the service.

        Arrange: Create mock service with count return value
        Act: Call count_app_conversations with sandbox_id__eq
        Assert: Service is called with the sandbox_id__eq parameter
        """
        # Arrange
        sandbox_id = 'test-sandbox-789'
        mock_service = _make_mock_service(count_return=5)

        # Act
        result = await count_app_conversations(
            sandbox_id__eq=sandbox_id,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.count_app_conversations.assert_called_once()
        call_kwargs = mock_service.count_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert result == 5

    async def test_count_without_sandbox_id_filter(self):
        """Test that sandbox_id__eq defaults to None when not provided.

        Arrange: Create mock service
        Act: Call count_app_conversations without sandbox_id__eq
        Assert: Service is called with sandbox_id__eq=None
        """
        # Arrange
        mock_service = _make_mock_service(count_return=10)

        # Act
        result = await count_app_conversations(
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.count_app_conversations.assert_called_once()
        call_kwargs = mock_service.count_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') is None
        assert result == 10

    async def test_count_with_sandbox_id_and_other_filters(self):
        """Test that sandbox_id__eq works correctly with other filters.

        Arrange: Create mock service
        Act: Call count_app_conversations with sandbox_id__eq and other filters
        Assert: Service is called with all parameters correctly
        """
        # Arrange
        sandbox_id = 'test-sandbox-abc'
        mock_service = _make_mock_service(count_return=3)

        # Act
        result = await count_app_conversations(
            title__contains='test',
            sandbox_id__eq=sandbox_id,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.count_app_conversations.assert_called_once()
        call_kwargs = mock_service.count_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert call_kwargs.get('title__contains') == 'test'
        assert result == 3


# ─── switch_conversation_profile ────────────────────────────────────────────


def _make_settings_with_profile(
    profile_name: str = 'gpt-5',
    model: str = 'openai/gpt-5',
    api_key: str = 'sk-test',
) -> Settings:
    """Build a Settings instance carrying one named LLM profile."""
    llm = LLM(model=model, api_key=api_key)
    return Settings(llm_profiles=LLMProfiles(profiles={profile_name: llm}))


def _make_agent_server_context(
    conversation_id, llm_model: str | None = 'openai/old-model'
) -> AgentServerContext:
    """Build a minimal AgentServerContext for the success path tests."""
    info = AppConversationInfo(
        id=conversation_id,
        created_by_user_id='test-user',
        sandbox_id=str(uuid4()),
        llm_model=llm_model,
    )
    return AgentServerContext(
        conversation=info,
        sandbox=MagicMock(status=SandboxStatus.RUNNING),
        sandbox_spec=MagicMock(),
        agent_server_url='http://agent.test',
        session_api_key='sess-key',
    )


def _make_httpx_client(post_return=None, post_side_effect=None) -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    if post_side_effect is not None:
        client.post = AsyncMock(side_effect=post_side_effect)
    else:
        response = post_return or MagicMock()
        client.post = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
class TestSwitchConversationProfile:
    """Test suite for the /switch_profile endpoint."""

    async def test_returns_404_when_user_settings_missing(self):
        """No user_settings → 404 (precondition for profile lookup)."""
        with pytest.raises(HTTPException) as exc_info:
            await switch_conversation_profile(
                conversation_id=uuid4(),
                request=SwitchProfileRequest(profile_name='gpt-5'),
                user_settings=None,
                app_conversation_service=MagicMock(),
                app_conversation_info_service=MagicMock(),
                sandbox_service=MagicMock(),
                sandbox_spec_service=MagicMock(),
                httpx_client=_make_httpx_client(),
            )

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert 'Settings not found' in exc_info.value.detail

    async def test_returns_404_when_profile_not_found(self):
        """Unknown profile name → 404 with the offending name in detail."""
        settings = _make_settings_with_profile(profile_name='default')

        with pytest.raises(HTTPException) as exc_info:
            await switch_conversation_profile(
                conversation_id=uuid4(),
                request=SwitchProfileRequest(profile_name='ghost'),
                user_settings=settings,
                app_conversation_service=MagicMock(),
                app_conversation_info_service=MagicMock(),
                sandbox_service=MagicMock(),
                sandbox_spec_service=MagicMock(),
                httpx_client=_make_httpx_client(),
            )

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "'ghost'" in exc_info.value.detail

    async def test_returns_409_when_sandbox_paused(self):
        """_get_agent_server_context returns None for paused sandboxes → 409."""
        settings = _make_settings_with_profile()

        with patch(
            'openhands.app_server.app_conversation.app_conversation_router.'
            '_get_agent_server_context',
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await switch_conversation_profile(
                    conversation_id=uuid4(),
                    request=SwitchProfileRequest(profile_name='gpt-5'),
                    user_settings=settings,
                    app_conversation_service=MagicMock(),
                    app_conversation_info_service=MagicMock(),
                    sandbox_service=MagicMock(),
                    sandbox_spec_service=MagicMock(),
                    httpx_client=_make_httpx_client(),
                )

        assert exc_info.value.status_code == status.HTTP_409_CONFLICT
        assert 'paused' in exc_info.value.detail.lower()

    async def test_propagates_status_when_conversation_not_reachable(self):
        """A JSONResponse from the helper is mirrored as an HTTPException."""
        settings = _make_settings_with_profile()
        helper_response = JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': 'Conversation not found'},
        )

        with patch(
            'openhands.app_server.app_conversation.app_conversation_router.'
            '_get_agent_server_context',
            new=AsyncMock(return_value=helper_response),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await switch_conversation_profile(
                    conversation_id=uuid4(),
                    request=SwitchProfileRequest(profile_name='gpt-5'),
                    user_settings=settings,
                    app_conversation_service=MagicMock(),
                    app_conversation_info_service=MagicMock(),
                    sandbox_service=MagicMock(),
                    sandbox_spec_service=MagicMock(),
                    httpx_client=_make_httpx_client(),
                )

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    async def test_returns_502_when_agent_server_returns_http_error(self):
        """A 4xx/5xx from switch_llm is folded into a 502."""
        conv_id = uuid4()
        settings = _make_settings_with_profile()
        ctx = _make_agent_server_context(conv_id)
        bad_response = MagicMock()
        bad_response.status_code = 500
        bad_response.text = 'agent crashed'
        bad_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                'boom', request=MagicMock(), response=bad_response
            ),
        )
        client = _make_httpx_client(post_return=bad_response)

        with patch(
            'openhands.app_server.app_conversation.app_conversation_router.'
            '_get_agent_server_context',
            new=AsyncMock(return_value=ctx),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await switch_conversation_profile(
                    conversation_id=conv_id,
                    request=SwitchProfileRequest(profile_name='gpt-5'),
                    user_settings=settings,
                    app_conversation_service=MagicMock(),
                    app_conversation_info_service=MagicMock(),
                    sandbox_service=MagicMock(),
                    sandbox_spec_service=MagicMock(),
                    httpx_client=client,
                )

        assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert '500' in exc_info.value.detail

    async def test_returns_502_when_agent_server_unreachable(self):
        """A network-level RequestError is folded into a 502."""
        conv_id = uuid4()
        settings = _make_settings_with_profile()
        ctx = _make_agent_server_context(conv_id)
        client = _make_httpx_client(
            post_side_effect=httpx.RequestError('connection refused'),
        )

        with patch(
            'openhands.app_server.app_conversation.app_conversation_router.'
            '_get_agent_server_context',
            new=AsyncMock(return_value=ctx),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await switch_conversation_profile(
                    conversation_id=conv_id,
                    request=SwitchProfileRequest(profile_name='gpt-5'),
                    user_settings=settings,
                    app_conversation_service=MagicMock(),
                    app_conversation_info_service=MagicMock(),
                    sandbox_service=MagicMock(),
                    sandbox_spec_service=MagicMock(),
                    httpx_client=client,
                )

        assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert 'reach agent server' in exc_info.value.detail.lower()

    async def test_success_persists_new_llm_model_on_conversation(self):
        """Happy path: agent-server returns 200, llm_model is saved."""
        conv_id = uuid4()
        new_model = 'openai/gpt-5'
        settings = _make_settings_with_profile(model=new_model)
        ctx = _make_agent_server_context(conv_id, llm_model='openai/old-model')

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()
        client = _make_httpx_client(post_return=ok_response)

        info_for_persist = AppConversationInfo(
            id=conv_id,
            created_by_user_id='test-user',
            sandbox_id=str(uuid4()),
            llm_model='openai/old-model',
        )
        info_service = MagicMock()
        info_service.get_app_conversation_info = AsyncMock(
            return_value=info_for_persist,
        )
        info_service.save_app_conversation_info = AsyncMock()

        with patch(
            'openhands.app_server.app_conversation.app_conversation_router.'
            '_get_agent_server_context',
            new=AsyncMock(return_value=ctx),
        ):
            await switch_conversation_profile(
                conversation_id=conv_id,
                request=SwitchProfileRequest(profile_name='gpt-5'),
                user_settings=settings,
                app_conversation_service=MagicMock(),
                app_conversation_info_service=info_service,
                sandbox_service=MagicMock(),
                sandbox_spec_service=MagicMock(),
                httpx_client=client,
            )

        info_service.save_app_conversation_info.assert_awaited_once()
        saved_info = info_service.save_app_conversation_info.await_args[0][0]
        assert saved_info.llm_model == new_model

        # The agent-server payload carries the new profile's LLM under the
        # `llm` key, with a usage_id derived from the profile name.
        client.post.assert_awaited_once()
        post_kwargs = client.post.await_args.kwargs
        assert post_kwargs['json']['llm']['model'] == new_model
        assert post_kwargs['json']['llm']['usage_id'].startswith('profile:gpt-5:')

    async def test_success_skips_persist_when_model_unchanged(self):
        """If the conversation already records the new model, save is skipped."""
        conv_id = uuid4()
        same_model = 'openai/gpt-5'
        settings = _make_settings_with_profile(model=same_model)
        ctx = _make_agent_server_context(conv_id)

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()
        client = _make_httpx_client(post_return=ok_response)

        info_already_correct = AppConversationInfo(
            id=conv_id,
            created_by_user_id='test-user',
            sandbox_id=str(uuid4()),
            llm_model=same_model,
        )
        info_service = MagicMock()
        info_service.get_app_conversation_info = AsyncMock(
            return_value=info_already_correct,
        )
        info_service.save_app_conversation_info = AsyncMock()

        with patch(
            'openhands.app_server.app_conversation.app_conversation_router.'
            '_get_agent_server_context',
            new=AsyncMock(return_value=ctx),
        ):
            await switch_conversation_profile(
                conversation_id=conv_id,
                request=SwitchProfileRequest(profile_name='gpt-5'),
                user_settings=settings,
                app_conversation_service=MagicMock(),
                app_conversation_info_service=info_service,
                sandbox_service=MagicMock(),
                sandbox_spec_service=MagicMock(),
                httpx_client=client,
            )

        info_service.save_app_conversation_info.assert_not_awaited()

    async def test_success_swallows_persistence_failures(self):
        """A save failure is logged but does not fail the request."""
        conv_id = uuid4()
        settings = _make_settings_with_profile(model='openai/gpt-5')
        ctx = _make_agent_server_context(conv_id, llm_model='openai/old-model')

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()
        client = _make_httpx_client(post_return=ok_response)

        info_service = MagicMock()
        info_service.get_app_conversation_info = AsyncMock(
            side_effect=RuntimeError('db down'),
        )
        info_service.save_app_conversation_info = AsyncMock()

        with patch(
            'openhands.app_server.app_conversation.app_conversation_router.'
            '_get_agent_server_context',
            new=AsyncMock(return_value=ctx),
        ):
            # Must not raise — the agent-server already accepted the swap.
            await switch_conversation_profile(
                conversation_id=conv_id,
                request=SwitchProfileRequest(profile_name='gpt-5'),
                user_settings=settings,
                app_conversation_service=MagicMock(),
                app_conversation_info_service=info_service,
                sandbox_service=MagicMock(),
                sandbox_spec_service=MagicMock(),
                httpx_client=client,
            )
