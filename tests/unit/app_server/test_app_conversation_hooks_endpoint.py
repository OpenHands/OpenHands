"""Unit tests for the V1 hooks endpoint in app_conversation_router.

This module tests the GET /{conversation_id}/hooks endpoint functionality.
"""

from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import status

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
)
from openhands.app_server.app_conversation.app_conversation_router import (
    get_conversation_hooks,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo


@pytest.mark.asyncio
class TestGetConversationHooks:
    async def test_get_hooks_returns_hook_events(self):
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        working_dir = '/workspace'

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            selected_repository='owner/repo',
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url='http://agent-server:8000', port=8000)
            ],
        )

        mock_sandbox_spec = SandboxSpecInfo(
            id=str(uuid4()), command=None, working_dir=working_dir
        )

        mock_app_conversation_service = MagicMock()
        mock_app_conversation_service.get_app_conversation = AsyncMock(
            return_value=mock_conversation
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()
        mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            'hook_config': {
                'stop': [
                    {
                        'matcher': '*',
                        'hooks': [
                            {
                                'type': 'command',
                                'command': '.openhands/hooks/on_stop.sh',
                                'timeout': 60,
                                'async': True,
                            }
                        ],
                    }
                ]
            }
        }

        mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        response = await get_conversation_hooks(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
            httpx_client=mock_httpx_client,
        )

        assert response.status_code == status.HTTP_200_OK

        data = __import__('json').loads(response.body.decode('utf-8'))
        assert 'hooks' in data
        assert data['hooks']
        assert data['hooks'][0]['event_type'] == 'stop'
        assert data['hooks'][0]['matchers'][0]['matcher'] == '*'
        assert data['hooks'][0]['matchers'][0]['hooks'][0]['type'] == 'command'
        assert (
            data['hooks'][0]['matchers'][0]['hooks'][0]['command']
            == '.openhands/hooks/on_stop.sh'
        )
        assert data['hooks'][0]['matchers'][0]['hooks'][0]['async'] is True
        assert 'async_' not in data['hooks'][0]['matchers'][0]['hooks'][0]

        mock_httpx_client.post.assert_called_once()
        called_url = mock_httpx_client.post.call_args[0][0]
        assert called_url == 'http://agent-server:8000/api/hooks'

    async def test_get_hooks_returns_502_when_agent_server_unreachable(self):
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            selected_repository=None,
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url='http://agent-server:8000', port=8000)
            ],
        )

        mock_sandbox_spec = SandboxSpecInfo(
            id=str(uuid4()), command=None, working_dir='/workspace'
        )

        mock_app_conversation_service = MagicMock()
        mock_app_conversation_service.get_app_conversation = AsyncMock(
            return_value=mock_conversation
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()
        mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)

        def _raise_request_error(*args, **_kwargs):
            request = httpx.Request('POST', args[0])
            raise httpx.RequestError('Connection error', request=request)

        mock_httpx_client.post = AsyncMock(side_effect=_raise_request_error)

        response = await get_conversation_hooks(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
            httpx_client=mock_httpx_client,
        )

        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        data = __import__('json').loads(response.body.decode('utf-8'))
        assert 'error' in data

    async def test_get_hooks_returns_502_when_agent_server_returns_error(self):
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            selected_repository=None,
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url='http://agent-server:8000', port=8000)
            ],
        )

        mock_sandbox_spec = SandboxSpecInfo(
            id=str(uuid4()), command=None, working_dir='/workspace'
        )

        mock_app_conversation_service = MagicMock()
        mock_app_conversation_service.get_app_conversation = AsyncMock(
            return_value=mock_conversation
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()
        mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)

        mock_response = Mock()
        mock_response.status_code = 500

        def _raise_http_status_error(*args, **_kwargs):
            request = httpx.Request('POST', args[0])
            response = httpx.Response(status_code=500, text='Internal Server Error')
            raise httpx.HTTPStatusError(
                'Server error', request=request, response=response
            )

        mock_httpx_client.post = AsyncMock(side_effect=_raise_http_status_error)

        response = await get_conversation_hooks(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
            httpx_client=mock_httpx_client,
        )

        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        data = __import__('json').loads(response.body.decode('utf-8'))
        assert 'error' in data

    async def test_get_hooks_returns_404_when_conversation_not_found(self):
        conversation_id = uuid4()

        mock_app_conversation_service = MagicMock()
        mock_app_conversation_service.get_app_conversation = AsyncMock(
            return_value=None
        )

        response = await get_conversation_hooks(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=MagicMock(),
            sandbox_spec_service=MagicMock(),
            httpx_client=AsyncMock(spec=httpx.AsyncClient),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_hooks_returns_404_when_sandbox_not_running(self):
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_app_conversation_service = MagicMock()
        mock_app_conversation_service.get_app_conversation = AsyncMock(
            return_value=mock_conversation
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=None)

        response = await get_conversation_hooks(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=MagicMock(),
            httpx_client=AsyncMock(spec=httpx.AsyncClient),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
