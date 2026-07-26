from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

from openhands.app_server.errors import SandboxError
from openhands.app_server.secrets.credential_binding import (
    CREDENTIAL_BINDING_CAPABILITIES,
    CREDENTIAL_BINDING_CONTEXT_HEADER,
    activate_codex_credential_binding,
    agent_server_supports_credential_binding,
    decode_binding_context,
)


def _response(status_code: int, body: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=body,
        request=httpx.Request('GET', 'http://agent-server'),
    )


@pytest.mark.parametrize(
    'capabilities, expected',
    [
        (sorted(CREDENTIAL_BINDING_CAPABILITIES), True),
        (sorted(CREDENTIAL_BINDING_CAPABILITIES | {'other'}), True),
        (['credential_binding_v1'], False),
        ([], False),
    ],
)
async def test_support_requires_all_capabilities(capabilities, expected):
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(200, {'capabilities': capabilities})

    supported = await agent_server_supports_credential_binding(
        client, 'http://agent-server', 'session-key'
    )

    assert supported is expected


async def test_missing_capability_metadata_is_legacy():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(200, {})

    supported = await agent_server_supports_credential_binding(
        client, 'http://agent-server', 'session-key'
    )

    assert supported is False


@pytest.mark.parametrize('missing', sorted(CREDENTIAL_BINDING_CAPABILITIES))
async def test_each_capability_is_required(missing):
    client = AsyncMock(spec=httpx.AsyncClient)
    capabilities = sorted(CREDENTIAL_BINDING_CAPABILITIES - {missing})
    client.get.return_value = _response(200, {'capabilities': capabilities})

    supported = await agent_server_supports_credential_binding(
        client, 'http://agent-server', 'session-key'
    )

    assert supported is False


async def test_malformed_capability_metadata_is_rejected():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(200, {'capabilities': 'all'})

    with pytest.raises(SandboxError, match='invalid capability'):
        await agent_server_supports_credential_binding(
            client, 'http://agent-server', 'session-key'
        )


async def test_optional_activation_falls_back_for_missing_capabilities():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(200, {'capabilities': []})

    activated = await activate_codex_credential_binding(
        client,
        MagicMock(),
        agent_server_url='http://agent-server',
        callback_url='https://app/callback',
        session_api_key='session-key',
        user_id='user-id',
        organization_id=None,
        sandbox_id='sandbox-id',
        conversation_id=uuid4(),
        start_task_id=uuid4(),
        required=False,
    )

    assert activated is False
    client.put.assert_not_awaited()


async def test_optional_activation_falls_back_for_501():
    conversation_id = uuid4()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(
        200, {'capabilities': sorted(CREDENTIAL_BINDING_CAPABILITIES)}
    )
    client.put.return_value = _response(501)
    jwt_service = MagicMock()
    jwt_service.create_jwe_token.return_value = 'context-token'

    activated = await activate_codex_credential_binding(
        client,
        jwt_service,
        agent_server_url='http://agent-server',
        callback_url='https://app/callback',
        session_api_key='session-key',
        user_id='user-id',
        organization_id=None,
        sandbox_id='sandbox-id',
        conversation_id=conversation_id,
        start_task_id=uuid4(),
        required=False,
    )

    assert activated is False
    assert client.put.await_args.kwargs['json'] == {
        'url': 'https://app/callback',
        'headers': {
            'X-Session-API-Key': 'session-key',
            CREDENTIAL_BINDING_CONTEXT_HEADER: 'context-token',
        },
    }


@pytest.mark.parametrize('status_code', [400, 401, 409, 500, 501])
async def test_required_activation_rejects_agent_server_error(status_code):
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(
        200, {'capabilities': sorted(CREDENTIAL_BINDING_CAPABILITIES)}
    )
    client.put.return_value = _response(status_code)
    jwt_service = MagicMock()
    jwt_service.create_jwe_token.return_value = 'context-token'

    with pytest.raises(SandboxError, match='rejected managed credential'):
        await activate_codex_credential_binding(
            client,
            jwt_service,
            agent_server_url='http://agent-server',
            callback_url='https://app/callback',
            session_api_key='session-key',
            user_id='user-id',
            organization_id=None,
            sandbox_id='sandbox-id',
            conversation_id=uuid4(),
            start_task_id=None,
            required=True,
        )


@pytest.mark.parametrize('status_code', [400, 401, 409, 500])
async def test_optional_activation_only_falls_back_for_501(status_code):
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(
        200, {'capabilities': sorted(CREDENTIAL_BINDING_CAPABILITIES)}
    )
    client.put.return_value = _response(status_code)
    jwt_service = MagicMock()
    jwt_service.create_jwe_token.return_value = 'context-token'

    with pytest.raises(SandboxError, match='rejected managed credential'):
        await activate_codex_credential_binding(
            client,
            jwt_service,
            agent_server_url='http://agent-server',
            callback_url='https://app/callback',
            session_api_key='session-key',
            user_id='user-id',
            organization_id=None,
            sandbox_id='sandbox-id',
            conversation_id=uuid4(),
            start_task_id=uuid4(),
            required=False,
        )


async def test_activation_context_is_bound_to_runtime_identity():
    conversation_id = uuid4()
    task_id = uuid4()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _response(
        200, {'capabilities': sorted(CREDENTIAL_BINDING_CAPABILITIES)}
    )
    client.put.return_value = _response(204)
    jwt_service = MagicMock()
    jwt_service.create_jwe_token.return_value = 'context-token'

    activated = await activate_codex_credential_binding(
        client,
        jwt_service,
        agent_server_url='http://agent-server',
        callback_url='https://app/callback',
        session_api_key='session-key',
        user_id='user-id',
        organization_id=None,
        sandbox_id='sandbox-id',
        conversation_id=conversation_id,
        start_task_id=task_id,
        required=False,
    )

    assert activated is True
    claims = jwt_service.create_jwe_token.call_args.args[0]
    assert claims['user_id'] == 'user-id'
    assert claims['sandbox_id'] == 'sandbox-id'
    assert claims['conversation_id'] == str(conversation_id)
    assert claims['start_task_id'] == str(task_id)
    assert claims['actions'] == ['load', 'replace']


def test_decode_context_rejects_extra_claims():
    jwt_service = MagicMock()
    jwt_service.decrypt_jwe_token.return_value = {
        'purpose': 'codex-credential-binding',
        'iat': 1,
        'user_id': 'user-id',
        'organization_id': None,
        'sandbox_id': 'sandbox-id',
        'conversation_id': str(uuid4()),
        'start_task_id': None,
        'secret_name': 'CODEX_AUTH_JSON',
        'actions': ['load', 'replace'],
        'other': True,
    }

    with pytest.raises(ValueError):
        decode_binding_context(jwt_service, 'token')
