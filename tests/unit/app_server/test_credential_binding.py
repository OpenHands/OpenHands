from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskPage,
    AppConversationStartTaskStatus,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.secrets import credential_binding
from openhands.app_server.secrets.credential_binding_models import (
    credential_binding_path,
)
from openhands.app_server.secrets.secrets_store import (
    CredentialVersionConflict,
    SecretsStore,
)
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey

_CONVERSATION_ID = UUID('11111111-1111-1111-1111-111111111111')
_ORG_ID = UUID('22222222-2222-2222-2222-222222222222')
_START_TASK_ID = UUID('33333333-3333-3333-3333-333333333333')
_OTHER_ID = UUID('44444444-4444-4444-4444-444444444444')
_MISSING = object()


@pytest.fixture
def jwt_service():
    return JwtService(
        [EncryptionKey(id='test', key=SecretStr('test-secret'), active=True)]
    )


@pytest.fixture
def store():
    result = AsyncMock(spec=SecretsStore)
    result.load_versioned.return_value = (
        '{"tokens":{"refresh_token":"r0"}}',
        'v0',
    )
    result.replace_versioned.return_value = 'v1'
    return result


@pytest.fixture
def client(monkeypatch, jwt_service, store):
    user_auth = AsyncMock()
    user_auth.get_secrets_store.return_value = store
    get_for_user = AsyncMock(return_value=user_auth)
    monkeypatch.setattr(credential_binding, 'get_for_user', get_for_user)
    monkeypatch.setattr(
        credential_binding,
        '_validate_active_binding',
        AsyncMock(),
    )
    app = FastAPI()
    app.include_router(credential_binding.router)
    app.dependency_overrides[credential_binding.jwt_service_dependency.dependency] = (
        lambda: jwt_service
    )
    with TestClient(app) as test_client:
        yield test_client, get_for_user


def _token(jwt_service: JwtService, **updates) -> str:
    payload = {
        'purpose': 'credential-binding',
        'user_id': 'user-id',
        'organization_id': str(_ORG_ID),
        'conversation_id': str(_CONVERSATION_ID),
        'runtime_id': 'runtime-id',
        'start_task_id': str(_START_TASK_ID),
        'secret_name': 'CODEX_AUTH_JSON',
        'actions': ['load', 'replace'],
    }
    for claim, value in updates.items():
        if value is _MISSING:
            payload.pop(claim, None)
        else:
            payload[claim] = value
    return jwt_service.create_jws_token(payload, expires_in=timedelta(minutes=5))


def _path(conversation_id: UUID = _CONVERSATION_ID) -> str:
    return credential_binding_path(conversation_id, 'CODEX_AUTH_JSON')


def _start_task(
    *,
    task_id: UUID = _START_TASK_ID,
    status: AppConversationStartTaskStatus = AppConversationStartTaskStatus.READY,
    sandbox_id: str = 'runtime-id',
    conversation_id: UUID | None = _CONVERSATION_ID,
    user_id: str | None = 'user-id',
) -> AppConversationStartTask:
    return AppConversationStartTask(
        id=task_id,
        created_by_user_id=user_id,
        status=status,
        app_conversation_id=conversation_id,
        sandbox_id=sandbox_id,
        request=AppConversationStartRequest(conversation_id=_CONVERSATION_ID),
    )


def _scope() -> credential_binding.CredentialBindingScope:
    return credential_binding.CredentialBindingScope(
        user_id='user-id',
        organization_id=_ORG_ID,
        conversation_id=_CONVERSATION_ID,
        runtime_id='runtime-id',
        start_task_id=_START_TASK_ID,
        secret_name='CODEX_AUTH_JSON',
    )


def _sandbox(
    *,
    status: SandboxStatus = SandboxStatus.RUNNING,
    user_id: str | None = 'user-id',
) -> SandboxInfo:
    return SandboxInfo(
        id='runtime-id',
        created_by_user_id=user_id,
        sandbox_spec_id='sandbox-spec',
        status=status,
        session_api_key=None,
    )


def _conversation(
    *,
    sandbox_id: str = 'runtime-id',
    user_id: str | None = 'user-id',
) -> AppConversationInfo:
    return AppConversationInfo(
        id=_CONVERSATION_ID,
        created_by_user_id=user_id,
        sandbox_id=sandbox_id,
    )


@pytest.fixture
def active_binding_services(monkeypatch):
    conversation_service = AsyncMock()
    sandbox_service = AsyncMock()
    start_task_service = AsyncMock()

    @asynccontextmanager
    async def conversation_context(state):
        yield conversation_service

    @asynccontextmanager
    async def start_task_context(state):
        yield start_task_service

    @asynccontextmanager
    async def sandbox_context(state):
        yield sandbox_service

    monkeypatch.setattr(
        credential_binding,
        'get_app_conversation_info_service',
        conversation_context,
    )
    monkeypatch.setattr(
        credential_binding,
        'get_app_conversation_start_task_service',
        start_task_context,
    )
    monkeypatch.setattr(
        credential_binding,
        'get_sandbox_service',
        sandbox_context,
    )
    return conversation_service, start_task_service, sandbox_service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('task', 'sandbox', 'conversation'),
    [
        pytest.param(None, _sandbox(), _conversation(), id='task-missing'),
        pytest.param(
            _start_task(task_id=_OTHER_ID),
            _sandbox(),
            _conversation(),
            id='task-id',
        ),
        pytest.param(
            _start_task(conversation_id=_OTHER_ID),
            _sandbox(),
            _conversation(),
            id='task-conversation',
        ),
        pytest.param(
            _start_task(sandbox_id='other-runtime'),
            _sandbox(),
            _conversation(),
            id='task-runtime',
        ),
        pytest.param(
            _start_task(user_id='other-user'),
            _sandbox(),
            _conversation(),
            id='task-owner',
        ),
        pytest.param(
            _start_task(status=AppConversationStartTaskStatus.WORKING),
            _sandbox(),
            _conversation(),
            id='task-status',
        ),
        pytest.param(_start_task(), None, _conversation(), id='runtime-missing'),
        pytest.param(
            _start_task(),
            _sandbox(status=SandboxStatus.PAUSED),
            _conversation(),
            id='runtime-status',
        ),
        pytest.param(
            _start_task(),
            _sandbox(user_id='other-user'),
            _conversation(),
            id='runtime-owner',
        ),
        pytest.param(
            _start_task(),
            _sandbox(),
            _conversation(user_id='other-user'),
            id='conversation-owner',
        ),
        pytest.param(
            _start_task(),
            _sandbox(),
            None,
            id='ready-conversation-missing',
        ),
        pytest.param(
            _start_task(),
            _sandbox(),
            _conversation(sandbox_id='other-runtime'),
            id='ready-conversation-runtime',
        ),
    ],
)
async def test_inactive_binding_clauses_return_generic_error(
    active_binding_services,
    task,
    sandbox,
    conversation,
):
    conversation_service, start_task_service, sandbox_service = active_binding_services
    start_task_service.search_app_conversation_start_tasks.return_value = (
        AppConversationStartTaskPage(items=[] if task is None else [task])
    )
    sandbox_service.get_sandbox_for_authorization.return_value = sandbox
    conversation_service.get_app_conversation_info.return_value = conversation

    with pytest.raises(HTTPException) as exc_info:
        await credential_binding._validate_active_binding(_scope())

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == 'Credential binding is no longer active'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('task', 'sandbox', 'conversation'),
    [
        pytest.param(_start_task(), _sandbox(), _conversation(), id='ready'),
        pytest.param(
            _start_task(status=AppConversationStartTaskStatus.STARTING_CONVERSATION),
            _sandbox(),
            _conversation(sandbox_id='previous-runtime'),
            id='starting-previous-runtime',
        ),
        pytest.param(
            _start_task(status=AppConversationStartTaskStatus.STARTING_CONVERSATION),
            _sandbox(),
            None,
            id='starting-before-conversation',
        ),
        pytest.param(
            _start_task(user_id=None),
            _sandbox(user_id=None),
            _conversation(user_id=None),
            id='legacy-unowned',
        ),
    ],
)
async def test_active_binding_clauses_are_accepted(
    active_binding_services,
    task,
    sandbox,
    conversation,
):
    conversation_service, start_task_service, sandbox_service = active_binding_services
    start_task_service.search_app_conversation_start_tasks.return_value = (
        AppConversationStartTaskPage(items=[task])
    )
    sandbox_service.get_sandbox_for_authorization.return_value = sandbox
    conversation_service.get_app_conversation_info.return_value = conversation

    await credential_binding._validate_active_binding(_scope())


def test_endpoint_enforces_active_binding(
    monkeypatch,
    jwt_service,
    store,
):
    user_auth = AsyncMock()
    user_auth.get_secrets_store.return_value = store
    monkeypatch.setattr(
        credential_binding,
        'get_for_user',
        AsyncMock(return_value=user_auth),
    )
    conversation_service = AsyncMock()
    conversation_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=_CONVERSATION_ID,
        created_by_user_id='user-id',
        sandbox_id='runtime-id',
    )
    sandbox = MagicMock()
    sandbox.created_by_user_id = 'user-id'
    sandbox.status = SandboxStatus.RUNNING
    sandbox_service = AsyncMock()
    sandbox_service.get_sandbox_for_authorization.return_value = sandbox
    start_task_service = AsyncMock()
    start_task_service.search_app_conversation_start_tasks.return_value = (
        AppConversationStartTaskPage(items=[_start_task()])
    )

    @asynccontextmanager
    async def conversation_context(state):
        yield conversation_service

    @asynccontextmanager
    async def start_task_context(state):
        yield start_task_service

    @asynccontextmanager
    async def sandbox_context(state):
        yield sandbox_service

    monkeypatch.setattr(
        credential_binding,
        'get_app_conversation_info_service',
        conversation_context,
    )
    monkeypatch.setattr(
        credential_binding,
        'get_app_conversation_start_task_service',
        start_task_context,
    )
    monkeypatch.setattr(
        credential_binding,
        'get_sandbox_service',
        sandbox_context,
    )
    app = FastAPI()
    app.include_router(credential_binding.router)
    app.dependency_overrides[credential_binding.jwt_service_dependency.dependency] = (
        lambda: jwt_service
    )

    with TestClient(app) as test_client:
        headers = {'Authorization': f'Bearer {_token(jwt_service)}'}
        assert test_client.get(_path(), headers=headers).status_code == 200
        sandbox.status = SandboxStatus.PAUSED
        assert test_client.get(_path(), headers=headers).status_code == 403
        assert (
            test_client.put(
                _path(),
                headers=headers,
                json={
                    'expected_version': 'v0',
                    'value': '{"tokens":{"refresh_token":"r1"}}',
                },
            ).status_code
            == 403
        )

    store.load_versioned.assert_awaited_once()
    store.replace_versioned.assert_not_awaited()


def test_load_uses_token_scope(client, jwt_service, store):
    test_client, get_for_user = client

    response = test_client.get(
        _path(), headers={'Authorization': f'Bearer {_token(jwt_service)}'}
    )

    assert response.status_code == 200
    assert response.json() == {
        'value': '{"tokens":{"refresh_token":"r0"}}',
        'version': 'v0',
    }
    assert response.headers['Cache-Control'] == 'no-store'
    get_for_user.assert_awaited_once_with('user-id')
    store.load_versioned.assert_awaited_once_with('CODEX_AUTH_JSON', _ORG_ID)


@pytest.mark.parametrize(
    ('organization_claim', 'organization_id'),
    [
        pytest.param(None, None, id='null'),
        pytest.param('', None, id='empty'),
        pytest.param(str(_OTHER_ID), _OTHER_ID, id='uuid'),
    ],
)
def test_organization_claim_is_parsed(
    client,
    jwt_service,
    store,
    organization_claim,
    organization_id,
):
    test_client, _ = client

    response = test_client.get(
        _path(),
        headers={
            'Authorization': f'Bearer {_token(jwt_service, organization_id=organization_claim)}'
        },
    )

    assert response.status_code == 200
    store.load_versioned.assert_awaited_once_with('CODEX_AUTH_JSON', organization_id)


def test_actions_are_order_independent(client, jwt_service):
    test_client, _ = client

    response = test_client.get(
        _path(),
        headers={
            'Authorization': f'Bearer {_token(jwt_service, actions=["replace", "load"])}'
        },
    )

    assert response.status_code == 200


def test_unsupported_store_is_distinct(client, jwt_service, store):
    test_client, _ = client
    store.load_versioned.side_effect = NotImplementedError

    response = test_client.get(
        _path(), headers={'Authorization': f'Bearer {_token(jwt_service)}'}
    )

    assert response.status_code == 501


def test_replace_uses_compare_and_swap(client, jwt_service, store):
    test_client, _ = client
    replacement = '{"tokens":{"refresh_token":"r1"}}'

    response = test_client.put(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
        json={'expected_version': 'v0', 'value': replacement},
    )

    assert response.status_code == 200
    assert response.json() == {'version': 'v1'}
    assert response.headers['Cache-Control'] == 'no-store'
    store.replace_versioned.assert_awaited_once_with(
        'CODEX_AUTH_JSON', 'v0', replacement, _ORG_ID
    )


@pytest.mark.parametrize(
    ('claim', 'value'),
    [
        pytest.param('purpose', _MISSING, id='purpose-missing'),
        pytest.param('purpose', 1, id='purpose-type'),
        pytest.param('user_id', _MISSING, id='user-missing'),
        pytest.param('user_id', 1, id='user-type'),
        pytest.param('organization_id', _MISSING, id='organization-missing'),
        pytest.param('organization_id', 1, id='organization-type'),
        pytest.param('organization_id', 'not-a-uuid', id='organization-format'),
        pytest.param('conversation_id', _MISSING, id='conversation-missing'),
        pytest.param('conversation_id', 1, id='conversation-type'),
        pytest.param('conversation_id', 'not-a-uuid', id='conversation-format'),
        pytest.param('runtime_id', _MISSING, id='runtime-missing'),
        pytest.param('runtime_id', 1, id='runtime-type'),
        pytest.param('start_task_id', _MISSING, id='start-task-missing'),
        pytest.param('start_task_id', 1, id='start-task-type'),
        pytest.param('start_task_id', 'not-a-uuid', id='start-task-format'),
        pytest.param('secret_name', _MISSING, id='secret-missing'),
        pytest.param('secret_name', 1, id='secret-type'),
        pytest.param('actions', _MISSING, id='actions-missing'),
        pytest.param('actions', 'load', id='actions-type'),
        pytest.param('actions', ['load', 1], id='action-type'),
    ],
)
def test_malformed_claims_are_unauthorized(
    client,
    jwt_service,
    store,
    claim,
    value,
):
    test_client, get_for_user = client

    response = test_client.get(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service, **{claim: value})}'},
    )

    assert response.status_code == 401
    assert response.json() == {'detail': 'Invalid credential binding token'}
    get_for_user.assert_not_awaited()
    store.load_versioned.assert_not_awaited()


@pytest.mark.parametrize(
    ('claim', 'value'),
    [
        pytest.param('purpose', 'other-purpose', id='purpose'),
        pytest.param('user_id', '', id='empty-user'),
        pytest.param('runtime_id', '', id='empty-runtime'),
        pytest.param('conversation_id', str(_OTHER_ID), id='conversation'),
        pytest.param('secret_name', 'OTHER_SECRET', id='secret'),
        pytest.param('actions', [], id='actions-empty'),
        pytest.param('actions', ['load'], id='actions-missing'),
        pytest.param('actions', ['load', 'load'], id='actions-duplicate'),
        pytest.param(
            'actions',
            ['load', 'replace', 'delete'],
            id='actions-extra',
        ),
    ],
)
def test_scope_mismatches_are_forbidden_before_store(
    client,
    jwt_service,
    store,
    claim,
    value,
):
    test_client, get_for_user = client

    response = test_client.get(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service, **{claim: value})}'},
    )

    assert response.status_code == 403
    assert response.json() == {'detail': 'Credential binding token scope mismatch'}
    get_for_user.assert_not_awaited()
    store.load_versioned.assert_not_awaited()


def test_unknown_credential_is_not_exposed(client, jwt_service, store):
    test_client, get_for_user = client

    response = test_client.get(
        credential_binding_path(_CONVERSATION_ID, 'OTHER_SECRET'),
        headers={
            'Authorization': f'Bearer {_token(jwt_service, secret_name="OTHER_SECRET")}'
        },
    )

    assert response.status_code == 404
    get_for_user.assert_not_awaited()
    store.load_versioned.assert_not_awaited()


def test_invalid_replacement_is_rejected(client, jwt_service, store):
    test_client, _ = client

    response = test_client.put(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
        json={'expected_version': 'v0', 'value': '{}'},
    )

    assert response.status_code == 422
    store.replace_versioned.assert_not_awaited()


def test_conflict_is_reported(client, jwt_service, store):
    test_client, _ = client
    store.replace_versioned.side_effect = CredentialVersionConflict

    response = test_client.put(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
        json={
            'expected_version': 'stale',
            'value': '{"tokens":{"refresh_token":"r1"}}',
        },
    )

    assert response.status_code == 409
