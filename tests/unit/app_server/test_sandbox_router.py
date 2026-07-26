import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from openhands.app_server.app_conversation.app_conversation_models import (
    CODEX_CREDENTIAL_BINDING_TAG_KEY,
    AppConversationInfo,
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_router import (
    _credential_binding_context,
    activate_conversation_credential_binding,
    load_credential_binding,
    replace_credential_binding,
)
from openhands.app_server.secrets.credential_binding import (
    CredentialBindingContext,
)
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict

USER_ID = 'user-id'
SANDBOX_ID = 'sandbox-id'
VALID_CODEX_AUTH = json.dumps(
    {'auth_mode': 'chatgpt', 'tokens': {'refresh_token': 'refresh-token'}}
)


def _sandbox() -> SandboxInfo:
    return SandboxInfo(
        id=SANDBOX_ID,
        created_by_user_id=USER_ID,
        sandbox_spec_id='spec-id',
        status=SandboxStatus.RUNNING,
        session_api_key='fresh-session-key',
        exposed_urls=[
            ExposedUrl(
                name=AGENT_SERVER,
                url='http://agent-server',
                port=8000,
            )
        ],
    )


def _context(
    *,
    user_id: str = USER_ID,
    sandbox_id: str = SANDBOX_ID,
    conversation_id=None,
    organization_id: str | None = None,
    start_task_id=None,
) -> CredentialBindingContext:
    return CredentialBindingContext(
        purpose='codex-credential-binding',
        iat=1,
        user_id=user_id,
        organization_id=organization_id,
        sandbox_id=sandbox_id,
        conversation_id=str(conversation_id or uuid4()),
        start_task_id=str(start_task_id) if start_task_id else None,
        secret_name='CODEX_AUTH_JSON',
        actions=('load', 'replace'),
    )


def _service_context(service):
    @asynccontextmanager
    async def context(*args, **kwargs):
        yield service

    return context


@pytest.mark.parametrize('mismatch', ['user', 'sandbox', 'conversation'])
async def test_callback_rejects_runtime_identity_mismatch(mismatch):
    conversation_id = uuid4()
    context = _context(conversation_id=conversation_id)
    updates = {
        'user': {'user_id': 'other-user'},
        'sandbox': {'sandbox_id': 'other-sandbox'},
        'conversation': {'conversation_id': str(uuid4())},
    }
    context = context.model_copy(update=updates[mismatch])

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.decode_binding_context',
            return_value=context,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _credential_binding_context(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_info=_sandbox(),
            context_token='token',
            jwt_service=MagicMock(),
        )

    assert exc_info.value.status_code == 403


async def test_callback_rejects_task_mismatch_before_conversation_exists():
    conversation_id = uuid4()
    task_id = uuid4()
    context = _context(
        conversation_id=conversation_id,
        start_task_id=task_id,
    )
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = None
    task_service = AsyncMock()
    task_service.get_app_conversation_start_task.return_value = (
        AppConversationStartTask(
            id=uuid4(),
            created_by_user_id=USER_ID,
            status=AppConversationStartTaskStatus.STARTING_CONVERSATION,
            sandbox_id=SANDBOX_ID,
            request=AppConversationStartRequest(conversation_id=conversation_id),
        )
    )

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.decode_binding_context',
            return_value=context,
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_info_service',
            _service_context(info_service),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_start_task_service',
            _service_context(task_service),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _credential_binding_context(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_info=_sandbox(),
            context_token='token',
            jwt_service=MagicMock(),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize('webhook_stub_exists', [False, True])
async def test_callback_accepts_exact_start_task(webhook_stub_exists):
    conversation_id = uuid4()
    task_id = uuid4()
    context = _context(
        conversation_id=conversation_id,
        start_task_id=task_id,
    )
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = (
        AppConversationInfo(
            id=conversation_id,
            created_by_user_id=USER_ID,
            sandbox_id=SANDBOX_ID,
            tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
        )
        if webhook_stub_exists
        else None
    )
    task_service = AsyncMock()
    task_service.get_app_conversation_start_task.return_value = (
        AppConversationStartTask(
            id=task_id,
            created_by_user_id=USER_ID,
            status=AppConversationStartTaskStatus.STARTING_CONVERSATION,
            sandbox_id=SANDBOX_ID,
            request=AppConversationStartRequest(conversation_id=conversation_id),
        )
    )

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.decode_binding_context',
            return_value=context,
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_info_service',
            _service_context(info_service),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_start_task_service',
            _service_context(task_service),
        ),
    ):
        result = await _credential_binding_context(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_info=_sandbox(),
            context_token='token',
            jwt_service=MagicMock(),
        )

    assert result == context


async def test_callback_rejects_organization_mismatch_after_start():
    conversation_id = uuid4()
    context = _context(
        conversation_id=conversation_id,
        organization_id=str(uuid4()),
    )
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
        tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
    )
    task_service = AsyncMock()

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.decode_binding_context',
            return_value=context,
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_info_service',
            _service_context(info_service),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_start_task_service',
            _service_context(task_service),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _credential_binding_context(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_info=_sandbox(),
            context_token='token',
            jwt_service=MagicMock(),
        )

    assert exc_info.value.status_code == 403
    task_service.get_app_conversation_start_task.assert_not_awaited()


async def test_callback_accepts_taskless_context_for_marked_conversation():
    conversation_id = uuid4()
    context = _context(conversation_id=conversation_id)
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
        tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
    )
    task_service = AsyncMock()

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.decode_binding_context',
            return_value=context,
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_info_service',
            _service_context(info_service),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_start_task_service',
            _service_context(task_service),
        ),
    ):
        result = await _credential_binding_context(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_info=_sandbox(),
            context_token='token',
            jwt_service=MagicMock(),
        )

    assert result == context
    task_service.get_app_conversation_start_task.assert_not_awaited()


async def test_callback_rejects_completed_task_context_after_marker_commit():
    conversation_id = uuid4()
    task_id = uuid4()
    context = _context(
        conversation_id=conversation_id,
        start_task_id=task_id,
    )
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
        tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
    )
    task_service = AsyncMock()
    task_service.get_app_conversation_start_task.return_value = (
        AppConversationStartTask(
            id=task_id,
            created_by_user_id=USER_ID,
            status=AppConversationStartTaskStatus.READY,
            sandbox_id=SANDBOX_ID,
            request=AppConversationStartRequest(conversation_id=conversation_id),
        )
    )

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.decode_binding_context',
            return_value=context,
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_info_service',
            _service_context(info_service),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_app_conversation_start_task_service',
            _service_context(task_service),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _credential_binding_context(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_info=_sandbox(),
            context_token='token',
            jwt_service=MagicMock(),
        )

    assert exc_info.value.status_code == 403
    task_service.get_app_conversation_start_task.assert_awaited_once_with(task_id)


@pytest.mark.parametrize(
    'error, status_code',
    [
        (KeyError(), 404),
        (NotImplementedError(), 501),
        (ValueError(), 422),
    ],
)
async def test_load_callback_maps_store_errors(error, status_code):
    store = AsyncMock()
    store.load_versioned.side_effect = error
    context = _context()

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router._credential_store',
            AsyncMock(return_value=store),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await load_credential_binding(context=context)

    assert exc_info.value.status_code == status_code


@pytest.mark.parametrize(
    'error, status_code',
    [
        (KeyError(), 404),
        (CredentialVersionConflict(), 409),
        (NotImplementedError(), 501),
    ],
)
async def test_replace_callback_maps_store_errors(error, status_code):
    store = AsyncMock()
    store.replace_versioned.side_effect = error
    context = _context()
    request = AsyncMock()
    request.json.return_value = {
        'expected_version': 'v1',
        'value': VALID_CODEX_AUTH,
    }

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router._credential_store',
            AsyncMock(return_value=store),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await replace_credential_binding(
            request=request,
            context=context,
        )

    assert exc_info.value.status_code == status_code


async def test_invalid_replacement_does_not_reflect_secret():
    canary = 'distinct-refresh-token-canary'
    request = AsyncMock()
    request.json.return_value = {
        'expected_version': 'v1',
        'value': canary * 3000,
    }

    with pytest.raises(HTTPException) as exc_info:
        await replace_credential_binding(
            request=request,
            context=_context(),
        )

    assert exc_info.value.status_code == 422
    assert canary not in json.dumps(exc_info.value.detail)


async def test_marked_resume_reactivates_with_stable_url_and_fresh_key():
    conversation_id = uuid4()
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
        tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
    )
    sandbox_service = AsyncMock()
    order = []

    async def resume(*args):
        order.append('resume-committed-key')
        return True

    async def activate_binding(*args, **kwargs):
        order.append('activate')
        return True

    sandbox_service.resume_sandbox.side_effect = resume
    sandbox_service.get_sandbox.return_value = _sandbox().model_copy(
        update={'status': SandboxStatus.PAUSED}
    )
    sandbox_service.wait_for_sandbox_running.return_value = _sandbox()
    sandbox_service._get_agent_server_url.return_value = 'http://agent-server'
    user_context = AsyncMock()
    user_context.get_user_id.return_value = USER_ID

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_global_config',
            return_value=SimpleNamespace(web_url='https://app.example.com/'),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.activate_codex_credential_binding',
            AsyncMock(side_effect=activate_binding),
        ) as activate,
    ):
        await activate_conversation_credential_binding(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_service=sandbox_service,
            user_context=user_context,
            app_conversation_info_service=info_service,
            httpx_client=AsyncMock(),
            jwt_service=MagicMock(),
        )

    assert activate.await_args.kwargs['callback_url'] == (
        f'https://app.example.com/api/v1/sandboxes/{SANDBOX_ID}/'
        f'credential-bindings/{conversation_id}/CODEX_AUTH_JSON'
    )
    assert activate.await_args.kwargs['session_api_key'] == 'fresh-session-key'
    assert activate.await_args.kwargs['required'] is True
    assert order == ['resume-committed-key', 'activate']


async def test_unmarked_resume_skips_activation():
    conversation_id = uuid4()
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
    )
    sandbox_service = AsyncMock()
    sandbox_service.resume_sandbox.return_value = True
    sandbox_service.get_sandbox.return_value = _sandbox().model_copy(
        update={'status': SandboxStatus.PAUSED}
    )
    sandbox_service.wait_for_sandbox_running.return_value = _sandbox()
    user_context = AsyncMock()
    user_context.get_user_id.return_value = USER_ID

    with patch(
        'openhands.app_server.sandbox.sandbox_router.activate_codex_credential_binding',
        AsyncMock(),
    ) as activate:
        await activate_conversation_credential_binding(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_service=sandbox_service,
            user_context=user_context,
            app_conversation_info_service=info_service,
            httpx_client=AsyncMock(),
            jwt_service=MagicMock(),
        )

    sandbox_service.resume_sandbox.assert_awaited_once_with(SANDBOX_ID)
    activate.assert_not_awaited()


async def test_running_marked_sandbox_reactivates_without_resume():
    conversation_id = uuid4()
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
        tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
    )
    sandbox_service = AsyncMock()
    sandbox_service.get_sandbox.return_value = _sandbox()
    sandbox_service.wait_for_sandbox_running.return_value = _sandbox()
    sandbox_service._get_agent_server_url.return_value = 'http://agent-server'
    user_context = AsyncMock()
    user_context.get_user_id.return_value = USER_ID

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_global_config',
            return_value=SimpleNamespace(web_url='https://app.example.com/'),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.activate_codex_credential_binding',
            AsyncMock(return_value=True),
        ) as activate,
    ):
        await activate_conversation_credential_binding(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_service=sandbox_service,
            user_context=user_context,
            app_conversation_info_service=info_service,
            httpx_client=AsyncMock(),
            jwt_service=MagicMock(),
        )

    sandbox_service.resume_sandbox.assert_not_awaited()
    activate.assert_awaited_once()


async def test_unauthorized_activation_does_not_resume_sandbox():
    conversation_id = uuid4()
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
        tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
    )
    sandbox_service = AsyncMock()
    user_context = AsyncMock()
    user_context.get_user_id.return_value = 'other-user'

    with pytest.raises(HTTPException) as exc_info:
        await activate_conversation_credential_binding(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_service=sandbox_service,
            user_context=user_context,
            app_conversation_info_service=info_service,
            httpx_client=AsyncMock(),
            jwt_service=MagicMock(),
        )

    assert exc_info.value.status_code == 403
    sandbox_service.resume_sandbox.assert_not_awaited()


async def test_concurrent_resume_is_accepted_after_refetch():
    conversation_id = uuid4()
    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = AppConversationInfo(
        id=conversation_id,
        created_by_user_id=USER_ID,
        sandbox_id=SANDBOX_ID,
        tags={CODEX_CREDENTIAL_BINDING_TAG_KEY: 'personal'},
    )
    paused = _sandbox().model_copy(update={'status': SandboxStatus.PAUSED})
    starting = _sandbox().model_copy(update={'status': SandboxStatus.STARTING})
    running = _sandbox()
    sandbox_service = AsyncMock()
    sandbox_service.get_sandbox.side_effect = [paused, starting]
    sandbox_service.resume_sandbox.return_value = False
    sandbox_service.wait_for_sandbox_running.return_value = running
    user_context = AsyncMock()
    user_context.get_user_id.return_value = USER_ID

    with (
        patch(
            'openhands.app_server.sandbox.sandbox_router.get_global_config',
            return_value=SimpleNamespace(web_url='https://app.example.com/'),
        ),
        patch(
            'openhands.app_server.sandbox.sandbox_router.activate_codex_credential_binding',
            AsyncMock(return_value=True),
        ),
    ):
        await activate_conversation_credential_binding(
            sandbox_id=SANDBOX_ID,
            conversation_id=conversation_id,
            sandbox_service=sandbox_service,
            user_context=user_context,
            app_conversation_info_service=info_service,
            httpx_client=AsyncMock(),
            jwt_service=MagicMock(),
        )

    sandbox_service.wait_for_sandbox_running.assert_awaited_once()
