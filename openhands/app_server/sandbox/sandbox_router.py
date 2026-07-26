"""Runtime Containers router for OpenHands App Server."""

import logging
from typing import Annotated, cast
from uuid import UUID

import httpx
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from openhands.agent_server.models import Success
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    CODEX_CREDENTIAL_BINDING_TAG_KEY,
    AppConversationStartTaskStatus,
)
from openhands.app_server.config import (
    depends_app_conversation_info_service,
    depends_httpx_client,
    depends_jwt_service,
    depends_sandbox_service,
    depends_user_context,
    get_app_conversation_info_service,
    get_app_conversation_start_task_service,
    get_global_config,
)
from openhands.app_server.sandbox.sandbox_models import (
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
    SecretNameItem,
    SecretNamesResponse,
)
from openhands.app_server.sandbox.sandbox_service import (
    SandboxService,
)
from openhands.app_server.sandbox.session_auth import validate_session_key
from openhands.app_server.secrets.credential_binding import (
    CODEX_AUTH_SECRET_NAME,
    CREDENTIAL_BINDING_CONTEXT_HEADER,
    CredentialBindingContext,
    CredentialReplacement,
    activate_codex_credential_binding,
    credential_binding_callback_path,
    decode_binding_context,
    marker_organization_id,
    validate_codex_credential,
)
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user_auth.user_auth import (
    get_for_user as get_user_auth_for_user,
)
from openhands.app_server.utils.dependencies import get_dependencies

_logger = logging.getLogger(__name__)

# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(
    prefix='/sandboxes', tags=['Sandbox'], dependencies=get_dependencies()
)
sandbox_service_dependency = depends_sandbox_service()
user_context_dependency = depends_user_context()
app_conversation_info_service_dependency = depends_app_conversation_info_service()
httpx_client_dependency = depends_httpx_client()
jwt_service_dependency = depends_jwt_service()

# Read methods


@router.get('/search')
async def search_sandboxes(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 100,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> SandboxPage:
    """Search / list sandboxes owned by the current user."""
    return await sandbox_service.search_sandboxes(page_id=page_id, limit=limit)


@router.get('')
async def batch_get_sandboxes(
    id: Annotated[list[str], Query()],
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> list[SandboxInfo | None]:
    """Get a batch of sandboxes given their ids, returning null for any missing."""
    if len(id) > 100:
        raise HTTPException(
            status_code=400,
            detail=f'Cannot request more than 100 sandboxes at once, got {len(id)}',
        )
    sandboxes = await sandbox_service.batch_get_sandboxes(id)
    return sandboxes


# Write Methods


@router.post('')
async def start_sandbox(
    sandbox_spec_id: str | None = None,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> SandboxInfo:
    info = await sandbox_service.start_sandbox(sandbox_spec_id)
    return info


@router.post('/{sandbox_id}/pause', responses={404: {'description': 'Item not found'}})
async def pause_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.pause_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


@router.post('/{sandbox_id}/resume', responses={404: {'description': 'Item not found'}})
async def resume_sandbox(
    sandbox_id: str,
    user_context: UserContext = user_context_dependency,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.resume_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    return Success()


@router.post(
    '/{sandbox_id}/credential-bindings/{conversation_id}/activate',
    responses={404: {'description': 'Item not found'}},
)
async def activate_conversation_credential_binding(
    sandbox_id: str,
    conversation_id: UUID,
    sandbox_service: SandboxService = sandbox_service_dependency,
    user_context: UserContext = user_context_dependency,
    app_conversation_info_service: AppConversationInfoService = app_conversation_info_service_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
    jwt_service: JwtService = jwt_service_dependency,
) -> Success:
    info = await app_conversation_info_service.get_app_conversation_info(
        conversation_id
    )
    if info is None or info.sandbox_id != sandbox_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    user_id = info.created_by_user_id or 'root'
    if user_id != (await user_context.get_user_id() or 'root'):
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    marker = info.tags.get(CODEX_CREDENTIAL_BINDING_TAG_KEY)
    organization_id = None
    if marker is not None:
        try:
            organization_id = marker_organization_id(marker)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail='Managed credential metadata is invalid',
            ) from exc

    if marker is None:
        exists = await sandbox_service.resume_sandbox(sandbox_id)
        if not exists:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return Success()

    sandbox = await sandbox_service.get_sandbox(sandbox_id)
    if sandbox is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    if sandbox.status not in (SandboxStatus.RUNNING, SandboxStatus.STARTING):
        exists = await sandbox_service.resume_sandbox(sandbox_id)
        if not exists:
            sandbox = await sandbox_service.get_sandbox(sandbox_id)
            if sandbox is None or sandbox.status not in (
                SandboxStatus.RUNNING,
                SandboxStatus.STARTING,
            ):
                raise HTTPException(status.HTTP_404_NOT_FOUND)
    sandbox = await sandbox_service.wait_for_sandbox_running(
        sandbox_id, httpx_client=httpx_client
    )

    base_url = get_global_config().web_url
    if base_url is None:
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxService,
        )

        if isinstance(sandbox_service, DockerSandboxService):
            base_url = f'http://host.docker.internal:{sandbox_service.host_port}'
    if not base_url or not sandbox.session_api_key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Managed credential callback is unavailable',
        )
    callback_url = base_url.rstrip('/') + credential_binding_callback_path(
        sandbox_id, conversation_id
    )
    await activate_codex_credential_binding(
        httpx_client,
        jwt_service,
        agent_server_url=sandbox_service._get_agent_server_url(sandbox),
        callback_url=callback_url,
        session_api_key=sandbox.session_api_key,
        user_id=user_id,
        organization_id=organization_id,
        sandbox_id=sandbox_id,
        conversation_id=conversation_id,
        start_task_id=None,
        required=True,
    )
    return Success()


@router.delete('/{id}', responses={404: {'description': 'Item not found'}})
async def delete_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    # delete_sandbox is sandbox-scoped (stop + delete) and never archives, so this
    # request handler can't block on a minutes-long capture. Workspace capture is
    # owned by the conversation-delete finalizer and the runtime-api idle reaper.
    exists = await sandbox_service.delete_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


# ---------------------------------------------------------------------------
# Sandbox-scoped secrets (authenticated via X-Session-API-Key)
# ---------------------------------------------------------------------------


async def _valid_sandbox_from_session_key(
    request: Request,
    sandbox_id: str,
    session_api_key: str = Depends(
        APIKeyHeader(name='X-Session-API-Key', auto_error=False)
    ),
) -> SandboxInfo:
    """Authenticate via ``X-Session-API-Key`` and verify sandbox ownership."""
    sandbox_info = await validate_session_key(session_api_key)

    if sandbox_info.id != sandbox_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Session API key does not match sandbox',
        )

    return sandbox_info


async def _get_user_context(sandbox_info: SandboxInfo) -> AuthUserContext:
    """Build an ``AuthUserContext`` for the sandbox owner."""
    if not sandbox_info.created_by_user_id:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Sandbox has no associated user',
        )
    user_auth = await get_user_auth_for_user(sandbox_info.created_by_user_id)
    return AuthUserContext(user_auth=user_auth)


async def _credential_binding_context(
    sandbox_id: str,
    conversation_id: UUID,
    sandbox_info: SandboxInfo = Depends(_valid_sandbox_from_session_key),
    context_token: str | None = Header(None, alias=CREDENTIAL_BINDING_CONTEXT_HEADER),
    jwt_service: JwtService = jwt_service_dependency,
) -> CredentialBindingContext:
    if not context_token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Credential binding context is required',
        )
    try:
        context = decode_binding_context(jwt_service, context_token)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Credential binding context is invalid',
        ) from exc
    owner_id = sandbox_info.created_by_user_id or 'root'
    if (
        context.user_id != owner_id
        or context.sandbox_id != sandbox_id
        or context.conversation_id != str(conversation_id)
        or context.secret_name != CODEX_AUTH_SECRET_NAME
        or context.actions != ('load', 'replace')
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Credential binding context does not match the runtime',
        )

    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)
    async with (
        get_app_conversation_info_service(state) as info_service,
        get_app_conversation_start_task_service(state) as task_service,
    ):
        info = await info_service.get_app_conversation_info(conversation_id)
        if info is not None:
            marker = info.tags.get(CODEX_CREDENTIAL_BINDING_TAG_KEY)
            if (
                info.sandbox_id != sandbox_id
                or (info.created_by_user_id or 'root') != owner_id
            ):
                raise HTTPException(status.HTTP_403_FORBIDDEN)
            if marker is not None:
                try:
                    organization_id = marker_organization_id(marker)
                except ValueError as exc:
                    raise HTTPException(status.HTTP_403_FORBIDDEN) from exc
                if context.organization_id != (
                    str(organization_id) if organization_id else None
                ):
                    raise HTTPException(status.HTTP_403_FORBIDDEN)
                if context.start_task_id is None:
                    return context

        if context.start_task_id is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN)
        try:
            task_id = UUID(context.start_task_id)
        except ValueError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN) from exc
        task = await task_service.get_app_conversation_start_task(task_id)
        if (
            task is None
            or task.id != task_id
            or task.status != AppConversationStartTaskStatus.STARTING_CONVERSATION
            or task.sandbox_id != sandbox_id
            or (task.created_by_user_id or 'root') != owner_id
            or task.request.conversation_id != conversation_id
        ):
            raise HTTPException(status.HTTP_403_FORBIDDEN)
    return context


async def _credential_store(context: CredentialBindingContext):
    user_auth = await get_user_auth_for_user(context.user_id)
    return await user_auth.get_secrets_store()


@router.get(
    '/{sandbox_id}/credential-bindings/{conversation_id}/CODEX_AUTH_JSON',
    include_in_schema=False,
)
async def load_credential_binding(
    context: CredentialBindingContext = Depends(_credential_binding_context),
) -> JSONResponse:
    store = await _credential_store(context)
    organization_id = UUID(context.organization_id) if context.organization_id else None
    try:
        value, version = await store.load_versioned(
            CODEX_AUTH_SECRET_NAME, organization_id
        )
        validate_codex_credential(value)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND) from exc
    except NotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY) from exc
    return JSONResponse(
        {'value': value, 'version': version},
        headers={'Cache-Control': 'no-store'},
    )


@router.put(
    '/{sandbox_id}/credential-bindings/{conversation_id}/CODEX_AUTH_JSON',
    include_in_schema=False,
)
async def replace_credential_binding(
    request: Request,
    context: CredentialBindingContext = Depends(_credential_binding_context),
) -> JSONResponse:
    try:
        replacement = CredentialReplacement.model_validate(await request.json())
        validate_codex_credential(replacement.value)
    except (TypeError, ValueError):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Invalid Codex credential replacement',
        ) from None
    store = await _credential_store(context)
    organization_id = UUID(context.organization_id) if context.organization_id else None
    try:
        version = await store.replace_versioned(
            CODEX_AUTH_SECRET_NAME,
            replacement.expected_version,
            replacement.value,
            organization_id,
        )
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND) from exc
    except CredentialVersionConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT) from exc
    except NotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED) from exc
    return JSONResponse(
        {'version': version},
        headers={'Cache-Control': 'no-store'},
    )


@router.get('/{sandbox_id}/settings/secrets')
async def list_secret_names(
    sandbox_info: SandboxInfo = Depends(_valid_sandbox_from_session_key),
) -> SecretNamesResponse:
    """List available secret names (no raw values).

    Includes both custom secrets and provider tokens (e.g. github_token).
    """
    user_context = await _get_user_context(sandbox_info)

    items: list[SecretNameItem] = []

    # Custom secrets
    secret_sources = await user_context.get_secrets()
    for name, source in secret_sources.items():
        items.append(SecretNameItem(name=name, description=source.description))

    # Provider tokens (github_token, gitlab_token, etc.)
    provider_env_vars = cast(
        dict[str, str] | None,
        await user_context.get_provider_tokens(as_env_vars=True),
    )
    if provider_env_vars:
        for env_key in provider_env_vars:
            items.append(
                SecretNameItem(name=env_key, description=f'{env_key} provider token')
            )

    return SecretNamesResponse(secrets=items)


@router.get('/{sandbox_id}/settings/secrets/{secret_name}')
async def get_secret_value(
    secret_name: str,
    sandbox_info: SandboxInfo = Depends(_valid_sandbox_from_session_key),
) -> Response:
    """Return a single secret value as plain text.

    Called by ``LookupSecret`` inside the sandbox. Checks custom secrets
    first, then falls back to provider tokens — always resolving the
    latest token at call time.
    """
    user_context = await _get_user_context(sandbox_info)

    # Check custom secrets first
    secret_sources = await user_context.get_secrets()
    source = secret_sources.get(secret_name)
    if source is not None:
        value = source.get_value()
        if value is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail='Secret has no value')
        return Response(content=value, media_type='text/plain')

    # Fall back to provider tokens (resolved fresh per request)
    provider_env_vars = cast(
        dict[str, str] | None,
        await user_context.get_provider_tokens(as_env_vars=True),
    )
    if provider_env_vars:
        token_value = provider_env_vars.get(secret_name)
        if token_value is not None:
            return Response(content=token_value, media_type='text/plain')

    raise HTTPException(status.HTTP_404_NOT_FOUND, detail='Secret not found')
