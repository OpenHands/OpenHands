"""Settings router for OpenHands App Server.

This module provides the V1 API routes for user settings under /api/v1/settings.
"""

import asyncio
import os
from collections import defaultdict
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from openhands.app_server.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderType,
)
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.settings.llm_profiles import (
    ProfileAlreadyExistsError,
    ProfileLimitExceededError,
    ProfileNotFoundError,
    StrictLLM,
)
from openhands.app_server.settings.settings_models import (
    GETSettingsModel,
    Settings,
)
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.user_auth import (
    get_provider_tokens,
    get_secrets_store,
    get_user_id,
    get_user_settings,
    get_user_settings_store,
)
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.core.logger import openhands_logger as logger
from openhands.sdk.llm import LLM
from openhands.sdk.settings import ConversationSettings
from openhands.server.shared import config
from openhands.utils.llm import (
    get_provider_api_base,
    is_openhands_model,
    resolve_llm_base_url,
)
from openhands.utils.sdk_settings_compat import (
    LLMAgentSettings,
    export_agent_settings_schema,
)

LITE_LLM_API_URL = os.environ.get(
    'LITE_LLM_API_URL', 'https://llm-proxy.app.all-hands.dev'
)

# Create router with /api/v1/settings prefix
router = APIRouter(
    prefix='/settings',
    tags=['Settings'],
    dependencies=get_dependencies(),
)


def _post_merge_llm_fixups(settings: Settings) -> None:
    """Apply LLM-specific fixups after merging settings.

    Delegates the empty-string → cleared and provider-default inference
    rules to :func:`openhands.utils.llm.resolve_llm_base_url` so the
    personal-save and enterprise org-defaults paths stay in lockstep.
    """
    if not isinstance(settings.agent_settings, LLMAgentSettings):
        return
    llm = settings.agent_settings.llm
    llm.base_url = resolve_llm_base_url(
        model=llm.model,
        base_url=llm.base_url,
        managed_proxy_url=LITE_LLM_API_URL,
    )


# NOTE: We use response_model=None for endpoints that return JSONResponse directly.
# This is because FastAPI's response_model expects a Pydantic model, but we're returning
# a response object directly. We document the possible responses using the 'responses'
# parameter and maintain proper type annotations for mypy.
@router.get(
    '',
    response_model=GETSettingsModel,
    responses={
        404: {'description': 'Settings not found', 'model': dict},
        401: {'description': 'Invalid token', 'model': dict},
    },
)
async def load_settings(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    settings_store: SettingsStore = Depends(get_user_settings_store),
    settings: Settings = Depends(get_user_settings),
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> GETSettingsModel | JSONResponse:
    """Load user settings.

    Retrieves the settings for the authenticated user, including LLM configuration,
    provider tokens, and other user preferences.

    Returns:
        GETSettingsModel: The user settings with token data

    Raises:
        404: Settings not found
        401: Invalid token
    """
    try:
        if not settings:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={'error': 'Settings not found'},
            )

        # On initial load, user secrets may not be populated with values migrated from settings store
        user_secrets = await invalidate_legacy_secrets_store(
            settings, settings_store, secrets_store
        )

        # If invalidation is successful, then the returned user secrets holds the most recent values
        git_providers = (
            user_secrets.provider_tokens if user_secrets else provider_tokens
        )

        provider_tokens_set: dict[ProviderType, str | None] = {}
        if git_providers:
            for provider_type, provider_token in git_providers.items():
                if provider_token.token or provider_token.user_id:
                    provider_tokens_set[provider_type] = provider_token.host

        llm = settings.agent_settings.llm
        settings_with_token_data = GETSettingsModel(
            **settings.model_dump(exclude={'secrets_store'}),
            llm_api_key_set=settings.llm_api_key_is_set,
            search_api_key_set=settings.search_api_key is not None
            and bool(settings.search_api_key),
            provider_tokens_set=provider_tokens_set,
        )

        # Convert litellm_proxy/ back to openhands/ for the frontend, but only
        # if the base_url is the OpenHands proxy. Custom litellm_proxy endpoints
        # should keep their litellm_proxy/ prefix.
        resp_llm = settings_with_token_data.agent_settings.llm
        normalized_base = (llm.base_url or '').rstrip('/')
        normalized_proxy = LITE_LLM_API_URL.rstrip('/')

        if resp_llm.model and resp_llm.model.startswith('litellm_proxy/'):
            # Only convert to openhands/ if using the OpenHands proxy URL
            if normalized_base == normalized_proxy:
                resp_llm.model = (
                    f'openhands/{resp_llm.model.removeprefix("litellm_proxy/")}'
                )

        # If the base url matches the default for the provider, we don't send it
        # So that the frontend can display basic mode.
        if is_openhands_model(llm.model):
            if normalized_base == normalized_proxy:
                resp_llm.base_url = None
        elif llm.model and llm.base_url == get_provider_api_base(llm.model):
            resp_llm.base_url = None

        resp_llm.api_key = None
        settings_with_token_data.search_api_key = None
        settings_with_token_data.sandbox_api_key = None
        return settings_with_token_data
    except Exception as e:
        logger.warning(f'Invalid token: {e}')
        # Get user_id from settings if available
        user_id = getattr(settings, 'user_id', 'unknown') if settings else 'unknown'
        logger.info(
            f'Returning 401 Unauthorized - Invalid token for user_id: {user_id}'
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Invalid token'},
        )


@router.post(
    '',
    response_model=None,
    responses={
        200: {'description': 'Settings stored successfully', 'model': dict},
        422: {
            'description': 'Legacy nested settings keys are not accepted',
            'model': dict,
        },
        500: {'description': 'Error storing settings', 'model': dict},
    },
)
async def store_settings(
    payload: dict[str, Any],
    settings_store: SettingsStore = Depends(get_user_settings_store),
) -> JSONResponse:
    """Store user settings.

    Accepts a partial payload and deep-merges ``agent_settings_diff`` and
    ``conversation_settings_diff`` with the existing persisted values so that
    saving one settings page never overwrites fields owned by another.

    Returns:
        200: Settings stored successfully
        422: Legacy nested settings keys are rejected
        500: Error storing settings
    """
    legacy_nested_keys = sorted(
        key for key in ('agent_settings', 'conversation_settings') if key in payload
    )
    if legacy_nested_keys:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                'error': 'Use *_diff nested settings payloads instead of legacy keys',
                'keys': legacy_nested_keys,
            },
        )

    try:
        existing_settings = await settings_store.load()
        settings = existing_settings.model_copy() if existing_settings else Settings()
        settings.update(payload)

        _post_merge_llm_fixups(settings)

        if existing_settings:
            if 'search_api_key' not in payload and settings.search_api_key is None:
                settings.search_api_key = existing_settings.search_api_key
            if settings.user_consents_to_analytics is None:
                settings.user_consents_to_analytics = (
                    existing_settings.user_consents_to_analytics
                )
            if settings.disabled_skills is None:
                settings.disabled_skills = existing_settings.disabled_skills

        # Update sandbox config with new settings
        if settings.remote_runtime_resource_factor is not None:
            config.sandbox.remote_runtime_resource_factor = (
                settings.remote_runtime_resource_factor
            )

        # Update git configuration with new settings
        git_config_updated = False
        if settings.git_user_name is not None:
            config.git_user_name = settings.git_user_name
            git_config_updated = True
        if settings.git_user_email is not None:
            config.git_user_email = settings.git_user_email
            git_config_updated = True

        if git_config_updated:
            logger.info(
                f'Updated global git configuration: name={config.git_user_name}, email={config.git_user_email}'
            )

        await settings_store.store(settings)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': 'Settings stored'},
        )
    except Exception as e:
        logger.warning(f'Something went wrong storing settings: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Something went wrong storing settings'},
        )


@router.get('/agent-schema')
async def load_settings_schema() -> dict[str, Any]:
    """Load the schema for settings"""
    return export_agent_settings_schema().model_dump(mode='json')


@router.get('/conversation-schema')
async def load_conversation_settings_schema() -> dict[str, Any]:
    """Load the schema for conversations"""
    return ConversationSettings.export_schema().model_dump(mode='json')


async def invalidate_legacy_secrets_store(
    settings: Settings, settings_store: SettingsStore, secrets_store: SecretsStore
) -> Secrets | None:
    """We are moving `secrets_store` (a field from `Settings` object) to its own dedicated store
    This function moves the values from Settings to Secrets, and deletes the values in Settings
    While this function in called multiple times, the migration only ever happens once
    """
    if len(settings.secrets_store.provider_tokens.items()) > 0:
        user_secrets = Secrets(provider_tokens=settings.secrets_store.provider_tokens)
        await secrets_store.store(user_secrets)

        # Invalidate old tokens via settings store serializer
        invalidated_secrets_settings = settings.model_copy(
            update={'secrets_store': Secrets()}
        )
        await settings_store.store(invalidated_secrets_settings)

        return user_secrets

    return None


# ── LLM Profile endpoints ─────────────────────────────────────────

# Profile name constraints: alphanumerics + . _ - only, 1-64 chars. This
# blocks empty names, path-traversal fragments, slash-in-name routing
# ambiguity, and pathological long inputs. Applied via ``Path`` so FastAPI
# rejects the request before the handler runs.
_NAME_PATTERN = r'^[A-Za-z0-9._-]{1,64}$'
ProfileName = Annotated[
    str,
    Path(min_length=1, max_length=64, pattern=_NAME_PATTERN),
]

# Per-user asyncio lock serializing the read-modify-write cycle for profile
# writes. This eliminates the lost-update race between concurrent
# save/delete/activate calls *within a single worker process*. In
# multi-worker SaaS deployments, DB-level row locks or optimistic
# concurrency tokens are still required for full safety — but this closes
# the single-process hole. The dict grows with distinct user ids; acceptable
# because each lock is tiny and users are bounded per-process in practice.
_user_profile_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _profile_lock_key(user_id: str | None) -> str:
    return user_id or '<anonymous>'


class ProfileInfo(BaseModel):
    """Profile summary returned by the list endpoint.

    ``api_key_set`` follows the same convention as ``llm_api_key_set`` on
    the main settings response — the frontend uses it to show "key stored"
    without exposing (or accidentally round-tripping) a mask string.
    """

    name: str
    model: str | None = None
    base_url: str | None = None
    api_key_set: bool = False


class ProfileListResponse(BaseModel):
    """Response body for listing profiles."""

    profiles: list[ProfileInfo]
    active_profile: str | None = None


class ProfileDetailResponse(BaseModel):
    """Response body for fetching a single profile.

    ``config.api_key`` is always ``None`` in the response; the sibling
    ``api_key_set`` bool reports whether a key is stored. This matches
    the ``/api/v1/settings`` convention and prevents the "GET → edit →
    POST" flow from poisoning the stored key with a mask string.
    """

    name: str
    config: dict[str, Any]
    api_key_set: bool = False


class ProfileMutationResponse(BaseModel):
    """Response body for save/delete operations."""

    name: str
    message: str


class ActivateProfileResponse(BaseModel):
    """Response body for activating a profile."""

    name: str
    message: str
    model: str | None = None


class RenameProfileRequest(BaseModel):
    """Request body for renaming a profile.

    ``new_name`` is validated against the same regex as the path-level
    ``{name}`` param so the two stay in sync.
    """

    new_name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=_NAME_PATTERN,
    )


class SaveProfileRequest(BaseModel):
    """Request body for saving a profile.

    If ``llm`` is provided, it is used as the profile config; otherwise the
    current ``agent_settings.llm`` is saved. The ``llm`` field is typed as
    :class:`StrictLLM`, which forbids unknown keys — so a typo like
    ``{"llm": {"custom_header": "x"}}`` returns 422 instead of being silently
    dropped.

    **Security note:** when ``llm.api_key`` is included in the request body,
    it is transmitted in plaintext over the wire and present in any request
    log or error trace that captures request bodies. ``SecretStr`` masks it
    in Pydantic reprs, but callers and operators should still avoid logging
    raw request bodies on this endpoint.
    """

    include_secrets: bool = True
    llm: StrictLLM | None = None


@router.get('/profiles', response_model=ProfileListResponse)
async def list_profiles(
    settings: Settings | None = Depends(get_user_settings),
) -> ProfileListResponse:
    """List all saved LLM profiles.

    Returns profile names with basic model info. API keys are never exposed.
    """
    if settings is None:
        return ProfileListResponse(profiles=[], active_profile=None)

    return ProfileListResponse(
        profiles=[ProfileInfo(**p) for p in settings.llm_profiles.summaries()],
        active_profile=settings.llm_profiles.active,
    )


@router.get('/profiles/{name}', response_model=ProfileDetailResponse)
async def get_profile(
    name: ProfileName,
    settings: Settings | None = Depends(get_user_settings),
) -> ProfileDetailResponse:
    """Get a specific profile's configuration.

    Returns the full LLM config with ``api_key`` nulled out; the sibling
    ``api_key_set`` flag reports whether a key is stored.
    """
    profile = settings.llm_profiles.get(name) if settings is not None else None
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{name}' not found",
        )

    api_key_set = profile.api_key is not None
    config = profile.model_dump(mode='json')
    config['api_key'] = None  # never echo a mask; use api_key_set instead

    return ProfileDetailResponse(name=name, config=config, api_key_set=api_key_set)


@router.post(
    '/profiles/{name}',
    response_model=ProfileMutationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_profile(
    name: ProfileName,
    request: Annotated[SaveProfileRequest | None, Body()] = None,
    user_id: str | None = Depends(get_user_id),
    settings_store: SettingsStore = Depends(get_user_settings_store),
) -> ProfileMutationResponse:
    """Save an LLM configuration as a named profile.

    If ``request.llm`` is supplied, it is saved as the profile's config.
    Otherwise the current ``agent_settings.llm`` is snapshotted. Existing
    profiles with the same name are overwritten.

    Runs inside a per-user lock to prevent lost updates between concurrent
    profile writes. Returns 409 if the user is already at the profile
    cap (:data:`MAX_PROFILES_PER_USER`).
    """
    if request is None:
        request = SaveProfileRequest()

    async with _user_profile_locks[_profile_lock_key(user_id)]:
        settings = await settings_store.load()
        if settings is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Settings not found',
            )

        llm: LLM
        if request.llm is not None:
            llm = request.llm
            # Preserve the existing api_key when the caller omits it on
            # update (e.g. a frontend round-tripping a GET response where
            # the key was nulled out). Mirrors the deep-merge behaviour
            # the main ``POST /api/v1/settings`` relies on.
            if llm.api_key is None:
                existing = settings.llm_profiles.get(name)
                if existing is not None and existing.api_key is not None:
                    llm = llm.model_copy(update={'api_key': existing.api_key})
        else:
            llm = settings.agent_settings.llm

        try:
            settings.llm_profiles.save(
                name, llm, include_secrets=request.include_secrets
            )
        except ProfileLimitExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        # Without this, overwriting the active profile leaves
        # agent_settings.llm stale — active would lie about what's running.
        settings.reconcile_active_profile()
        await settings_store.store(settings)

    return ProfileMutationResponse(name=name, message=f"Profile '{name}' saved")


@router.delete('/profiles/{name}', response_model=ProfileMutationResponse)
async def delete_profile(
    name: ProfileName,
    user_id: str | None = Depends(get_user_id),
    settings_store: SettingsStore = Depends(get_user_settings_store),
) -> ProfileMutationResponse:
    """Delete a saved profile.

    Idempotent: returns success even if the profile didn't exist.
    """
    async with _user_profile_locks[_profile_lock_key(user_id)]:
        settings = await settings_store.load()
        if settings is not None and settings.llm_profiles.delete(name):
            await settings_store.store(settings)

    return ProfileMutationResponse(name=name, message=f"Profile '{name}' deleted")


@router.post('/profiles/{name}/activate', response_model=ActivateProfileResponse)
async def activate_profile(
    name: ProfileName,
    user_id: str | None = Depends(get_user_id),
    settings_store: SettingsStore = Depends(get_user_settings_store),
) -> ActivateProfileResponse:
    """Switch ``agent_settings.llm`` to use a saved profile.

    Applies the same ``base_url`` fixups as ``POST /api/v1/settings``, so
    activating a profile that omits ``base_url`` still points at the
    provider's default endpoint.
    """
    async with _user_profile_locks[_profile_lock_key(user_id)]:
        settings = await settings_store.load()
        if settings is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile '{name}' not found",
            )

        try:
            settings.switch_to_profile(name)
        except ProfileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        _post_merge_llm_fixups(settings)
        await settings_store.store(settings)

    return ActivateProfileResponse(
        name=name,
        message=f"Switched to profile '{name}'",
        model=settings.agent_settings.llm.model,
    )


@router.post('/profiles/{name}/rename', response_model=ProfileMutationResponse)
async def rename_profile(
    name: ProfileName,
    request: RenameProfileRequest,
    user_id: str | None = Depends(get_user_id),
    settings_store: SettingsStore = Depends(get_user_settings_store),
) -> ProfileMutationResponse:
    """Rename a saved profile.

    Preserves the stored LLM config (including the api_key) and the active
    flag if the renamed profile was active. Returns 409 if ``new_name`` is
    already in use by a different profile.
    """
    async with _user_profile_locks[_profile_lock_key(user_id)]:
        settings = await settings_store.load()
        if settings is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Settings not found',
            )
        try:
            settings.llm_profiles.rename(name, request.new_name)
        except ProfileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except ProfileAlreadyExistsError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        await settings_store.store(settings)

    return ProfileMutationResponse(
        name=request.new_name,
        message=f"Profile '{name}' renamed to '{request.new_name}'",
    )
