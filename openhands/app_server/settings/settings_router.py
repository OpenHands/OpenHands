"""Settings router for OpenHands App Server.

This module provides the V1 API routes for user settings under /api/v1/settings.
"""

import os
from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from openhands.app_server.utils.dependencies import get_dependencies
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderType,
)
from openhands.sdk.settings import AgentSettings, ConversationSettings
from openhands.server.settings import (
    GETSettingsModel,
)
from openhands.server.shared import config
from openhands.server.user_auth import (
    get_provider_tokens,
    get_secrets_store,
    get_user_settings,
    get_user_settings_store,
)
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.data_models.settings import Settings
from openhands.storage.secrets.secrets_store import SecretsStore
from openhands.storage.settings.settings_store import SettingsStore
from openhands.utils.llm import (
    get_provider_api_base,
    is_openhands_model,
    resolve_llm_base_url,
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

        # Convert litellm_proxy/ back to openhands/ for the frontend
        resp_llm = settings_with_token_data.agent_settings.llm
        if resp_llm.model and resp_llm.model.startswith('litellm_proxy/'):
            resp_llm.model = (
                f'openhands/{resp_llm.model.removeprefix("litellm_proxy/")}'
            )

        # If the base url matches the default for the provider, we don't send it
        # So that the frontend can display basic mode.
        # Normalize trailing slashes for comparison since the SDK may add one.
        normalized_base = (llm.base_url or '').rstrip('/')
        normalized_proxy = LITE_LLM_API_URL.rstrip('/')
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
        500: {'description': 'Error storing settings', 'model': dict},
    },
)
async def store_settings(
    payload: dict[str, Any],
    settings_store: SettingsStore = Depends(get_user_settings_store),
) -> JSONResponse:
    """Store user settings.

    Accepts a partial payload and deep-merges ``agent_settings`` and
    ``conversation_settings`` with the existing persisted values so that
    saving one settings page never overwrites fields owned by another.

    Returns:
        200: Settings stored successfully
        500: Error storing settings
    """
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
    return AgentSettings.export_schema().model_dump(mode='json')


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
