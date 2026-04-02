import os

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderType,
)
from openhands.server.dependencies import get_dependencies
from openhands.server.routes.secrets import invalidate_legacy_secrets_store
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
from openhands.storage.data_models.settings import Settings
from openhands.storage.secrets.secrets_store import SecretsStore
from openhands.storage.settings.settings_store import SettingsStore
from openhands.utils.llm import get_provider_api_base, is_openhands_model

LITE_LLM_API_URL = os.environ.get(
    "LITE_LLM_API_URL", "https://llm-proxy.app.all-hands.dev"
)

router = APIRouter(prefix="/api/v1", dependencies=get_dependencies())


async def store_llm_settings(
    settings: Settings, existing_settings: Settings
) -> Settings:
    if existing_settings:
        if settings.llm_api_key is None:
            settings.llm_api_key = existing_settings.llm_api_key
        if settings.llm_model is None:
            settings.llm_model = existing_settings.llm_model
        if settings.llm_base_url is None:
            if existing_settings.llm_base_url:
                settings.llm_base_url = existing_settings.llm_base_url
            elif is_openhands_model(settings.llm_model):
                settings.llm_base_url = LITE_LLM_API_URL
            elif settings.llm_model:
                try:
                    api_base = get_provider_api_base(settings.llm_model)
                    if api_base:
                        settings.llm_base_url = api_base
                except Exception as e:
                    logger.error(
                        f"Failed to get api_base from litellm for model {settings.llm_model}: {e}"
                    )
        elif settings.llm_base_url == "":
            settings.llm_base_url = None
        if not settings.search_api_key:
            settings.search_api_key = existing_settings.search_api_key
    return settings


def convert_to_settings(settings_with_token_data: Settings) -> Settings:
    settings_data = settings_with_token_data.model_dump()
    filtered_settings_data = {
        key: value
        for key, value in settings_data.items()
        if key in Settings.model_fields
    }
    filtered_settings_data["llm_api_key"] = settings_with_token_data.llm_api_key
    filtered_settings_data["search_api_key"] = settings_with_token_data.search_api_key
    settings = Settings(**filtered_settings_data)
    return settings


@router.get(
    "/settings",
    response_model=GETSettingsModel,
    responses={
        404: {"description": "Settings not found", "model": dict},
        401: {"description": "Invalid token", "model": dict},
    },
)
async def load_settings(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    settings_store: SettingsStore = Depends(get_user_settings_store),
    settings: Settings = Depends(get_user_settings),
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> GETSettingsModel | JSONResponse:
    try:
        if not settings:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "Settings not found"},
            )

        user_secrets = await invalidate_legacy_secrets_store(
            settings, settings_store, secrets_store
        )

        git_providers = (
            user_secrets.provider_tokens if user_secrets else provider_tokens
        )

        provider_tokens_set: dict[ProviderType, str | None] = {}
        if git_providers:
            for provider_type, provider_token in git_providers.items():
                if provider_token.token or provider_token.user_id:
                    provider_tokens_set[provider_type] = provider_token.host

        settings_with_token_data = GETSettingsModel(
            **settings.model_dump(exclude={"secrets_store"}),
            llm_api_key_set=settings.llm_api_key is not None
            and bool(settings.llm_api_key),
            search_api_key_set=settings.search_api_key is not None
            and bool(settings.search_api_key),
            provider_tokens_set=provider_tokens_set,
        )

        if is_openhands_model(settings.llm_model):
            if settings.llm_base_url == LITE_LLM_API_URL:
                settings_with_token_data.llm_base_url = None
        elif settings.llm_model and settings.llm_base_url == get_provider_api_base(
            settings.llm_model
        ):
            settings_with_token_data.llm_base_url = None

        settings_with_token_data.llm_api_key = None
        settings_with_token_data.search_api_key = None
        settings_with_token_data.sandbox_api_key = None
        return settings_with_token_data
    except Exception as e:
        logger.warning(f"Invalid token: {e}")
        user_id = getattr(settings, "user_id", "unknown") if settings else "unknown"
        logger.info(
            f"Returning 401 Unauthorized - Invalid token for user_id: {user_id}"
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Invalid token"},
        )


@router.post(
    "/settings",
    response_model=None,
    responses={
        200: {"description": "Settings stored successfully", "model": dict},
        500: {"description": "Error storing settings", "model": dict},
    },
)
async def store_settings(
    settings: Settings,
    settings_store: SettingsStore = Depends(get_user_settings_store),
) -> JSONResponse:
    try:
        existing_settings = await settings_store.load()

        if existing_settings:
            settings = await store_llm_settings(settings, existing_settings)

            if settings.user_consents_to_analytics is None:
                settings.user_consents_to_analytics = (
                    existing_settings.user_consents_to_analytics
                )

            if settings.disabled_skills is None:
                settings.disabled_skills = existing_settings.disabled_skills

        if settings.remote_runtime_resource_factor is not None:
            config.sandbox.remote_runtime_resource_factor = (
                settings.remote_runtime_resource_factor
            )

        git_config_updated = False
        if settings.git_user_name is not None:
            config.git_user_name = settings.git_user_name
            git_config_updated = True
        if settings.git_user_email is not None:
            config.git_user_email = settings.git_user_email
            git_config_updated = True

        if git_config_updated:
            logger.info(
                f"Updated global git configuration: name={config.git_user_name}, email={config.git_user_email}"
            )

        settings = convert_to_settings(settings)
        await settings_store.store(settings)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Settings stored"},
        )
    except Exception as e:
        logger.warning(f"Something went wrong storing settings: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Something went wrong storing settings"},
        )
