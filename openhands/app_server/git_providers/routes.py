from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE, CustomSecret
from openhands.integrations.service_types import ProviderType
from openhands.integrations.utils import validate_provider_token
from openhands.server.dependencies import get_dependencies
from openhands.server.settings import (
    CustomSecretModel,
    CustomSecretWithoutValueModel,
    GETCustomSecrets,
    POSTProviderModel,
)
from openhands.server.user_auth import (
    get_provider_tokens,
    get_secrets,
    get_secrets_store,
)
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.data_models.settings import Settings
from openhands.storage.secrets.secrets_store import SecretsStore
from openhands.storage.settings.settings_store import SettingsStore

router = APIRouter(prefix="/api/v1", dependencies=get_dependencies())


async def invalidate_legacy_secrets_store(
    settings: Settings, settings_store: SettingsStore, secrets_store: SecretsStore
) -> Secrets | None:
    if len(settings.secrets_store.provider_tokens.items()) > 0:
        user_secrets = Secrets(provider_tokens=settings.secrets_store.provider_tokens)
        await secrets_store.store(user_secrets)

        invalidated_secrets_settings = settings.model_copy(
            update={"secrets_store": Secrets()}
        )
        await settings_store.store(invalidated_secrets_settings)

        return user_secrets

    return None


def process_token_validation_result(
    confirmed_token_type: ProviderType | None, token_type: ProviderType
) -> str:
    if not confirmed_token_type or confirmed_token_type != token_type:
        return (
            f"Invalid token. Please make sure it is a valid {token_type.value} token."
        )

    return ""


async def check_provider_tokens(
    incoming_provider_tokens: POSTProviderModel,
    existing_provider_tokens: PROVIDER_TOKEN_TYPE | None,
) -> str:
    msg = ""
    if incoming_provider_tokens.provider_tokens:
        for token_type, token_value in incoming_provider_tokens.provider_tokens.items():
            if token_value.token:
                confirmed_token_type = await validate_provider_token(
                    token_value.token, token_value.host
                )
                msg = process_token_validation_result(confirmed_token_type, token_type)

            existing_token = (
                existing_provider_tokens.get(token_type, None)
                if existing_provider_tokens
                else None
            )
            if (
                existing_token
                and (existing_token.host != token_value.host)
                and existing_token.token
            ):
                confirmed_token_type = await validate_provider_token(
                    existing_token.token, token_value.host
                )
                if not confirmed_token_type or confirmed_token_type != token_type:
                    msg = process_token_validation_result(
                        confirmed_token_type, token_type
                    )

    return msg


@router.post("/git-providers")
async def store_provider_tokens(
    provider_info: POSTProviderModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
) -> JSONResponse:
    provider_err_msg = await check_provider_tokens(provider_info, provider_tokens)
    if provider_err_msg:
        logger.info(
            f"Returning 401 Unauthorized - Provider token error: {provider_err_msg}"
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": provider_err_msg},
        )

    try:
        user_secrets = await secrets_store.load()
        if not user_secrets:
            user_secrets = Secrets()

        if provider_info.provider_tokens:
            existing_providers = [provider for provider in user_secrets.provider_tokens]

            for provider, token_value in list(provider_info.provider_tokens.items()):
                if provider in existing_providers and not token_value.token:
                    existing_token = user_secrets.provider_tokens.get(provider)
                    if existing_token and existing_token.token:
                        provider_info.provider_tokens[provider] = existing_token

                provider_info.provider_tokens[provider] = provider_info.provider_tokens[
                    provider
                ].model_copy(update={"host": token_value.host})

        updated_secrets = user_secrets.model_copy(
            update={"provider_tokens": provider_info.provider_tokens}
        )
        await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Git providers stored"},
        )
    except Exception as e:
        logger.warning(f"Something went wrong storing git providers: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Something went wrong storing git providers"},
        )


@router.delete("/git-providers/tokens", response_model=dict[str, str])
async def unset_provider_tokens(
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
    try:
        user_secrets = await secrets_store.load()
        if user_secrets:
            updated_secrets = user_secrets.model_copy(update={"provider_tokens": {}})
            await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Unset Git provider tokens"},
        )

    except Exception as e:
        logger.warning(f"Something went wrong unsetting tokens: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Something went wrong unsetting tokens"},
        )
