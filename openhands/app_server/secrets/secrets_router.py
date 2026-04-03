"""Secrets router for OpenHands App Server.

This module provides the V1 API routes for secrets under /api/v1/secrets.
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from openhands.app_server.utils.dependencies import get_dependencies
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    CustomSecret,
    ProviderType,
)
from openhands.integrations.utils import validate_provider_token
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
from openhands.storage.secrets.secrets_store import SecretsStore

# Create router with /api/v1/secrets prefix
router = APIRouter(
    prefix='/secrets',
    tags=['Secrets'],
    dependencies=get_dependencies(),
)


# =================================================
# SECTION: Helper functions for git providers
# =================================================


def get_token_validation_error_message(
    confirmed_token_type: ProviderType | None, token_type: ProviderType
) -> str:
    """Returns error message if token type doesn't match, empty string otherwise."""
    if not confirmed_token_type or confirmed_token_type != token_type:
        return (
            f'Invalid token. Please make sure it is a valid {token_type.value} token.'
        )

    return ''


async def check_provider_tokens(
    incoming_provider_tokens: POSTProviderModel,
    existing_provider_tokens: PROVIDER_TOKEN_TYPE | None,
) -> str:
    msg = ''
    if incoming_provider_tokens.provider_tokens:
        # Determine whether tokens are valid
        for token_type, token_value in incoming_provider_tokens.provider_tokens.items():
            if token_value.token:
                confirmed_token_type = await validate_provider_token(
                    token_value.token, token_value.host
                )  # FE always sends latest host
                msg = get_token_validation_error_message(
                    confirmed_token_type, token_type
                )

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
                )  # Host has changed, check it against existing token
                if not confirmed_token_type or confirmed_token_type != token_type:
                    msg = get_token_validation_error_message(
                        confirmed_token_type, token_type
                    )

    return msg


# =================================================
# SECTION: Git Provider Token Endpoints
# =================================================


@router.post(
    '/git-providers',
    response_model=None,
    responses={
        200: {'description': 'Git providers stored successfully', 'model': dict},
        401: {'description': 'Invalid token', 'model': dict},
        500: {'description': 'Error storing git providers', 'model': dict},
    },
    tags=['Git Providers'],
)
async def store_provider_tokens(
    provider_info: POSTProviderModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
) -> JSONResponse:
    """Store git provider tokens.

    Saves the git provider tokens (GitHub, GitLab, Bitbucket, etc.) for the authenticated user.

    Returns:
        200: Git providers stored successfully
        401: Invalid token
        500: Error storing git providers
    """
    provider_err_msg = await check_provider_tokens(provider_info, provider_tokens)
    if provider_err_msg:
        # We don't have direct access to user_id here, but we can log the provider info
        logger.info(
            f'Returning 401 Unauthorized - Provider token error: {provider_err_msg}'
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': provider_err_msg},
        )

    try:
        user_secrets = await secrets_store.load()
        if not user_secrets:
            user_secrets = Secrets()

        if provider_info.provider_tokens:
            existing_providers = [provider for provider in user_secrets.provider_tokens]

            # Merge incoming settings store with the existing one
            for provider, token_value in list(provider_info.provider_tokens.items()):
                if provider in existing_providers and not token_value.token:
                    existing_token = user_secrets.provider_tokens.get(provider)
                    if existing_token and existing_token.token:
                        provider_info.provider_tokens[provider] = existing_token

                provider_info.provider_tokens[provider] = provider_info.provider_tokens[
                    provider
                ].model_copy(update={'host': token_value.host})

        updated_secrets = user_secrets.model_copy(
            update={'provider_tokens': provider_info.provider_tokens}
        )
        await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': 'Git providers stored'},
        )
    except Exception as e:
        logger.warning(f'Something went wrong storing git providers: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Something went wrong storing git providers'},
        )


@router.delete(
    '/git-providers/tokens',
    response_model=None,
    responses={
        200: {'description': 'Git provider tokens unset successfully', 'model': dict},
        500: {'description': 'Error unsetting git provider tokens', 'model': dict},
    },
    tags=['Git Providers'],
)
async def unset_provider_tokens(
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
    """Unset (delete) all git provider tokens.

    Removes all git provider tokens for the authenticated user.

    Returns:
        200: Git provider tokens unset successfully
        500: Error unsetting git provider tokens
    """
    try:
        user_secrets = await secrets_store.load()
        if user_secrets:
            updated_secrets = user_secrets.model_copy(update={'provider_tokens': {}})
            await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': 'Unset Git provider tokens'},
        )

    except Exception as e:
        logger.warning(f'Something went wrong unsetting tokens: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Something went wrong unsetting tokens'},
        )


# =================================================
# SECTION: Custom Secrets Endpoints
# =================================================


@router.get(
    '',
    response_model=GETCustomSecrets,
    responses={
        401: {'description': 'Failed to get secret names', 'model': dict},
    },
)
async def load_custom_secrets_names(
    user_secrets: Secrets | None = Depends(get_secrets),
) -> GETCustomSecrets | JSONResponse:
    """Load custom secret names.

    Retrieves the names and descriptions of all custom secrets for the authenticated user.

    Returns:
        GETCustomSecrets: List of custom secrets (without values)
    """
    try:
        if not user_secrets:
            return GETCustomSecrets(custom_secrets=[])

        custom_secrets: list[CustomSecretWithoutValueModel] = []
        if user_secrets.custom_secrets:
            for secret_name, secret_value in user_secrets.custom_secrets.items():
                custom_secret = CustomSecretWithoutValueModel(
                    name=secret_name,
                    description=secret_value.description,
                )
                custom_secrets.append(custom_secret)

        return GETCustomSecrets(custom_secrets=custom_secrets)

    except Exception as e:
        logger.warning(f'Failed to load secret names: {e}')
        logger.info('Returning 401 Unauthorized - Failed to get secret names')
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Failed to get secret names'},
        )


@router.post(
    '',
    response_model=None,
    responses={
        201: {'description': 'Secret created successfully', 'model': dict},
        400: {'description': 'Secret already exists', 'model': dict},
        500: {'description': 'Error creating secret', 'model': dict},
    },
)
async def create_custom_secret(
    incoming_secret: CustomSecretModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
    """Create a custom secret.

    Creates a new custom secret for the authenticated user.

    Returns:
        201: Secret created successfully
        400: Secret already exists
        500: Error creating secret
    """
    try:
        existing_secrets = await secrets_store.load()
        custom_secrets = (
            dict(existing_secrets.custom_secrets) if existing_secrets else {}
        )

        secret_name = incoming_secret.name
        secret_value = incoming_secret.value
        secret_description = incoming_secret.description

        if secret_name in custom_secrets:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={'error': f'Secret {secret_name} already exists'},
            )

        custom_secrets[secret_name] = CustomSecret(
            secret=secret_value,
            description=secret_description or '',
        )

        # Create a new Secrets that preserves provider tokens
        updated_user_secrets = Secrets(
            custom_secrets=custom_secrets,  # type: ignore[arg-type]
            provider_tokens=existing_secrets.provider_tokens
            if existing_secrets
            else {},  # type: ignore[arg-type]
        )

        await secrets_store.store(updated_user_secrets)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={'message': 'Secret created successfully'},
        )
    except Exception as e:
        logger.warning(f'Something went wrong creating secret: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Something went wrong creating secret'},
        )


@router.put(
    '/{secret_id}',
    response_model=None,
    responses={
        200: {'description': 'Secret updated successfully', 'model': dict},
        400: {'description': 'Secret name already exists', 'model': dict},
        404: {'description': 'Secret not found', 'model': dict},
        500: {'description': 'Error updating secret', 'model': dict},
    },
)
async def update_custom_secret(
    secret_id: str,
    incoming_secret: CustomSecretWithoutValueModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
    """Update a custom secret.

    Updates the name and/or description of an existing custom secret.

    Returns:
        200: Secret updated successfully
        400: Secret name already exists
        404: Secret not found
        500: Error updating secret
    """
    try:
        existing_secrets = await secrets_store.load()
        if existing_secrets:
            # Check if the secret to update exists
            if secret_id not in existing_secrets.custom_secrets:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={'error': f'Secret with ID {secret_id} not found'},
                )

            secret_name = incoming_secret.name
            secret_description = incoming_secret.description

            custom_secrets = dict(existing_secrets.custom_secrets)
            existing_secret = custom_secrets.pop(secret_id)

            if secret_name != secret_id and secret_name in custom_secrets:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={'error': f'Secret {secret_name} already exists'},
                )

            custom_secrets[secret_name] = CustomSecret(
                secret=existing_secret.secret,
                description=secret_description or '',
            )

            updated_secrets = Secrets(
                custom_secrets=custom_secrets,  # type: ignore[arg-type]
                provider_tokens=existing_secrets.provider_tokens,
            )

            await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': 'Secret updated successfully'},
        )
    except Exception as e:
        logger.warning(f'Something went wrong updating secret: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Something went wrong updating secret'},
        )


@router.delete(
    '/{secret_id}',
    response_model=None,
    responses={
        200: {'description': 'Secret deleted successfully', 'model': dict},
        404: {'description': 'Secret not found', 'model': dict},
        500: {'description': 'Error deleting secret', 'model': dict},
    },
)
async def delete_custom_secret(
    secret_id: str,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
    """Delete a custom secret.

    Removes a custom secret for the authenticated user.

    Returns:
        200: Secret deleted successfully
        404: Secret not found
        500: Error deleting secret
    """
    try:
        existing_secrets = await secrets_store.load()
        if existing_secrets:
            # Get existing custom secrets
            custom_secrets = dict(existing_secrets.custom_secrets)

            # Check if the secret to delete exists
            if secret_id not in custom_secrets:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={'error': f'Secret with ID {secret_id} not found'},
                )

            # Remove the secret
            custom_secrets.pop(secret_id)

            # Create a new Secrets that preserves provider tokens and remaining secrets
            updated_secrets = Secrets(
                custom_secrets=custom_secrets,  # type: ignore[arg-type]
                provider_tokens=existing_secrets.provider_tokens,
            )

            await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': 'Secret deleted successfully'},
        )
    except Exception as e:
        logger.warning(f'Something went wrong deleting secret: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Something went wrong deleting secret'},
        )
