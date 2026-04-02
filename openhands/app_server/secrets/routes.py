from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from openhands.core.logger import openhands_logger as logger
from openhands.server.dependencies import get_dependencies
from openhands.server.settings import (
    CustomSecretModel,
    CustomSecretWithoutValueModel,
    GETCustomSecrets,
)
from openhands.server.user_auth import (
    get_secrets,
    get_secrets_store,
)
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.secrets.secrets_store import SecretsStore

router = APIRouter(prefix="/api/v1", dependencies=get_dependencies())


@router.get("/secrets", response_model=GETCustomSecrets)
async def load_custom_secrets_names(
    user_secrets: Secrets | None = Depends(get_secrets),
) -> GETCustomSecrets | JSONResponse:
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
        logger.warning(f"Failed to load secret names: {e}")
        logger.info("Returning 401 Unauthorized - Failed to get secret names")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Failed to get secret names"},
        )


@router.post("/secrets", response_model=dict[str, str])
async def create_custom_secret(
    incoming_secret: CustomSecretModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
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
                content={"message": f"Secret {secret_name} already exists"},
            )

        from openhands.integrations.provider import CustomSecret

        custom_secrets[secret_name] = CustomSecret(
            secret=secret_value,
            description=secret_description or "",
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
            content={"message": "Secret created successfully"},
        )
    except Exception as e:
        logger.warning(f"Something went wrong creating secret: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Something went wrong creating secret"},
        )


@router.put("/secrets/{secret_id}", response_model=dict[str, str])
async def update_custom_secret(
    secret_id: str,
    incoming_secret: CustomSecretWithoutValueModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
    try:
        existing_secrets = await secrets_store.load()
        if existing_secrets:
            # Check if the secret to update exists
            if secret_id not in existing_secrets.custom_secrets:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"error": f"Secret with ID {secret_id} not found"},
                )

            secret_name = incoming_secret.name
            secret_description = incoming_secret.description

            custom_secrets = dict(existing_secrets.custom_secrets)
            existing_secret = custom_secrets.pop(secret_id)

            if secret_name != secret_id and secret_name in custom_secrets:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"message": f"Secret {secret_name} already exists"},
                )

            from openhands.integrations.provider import CustomSecret

            custom_secrets[secret_name] = CustomSecret(
                secret=existing_secret.secret,
                description=secret_description or "",
            )

            updated_secrets = Secrets(
                custom_secrets=custom_secrets,  # type: ignore[arg-type]
                provider_tokens=existing_secrets.provider_tokens,
            )

            await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Secret updated successfully"},
        )
    except Exception as e:
        logger.warning(f"Something went wrong updating secret: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Something went wrong updating secret"},
        )


@router.delete("/secrets/{secret_id}")
async def delete_custom_secret(
    secret_id: str,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> JSONResponse:
    try:
        existing_secrets = await secrets_store.load()
        if existing_secrets:
            # Get existing custom secrets
            custom_secrets = dict(existing_secrets.custom_secrets)

            # Check if the secret to delete exists
            if secret_id not in custom_secrets:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"error": f"Secret with ID {secret_id} not found"},
                )

            # Remove the secret
            del custom_secrets[secret_id]

            updated_secrets = Secrets(
                custom_secrets=custom_secrets,  # type: ignore[arg-type]
                provider_tokens=existing_secrets.provider_tokens,
            )

            await secrets_store.store(updated_secrets)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Secret deleted successfully"},
        )
    except Exception as e:
        logger.warning(f"Something went wrong deleting secret: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Something went wrong deleting secret"},
        )