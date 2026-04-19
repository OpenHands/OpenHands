# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from fastapi import APIRouter, Depends, status

from openhands.app_server.secrets.secrets_router import (
    create_custom_secret as v1_create_custom_secret,
)
from openhands.app_server.secrets.secrets_router import (
    delete_custom_secret as v1_delete_custom_secret,
)
from openhands.app_server.secrets.secrets_router import search_custom_secrets
from openhands.app_server.secrets.secrets_router import (
    store_provider_tokens as v1_store_provider_tokens,
)
from openhands.app_server.secrets.secrets_router import (
    unset_provider_tokens as v1_unset_provider_tokens,
)
from openhands.app_server.secrets.secrets_router import (
    update_custom_secret as v1_update_custom_secret,
)
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.models import EditResponse
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.sdk.utils.paging import page_iterator
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

app = APIRouter(prefix='/api', dependencies=get_dependencies())


# =================================================
# SECTION: Handle git provider tokens
# =================================================


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


@app.post('/add-git-providers', deprecated=True)
async def store_provider_tokens(
    provider_info: POSTProviderModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
) -> EditResponse:
    """Store git provider tokens.

    .. deprecated::
        Use ``POST /api/v1/secrets/git-providers`` instead. This V0 endpoint will be removed
        April 1, 2026.
    """
    return await v1_store_provider_tokens(provider_info, secrets_store, provider_tokens)


@app.post('/unset-provider-tokens', deprecated=True)
async def unset_provider_tokens(
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> EditResponse:
    """Unset (delete) all git provider tokens.

    .. deprecated::
        Use ``DELETE /api/v1/secrets/git-providers`` instead. This V0 endpoint will be removed
        April 1, 2026.
    """
    return await v1_unset_provider_tokens(secrets_store)


# =================================================
# SECTION: Handle custom secrets
# =================================================


@app.get('/secrets', response_model=GETCustomSecrets, deprecated=True)
async def load_custom_secrets_names(
    user_secrets: Secrets | None = Depends(get_secrets),
) -> GETCustomSecrets:
    """Load custom secret names.

    .. deprecated::
        Use ``GET /api/v1/secrets/search`` instead. This V0 endpoint will be removed
        April 1, 2026. The V1 endpoint provides pagination and filtering support.
    """
    custom_secrets = []
    async for custom_secret in page_iterator(
        search_custom_secrets, user_secrets=user_secrets
    ):
        custom_secrets.append(
            CustomSecretWithoutValueModel(
                name=custom_secret.name, description=custom_secret.description
            )
        )
    result = GETCustomSecrets(custom_secrets=custom_secrets)
    return result


@app.post('/secrets', status_code=status.HTTP_201_CREATED, deprecated=True)
async def create_custom_secret(
    incoming_secret: CustomSecretModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> EditResponse:
    """Create a custom secret.

    .. deprecated::
        Use ``POST /api/v1/secrets`` instead. This V0 endpoint will be removed
        April 1, 2026.
    """
    return await v1_create_custom_secret(incoming_secret, secrets_store)


@app.put('/secrets/{secret_id}', deprecated=True)
async def update_custom_secret(
    secret_id: str,
    incoming_secret: CustomSecretWithoutValueModel,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> EditResponse:
    """Update a custom secret.

    .. deprecated::
        Use ``PUT /api/v1/secrets/{secret_id}`` instead. This V0 endpoint will be removed
        April 1, 2026.
    """
    return await v1_update_custom_secret(secret_id, incoming_secret, secrets_store)


@app.delete('/secrets/{secret_id}', deprecated=True)
async def delete_custom_secret(
    secret_id: str,
    secrets_store: SecretsStore = Depends(get_secrets_store),
) -> EditResponse:
    """Delete a custom secret.

    .. deprecated::
        Use ``DELETE /api/v1/secrets/{secret_id}`` instead. This V0 endpoint will be removed
        April 1, 2026.
    """
    return await v1_delete_custom_secret(secret_id, secrets_store)
