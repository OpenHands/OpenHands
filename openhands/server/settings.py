from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    SecretStr,
)

from openhands.core.config.mcp_config import MCPConfig
from openhands.integrations.provider import CustomSecret, ProviderToken
from openhands.integrations.service_types import ProviderType
from openhands.storage.data_models.settings import Settings


class POSTProviderModel(BaseModel):
    """Settings for POST requests"""

    mcp_config: MCPConfig | None = None
    provider_tokens: dict[ProviderType, ProviderToken] = {}


class POSTCustomSecrets(BaseModel):
    """Adding new custom secret"""

    custom_secrets: dict[str, CustomSecret] = {}


class GETSettingsModel(Settings):
    """Settings with additional token data for the frontend"""

    provider_tokens_set: dict[ProviderType, str | None] | None = (
        None  # provider + base_domain key-value pair
    )
    llm_api_key_set: bool
    search_api_key_set: bool = False

    model_config = ConfigDict(use_enum_values=True)


class CustomSecretWithoutValueModel(BaseModel):
    """Custom secret model without value"""

    name: str
    description: str | None = None


class CustomSecretModel(CustomSecretWithoutValueModel):
    """Custom secret model with value"""

    value: SecretStr


class GETCustomSecrets(BaseModel):
    """Custom secrets names"""

    custom_secrets: list[CustomSecretWithoutValueModel] | None = None
