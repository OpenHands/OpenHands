from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.mcp_config import MCPConfig
from openhands.core.config.utils import load_openhands_config
from openhands.storage.data_models.secrets import Secrets


class Settings(BaseModel):
    """Persisted settings for OpenHands sessions"""

    language: str | None = None
    agent: str | None = None
    max_iterations: int | None = None
    security_analyzer: str | None = None
    confirmation_mode: bool | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    user_version: int | None = None
    remote_runtime_resource_factor: int | None = None
    # Planned to be removed from settings
    secrets_store: Secrets = Field(default_factory=Secrets, frozen=True)
    enable_default_condenser: bool = True
    enable_sound_notifications: bool = False
    enable_proactive_conversation_starters: bool = True
    enable_solvability_analysis: bool = True
    user_consents_to_analytics: bool | None = None
    sandbox_base_container_image: str | None = None
    sandbox_runtime_container_image: str | None = None
    mcp_config: MCPConfig | None = None
    search_api_key: SecretStr | None = None
    sandbox_api_key: SecretStr | None = None
    max_budget_per_task: float | None = None
    # Maximum number of events in the conversation view before condensation runs
    condenser_max_size: int | None = None
    email: str | None = None
    email_verified: bool | None = None
    git_user_name: str | None = None
    git_user_email: str | None = None
    v1_enabled: bool = True

    model_config = ConfigDict(
        validate_assignment=True,
    )

    @field_serializer('llm_api_key', 'search_api_key')
    def api_key_serializer(self, api_key: SecretStr | None, info: SerializationInfo):
        """Custom serializer for API keys.

        To serialize the API key instead of ********, set expose_secrets to True in the serialization context.
        """
        if api_key is None:
            return None

        # Get the secret value to check if it's empty
        secret_value = api_key.get_secret_value()
        if not secret_value or not secret_value.strip():
            return None

        context = info.context
        if context and context.get('expose_secrets', False):
            return secret_value

        return str(api_key)

    @model_validator(mode='before')
    @classmethod
    def convert_provider_tokens(cls, data: dict | object) -> dict | object:
        """Convert provider tokens from JSON format to Secrets format."""
        if not isinstance(data, dict):
            return data

        secrets_store = data.get('secrets_store')
        if not isinstance(secrets_store, dict):
            return data

        custom_secrets = secrets_store.get('custom_secrets')
        tokens = secrets_store.get('provider_tokens')

        secret_store = Secrets(provider_tokens={}, custom_secrets={})  # type: ignore[arg-type]

        if isinstance(tokens, dict):
            converted_store = Secrets(provider_tokens=tokens)  # type: ignore[arg-type]
            secret_store = secret_store.model_copy(
                update={'provider_tokens': converted_store.provider_tokens}
            )
        else:
            secret_store.model_copy(update={'provider_tokens': tokens})

        if isinstance(custom_secrets, dict):
            converted_store = Secrets(custom_secrets=custom_secrets)  # type: ignore[arg-type]
            secret_store = secret_store.model_copy(
                update={'custom_secrets': converted_store.custom_secrets}
            )
        else:
            secret_store = secret_store.model_copy(
                update={'custom_secrets': custom_secrets}
            )
        data['secret_store'] = secret_store
        return data

    @field_validator('condenser_max_size')
    @classmethod
    def validate_condenser_max_size(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 20:
            raise ValueError('condenser_max_size must be at least 20')
        return v

    @field_serializer('secrets_store')
    def secrets_store_serializer(self, secrets: Secrets, info: SerializationInfo):
        """Custom serializer for secrets store."""
        """Force invalidate secret store"""
        return {'provider_tokens': {}}

    @staticmethod
    def from_config() -> Settings | None:
        app_config = load_openhands_config()
        llm_config: LLMConfig = app_config.get_llm_config()
        if llm_config.api_key is None:
            # If no api key has been set, we take this to mean that there is no reasonable default
            return None
        security = app_config.security

        # Get MCP config if available
        mcp_config = None
        if hasattr(app_config, 'mcp'):
            mcp_config = app_config.mcp

        settings = Settings(
            language='en',
            agent=app_config.default_agent,
            max_iterations=app_config.max_iterations,
            security_analyzer=security.security_analyzer,
            confirmation_mode=security.confirmation_mode,
            llm_model=llm_config.model,
            llm_api_key=llm_config.api_key,
            llm_base_url=llm_config.base_url,
            remote_runtime_resource_factor=app_config.sandbox.remote_runtime_resource_factor,
            mcp_config=mcp_config,
            search_api_key=app_config.search_api_key,
            max_budget_per_task=app_config.max_budget_per_task,
        )
        return settings

    def merge_with_config_settings(self) -> 'Settings':
        """Merge config.toml settings with stored settings.

        Config.toml takes priority for MCP settings, but they are merged rather than replaced.
        This method can be used by both server mode and CLI mode.
        """
        # Get config.toml settings
        config_settings = Settings.from_config()
        if not config_settings or not config_settings.mcp_config:
            return self

        # If stored settings don't have MCP config, use config.toml MCP config
        if not self.mcp_config:
            self.mcp_config = config_settings.mcp_config
            return self

        # Both have MCP config - merge them with config.toml taking priority
        merged_mcp = MCPConfig(
            sse_servers=list(config_settings.mcp_config.sse_servers)
            + list(self.mcp_config.sse_servers),
            stdio_servers=list(config_settings.mcp_config.stdio_servers)
            + list(self.mcp_config.stdio_servers),
            shttp_servers=list(config_settings.mcp_config.shttp_servers)
            + list(self.mcp_config.shttp_servers),
        )

        # Create new settings with merged MCP config
        self.mcp_config = merged_mcp
        return self
