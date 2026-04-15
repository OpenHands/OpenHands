from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from fastmcp.mcp_config import MCPConfig as SDKMCPConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.utils import load_openhands_config
from openhands.sdk.settings import AgentSettings, ConversationSettings
from openhands.storage.data_models.secrets import Secrets
from openhands.utils.jsonpatch_compat import deep_merge


def _coerce_value(value: Any) -> Any:
    """Unwrap SecretStr to plain values."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, SDKMCPConfig):
        return value.model_dump(exclude_none=True, exclude_defaults=True) or None
    return value


def _coerce_dict_secrets(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively coerce SecretStr / MCPConfig leaves to plain values."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _coerce_dict_secrets(v)
        else:
            out[k] = _coerce_value(v)
    return out


def _merge_sdk_mcp_configs(
    base_config: SDKMCPConfig | None, extra_config: SDKMCPConfig | None
) -> SDKMCPConfig | None:
    if base_config is None:
        return extra_config
    if extra_config is None:
        return base_config

    merged_servers: dict[str, Any] = {}

    def _add_server(server_name: str, server_config: dict[str, Any]) -> None:
        candidate = server_name or 'server'
        if candidate not in merged_servers:
            merged_servers[candidate] = server_config
            return

        suffix = 1
        while f'{candidate}_{suffix}' in merged_servers:
            suffix += 1
        merged_servers[f'{candidate}_{suffix}'] = server_config

    for config in (base_config, extra_config):
        raw_config = config.model_dump(exclude_none=True)
        for server_name, server_config in raw_config.get('mcpServers', {}).items():
            _add_server(server_name, server_config)

    if not merged_servers:
        return None

    return SDKMCPConfig.model_validate({'mcpServers': merged_servers})


class SandboxGroupingStrategy(str, Enum):
    """Strategy for grouping conversations within sandboxes."""

    NO_GROUPING = 'NO_GROUPING'  # Default - each conversation gets its own sandbox
    GROUP_BY_NEWEST = 'GROUP_BY_NEWEST'  # Add to the most recently created sandbox
    LEAST_RECENTLY_USED = (
        'LEAST_RECENTLY_USED'  # Add to the least recently used sandbox
    )
    FEWEST_CONVERSATIONS = (
        'FEWEST_CONVERSATIONS'  # Add to sandbox with fewest conversations
    )
    ADD_TO_ANY = 'ADD_TO_ANY'  # Add to any available sandbox (first found)


_SETTINGS_FROZEN_FIELDS = frozenset(['secrets_store'])


class Settings(BaseModel):
    """Persisted settings for OpenHands sessions.

    Agent settings (agent, llm, mcp, condenser) live in ``agent_settings``.
    Conversation settings (max_iterations, confirmation_mode, security_analyzer)
    live in ``conversation_settings``.
    Product settings remain as top-level fields.
    """

    language: str | None = None
    user_version: int | None = None
    remote_runtime_resource_factor: int | None = None
    # Planned to be removed from settings
    secrets_store: Annotated[Secrets, Field(frozen=True)] = Field(
        default_factory=Secrets
    )
    enable_sound_notifications: bool = False
    enable_proactive_conversation_starters: bool = True
    enable_solvability_analysis: bool = True
    user_consents_to_analytics: bool | None = None
    sandbox_base_container_image: str | None = None
    sandbox_runtime_container_image: str | None = None
    disabled_skills: list[str] | None = None
    search_api_key: SecretStr | None = None
    sandbox_api_key: SecretStr | None = None
    max_budget_per_task: float | None = None
    email: str | None = None
    email_verified: bool | None = None
    git_user_name: str | None = None
    git_user_email: str | None = None
    v1_enabled: bool = True
    agent_settings: AgentSettings = Field(default_factory=AgentSettings)
    conversation_settings: ConversationSettings = Field(
        default_factory=ConversationSettings
    )
    sandbox_grouping_strategy: SandboxGroupingStrategy = (
        SandboxGroupingStrategy.NO_GROUPING
    )

    model_config = ConfigDict(populate_by_name=True)

    @property
    def llm_api_key_is_set(self) -> bool:
        raw = self.agent_settings.llm.api_key
        if raw is None:
            return False
        secret_value = (
            raw.get_secret_value() if isinstance(raw, SecretStr) else str(raw)
        )
        return bool(secret_value and secret_value.strip())

    # ── Batch update ────────────────────────────────────────────────

    def update(self, payload: dict[str, Any]) -> None:
        """Apply a batch of changes from a nested dict.

        ``agent_settings`` values use nested dict shape (matching model_dump).
        ``conversation_settings`` values likewise.
        Top-level keys are set directly on the model.
        """
        if 'agent_settings' in payload:
            agent_update = payload['agent_settings']
            if isinstance(agent_update, dict):
                coerced: dict[str, Any] = {}
                for key, value in agent_update.items():
                    coerced[key] = (
                        _coerce_value(value) if not isinstance(value, dict) else value
                    )

                replace_mcp_config = 'mcp_config' in agent_update
                mcp_config = (
                    coerced.pop('mcp_config', None) if replace_mcp_config else None
                )

                merged = deep_merge(
                    self.agent_settings.model_dump(
                        mode='json', context={'expose_secrets': True}
                    ),
                    coerced,
                )
                if replace_mcp_config:
                    merged['mcp_config'] = mcp_config

                # Use object.__setattr__ to avoid validate_assignment
                # side-effects on other fields.
                object.__setattr__(
                    self, 'agent_settings', AgentSettings.model_validate(merged)
                )

        if 'conversation_settings' in payload:
            conv_update = payload['conversation_settings']
            if isinstance(conv_update, dict):
                merged = deep_merge(
                    self.conversation_settings.model_dump(mode='json'),
                    conv_update,
                )
                object.__setattr__(
                    self,
                    'conversation_settings',
                    ConversationSettings.model_validate(merged),
                )

        for key, value in payload.items():
            if key in ('agent_settings', 'conversation_settings'):
                continue
            if key in Settings.model_fields and key not in _SETTINGS_FROZEN_FIELDS:
                field_info = Settings.model_fields[key]
                # Coerce plain strings to SecretStr when the field type expects it
                if value is not None and isinstance(value, str):
                    annotation = field_info.annotation
                    if annotation is SecretStr or (
                        hasattr(annotation, '__args__')
                        and SecretStr in getattr(annotation, '__args__', ())
                    ):
                        value = SecretStr(value) if value else None
                setattr(self, key, value)

    # ── Serialization ───────────────────────────────────────────────

    @field_serializer('search_api_key')
    def api_key_serializer(self, api_key: SecretStr | None, info: SerializationInfo):
        if api_key is None:
            return None
        secret_value = api_key.get_secret_value()
        if not secret_value or not secret_value.strip():
            return None
        context = info.context
        if context and context.get('expose_secrets', False):
            return secret_value
        return str(api_key)

    @field_serializer('agent_settings')
    def agent_settings_serializer(
        self, agent_settings: AgentSettings, info: SerializationInfo
    ) -> dict[str, Any]:
        context = info.context or {}
        if context.get('expose_secrets', False):
            return agent_settings.model_dump(
                mode='json', context={'expose_secrets': True}
            )
        return agent_settings.model_dump(mode='json')

    @model_validator(mode='before')
    @classmethod
    def _normalize_inputs(cls, data: dict | object) -> dict | object:
        """Normalize agent_settings and secrets_store inputs."""
        if not isinstance(data, dict):
            return data

        # --- Agent settings: coerce SecretStr leaves to plain strings ---
        agent_settings = data.get('agent_settings')
        if isinstance(agent_settings, dict):
            data['agent_settings'] = _coerce_dict_secrets(agent_settings)
        elif isinstance(agent_settings, AgentSettings):
            data['agent_settings'] = agent_settings.model_dump(
                mode='json', context={'expose_secrets': True}
            )

        # --- Conversation settings: normalize ---
        conversation_settings = data.get('conversation_settings')
        if isinstance(conversation_settings, ConversationSettings):
            data['conversation_settings'] = conversation_settings.model_dump(
                mode='json'
            )

        # --- Secrets store ---
        secrets_store = data.get('secrets_store')
        if isinstance(secrets_store, dict):
            custom_secrets = secrets_store.get('custom_secrets')
            tokens = secrets_store.get('provider_tokens')
            secret_store = Secrets.model_validate(
                {'provider_tokens': {}, 'custom_secrets': {}}
            )
            if isinstance(tokens, dict):
                converted_store = Secrets.model_validate({'provider_tokens': tokens})
                secret_store = secret_store.model_copy(
                    update={'provider_tokens': converted_store.provider_tokens}
                )
            if isinstance(custom_secrets, dict):
                converted_store = Secrets.model_validate(
                    {'custom_secrets': custom_secrets}
                )
                secret_store = secret_store.model_copy(
                    update={'custom_secrets': converted_store.custom_secrets}
                )
            data['secret_store'] = secret_store

        return data

    @field_serializer('secrets_store')
    def secrets_store_serializer(self, secrets: Secrets, info: SerializationInfo):
        return {'provider_tokens': {}}

    # ── Factory methods ─────────────────────────────────────────────

    @staticmethod
    def from_config() -> Settings | None:
        app_config = load_openhands_config()
        llm_config: LLMConfig = app_config.get_llm_config()
        if llm_config.api_key is None:
            return None

        agent_settings_dict: dict[str, Any] = {
            'agent': app_config.default_agent,
            'llm': {
                'model': llm_config.model,
                'api_key': (
                    llm_config.api_key.get_secret_value()
                    if isinstance(llm_config.api_key, SecretStr)
                    else llm_config.api_key
                ),
                'base_url': llm_config.base_url,
            },
        }
        if hasattr(app_config, 'mcp') and app_config.mcp:
            agent_settings_dict['mcp_config'] = _coerce_value(app_config.mcp)

        return Settings(
            language='en',
            remote_runtime_resource_factor=app_config.sandbox.remote_runtime_resource_factor,
            search_api_key=app_config.search_api_key,
            max_budget_per_task=app_config.max_budget_per_task,
            agent_settings=AgentSettings(**agent_settings_dict),
            conversation_settings=ConversationSettings.model_validate(
                {
                    'confirmation_mode': bool(app_config.security.confirmation_mode),
                    'security_analyzer': app_config.security.security_analyzer,
                    'max_iterations': app_config.max_iterations,
                }
            ),
        )

    def merge_with_config_settings(self) -> 'Settings':
        """Merge config.toml MCP settings with stored SDK agent_settings."""
        config_settings = Settings.from_config()
        if not config_settings:
            return self

        merged_mcp = _merge_sdk_mcp_configs(
            config_settings.agent_settings.mcp_config,
            self.agent_settings.mcp_config,
        )
        if merged_mcp is None:
            return self

        self.agent_settings.mcp_config = merged_mcp
        return self

    def to_agent_settings(self) -> AgentSettings:
        return self.agent_settings

    def get_agent_settings_display(self) -> dict[str, Any]:
        """Return agent_settings dict with display-friendly model names.

        ``litellm_proxy/`` prefixes are normalised to ``openhands/``.
        The LiteLLM proxy ``base_url`` is cleared for managed models so
        that the frontend can display "basic" mode.
        Secrets are masked by Pydantic's default serialiser.
        """
        from openhands.utils.llm import is_openhands_model

        data = self.agent_settings.model_dump(mode='json')
        llm = data.get('llm')
        if isinstance(llm, dict):
            model = llm.get('model')
            if isinstance(model, str) and model.startswith('litellm_proxy/'):
                llm['model'] = f'openhands/{model.removeprefix("litellm_proxy/")}'
            # Clear the proxy base_url for managed models so the frontend
            # sees null and can display the simple "basic" settings view.
            if is_openhands_model(model):
                base_url = llm.get('base_url')
                if isinstance(base_url, str) and base_url.rstrip('/').endswith(
                    'llm-proxy.app.all-hands.dev'
                ):
                    llm['base_url'] = None
        return data
