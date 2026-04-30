"""Settings models for OpenHands App Server.

This module contains:
- Settings: Persisted settings for OpenHands sessions
- SandboxGroupingStrategy: Strategy enum for grouping conversations
- GETSettingsModel: Settings response model with additional token data
- POSTProviderModel: Settings for POST requests
- CustomSecretWithoutValueModel: Custom secret model without value (legacy)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from fastmcp.mcp_config import MCPConfig
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

from openhands.app_server.integrations.provider import ProviderToken
from openhands.app_server.integrations.service_types import ProviderType
from openhands.app_server.settings.llm_profiles import LLMProfiles
from openhands.app_server.utils.jsonpatch_compat import deep_merge
from openhands.app_server.utils.sdk_settings_compat import (
    ACPAgentSettings,
    AgentSettingsConfig,
    LLMAgentSettings,
    default_agent_settings,
    validate_agent_settings,
)
from openhands.sdk.settings import ConversationSettings


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


# Fields the batch ``update()`` method refuses to touch:
# - ``secrets_store`` is frozen (Pydantic would raise).
# - ``llm_profiles`` is off-limits for the generic settings POST; profile
#   mutations go through ``/api/v1/settings/profiles/...`` which validate
#   inputs, enforce the count cap, and take the per-user lock. Accepting a
#   raw dict here both bypassed those guards and crashed downstream
#   serialisation.
_SETTINGS_UPDATE_IGNORED_FIELDS = frozenset(['secrets_store', 'llm_profiles'])


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
    # Planned to be removed from settings - import Secrets lazily to avoid circular imports
    secrets_store: Annotated[Any, Field(frozen=True)] = Field(default=None)
    enable_sound_notifications: bool = False
    enable_proactive_conversation_starters: bool = True
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
    agent_settings: AgentSettingsConfig = Field(default_factory=default_agent_settings)
    conversation_settings: ConversationSettings = Field(
        default_factory=ConversationSettings
    )
    sandbox_grouping_strategy: SandboxGroupingStrategy = (
        SandboxGroupingStrategy.NO_GROUPING
    )
    llm_profiles: LLMProfiles = Field(
        default_factory=LLMProfiles,
        description=(
            'Saved LLM profiles and the currently active profile name. '
            'See ``LLMProfiles`` for the profile-management API.'
        ),
    )

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, **data: Any):
        # Import Secrets here to avoid circular imports
        from openhands.app_server.secrets.secrets_models import Secrets

        if 'secrets_store' not in data or data['secrets_store'] is None:
            data['secrets_store'] = Secrets()
        super().__init__(**data)

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

    def reconcile_active_profile(self) -> None:
        """Clear ``llm_profiles.active`` when the current LLM diverges from it.

        The active profile is a pointer into ``llm_profiles.profiles``; if the
        user edits ``agent_settings.llm`` directly (via the main settings
        endpoint), the pointer becomes a lie. Rather than mutate the saved
        profile, we drop the active marker so the frontend stops claiming a
        profile is "in use" that no longer matches what's actually running.
        """
        active = self.llm_profiles.active
        if active is None:
            return
        saved = self.llm_profiles.get(active)
        if saved is None or saved != self.agent_settings.llm:
            self.llm_profiles.active = None

    def update(self, payload: dict[str, Any]) -> None:
        """Apply a batch of changes from a nested dict.

        ``agent_settings_diff`` and ``conversation_settings_diff`` use nested
        dict shape (matching model_dump). Top-level keys are set directly on the
        model.
        """
        legacy_nested_keys = [
            key for key in ('agent_settings', 'conversation_settings') if key in payload
        ]
        if legacy_nested_keys:
            raise ValueError(
                'Use *_diff nested settings payloads instead of legacy '
                + ', '.join(sorted(legacy_nested_keys))
            )

        agent_update = payload.get('agent_settings_diff')
        if isinstance(agent_update, dict):
            coerced: dict[str, Any] = {}
            for key, value in agent_update.items():
                coerced[key] = (
                    _coerce_value(value) if not isinstance(value, dict) else value
                )

            replace_mcp_config = 'mcp_config' in agent_update
            mcp_config = coerced.pop('mcp_config', None) if replace_mcp_config else None

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
            object.__setattr__(self, 'agent_settings', validate_agent_settings(merged))

        conv_update = payload.get('conversation_settings_diff')
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
            if key in ('agent_settings_diff', 'conversation_settings_diff'):
                continue
            if (
                key in Settings.model_fields
                and key not in _SETTINGS_UPDATE_IGNORED_FIELDS
            ):
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

        self.reconcile_active_profile()

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
        self,
        agent_settings: LLMAgentSettings | ACPAgentSettings,
        info: SerializationInfo,
    ) -> dict[str, Any]:
        context = info.context or {}
        if context.get('expose_secrets', False):
            return agent_settings.model_dump(
                mode='json', context={'expose_secrets': True}
            )
        return agent_settings.model_dump(mode='json')

    # ── Profile management ─────────────────────────────────────────
    #
    # Pure profile operations (get/save/delete/summaries) live on
    # ``LLMProfiles``. ``switch_to_profile`` remains here because it
    # touches ``agent_settings.llm``.

    def switch_to_profile(self, name: str) -> None:
        """Switch ``agent_settings.llm`` to a saved profile.

        Raises :class:`ProfileNotFoundError` if ``name`` isn't a saved profile.
        """
        llm = self.llm_profiles.require(name)
        self.agent_settings = self.agent_settings.model_copy(update={'llm': llm})
        self.llm_profiles.active = name

    @model_validator(mode='before')
    @classmethod
    def _normalize_inputs(cls, data: dict | object) -> dict | object:
        """Normalize agent_settings and secrets_store inputs."""
        # Import Secrets here to avoid circular imports
        from openhands.app_server.secrets.secrets_models import Secrets

        if not isinstance(data, dict):
            return data

        # --- Agent settings: coerce SecretStr leaves to plain strings ---
        agent_settings = data.get('agent_settings')
        if isinstance(agent_settings, dict):
            data['agent_settings'] = _coerce_dict_secrets(agent_settings)
        elif isinstance(agent_settings, (LLMAgentSettings, ACPAgentSettings)):
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
            data['secrets_store'] = secret_store

        return data

    @field_serializer('secrets_store')
    def secrets_store_serializer(self, secrets: Any, info: SerializationInfo):
        return {'provider_tokens': {}}

    def to_agent_settings(self) -> LLMAgentSettings | ACPAgentSettings:
        return self.agent_settings

    def get_agent_settings_display(self) -> dict[str, Any]:
        """Return agent_settings dict with display-friendly model names.

        ``litellm_proxy/`` prefixes are normalised to ``openhands/`` only when
        using the OpenHands proxy URL. Custom litellm_proxy endpoints keep their
        litellm_proxy/ prefix.
        The LiteLLM proxy ``base_url`` is cleared for managed models so
        that the frontend can display "basic" mode.
        Secrets are masked by Pydantic's default serialiser.
        """
        from openhands.app_server.settings.settings_router import LITE_LLM_API_URL
        from openhands.app_server.utils.llm import is_openhands_model

        data = self.agent_settings.model_dump(mode='json')
        llm = data.get('llm')
        if isinstance(llm, dict):
            model = llm.get('model')
            base_url = llm.get('base_url')

            # Only convert litellm_proxy/ to openhands/ if using the OpenHands proxy
            if isinstance(model, str) and model.startswith('litellm_proxy/'):
                normalized_base = (base_url or '').rstrip('/')
                normalized_proxy = LITE_LLM_API_URL.rstrip('/')
                if normalized_base == normalized_proxy:
                    llm['model'] = f'openhands/{model.removeprefix("litellm_proxy/")}'

            # Clear the proxy base_url for managed models so the frontend
            # sees null and can display the simple "basic" settings view.
            if is_openhands_model(model):
                normalized_base = (base_url or '').rstrip('/')
                normalized_proxy = LITE_LLM_API_URL.rstrip('/')
                if normalized_base == normalized_proxy:
                    llm['base_url'] = None
        return data


# ── Legacy V0 Models (scheduled for removal April 1, 2026) ──────────


class POSTProviderModel(BaseModel):
    """Settings for POST requests"""

    mcp_config: MCPConfig | None = None
    provider_tokens: dict[ProviderType, ProviderToken] = {}


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
