"""Settings models for OpenHands App Server.

This module contains:
- Settings: Persisted settings for OpenHands sessions
- SandboxGroupingStrategy: Strategy enum for grouping conversations
- GETSettingsModel: Settings response model with additional token data
- POSTProviderModel: Settings for POST requests
- CustomSecretWithoutValueModel: Custom secret model without value (legacy)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from typing import Annotated, Any, Sequence

from fastmcp.mcp_config import MCPConfig
from fastmcp.mcp_config import MCPConfig as SDKMCPConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    SerializationInfo,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from openhands.app_server.integrations.provider import ProviderToken
from openhands.app_server.integrations.service_types import ProviderType
from openhands.app_server.settings.llm_profiles import LLMProfiles
from openhands.app_server.utils.jsonpatch_compat import deep_merge
from openhands.sdk.mcp.config import MCPServer, dump_mcp_config
from openhands.sdk.settings import (
    ACPAgentSettings,
    AgentSettingsConfig,
    ConversationSettings,
    OpenHandsAgentSettings,
    apply_agent_settings_diff,
    default_agent_settings,
    validate_agent_settings,
)
from openhands.sdk.utils.pydantic_secrets import REDACTED_SECRET_VALUE

logger = logging.getLogger(__name__)

# Valid source patterns for MarketplaceRegistration
# - github:owner/repo format
_GITHUB_SOURCE_PATTERN = re.compile(r'^github:[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$')
# - Git URLs (https, git, ssh protocols). '~' in paths covers Bitbucket Data
#   Center personal repos (/scm/~user/repo); userinfo may only be a bare
#   username — 'user:password@' forms stay rejected.
_GIT_URL_PATTERN = re.compile(
    r'^(https?://(?:[a-zA-Z0-9_.-]+@)?|git@|ssh://(?:[a-zA-Z0-9_.-]+@)?|git://)'
    r'[a-zA-Z0-9_.-]+[:/][a-zA-Z0-9_.~/-]+$'
)
# - Relative local paths (no absolute paths, no parent traversal)
_LOCAL_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_][a-zA-Z0-9_./-]*$')


class MarketplaceScope(str, Enum):
    """Scope of a marketplace registration."""

    INSTANCE = 'instance'
    ORG = 'org'
    PERSONAL = 'personal'


class MarketplaceRegistration(BaseModel):
    """Registration for a plugin marketplace.

    Represents a marketplace that can be registered for plugin resolution.
    Marketplaces can be auto-loaded (plugins loaded at conversation start)
    or registered only (available for explicit plugin references).

    Wire-compatible with ``openhands.sdk.marketplace.MarketplaceRegistration``
    (the model the agent-server ``/api/skills`` endpoint consumes): dumping this
    model with ``exclude={'scope'}`` yields exactly the SDK model's fields
    (``name``/``source``/``ref``/``repo_path``/``auto_load``), and our ``bool``
    ``auto_load`` and stricter field validators are a subset of what the SDK
    accepts, so any value we produce validates upstream.

    This is intentionally kept as a separate model rather than importing the SDK
    one, because it carries a backend-only ``scope`` (set per storage layer for
    API responses/UI, stripped at the wire boundary and re-derived during
    composition) and enforces input validation the SDK model does not (source /
    ``repo_path`` traversal guards, name format).

    Examples:
        >>> # Auto-load all plugins from a marketplace
        >>> MarketplaceRegistration(
        ...     name="public",
        ...     source="github:OpenHands/skills",
        ...     auto_load=True
        ... )

        >>> # Register marketplace without auto-loading
        >>> MarketplaceRegistration(
        ...     name="experimental",
        ...     source="github:acme/experimental"
        ... )

        >>> # Marketplace in monorepo subdirectory
        >>> MarketplaceRegistration(
        ...     name="team",
        ...     source="github:acme/monorepo",
        ...     repo_path="marketplaces/internal",
        ...     auto_load=True
        ... )
    """

    name: str = Field(description='Identifier for this marketplace registration')
    source: str = Field(
        description="Marketplace source: 'github:owner/repo', git URL, or local path"
    )
    ref: str | None = Field(
        default=None,
        description='Optional branch, tag, or commit (only for git sources)',
    )
    repo_path: str | None = Field(
        default=None,
        description=(
            'Subdirectory path within the git repository containing the marketplace '
            "(e.g., 'marketplaces/internal' for monorepos). "
            'Only relevant for git sources, not local paths.'
        ),
    )
    auto_load: bool = Field(
        default=False,
        description=(
            'Auto-load behavior for this marketplace. '
            'True = load all plugins at conversation start. '
            'False = registered for resolution but not auto-loaded.'
        ),
    )
    scope: 'MarketplaceScope | None' = Field(
        default=None,
        description=(
            'Scope of this marketplace registration. '
            'Set automatically by backend based on storage layer: '
            '"instance" for system defaults, "org" for organization-level, '
            '"personal" for user-level. '
            'Frontend should NOT send this field in save requests.'
        ),
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty and contains only valid identifier characters."""
        if not v or not v.strip():
            raise ValueError('name cannot be empty')
        v = v.strip()
        # Name should be a valid identifier (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v):
            raise ValueError(
                'name must start with a letter and contain only '
                'letters, numbers, hyphens, and underscores'
            )
        return v

    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate source matches expected patterns (github:owner/repo, git URL, or local path)."""
        if not v or not v.strip():
            raise ValueError('source cannot be empty')
        v = v.strip()

        # Check for valid source patterns
        if _GITHUB_SOURCE_PATTERN.match(v):
            return v
        if _GIT_URL_PATTERN.match(v):
            return v
        # Local path: must be relative, no parent traversal
        if v.startswith('/'):
            raise ValueError('local path source must be relative, not absolute')
        if '..' in v:
            raise ValueError("source cannot contain '..' (parent directory traversal)")
        if _LOCAL_PATH_PATTERN.match(v):
            return v

        raise ValueError(
            "source must be 'github:owner/repo', a git URL "
            '(https/git/ssh), or a relative local path'
        )

    @field_validator('repo_path')
    @classmethod
    def validate_repo_path(cls, v: str | None) -> str | None:
        """Validate repo_path is a safe relative path within the repository."""
        if v is None:
            return v
        # A common mistake is pasting the repository URL here; repo_path is a
        # subdirectory *within* the already-specified source repository.
        if '://' in v:
            raise ValueError(
                'repo_path must be a subdirectory within the repository, not a URL'
            )
        # Must be relative (no absolute paths)
        if v.startswith('/'):
            raise ValueError('repo_path must be relative, not absolute')
        # No parent directory traversal
        if '..' in v:
            raise ValueError(
                "repo_path cannot contain '..' (parent directory traversal)"
            )
        return v


def validate_and_convert_marketplaces(
    raw_marketplaces: Sequence[dict[str, Any] | MarketplaceRegistration] | None,
    source_name: str = 'marketplaces',
) -> list[MarketplaceRegistration]:
    """Validate and convert raw marketplace data to MarketplaceRegistration objects.

    This function handles the common pattern of validating marketplace data
    that comes from database storage (stored as dicts) or direct model instances.
    Invalid entries are logged and skipped (graceful degradation).

    Args:
        raw_marketplaces: List of raw marketplace data (dicts or model instances)
        source_name: Descriptive name for logging (e.g., "org", "user settings")

    Returns:
        List of validated MarketplaceRegistration objects.

    Example:
        >>> data = [{'name': 'test', 'source': 'github:owner/repo'}]
        >>> registrations = validate_and_convert_marketplaces(data, "my-org")
        >>> len(registrations)
        1
    """
    if not raw_marketplaces:
        return []

    validated = []
    for i, mp in enumerate(raw_marketplaces):
        try:
            if isinstance(mp, dict):
                validated.append(MarketplaceRegistration.model_validate(mp))
            elif isinstance(mp, MarketplaceRegistration):
                validated.append(mp)
            else:
                raise ValueError(
                    f'Expected dict or MarketplaceRegistration, got {type(mp).__name__}'
                )
        except (ValidationError, ValueError) as e:
            logger.warning(
                f'Skipping invalid marketplace at index {i} in {source_name}: {e}'
            )
            continue

    return validated


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


_MISSING_SECRET = object()
_MCP_SECRET_FIELDS = ('headers', 'env', 'auth')


def _is_redacted_mcp_secret(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value == REDACTED_SECRET_VALUE or bool(
        re.fullmatch(
            rf'Bearer\s+{re.escape(REDACTED_SECRET_VALUE)}',
            value,
            flags=re.IGNORECASE,
        )
    )


def _has_redacted_mcp_secret(value: object) -> bool:
    if _is_redacted_mcp_secret(value):
        return True
    if isinstance(value, Mapping):
        return any(_has_redacted_mcp_secret(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_redacted_mcp_secret(item) for item in value)
    return False


def _restore_redacted_secret(value: Any, existing: Any) -> Any:
    if _is_redacted_mcp_secret(value):
        if isinstance(existing, str) and not _is_redacted_mcp_secret(existing):
            return existing
        return _MISSING_SECRET
    if isinstance(value, dict):
        existing_dict = existing if isinstance(existing, Mapping) else {}
        restored = {}
        for key, item in value.items():
            restored_item = _restore_redacted_secret(item, existing_dict.get(key))
            if restored_item is not _MISSING_SECRET:
                restored[key] = restored_item
        return restored
    return value


def _is_authorization_header(key: object) -> bool:
    return isinstance(key, str) and key.lower() == 'authorization'


def _header_value(headers: object, key: object) -> object:
    if not isinstance(headers, Mapping):
        return None
    for existing_key, value in headers.items():
        if (
            isinstance(key, str)
            and isinstance(existing_key, str)
            and existing_key.lower() == key.lower()
        ):
            return value
    return None


def _restore_submitted_mcp_headers(
    incoming: object,
    existing_server: Mapping[str, Any],
) -> object:
    if not isinstance(incoming, Mapping):
        return incoming
    existing_headers = existing_server.get('headers')
    restored = (
        dict(existing_headers)
        if _has_redacted_mcp_secret(incoming) and isinstance(existing_headers, Mapping)
        else {}
    )
    for key, value in incoming.items():
        existing_value = _header_value(existing_headers, key)
        for existing_key in tuple(restored):
            if (
                isinstance(existing_key, str)
                and isinstance(key, str)
                and existing_key.lower() == key.lower()
            ):
                restored.pop(existing_key)
        if (
            _is_authorization_header(key)
            and _is_redacted_mcp_secret(value)
            and existing_value is None
        ):
            continue
        restored_value = _restore_redacted_secret(value, existing_value)
        if restored_value is not _MISSING_SECRET:
            restored[key] = restored_value
    return restored


def _merge_redacted_mcp_auth(
    value: Mapping[str, Any],
    existing: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a submitted ``auth`` credential onto the stored one.

    The ``GET /settings`` round-trip strips secret sub-fields: the redaction
    marker validates back to ``None`` (``validate_secret``) and is then dropped
    from the response, so an unchanged credential reaches ``store`` as e.g.
    ``{'strategy': 'bearer'}`` with no ``value`` at all — not as the
    ``"**********"`` sentinel the plain restore logic looks for. Keep every
    field the client actually sent (restoring any surviving redaction marker),
    then carry over the stored sub-fields the client omitted. Only the secrets
    are ever stripped, so an omitted key is always a stored secret to recover.

    Recurses so nested secret carriers (``header`` strategy ``headers``,
    ``oauth2`` ``authentication``/``state``) are restored the same way. Gated by
    an equal ``strategy`` in the caller so a genuine credential-type switch is
    honored as submitted rather than merged.
    """
    merged: dict[str, Any] = {}
    for key, item in value.items():
        existing_item = existing.get(key)
        if isinstance(item, Mapping):
            merged[key] = _merge_redacted_mcp_auth(
                item, existing_item if isinstance(existing_item, Mapping) else {}
            )
            continue
        restored = _restore_redacted_secret(item, existing_item)
        if restored is not _MISSING_SECRET:
            merged[key] = restored
    for key, item in existing.items():
        if key not in value:
            merged[key] = deepcopy(item)
    return merged


def _restore_submitted_mcp_auth(value: object, existing: object) -> object:
    # Same credential type: keep what the client sent, restore any surviving
    # redaction marker, and recover stored secret sub-fields the GET round-trip
    # stripped to absent (bearer ``value``, api_key ``value``, basic
    # ``password``, header ``headers``, oauth2 ``state``/``client_secret``).
    if (
        isinstance(value, Mapping)
        and isinstance(existing, Mapping)
        and value.get('strategy') == existing.get('strategy')
    ):
        return _merge_redacted_mcp_auth(value, existing)
    if _has_redacted_mcp_secret(value) and existing is not None:
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            if value.get('strategy') != existing.get('strategy'):
                return deepcopy(existing)
        elif not isinstance(value, str) or not isinstance(existing, str):
            return deepcopy(existing)
    return _restore_redacted_secret(value, existing)


def _restore_submitted_mcp_secrets(
    incoming_server: dict[str, Any],
    existing_server: Mapping[str, Any],
    submitted_fields: tuple[str, ...],
) -> None:
    for field in submitted_fields:
        existing_value = existing_server.get(field)
        if field == 'headers':
            restored = _restore_submitted_mcp_headers(
                incoming_server[field],
                existing_server,
            )
        elif field == 'auth':
            restored = _restore_submitted_mcp_auth(
                incoming_server[field], existing_value
            )
        else:
            restored = _restore_redacted_secret(
                incoming_server[field],
                existing_value,
            )
        if restored is _MISSING_SECRET:
            incoming_server.pop(field)
        else:
            incoming_server[field] = restored


def _carry_omitted_mcp_secrets(
    incoming_server: dict[str, Any],
    existing_server: Mapping[str, Any],
    submitted_fields: tuple[str, ...],
) -> None:
    if 'env' not in submitted_fields and existing_server.get('env') is not None:
        incoming_server['env'] = existing_server['env']

    if 'headers' not in submitted_fields:
        existing_headers = existing_server.get('headers')
        if isinstance(existing_headers, Mapping):
            headers = dict(existing_headers)
            if 'auth' in submitted_fields:
                headers = {
                    key: value
                    for key, value in headers.items()
                    if not _is_authorization_header(key)
                }
            if headers:
                incoming_server['headers'] = headers

    if 'auth' not in submitted_fields and existing_server.get('auth') is not None:
        submitted_headers = incoming_server.get('headers')
        has_plain_authorization = isinstance(submitted_headers, Mapping) and any(
            _is_authorization_header(key) and not _is_redacted_mcp_secret(value)
            for key, value in submitted_headers.items()
        )
        if not has_plain_authorization:
            incoming_server['auth'] = existing_server['auth']


def _mcp_server_map(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    servers = value.get('mcpServers', value)
    return servers if isinstance(servers, dict) else None


def _mcp_endpoint_identity(server: Mapping[str, Any]) -> tuple[Any, ...] | None:
    url = server.get('url')
    if isinstance(url, str) and url:
        return ('url', url)
    command = server.get('command')
    if not isinstance(command, str) or not command:
        return None
    args = server.get('args')
    # Bind environment secrets to the full process invocation.
    return (
        'stdio',
        command,
        tuple(args) if isinstance(args, list) else (),
        server.get('cwd'),
    )


def _matching_existing_mcp_server(
    name: str,
    incoming_server: Mapping[str, Any],
    incoming_servers: Mapping[str, Any],
    existing_servers: Mapping[str, Any],
) -> Mapping[str, Any]:
    incoming_endpoint = _mcp_endpoint_identity(incoming_server)
    if incoming_endpoint is None:
        return {}

    named_server = existing_servers.get(name)
    if isinstance(named_server, Mapping):
        return (
            named_server
            if incoming_endpoint == _mcp_endpoint_identity(named_server)
            else {}
        )

    candidates = [
        server
        for existing_name, server in existing_servers.items()
        if existing_name not in incoming_servers
        and isinstance(server, Mapping)
        and incoming_endpoint == _mcp_endpoint_identity(server)
    ]
    competing_updates = sum(
        1
        for incoming_name, server in incoming_servers.items()
        if incoming_name not in existing_servers
        and isinstance(server, Mapping)
        and incoming_endpoint == _mcp_endpoint_identity(server)
    )
    return candidates[0] if len(candidates) == competing_updates == 1 else {}


def _preserve_redacted_mcp_secrets(
    value: Any,
    existing: Mapping[str, MCPServer] | None,
) -> Any:
    incoming_value = deepcopy(value)
    incoming_servers = _mcp_server_map(incoming_value)
    if incoming_servers is None:
        return incoming_value

    existing = existing or {}
    existing_dump = dump_mcp_config(
        existing,
        context={'expose_secrets': 'plaintext'},
    )
    for name, incoming_server in incoming_servers.items():
        if not isinstance(name, str) or not isinstance(incoming_server, dict):
            continue
        existing_server = _matching_existing_mcp_server(
            name,
            incoming_server,
            incoming_servers,
            existing_dump,
        )

        submitted_fields = tuple(
            field for field in _MCP_SECRET_FIELDS if field in incoming_server
        )
        _restore_submitted_mcp_secrets(
            incoming_server,
            existing_server,
            submitted_fields,
        )
        _carry_omitted_mcp_secrets(
            incoming_server,
            existing_server,
            submitted_fields,
        )

    return incoming_value


def _load_persisted_agent_settings(
    data: Any,
) -> OpenHandsAgentSettings | ACPAgentSettings:
    """Load persisted agent settings via the SDK loader.

    Routes the raw payload through :func:`validate_agent_settings`, which
    applies registered schema migrations, canonicalizes the legacy
    ``agent_kind: 'llm'`` tag to ``'openhands'``, and validates against the
    discriminated :data:`AgentSettingsConfig` union.
    """
    return validate_agent_settings(data or {})


def _load_persisted_conversation_settings(data: Any) -> ConversationSettings:
    """Load persisted conversation settings via the SDK loader."""
    return ConversationSettings.from_persisted(data or {})


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


def grouped_workspace_dir(
    base_working_dir: str,
    grouping_strategy: SandboxGroupingStrategy,
    conversation_id_hex: str,
) -> str:
    """Workspace dir for a conversation given the grouping strategy.

    Single source of truth for the relocation used at conversation start and at
    archive time. Under any grouping strategy the workspace is nested under the
    conversation id so co-located conversations stay isolated; NO_GROUPING keeps
    the bare base dir.
    """
    if grouping_strategy == SandboxGroupingStrategy.NO_GROUPING:
        return base_working_dir
    return f'{base_working_dir}/{conversation_id_hex}'


# Fields the batch ``update()`` method refuses to touch:
# - ``secrets_store`` is frozen (Pydantic would raise).
# - ``llm_profiles`` is off-limits for the generic settings POST; profile
#   mutations go through ``/api/v1/settings/profiles/...`` which validate
#   inputs, enforce the count cap, and take the per-user lock. Accepting a
#   raw dict here both bypassed those guards and crashed downstream
#   serialisation.
# - ``active_agent_profile_id`` / ``active_agent_profile_revision`` are launch
#   provenance, not persisted settings — ``store()`` refuses instances that
#   carry them. The pointer is set via the agent-profiles activate endpoint, so
#   a client echoing a non-null value back on a settings PUT must be ignored,
#   not written (which would 500 in ``store()``).
_SETTINGS_UPDATE_IGNORED_FIELDS = frozenset(
    [
        'secrets_store',
        'llm_profiles',
        'active_agent_profile_id',
        'active_agent_profile_revision',
    ]
)


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
    title_llm_profile: str | None = None
    git_full_clone: bool = False
    v1_enabled: bool = True
    agent_settings: AgentSettingsConfig = Field(default_factory=default_agent_settings)
    conversation_settings: ConversationSettings = Field(
        default_factory=ConversationSettings
    )
    sandbox_grouping_strategy: SandboxGroupingStrategy = (
        SandboxGroupingStrategy.NO_GROUPING
    )
    default_sandbox_spec_id: str | None = None
    llm_profiles: LLMProfiles = Field(
        default_factory=LLMProfiles,
        description=(
            'Saved LLM profiles and the currently active profile name. '
            'See ``LLMProfiles`` for the profile-management API.'
        ),
    )
    # Active Agent Profile provenance (cloud SaaS), surfaced by
    # ``SaasSettingsStore.load`` only when the caller requested profile
    # resolution (``resolve_agent_profile=True`` / an explicit override).
    # ``active_agent_profile_id`` mirrors the SDK persisted-settings pointer;
    # the revision rides alongside so conversation-start can stamp
    # ``LaunchedAgentProfile``. Neither is a persisted setting — they default to
    # None and ``store()`` refuses instances that carry them (no backing column).
    active_agent_profile_id: str | None = None
    active_agent_profile_revision: int | None = None

    # True when this instance came from a resolve-requested load: its
    # agent_settings are the *effective* launch view (possibly an Agent
    # Profile's resolved dump), not user-authored persisted settings, so it
    # must never be passed back to ``store()``. Survives ``model_copy`` but
    # not dump/reconstruct — ``store()`` also guards on
    # ``active_agent_profile_id`` for reconstructed copies.
    _resolved_view: bool = PrivateAttr(default=False)
    _mcp_config_updated: bool = PrivateAttr(default=False)

    # Marketplace registrations for plugin resolution
    # Users can register multiple marketplaces with different auto-load behaviors
    registered_marketplaces: list[MarketplaceRegistration] = Field(
        default_factory=list,
        description=(
            'List of marketplace registrations for plugin resolution. '
            'Marketplaces with auto_load=True will have their plugins loaded '
            'automatically at conversation start. '
            'See MarketplaceRegistration for details.'
        ),
    )
    # Inherited marketplaces from instance/org level (read-only for user)
    # This is computed at runtime from environment variables and org settings
    inherited_marketplaces: list[MarketplaceRegistration] = Field(
        default_factory=list,
        description=(
            'Marketplaces inherited from instance or organization level. '
            'These are read-only and cannot be modified by the user. '
            'Computed at runtime: Instance defaults + Org defaults.'
        ),
    )

    model_config = ConfigDict(populate_by_name=True)

    # NOTE: marketplace name uniqueness is enforced on the write paths (personal
    # settings + org store) and deduplicated defensively during composition
    # (``marketplace_composition``). It is intentionally NOT a model validator:
    # validating on construction would run on every ``load()`` and could lock a
    # user out of settings entirely if legacy stored data contained a duplicate.

    def __init__(self, **data: Any):
        raw_agent_settings = data.get('agent_settings')
        mcp_config_updated = (
            'mcp_config' in raw_agent_settings
            if isinstance(raw_agent_settings, Mapping)
            else 'mcp_config'
            in getattr(raw_agent_settings, 'model_fields_set', frozenset())
        )
        # Import Secrets here to avoid circular imports
        from openhands.app_server.secrets.secrets_models import Secrets

        if 'secrets_store' not in data or data['secrets_store'] is None:
            data['secrets_store'] = Secrets()
        super().__init__(**data)
        self._mcp_config_updated = mcp_config_updated

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

            # ``mcp_config`` replaces wholesale rather than deep-merging, so
            # hold it back from the variant-aware merge and apply it after.
            replace_mcp_config = 'mcp_config' in agent_update
            mcp_config = (
                _preserve_redacted_mcp_secrets(
                    coerced.pop('mcp_config', None),
                    self.agent_settings.mcp_config,
                )
                if replace_mcp_config
                else None
            )

            # The SDK owns the discriminated-union merge: replace on
            # ``agent_kind`` change, deep-merge within a variant. Cross-kind
            # config preservation tracked in OpenHands/OpenHands#14370.
            new_settings = apply_agent_settings_diff(self.agent_settings, coerced)
            if replace_mcp_config:
                dumped = new_settings.model_dump(
                    mode='json', context={'expose_secrets': True}
                )
                dumped['mcp_config'] = mcp_config
                new_settings = validate_agent_settings(dumped)

            # Use object.__setattr__ to avoid validate_assignment
            # side-effects on other fields.
            object.__setattr__(self, 'agent_settings', new_settings)
            if replace_mcp_config:
                self._mcp_config_updated = True

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
                # Validate registered_marketplaces before setting
                if key == 'registered_marketplaces' and value is not None:
                    validated = []
                    for i, mp in enumerate(value):
                        try:
                            if isinstance(mp, dict):
                                # Strip scope from incoming request - backend will set it
                                mp_dict = {k: v for k, v in mp.items() if k != 'scope'}
                                # Ensure auto_load defaults to False if not provided
                                if 'auto_load' not in mp_dict:
                                    mp_dict['auto_load'] = False
                                mp_obj = MarketplaceRegistration.model_validate(mp_dict)
                                # Set scope='personal' for user-level settings
                                mp_obj.scope = MarketplaceScope.PERSONAL
                                validated.append(mp_obj)
                            elif isinstance(mp, MarketplaceRegistration):
                                # Set scope='personal' for user-level settings
                                mp.scope = MarketplaceScope.PERSONAL
                                validated.append(mp)
                            else:
                                raise ValueError(
                                    f'Expected dict or MarketplaceRegistration, '
                                    f'got {type(mp).__name__}'
                                )
                        except ValidationError as e:
                            raise ValueError(
                                f'Invalid marketplace at index {i}: {e.errors()[0]["msg"]}'
                            ) from e
                    value = validated
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
        agent_settings: OpenHandsAgentSettings | ACPAgentSettings,
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
        # Copy the LLM so post-activation fixups (e.g. resolving ``base_url``
        # against the provider default) don't bleed back into the saved
        # profile. ``model_copy(update={'llm': llm})`` is shallow, so the
        # update value is shared with ``llm_profiles.profiles[name]``.
        llm = self.llm_profiles.require(name)
        self.agent_settings = self.agent_settings.model_copy(
            update={'llm': llm.model_copy()}
        )
        self.llm_profiles.active = name

    def delete_profile(self, name: str) -> bool:
        """Delete a saved profile, promoting a fallback when it was active.

        Returns False if the profile didn't exist; True otherwise. When the
        deleted profile was active and other profiles remain, switches to
        the first remaining one (insertion order — same ordering ``rename``
        relies on) so the user isn't left without an active LLM.
        """
        was_active = self.llm_profiles.active == name
        if not self.llm_profiles.delete(name):
            return False
        if self.title_llm_profile == name:
            self.title_llm_profile = None
        if was_active and self.llm_profiles.profiles:
            fallback = next(iter(self.llm_profiles.profiles))
            self.switch_to_profile(fallback)
        return True

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
            data['agent_settings'] = _load_persisted_agent_settings(
                _coerce_dict_secrets(agent_settings)
            ).model_dump(mode='json', context={'expose_secrets': True})
        elif isinstance(agent_settings, (OpenHandsAgentSettings, ACPAgentSettings)):
            data['agent_settings'] = agent_settings.model_dump(
                mode='json', context={'expose_secrets': True}
            )

        # --- Conversation settings: normalize ---
        conversation_settings = data.get('conversation_settings')
        if isinstance(conversation_settings, dict):
            data['conversation_settings'] = _load_persisted_conversation_settings(
                conversation_settings
            ).model_dump(mode='json')
        elif isinstance(conversation_settings, ConversationSettings):
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

    def to_agent_settings(self) -> OpenHandsAgentSettings | ACPAgentSettings:
        return self.agent_settings

    def get_agent_settings_display(self) -> dict[str, Any]:
        """Return agent_settings with display-only defaults removed."""
        from openhands.app_server.settings.settings_router import LITE_LLM_API_URL
        from openhands.app_server.utils.llm import is_openhands_model

        data = self.agent_settings.model_dump(mode='json')
        llm = data.get('llm')
        if isinstance(llm, dict):
            model = llm.get('model')
            base_url = llm.get('base_url')
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
