from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.constants import LITE_LLM_API_URL
from server.logger import logger
from server.routes.org_models import (
    MEMBER_PRIVATE_AGENT_KEYS,
    OrgMemberSettingsUpdate,
)
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from storage.agent_profile_resolution import (
    OrgLLMProfileLoader,
    load_agent_profiles,
    load_llm_profiles,
)
from storage.database import a_session_maker
from storage.lite_llm_manager import LiteLlmManager, get_openhands_cloud_key_alias
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore, serialize_mcp_config
from storage.org_store import OrgStore
from storage.user import User
from storage.user_settings import UserSettings
from storage.user_store import UserStore

from openhands.app_server.settings.llm_profiles import LLMProfiles, resolve_profile_llm
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.utils.jsonpatch_compat import (
    deep_merge,
    deep_merge_with_wholesale_keys,
)
from openhands.app_server.utils.llm import is_openhands_model
from openhands.sdk.llm.utils.openhands_provider import (
    canonicalize_openhands_llm_payload,
)
from openhands.sdk.mcp.config import MCPServer, coerce_mcp_config
from openhands.sdk.profiles import resolve_agent_profile


class ManagedLlmKeyStatus:
    """Outcomes for ``SaasSettingsStore.rotate_managed_llm_key``.

    Kept as plain string constants (not an Enum) so callers can compare
    against the literals without importing a type.
    """

    ROTATED = 'rotated'
    NOT_MANAGED = 'not_managed'
    BYOK = 'byok'
    MISSING_MEMBER = 'missing_member'


@dataclass(frozen=True)
class ManagedLlmKeyConfig:
    """Whether an effective LLM config uses a managed LiteLLM/OpenHands key."""

    openhands_type: bool


@dataclass(frozen=True)
class ManagedLlmKeyRotation:
    """Result of a managed-key rotation attempt."""

    status: str
    old_key: str | None = None
    new_key: str | None = None
    openhands_type: bool = False


def managed_llm_key_config_from_model(
    llm_model: str | None, llm_base_url: str | None
) -> ManagedLlmKeyConfig | None:
    """Classify an effective LLM config as managed.

    Uses the same model/base_url logic as ``store()`` and
    ``OrgStore._maybe_get_managed_llm_key_for_user``.

    Returns ``None`` when the config is not a managed LiteLLM/OpenHands-provider
    configuration (e.g. member/org BYOK pointing at a third-party base_url).
    """
    openhands_type = is_openhands_model(llm_model)
    normalized_llm_base_url = llm_base_url.rstrip('/') if llm_base_url else None
    normalized_managed_base_url = (
        LITE_LLM_API_URL.rstrip('/') if LITE_LLM_API_URL else None
    )
    uses_openhands_provider_proxy = openhands_type and (
        normalized_llm_base_url is None
        or 'all-hands.dev' in normalized_llm_base_url.lower()
    )
    uses_managed_llm_key = (
        normalized_managed_base_url is not None
        and normalized_llm_base_url == normalized_managed_base_url
    ) or uses_openhands_provider_proxy
    if not uses_managed_llm_key:
        return None
    return ManagedLlmKeyConfig(openhands_type=openhands_type)


@dataclass
class SaasSettingsStore(SettingsStore):
    user_id: str
    # When set, overrides the user's `current_org_id` for both `load()` and
    # `store()`. Used to honor a request's effective org (api_key_org_id >
    # X-Org-Id header > user.current_org_id) so an API key minted for org A
    # used by a user whose `current_org_id` later switched to B still
    # reads/writes settings under org A.
    effective_org_id: UUID | None = None

    def _resolve_org_id(self, user: User) -> UUID:
        """Return the effective org id for this request.

        Falls back to the user's current org id. The caller still needs to verify
        that the user is a member of the returned org (handled in load/store by
        the existing org_members lookup).

        `user.current_org_id` is non-nullable on the ORM model, so the result is
        always a UUID.
        """
        return self.effective_org_id or user.current_org_id

    async def _get_user_settings_by_keycloak_id_async(
        self, keycloak_user_id: str, session=None
    ) -> UserSettings | None:
        """Get UserSettings by keycloak_user_id (async version).

        Args:
            keycloak_user_id: The keycloak user ID to search for
            session: Optional existing async database session. If not provided, creates a new one.

        Returns:
            UserSettings object if found, None otherwise
        """
        if not keycloak_user_id:
            return None

        if session:
            # Use provided session
            result = await session.execute(
                select(UserSettings).filter(
                    UserSettings.keycloak_user_id == keycloak_user_id
                )
            )
            return result.scalars().first()
        else:
            # Create new session
            async with a_session_maker() as new_session:
                result = await new_session.execute(
                    select(UserSettings).filter(
                        UserSettings.keycloak_user_id == keycloak_user_id
                    )
                )
                return result.scalars().first()

    @staticmethod
    def _get_effective_llm_api_key(
        org: Org,
        org_member: OrgMember,
    ) -> SecretStr | None:
        if org_member.has_custom_llm_api_key:
            return org_member.llm_api_key
        if org.llm_api_key:
            return org.llm_api_key
        # Managed keys are stored on the member row (has_custom=False); fall back to
        # it, but only decrypt when actually set to avoid the empty-value error (#14898).
        if org_member._llm_api_key:
            return org_member.llm_api_key
        return None

    @staticmethod
    def _get_persisted_agent_settings(item: Settings) -> dict[str, Any]:
        persisted = item.agent_settings.model_dump(
            mode='json',
            exclude={'llm': {'api_key'}},
        )
        persisted.pop('mcp_config', None)
        return persisted

    @staticmethod
    def _get_persisted_mcp_config(item: Settings) -> dict[str, Any] | None:
        return serialize_mcp_config(item.agent_settings.mcp_config)

    def _resolve_active_agent_profile(
        self,
        org: Org,
        org_member: OrgMember,
        merged_agent_settings: dict[str, Any],
        effective_llm_api_key: SecretStr | None,
        override_agent_profile_id: str | None = None,
    ) -> tuple[dict[str, Any], str, int] | None:
        """Resolve an agent profile into an ``agent_settings`` dump.

        Resolves ``override_agent_profile_id`` when given (a one-off,
        non-persisted launch override — never written back to
        ``org_member.active_agent_profile_id``), else the member's ambient
        ``active_agent_profile_id``, else the org-wide default pointer
        (``AgentProfiles.active``, if one has been set). Returns
        ``(agent_settings_dump, profile_id, revision)``, or ``None`` to fall
        back to the composed ``agent_settings``.
        Delegates the ``llm_profile_ref`` + ``mcp_server_refs`` join entirely to
        the SDK ``resolve_agent_profile``; only the cloud-specific glue (the
        org-backed ``llm_store`` adapter, the member-effective ``mcp_config``,
        and the managed-key / base-url overlay via ``resolve_profile_llm``)
        lives here.

        Fail-safe by design: a stale pointer (profile deleted) or any resolution
        error returns ``None`` and logs, because bricking *every* settings load
        on a dangling pointer is far worse than launching the composed default.
        """
        agent_profiles = load_agent_profiles(org)
        active_id = (
            override_agent_profile_id
            or org_member.active_agent_profile_id
            or agent_profiles.active
        )
        if not active_id:
            return None

        name = agent_profiles.name_for_id(active_id)
        if name is None:
            logger.warning(
                'Active agent profile %s not found for user %s in org %s; '
                'falling back to composed settings',
                active_id,
                self.user_id,
                org.id,
            )
            return None
        try:
            profile = agent_profiles.load(name)
        except FileNotFoundError:
            return None

        mcp_config: dict[str, MCPServer] = {}
        mcp_raw = merged_agent_settings.get('mcp_config')
        if mcp_raw:
            try:
                mcp_config = coerce_mcp_config(mcp_raw)
            except Exception:
                mcp_config = {}

        llm_store = OrgLLMProfileLoader(load_llm_profiles(org))
        try:
            resolved = resolve_agent_profile(
                profile,
                llm_store=llm_store,
                mcp_config=mcp_config,
                available_skills=None,
                cipher=None,
            )

            # Apply the cloud managed-key / base-url overlay to the resolved LLM
            # (OpenHands kind only), so managed OpenHands keys and provider-default
            # base URLs behave exactly as the non-profile path. Reuses the existing
            # resolve_profile_llm helper rather than re-deriving key resolution.
            if resolved.agent_kind == 'openhands':
                resolved = resolved.model_copy(
                    update={
                        'llm': resolve_profile_llm(
                            resolved.llm,
                            managed_proxy_url=LITE_LLM_API_URL,
                            fallback_api_key=effective_llm_api_key,
                        )
                    }
                )

            # expose_secrets so the resolved LLM key lands in agent_settings the
            # same way the composed path sets
            # merged_agent_settings['llm']['api_key'].
            resolved_dump = resolved.model_dump(
                mode='json', context={'expose_secrets': True}
            )
            # Canonicalize legacy managed OpenHands model names/base_urls on the
            # resolved LLM, mirroring the composed path (merged_agent_settings
            # ['llm'], line ~348) so a profile launch and a non-profile launch
            # normalize an org's pre-canonical llm_profiles identically.
            resolved_llm = resolved_dump.get('llm')
            if isinstance(resolved_llm, dict):
                resolved_dump['llm'] = canonicalize_openhands_llm_payload(resolved_llm)
        except Exception as exc:
            # Never-brick contract: catch broadly, not just the known resolver
            # errors — SDK contract drift (e.g. a new required kwarg raising
            # TypeError) must degrade to the composed settings, not 500 every
            # settings load.
            logger.warning(
                'Failed to resolve active agent profile %r for user %s: %s; '
                'falling back to composed settings',
                name,
                self.user_id,
                exc,
            )
            return None
        return resolved_dump, str(profile.id), profile.revision

    async def load(
        self,
        *,
        resolve_agent_profile: bool = False,
        override_agent_profile_id: str | None = None,
    ) -> Settings | None:
        """Load settings; opt into the resolved Agent-Profile launch view.

        The default returns the PERSISTED (composed org + member) settings —
        the view every load() -> store() round-trip must operate on, so an
        active Agent Profile's *resolved* dump (ref-filtered ``mcp_config``,
        the referenced LLM profile's key) can never masquerade as
        user-authored settings and get written back.

        ``resolve_agent_profile=True`` is the conversation-start opt-in: the
        active Agent Profile resolves and REPLACES ``agent_settings`` (the
        sole launch authority, #15044 §3), and the result is marked
        ``_resolved_view`` — ``store()`` refuses it. Falls back to the
        composed settings when no pointer is set or the pointer is stale /
        unresolvable, so a broken pointer can never brick the load.

        ``override_agent_profile_id`` is a one-off launch override (e.g. a
        per-conversation-start request): it implies resolution, changes which
        profile resolves for *this call only*, and is never persisted to
        ``org_member.active_agent_profile_id``.
        """
        user = await UserStore.get_user_by_id(self.user_id)
        if not user:
            logger.error(f'User not found for ID {self.user_id}')
            return None

        org_id = self._resolve_org_id(user)
        org_member: OrgMember | None = None
        for om in user.org_members:
            if om.org_id == org_id:
                org_member = om
                break
        if not org_member:
            return None
        org = await OrgStore.get_org_by_id_async(org_id)
        if not org:
            logger.error(
                f'Org not found for ID {org_id} as the current org for user {self.user_id}'
            )
            return None
        org_agent_settings = OrgStore.get_agent_settings_from_org(org)
        member_agent_settings_diff = dict(org_member.agent_settings_diff)
        member_agent_settings_diff.pop('mcp_config', None)
        member_mcp_config = org_member.effective_mcp_config

        kwargs = {
            **{
                normalized: getattr(org, c.name)
                for c in Org.__table__.columns
                if (
                    normalized := c.name.removeprefix('_default_')
                    .removeprefix('default_')
                    .lstrip('_')
                )
                in Settings.model_fields
            },
            **{
                normalized: getattr(user, c.name)
                for c in User.__table__.columns
                if (normalized := c.name.lstrip('_')) in Settings.model_fields
            },
        }
        # Drop member-private keys from the org dump before merging so
        # legacy org-level values (older code paths broadcast mcp_config)
        # can no longer leak one member's private config to another. Each
        # member's own ``agent_settings_diff`` still supplies their values.
        org_agent_settings_dump = org_agent_settings.model_dump(mode='json')
        for private_key in MEMBER_PRIVATE_AGENT_KEYS:
            org_agent_settings_dump.pop(private_key, None)
        merged_agent_settings = deep_merge(
            org_agent_settings_dump,
            member_agent_settings_diff,
        )
        if member_mcp_config is not None:
            merged_agent_settings['mcp_config'] = member_mcp_config
        effective_llm_api_key = self._get_effective_llm_api_key(org, org_member)
        if effective_llm_api_key is not None:
            merged_agent_settings.setdefault('llm', {})['api_key'] = (
                effective_llm_api_key.get_secret_value()
                if isinstance(effective_llm_api_key, SecretStr)
                else effective_llm_api_key
            )
        else:
            logger.warning(
                f'No effective LLM API key found for user {self.user_id} '
                f'in org {org_id} (org key and member key are both unset)'
            )
        # Canonicalize legacy managed OpenHands LLM payloads before Settings
        # validation so current settings and seeded profiles use the public
        # openhands/ prefix.
        llm_dict = merged_agent_settings.get('llm')
        if isinstance(llm_dict, dict):
            merged_agent_settings['llm'] = canonicalize_openhands_llm_payload(llm_dict)

        kwargs['agent_settings'] = merged_agent_settings
        org_conversation = OrgStore.get_conversation_settings_from_org(org)
        member_conversation_diff = dict(org_member.conversation_settings_diff)
        kwargs['conversation_settings'] = deep_merge(
            org_conversation.model_dump(mode='json'),
            member_conversation_diff,
        )
        if org.v1_enabled is None:
            kwargs['v1_enabled'] = True
        # Apply default if sandbox_grouping_strategy is None in the database
        if kwargs.get('sandbox_grouping_strategy') is None:
            kwargs.pop('sandbox_grouping_strategy', None)
        # Apply default if git_full_clone is None in the database (pre-existing rows)
        if kwargs.get('git_full_clone') is None:
            kwargs.pop('git_full_clone', None)
        # Apply default if registered_marketplaces is None in the database
        if kwargs.get('registered_marketplaces') is None:
            kwargs.pop('registered_marketplaces', None)

        # Load personal registered_marketplaces from user_settings table
        user_settings = await self._get_user_settings_by_keycloak_id_async(self.user_id)
        if user_settings and user_settings.registered_marketplaces:
            # Normalize marketplaces: use 'personal' scope for legacy data without scope
            # The merge function will override with the correct scope value
            normalized_mps: list[dict[str, Any]] = []
            for mp in user_settings.registered_marketplaces:
                if isinstance(mp, dict):
                    if mp.get('scope') is None:
                        mp = {**mp, 'scope': 'personal'}
                    normalized_mps.append(mp)
                else:
                    # Convert MarketplaceRegistration to dict
                    normalized_mps.append(mp.model_dump())
            kwargs['registered_marketplaces'] = normalized_mps
        # Profiles in SaaS live on the org (managed via
        # /api/organizations/{org_id}/profiles). Surface them through
        # Settings.llm_profiles so the chat-layer endpoints
        # (/api/v1/settings/profiles and /switch_profile) see them without
        # needing a separate code path. Falls back to user.llm_profiles when
        # the org has none — handles older personal accounts whose profiles
        # never moved to the org column.
        if org.llm_profiles:
            profiles_data = dict(org.llm_profiles)
            raw_profiles = profiles_data.get('profiles')
            if isinstance(raw_profiles, dict):
                profiles_data['profiles'] = {
                    name: canonicalize_openhands_llm_payload(prof)
                    if isinstance(prof, dict)
                    else prof
                    for name, prof in raw_profiles.items()
                }
            kwargs['llm_profiles'] = profiles_data
        # When no profiles exist yet, seed a Default profile from the legacy
        # LLM config so users (and orgs) upgrading from pre-llm_profiles
        # settings keep their previous LLM as the active profile instead of
        # landing on an empty profiles UI (mirrors the OSS FileSettingsStore).
        # Covers both pre-migration rows (llm_profiles is None) and
        # already-migrated orgs whose profiles map is empty.
        seeded_default = False
        if not (kwargs.get('llm_profiles') or {}).get('profiles'):
            legacy_llm = merged_agent_settings.get('llm')
            if isinstance(legacy_llm, dict) and legacy_llm.get('model'):
                kwargs['llm_profiles'] = {
                    'profiles': {'Default': dict(legacy_llm)},
                    'active': 'Default',
                }
                seeded_default = True
            else:
                # No legacy LLM to seed; drop a None value so the non-nullable
                # Settings.llm_profiles falls back to its default_factory.
                kwargs.pop('llm_profiles', None)

        # Agent Profiles: only on a resolve-requested load (conversation
        # start), resolve the active agent profile and let the result REPLACE
        # the composed agent_settings — the active Agent Profile is the sole
        # launch authority (#15044 §3). Plain loads keep the persisted
        # composed settings so settings round-trips never see (and never
        # write back) a profile's resolved dump.
        resolution_requested = (
            resolve_agent_profile or override_agent_profile_id is not None
        )
        if resolution_requested:
            resolved = self._resolve_active_agent_profile(
                org,
                org_member,
                merged_agent_settings,
                effective_llm_api_key,
                override_agent_profile_id,
            )
            if resolved is not None:
                resolved_dump, resolved_id, resolved_revision = resolved
                kwargs['agent_settings'] = resolved_dump
                kwargs['active_agent_profile_id'] = resolved_id
                kwargs['active_agent_profile_revision'] = resolved_revision

        settings = Settings(**kwargs)
        settings._mcp_config_updated = False
        if resolution_requested:
            # Launch view (even when resolution fell back): never persistable.
            settings._resolved_view = True

        # The seed above is in-memory only. Persist it onto the org row so the
        # legacy LLM becomes a real stored profile — otherwise the profiles
        # management API (server/routes/org_profiles.py, which reads
        # org.llm_profiles directly) would still see an empty list and the
        # user's previous model would never land "inside the profiles".
        # Persist is best-effort: a transient DB failure here must not block
        # returning settings the caller already has in memory.
        if seeded_default:
            try:
                await self._persist_seeded_default_profile(
                    org_id, settings.llm_profiles
                )
            except Exception:
                logger.warning(
                    'Failed to persist seeded Default profile for org %s',
                    org_id,
                    exc_info=True,
                )

        return settings

    async def _persist_seeded_default_profile(
        self, org_id: UUID, llm_profiles: LLMProfiles
    ) -> None:
        """Backfill the seeded ``Default`` profile onto ``org.llm_profiles``.

        Runs once per org during the pre-llm_profiles → llm_profiles upgrade:
        ``load()`` seeds the profile in memory, and this writes it back so it
        becomes a real stored profile that the org-profiles management API can
        list and mutate. Re-checks emptiness under a row lock so a concurrent
        ``load()`` doesn't double-seed and so a profile a member just created
        through the management API is never clobbered.
        """
        serialized = llm_profiles.model_dump(
            mode='json', context={'expose_secrets': True}
        )
        async with a_session_maker() as session:
            result = await session.execute(
                select(Org).filter(Org.id == org_id).with_for_update()
            )
            org = result.scalars().first()
            if org is None:
                return
            # Only seed while the column is still empty — another request may
            # have populated it between this load() and acquiring the lock.
            if (org.llm_profiles or {}).get('profiles'):
                return
            org.llm_profiles = serialized
            await session.commit()

    async def store(self, item: Settings):
        if item is not None and (
            item._resolved_view or item.active_agent_profile_id is not None
        ):
            # A resolved Agent-Profile launch view (from
            # load(resolve_agent_profile=True) / an override) is the profile's
            # dump, not user-authored settings: persisting it would bake the
            # ref-filtered mcp_config and the referenced LLM profile's key into
            # the member/org rows. Only plain load() results may round-trip.
            raise ValueError(
                'Refusing to persist a resolved Agent-Profile settings view; '
                'store() only accepts persisted settings from a plain load()'
            )
        async with a_session_maker() as session:
            if not item:
                return None
            result = await session.execute(
                select(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(self.user_id))
            )
            user = result.scalars().first()

            if not user:
                # Check if we need to migrate from user_settings
                user_settings = None
                async with a_session_maker() as new_session:
                    user_settings = await self._get_user_settings_by_keycloak_id_async(
                        self.user_id, new_session
                    )
                if user_settings:
                    token_manager = TokenManager()
                    user_info = await token_manager.get_user_info_from_user_id(
                        self.user_id
                    )
                    if not user_info:
                        logger.error(f'User info not found for ID {self.user_id}')
                        return None
                    user = await UserStore.migrate_user(
                        self.user_id, user_settings, user_info
                    )
                    if not user:
                        logger.error(f'Failed to migrate user {self.user_id}')
                        return None
                else:
                    logger.error(f'User not found for ID {self.user_id}')
                    return None

            org_id = self._resolve_org_id(user)

            org_member: OrgMember | None = None
            for om in user.org_members:
                if om.org_id == org_id:
                    org_member = om
                    break
            if not org_member:
                return None

            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
            if not org:
                logger.error(
                    f'Org not found for ID {org_id} as the current org for user {self.user_id}'
                )
                return None

            llm_model = item.agent_settings.llm.model
            llm_base_url = item.agent_settings.llm.base_url
            normalized_llm_base_url = llm_base_url.rstrip('/') if llm_base_url else None
            normalized_managed_base_url = LITE_LLM_API_URL.rstrip('/')
            uses_managed_llm_key = (
                normalized_llm_base_url == normalized_managed_base_url
                or (normalized_llm_base_url is None and is_openhands_model(llm_model))
            )
            logger.info(
                'saas_settings_store:store:managed_llm_config_decision',
                extra={
                    'user_id': self.user_id,
                    'org_id': str(org_id),
                    'model': llm_model,
                    'base_url': llm_base_url or '',
                    'uses_managed_llm_key': uses_managed_llm_key,
                },
            )

            if uses_managed_llm_key:
                await self._ensure_api_key(
                    item, str(org_id), openhands_type=is_openhands_model(llm_model)
                )

            effective_agent_settings_diff = self._get_persisted_agent_settings(item)
            shared_agent_settings_diff = {
                key: value
                for key, value in effective_agent_settings_diff.items()
                if key not in MEMBER_PRIVATE_AGENT_KEYS
            }

            # Strip any pre-existing private keys from the org dump before
            # merging, so legacy values written by older code paths are
            # cleaned up on the next save and stop leaking to other members.
            org_agent_settings_dump = OrgStore.get_agent_settings_from_org(
                org
            ).model_dump(mode='json')
            for private_key in MEMBER_PRIVATE_AGENT_KEYS:
                org_agent_settings_dump.pop(private_key, None)

            # Single assignment so SQLAlchemy tracks the change
            org.agent_settings = deep_merge_with_wholesale_keys(
                org_agent_settings_dump,
                shared_agent_settings_diff,
            )

            effective_conversation_diff = item.conversation_settings.model_dump(
                mode='json'
            )
            org.conversation_settings = deep_merge(
                OrgStore.get_conversation_settings_from_org(org).model_dump(
                    mode='json'
                ),
                effective_conversation_diff,
            )

            kwargs = item.model_dump(context={'expose_secrets': True})
            kwargs.pop('agent_settings', None)
            kwargs.pop('conversation_settings', None)

            # Get or create user_settings for this user
            user_settings_result = await session.execute(
                select(UserSettings).filter(
                    UserSettings.keycloak_user_id == self.user_id
                )
            )
            user_settings = user_settings_result.scalars().first()
            if not user_settings:
                user_settings = UserSettings(keycloak_user_id=self.user_id)
                session.add(user_settings)

            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
                if key == 'registered_marketplaces':
                    # Save personal marketplace settings to user_settings table
                    user_settings.registered_marketplaces = value
                elif hasattr(org, key) and key not in {
                    'llm_api_key',
                    'agent_settings',
                    'conversation_settings',
                }:
                    setattr(org, key, value)

            current_member_llm_api_key = item.agent_settings.llm.api_key
            org_default_llm_api_key = org.llm_api_key
            org_default_llm_api_key_raw = (
                org_default_llm_api_key.get_secret_value()
                if org_default_llm_api_key
                else None
            )
            current_member_llm_api_key_raw = (
                current_member_llm_api_key.get_secret_value()  # type: ignore[union-attr]
                if current_member_llm_api_key
                else None
            )

            await OrgMemberStore.update_all_members_settings_async(
                session,
                org_id,
                OrgMemberSettingsUpdate(
                    agent_settings_diff=shared_agent_settings_diff,
                    conversation_settings_diff=effective_conversation_diff,
                    llm_api_key=(
                        current_member_llm_api_key_raw  # type: ignore[arg-type]
                        if not uses_managed_llm_key
                        else None
                    ),
                ),
            )

            member_mcp_config = org_member.effective_mcp_config
            member_agent_settings_diff = dict(org_member.agent_settings_diff)
            for private_key in MEMBER_PRIVATE_AGENT_KEYS:
                member_agent_settings_diff.pop(private_key, None)
            org_member.agent_settings_diff = member_agent_settings_diff
            if item._mcp_config_updated:
                org_member.mcp_config = self._get_persisted_mcp_config(item)
            elif org_member.mcp_config is None and member_mcp_config is not None:
                org_member.mcp_config = serialize_mcp_config(member_mcp_config)

            if uses_managed_llm_key and current_member_llm_api_key is not None:
                # Managed/proxy key — store on this member but mark as org-managed
                org_member.llm_api_key = current_member_llm_api_key  # type: ignore[assignment]
                org_member.has_custom_llm_api_key = False
            elif current_member_llm_api_key_raw is not None:
                # BYOR: member supplied their own (non-managed) API key
                org_member.llm_api_key = current_member_llm_api_key  # type: ignore[assignment]
                org_member.has_custom_llm_api_key = True
            elif org_default_llm_api_key_raw is not None:
                # No member key, falling back to org default
                org_member.has_custom_llm_api_key = False

            await session.commit()

    @classmethod
    async def get_instance(  # type: ignore[override]
        cls,
        user_id: str,
        effective_org_id: UUID | None = None,
    ) -> SaasSettingsStore:
        """Get a SaasSettingsStore instance for the given user.

        Args:
            user_id: Keycloak user id.
            effective_org_id: Optional org id resolved from the request
                (see SaasUserAuth.get_effective_org_id). When None the
                store falls back to ``user.current_org_id`` to preserve
                legacy behavior for background / non-request callers.

        TODO: This method should be replaced with dependency injection.
        """
        logger.debug(f'saas_settings_store.get_instance::{user_id}')
        return SaasSettingsStore(user_id, effective_org_id=effective_org_id)

    async def get_org_marketplaces(self, user_id: str | None) -> list[dict]:
        """Get organization-level marketplaces from the org's registered_marketplaces.

        Uses the effective_org_id if set, otherwise resolves via user.current_org_id.
        Returns empty list if no org or org has no registered marketplaces.
        """
        if not user_id:
            return []

        try:
            user = await UserStore.get_user_by_id(user_id)
            if not user:
                logger.debug(f'No user found for ID {user_id}')
                return []

            org_id = self.effective_org_id or user.current_org_id
            if not org_id:
                logger.debug(f'No org_id for user {user_id}')
                return []

            # Validate org_id is a valid UUID before calling get_org_by_id_async
            try:
                from uuid import UUID

                UUID(str(org_id))
            except (ValueError, AttributeError, TypeError) as uuid_error:
                logger.warning(
                    f'Invalid org_id format for user {user_id}: {org_id} - {uuid_error}'
                )
                return []

            org = await OrgStore.get_org_by_id_async(org_id)
            if not org:
                logger.debug(f'Org {org_id} not found for user {user_id}')
                return []

            if org.registered_marketplaces:
                # Normalize: use 'org' scope for legacy data without scope
                normalized: list[dict[str, Any]] = []
                for mp in org.registered_marketplaces:
                    if isinstance(mp, dict):
                        # Set scope='org' if missing (backward compatibility)
                        if mp.get('scope') is None:
                            mp = {**mp, 'scope': 'org'}
                        # Ensure auto_load defaults to False if missing
                        if 'auto_load' not in mp:
                            mp = {**mp, 'auto_load': False}
                        normalized.append(mp)
                    else:
                        # Convert MarketplaceRegistration to dict
                        normalized.append(mp.model_dump())
                return normalized
            return []
        except Exception:
            logger.exception('Error fetching org marketplaces', stack_info=True)
            return []

    async def _ensure_api_key(
        self, item: Settings, org_id: str, openhands_type: bool = False
    ) -> None:
        """Generate and set the OpenHands API key for the given settings.

        First checks if an existing key exists for the user and verifies it
        is valid in LiteLLM. If valid, reuses it. Otherwise, generates a new key.
        """
        llm_api_key = item.agent_settings.llm.api_key
        logger.info(
            'saas_settings_store:ensure_api_key:evaluate',
            extra={
                'user_id': self.user_id,
                'org_id': org_id,
                'openhands_type': openhands_type,
                'has_api_key': bool(llm_api_key),
            },
        )

        if not llm_api_key:
            logger.info(
                'saas_settings_store:ensure_api_key:skip_missing_api_key',
                extra={'user_id': self.user_id, 'org_id': org_id},
            )
            return

        existing_key_valid = await LiteLlmManager.verify_existing_key(
            llm_api_key.get_secret_value(),  # type: ignore[union-attr]
            self.user_id,
            org_id,
            openhands_type=openhands_type,
        )
        logger.info(
            'saas_settings_store:ensure_api_key:verify_existing_key',
            extra={
                'user_id': self.user_id,
                'org_id': org_id,
                'openhands_type': openhands_type,
                'existing_key_valid': existing_key_valid,
            },
        )

        # First, check if our current key is valid
        if not existing_key_valid:
            # Both branches mint one managed key per (user, org) under the same
            # deterministic alias, deleting any prior key first — so switching
            # the default to/from an openhands/* model never orphans a key.
            key_alias = get_openhands_cloud_key_alias(self.user_id, org_id)
            await LiteLlmManager.delete_key_by_alias(key_alias=key_alias)
            generated_key = await LiteLlmManager.generate_key(
                self.user_id,
                org_id,
                key_alias,
                {'type': 'openhands'} if openhands_type else None,
            )

            item.agent_settings.llm.api_key = SecretStr(generated_key)
            logger.info(
                'saas_settings_store:store:generated_openhands_key',
                extra={
                    'user_id': self.user_id,
                    'org_id': org_id,
                    'openhands_type': openhands_type,
                },
            )

    async def get_current_managed_llm_key(self) -> str | None:
        """Return the acting member's current managed key, if effective config uses it.

        This is intentionally stricter than ``load()``'s effective-key resolution:
        org-level keys and member BYOK/custom keys are treated as non-managed for
        this helper because rotating the member managed key would not affect the
        effective key used at runtime.
        """
        settings = await self.load()
        if settings is None:
            return None

        llm = settings.agent_settings.llm
        if managed_llm_key_config_from_model(llm.model, llm.base_url) is None:
            return None

        async with a_session_maker() as session:
            result = await session.execute(
                select(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(self.user_id))
            )
            user = result.scalars().first()
            if user is None:
                return None

            org_id = self._resolve_org_id(user)
            org = await session.get(Org, org_id)
            if org is None or org._llm_api_key:
                return None

            org_member = next(
                (om for om in user.org_members if om.org_id == org_id), None
            )
            if (
                org_member is None
                or org_member.has_custom_llm_api_key
                or not org_member._llm_api_key
            ):
                return None
            return org_member.llm_api_key.get_secret_value()

    async def rotate_managed_llm_key(self) -> ManagedLlmKeyRotation:
        """Force-rotate the managed LiteLLM/OpenHands key for this user/org.

        Centralizes the managed-key lifecycle so callers (e.g. the API-key
        refresh endpoint) don't re-implement it. The effective LLM config is
        resolved through ``load()`` (org defaults merged with the member
        diff, with org-default precedence), so non-managed configs — member
        BYOK or org-level BYOK pointing at a third-party base_url — are
        rejected before any key is generated.

        The new key is generated under the same deterministic alias as
        ``_ensure_api_key`` / ``OrgStore._maybe_get_managed_llm_key_for_user``
        (deleting any prior alias first to avoid orphaned keys) and carries
        ``{'type': 'openhands'}`` metadata when the effective model is an
        ``openhands/*`` model, matching ``verify_existing_key``'s contract.

        The replacement key is persisted on the acting member's row in the
        same session used to load it, so a member that disappears mid-rotation
        is reported explicitly (``MISSING_MEMBER``) rather than silently
        swallowed. The previous key token is returned for best-effort cleanup
        and is only exposed after a successful persist.
        """
        settings = await self.load()
        if settings is None:
            return ManagedLlmKeyRotation(status=ManagedLlmKeyStatus.MISSING_MEMBER)

        llm = settings.agent_settings.llm
        config = managed_llm_key_config_from_model(llm.model, llm.base_url)
        if config is None:
            return ManagedLlmKeyRotation(status=ManagedLlmKeyStatus.NOT_MANAGED)

        async with a_session_maker() as session:
            result = await session.execute(
                select(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(self.user_id))
            )
            user = result.scalars().first()
            if user is None:
                return ManagedLlmKeyRotation(status=ManagedLlmKeyStatus.MISSING_MEMBER)

            org_id = self._resolve_org_id(user)
            org = await session.get(Org, org_id)
            if org is None:
                return ManagedLlmKeyRotation(status=ManagedLlmKeyStatus.MISSING_MEMBER)
            if org._llm_api_key:
                return ManagedLlmKeyRotation(status=ManagedLlmKeyStatus.BYOK)

            org_member = next(
                (om for om in user.org_members if om.org_id == org_id), None
            )
            if org_member is None:
                return ManagedLlmKeyRotation(status=ManagedLlmKeyStatus.MISSING_MEMBER)

            if org_member.has_custom_llm_api_key:
                return ManagedLlmKeyRotation(status=ManagedLlmKeyStatus.BYOK)

            existing_key = org_member.llm_api_key if org_member._llm_api_key else None
            old_key = existing_key.get_secret_value() if existing_key else None

            org_id_str = str(org_id)
            # One managed key per (user, org) under the deterministic alias;
            # delete the alias first so rotation never orphans a prior key.
            key_alias = get_openhands_cloud_key_alias(self.user_id, org_id_str)
            await LiteLlmManager.delete_key_by_alias(key_alias=key_alias)
            new_key = await LiteLlmManager.generate_key(
                self.user_id,
                org_id_str,
                key_alias,
                {'type': 'openhands'} if config.openhands_type else None,
            )

            # Persist on the same member row we loaded, in the same session,
            # before exposing the old token for cleanup.
            org_member.llm_api_key = SecretStr(new_key)
            org_member.has_custom_llm_api_key = False
            await session.commit()

            logger.info(
                'saas_settings_store:rotate_managed_llm_key:rotated',
                extra={'user_id': self.user_id, 'org_id': org_id_str},
            )
            return ManagedLlmKeyRotation(
                status=ManagedLlmKeyStatus.ROTATED,
                old_key=old_key,
                new_key=new_key,
                openhands_type=config.openhands_type,
            )
