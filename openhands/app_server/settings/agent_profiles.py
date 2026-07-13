"""Cloud container for saved ``AgentProfile`` launch specs.

The Agent-Profile analogue of :class:`~openhands.app_server.settings.llm_profiles.LLMProfiles`,
one level up: an :class:`~openhands.sdk.profiles.AgentProfile` *references* an LLM
profile (``llm_profile_ref``) and a subset of the user's MCP servers
(``mcp_server_refs``) rather than embedding the resolved ``llm`` / ``mcp_config``.

Two responsibilities, kept deliberately thin:

* a pydantic envelope ``{profiles: {<id>: AgentProfile}, active: <id> | null}`` stored
  on the ``org.agent_profiles`` ``EncryptedJSON`` column (mirrors ``org.llm_profiles``);
* an :class:`~openhands.sdk.profiles.AgentProfileStoreProtocol` implementation, so the
  SDK FK / seed / id-stamping helpers (``find_referrers`` / ``cascade_rename`` /
  ``save_profile_preserving_identity`` / ``build_seed_profile``) drive a cloud DB store
  verbatim — never re-implemented here (the #3730 thin-boundary contract).

The collection is **id-keyed** (the active pointer and ``LaunchedAgentProfile`` key on
the stable UUID), but the Protocol's mutation surface (``load`` / ``rename`` / ``delete``
/ ``set_llm_profile_ref``) is **name-based** — names are the user-facing, FK-referenced
keys — so those methods look profiles up by ``name`` across the id-keyed dict. Names are
unique because every write goes through ``save_profile_preserving_identity``, which keys
identity on ``name``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final, TypeAlias
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.profiles import (
    ACPAgentProfile,
    OpenHandsAgentProfile,
    ProfileLimitExceeded,
    validate_agent_profile,
)

# Mirrors ``MAX_AGENT_PROFILES`` in the local agent-server router
# (openhands-agent-server/.../agent_profiles_router.py) so both backends share
# one cap and ``save_profile_preserving_identity`` enforces it identically.
MAX_AGENT_PROFILES: Final[int] = 50

_AgentProfile: TypeAlias = OpenHandsAgentProfile | ACPAgentProfile
# Return-type aliases for the ``list`` / ``list_summaries`` Protocol methods:
# the ``list()`` method shadows the ``list`` builtin in the class body, so a
# bare ``list[...]`` return annotation resolves to the method under mypy.
# Bind the builtin here at module scope instead.
_ProfileNames: TypeAlias = list[str]
_ProfileSummaries: TypeAlias = list[dict[str, Any]]


class AgentProfiles(BaseModel):
    """Id-keyed collection of ``AgentProfile``\\ s + an org-default active pointer.

    Conforms to :class:`~openhands.sdk.profiles.AgentProfileStoreProtocol`: the
    SDK FK / identity / seed helpers operate over this model with no filesystem
    access. ``lock()`` is a re-entrant no-op because the router holds the
    ``org`` row transaction (``SELECT ... FOR UPDATE``) for the whole mutation,
    exactly as ``org_profiles._org_profiles_transaction`` does for LLM profiles.

    Invariants (enforced on validate + assignment):
    - ``active`` is either ``None`` or a key (id) of ``profiles``.
    - Individual profiles that fail to parse (schema drift) are dropped with a
      warning rather than failing the whole ``Settings`` load — mirrors
      ``LLMProfiles._skip_invalid_profiles``.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Keyed by ``str(profile.id)`` — the stable UUID the active pointer and
    # ``LaunchedAgentProfile.agent_profile_id`` reference.
    profiles: dict[str, _AgentProfile] = Field(default_factory=dict)
    active: str | None = None

    # ── Validation ─────────────────────────────────────────────────

    @field_validator('profiles', mode='before')
    @classmethod
    def _skip_invalid_profiles(cls, value: Any) -> Any:
        """Best-effort per-profile load: skip entries that fail to validate.

        Guards against schema drift — one stored profile going invalid after an
        SDK upgrade must not fail the whole ``Settings`` load. Delegates parsing
        to the SDK's :func:`validate_agent_profile` (never re-validated here).
        """
        if not isinstance(value, dict):
            return value
        valid: dict[str, Any] = {}
        for key, raw in value.items():
            try:
                valid[key] = validate_agent_profile(raw)
            except Exception as exc:  # noqa: BLE001 - schema drift is non-fatal
                logger.warning('Skipping invalid agent profile %r: %s', key, exc)
        return valid

    @model_validator(mode='after')
    def _reconcile_active(self) -> AgentProfiles:
        if self.active is not None and self.active not in self.profiles:
            # Bypass validate_assignment to avoid re-entering this validator.
            object.__setattr__(self, 'active', None)
        return self

    # ── Internal name↔id helpers ───────────────────────────────────

    def _entry_for_name(self, name: str) -> tuple[str, _AgentProfile] | None:
        for pid, profile in self.profiles.items():
            if profile.name == name:
                return pid, profile
        return None

    # ── AgentProfileStoreProtocol ──────────────────────────────────

    @contextmanager
    def lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Re-entrant no-op — the router holds the ``org`` row transaction."""
        yield

    def list(self) -> _ProfileNames:
        """Profile "filenames" (``<name>.json``) — used only for emptiness checks."""
        return [f'{profile.name}.json' for profile in self.profiles.values()]

    def list_summaries(self) -> _ProfileSummaries:
        """Metadata projection ``{id, name, agent_kind, revision, llm_profile_ref,
        mcp_server_refs}`` per profile — the shape the SDK FK scans and the list
        endpoint consume. ``llm_profile_ref`` is ``None`` for ACP profiles."""
        summaries: list[dict[str, Any]] = []
        for pid, profile in self.profiles.items():
            summaries.append(
                {
                    'id': pid,
                    'name': profile.name,
                    'agent_kind': profile.agent_kind,
                    'revision': profile.revision,
                    'llm_profile_ref': (
                        profile.llm_profile_ref
                        if isinstance(profile, OpenHandsAgentProfile)
                        else None
                    ),
                    'mcp_server_refs': profile.mcp_server_refs,
                }
            )
        return summaries

    def save(
        self,
        profile: _AgentProfile,
        *,
        max_profiles: int | None = None,
    ) -> None:
        """Store ``profile`` under ``str(profile.id)``, overwriting any namesake.

        The id is set by the caller (``save_profile_preserving_identity``). No
        cipher is involved — the profile is secret-free at rest (#4017), and
        the ``org.agent_profiles`` ``EncryptedJSON`` column is the encryption
        boundary regardless (parity with ``org.llm_profiles``).
        """
        pid = str(profile.id)
        if (
            max_profiles is not None
            and pid not in self.profiles
            and len(self.profiles) >= max_profiles
        ):
            raise ProfileLimitExceeded(f'Profile limit reached ({max_profiles}).')
        self.profiles[pid] = profile

    def load(self, name: str) -> _AgentProfile:
        entry = self._entry_for_name(name)
        if entry is None:
            raise FileNotFoundError(f'Agent profile {name!r} not found')
        return entry[1]

    def delete(self, name: str) -> None:
        """Remove the profile named ``name`` (idempotent). Clears the org-default
        ``active`` pointer if it referenced the deleted profile."""
        entry = self._entry_for_name(name)
        if entry is None:
            return
        pid = entry[0]
        del self.profiles[pid]
        if self.active == pid:
            object.__setattr__(self, 'active', None)

    def rename(self, old_name: str, new_name: str) -> None:
        entry = self._entry_for_name(old_name)
        if entry is None:
            raise FileNotFoundError(f'Agent profile {old_name!r} not found')
        if old_name == new_name:
            return
        if self._entry_for_name(new_name) is not None:
            raise FileExistsError(f'Agent profile {new_name!r} already exists')
        pid, profile = entry
        # Preserve the id (and thus the slot + active pointer): rename is a
        # ``name`` edit only.
        self.profiles[pid] = profile.model_copy(update={'name': new_name})

    def set_llm_profile_ref(self, name: str, new_ref: str) -> None:
        """Repoint one OpenHands profile's ``llm_profile_ref`` (the FK cascade
        write primitive). No-op for a missing or ACP-kind profile."""
        entry = self._entry_for_name(name)
        if entry is None:
            return
        pid, profile = entry
        if isinstance(profile, OpenHandsAgentProfile):
            self.profiles[pid] = profile.model_copy(update={'llm_profile_ref': new_ref})

    def name_for_id(self, profile_id: str | UUID) -> str | None:
        profile = self.profiles.get(str(profile_id))
        return profile.name if profile is not None else None

    # ── Serialization ──────────────────────────────────────────────

    @field_serializer('profiles')
    def _profiles_serializer(
        self,
        profiles: dict[str, _AgentProfile],
        info: SerializationInfo,
    ) -> dict[str, Any]:
        # Thread the serialization context through — a no-op for AgentProfile
        # today (secret-free since #4017), kept for parity with LLMProfiles'
        # write-back pattern.
        return {
            pid: profile.model_dump(mode='json', context=info.context)
            for pid, profile in profiles.items()
        }
