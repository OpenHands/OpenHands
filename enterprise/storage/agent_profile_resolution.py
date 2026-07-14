"""Shared cloud adapters for resolving Agent Profiles against an org row.

The thin glue (#15044) that lets the SDK profile substrate
(``resolve_agent_profile`` / ``find_referrers`` / ``cascade_rename``) operate
over the SaaS storage model. Everything domain-level is imported from
``openhands.sdk.profiles``; this module only adapts the ``org.agent_profiles`` /
``org.llm_profiles`` columns to the SDK's store/loader/mutator Protocols.

Used by the ``/api/agent-profiles`` router, the LLM-profile FK guard in
``org_profiles``, and ``SaasSettingsStore.load``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from openhands.app_server.settings.agent_profiles import AgentProfiles
from openhands.app_server.settings.llm_profiles import LLMProfiles
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.mcp.config import MCPServer, coerce_mcp_config

if TYPE_CHECKING:
    from storage.org import Org
    from storage.org_member import OrgMember

    from openhands.sdk.llm import LLM
    from openhands.sdk.utils.cipher import Cipher


def load_agent_profiles(org: Org) -> AgentProfiles:
    """Load ``AgentProfiles`` from the org row, defaulting to empty if unset.

    Degrades to an empty collection on schema drift rather than 500-ing — the
    same contract as ``org_profiles._load_profiles`` for ``llm_profiles``.
    """
    if org.agent_profiles is None:
        return AgentProfiles()
    try:
        return AgentProfiles.model_validate(org.agent_profiles)
    except ValidationError as exc:
        logger.warning('Failed to load org agent profiles for %s: %s', org.id, exc)
        return AgentProfiles()


def load_llm_profiles(org: Org) -> LLMProfiles:
    """Load ``LLMProfiles`` from the org row, defaulting to empty if unset."""
    if org.llm_profiles is None:
        return LLMProfiles()
    try:
        return LLMProfiles.model_validate(org.llm_profiles)
    except ValidationError as exc:
        logger.warning('Failed to load org LLM profiles for %s: %s', org.id, exc)
        return LLMProfiles()


class OrgLLMProfileLoader:
    """``LLMProfileLoader`` over an org's LLM profiles — the resolver's ``llm_store``.

    The ``org.llm_profiles`` ``EncryptedJSON`` column decrypts at the column
    boundary, so the wrapped ``LLMProfiles`` already holds cleartext keys; the
    ``cipher`` arg is accepted for Protocol parity and ignored.
    """

    def __init__(self, profiles: LLMProfiles) -> None:
        self._profiles = profiles

    def load(self, name: str, *, cipher: Cipher | None = None) -> LLM:
        llm = self._profiles.get(name)
        if llm is None:
            # The resolver maps this to ProfileNotFound (HTTP 4xx).
            raise FileNotFoundError(f'LLM profile {name!r} not found')
        return llm


class OrgLLMProfileMutator:
    """``LLMProfileMutator`` over an org's LLM profiles — drives the FK delete/rename.

    Mutates the in-memory ``LLMProfiles`` container; the caller persists it back
    onto the org row under the same locked transaction.
    """

    def __init__(self, profiles: LLMProfiles) -> None:
        self._profiles = profiles

    def delete(self, name: str) -> None:
        self._profiles.delete(name)

    def rename(self, old_name: str, new_name: str) -> None:
        self._profiles.rename(old_name, new_name)


def member_mcp_config(member: OrgMember) -> dict[str, MCPServer]:
    """Return the acting member's configured MCP servers."""
    raw = member.effective_mcp_config
    if not raw:
        return {}
    try:
        return coerce_mcp_config(raw)
    except Exception as exc:
        # Catch broadly, not just ValidationError: coerce_mcp_config also raises
        # on fastmcp normalization / contract drift, and a malformed member
        # config must resolve to "no servers" rather than 500 the materialize
        # endpoint. Matches _resolve_active_agent_profile's inline coerce.
        logger.warning('Failed to parse member MCP config for resolve: %s', exc)
        return {}
