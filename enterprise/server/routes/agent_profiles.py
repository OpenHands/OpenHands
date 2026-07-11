"""Organization Agent Profiles router (cloud / SaaS).

Serves the flat, member-facing ``/api/agent-profiles`` surface that
agent-canvas's backend-agnostic ``AgentProfilesClient`` calls (the same wire
contract as the local agent-server ``agent_profiles_router.py``). The org is
derived from the authenticated session (``EFFECTIVE_ORG_ID``) rather than the
path, because each member activates *their own* profile (#15044 §2); everything
else — the org-owned column, the locked org-row transaction, the
``VIEW``/``EDIT_ORG_SETTINGS`` permission model — mirrors the LLM-profile router
``org_profiles.py``.

Domain logic is **imported, never re-implemented** (the #3730 thin-boundary
contract): validation → ``validate_agent_profile``; id/revision stamping →
``save_profile_preserving_identity``; the dry-run resolve →
``resolve_agent_profile_dry_run``; the 422 redaction →
``safe_validation_error_detail``. Agent profiles are secret-free at rest
(software-agent-sdk#4017 dropped the last secret-bearing field, embedded
``skills``) — unlike ``org_profiles``, there is no api-key lift/mask on
activate, and GET/POST need no secret masking or restore-on-save at all.

Activation is **pointer-only**: it writes the per-member
``OrgMember.active_agent_profile_id`` and nothing else (the creation-time-only
contract, #15044 §3). Resolution into ``agent_settings`` happens at
conversation-start (``SaasSettingsStore.load``), not here.
"""

from __future__ import annotations

import contextlib
from typing import Annotated, Any, AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field, ValidationError
from server.auth.authorization import Permission, require_permission
from server.auth.org_context import EFFECTIVE_ORG_ID
from server.routes.org_models import OrgNotFoundError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from storage.agent_profile_resolution import (
    OrgLLMProfileLoader,
    load_agent_profiles,
    load_llm_profiles,
    member_mcp_config,
)
from storage.database import a_session_maker
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_service import OrgService

from openhands.app_server.settings.agent_profiles import (
    MAX_AGENT_PROFILES,
    AgentProfiles,
)
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.profiles import (
    AgentProfile,
    AgentProfileDiagnostics,
    ProfileLimitExceeded,
    resolve_agent_profile_dry_run,
    safe_validation_error_detail,
    save_profile_preserving_identity,
    validate_agent_profile,
)
from openhands.sdk.profiles.agent_profile_store import PROFILE_NAME_PATTERN

router = APIRouter(prefix='/api/agent-profiles', tags=['Agent Profiles'])

ProfileName = Annotated[
    str, Path(min_length=1, max_length=64, pattern=PROFILE_NAME_PATTERN)
]
ProfileId = Annotated[str, Path(min_length=1, max_length=128)]


# ── Request/Response Models (mirror the local agent_profiles_router contract) ──


class AgentProfileInfo(BaseModel):
    """Summary projection of a stored profile (no secret instantiation)."""

    id: str | None = None
    name: str
    agent_kind: str = 'openhands'
    revision: int | None = None
    llm_profile_ref: str | None = None
    mcp_server_refs: list[str] | None = None


class AgentProfileListResponse(BaseModel):
    profiles: list[AgentProfileInfo]
    active_agent_profile_id: str | None = None


class AgentProfileDetailResponse(BaseModel):
    name: str
    # The stored profile, secret-free at rest. Typed as the SDK discriminated
    # union (not ``dict``) so the OpenAPI schema matches the ``AgentProfile``
    # the ts-client consumes.
    profile: AgentProfile


class AgentProfileMutationResponse(BaseModel):
    name: str
    message: str


class ActivateAgentProfileResponse(BaseModel):
    id: str
    message: str
    # Always False: activation is pointer-only by contract. agent_settings is
    # resolved at conversation-start, not on activate.
    agent_settings_applied: bool = False


class RenameAgentProfileRequest(BaseModel):
    new_name: str = Field(
        ..., min_length=1, max_length=64, pattern=PROFILE_NAME_PATTERN
    )


# ── Helpers ────────────────────────────────────────────────────────────────


async def _get_org(org_id: UUID, user_id: str) -> Org:
    """Load the org, raising 404 if the user is not a member / it is missing."""
    try:
        return await OrgService.get_org_by_id(org_id=org_id, user_id=user_id)
    except OrgNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


async def _get_member(
    session: AsyncSession, org_id: UUID, user_id: str
) -> OrgMember | None:
    result = await session.execute(
        select(OrgMember).filter(
            OrgMember.org_id == org_id, OrgMember.user_id == UUID(user_id)
        )
    )
    return result.scalars().first()


async def _get_org_and_member(
    org_id: UUID, user_id: str
) -> tuple[Org, OrgMember | None]:
    """Fetch the org (raising 404 if missing) and the acting member's row.

    The member is read in its own short-lived session — matches the org read,
    which resolves through ``OrgService.get_org_by_id`` rather than this
    module's session directly.
    """
    org = await _get_org(org_id, user_id)
    async with a_session_maker() as session:
        member = await _get_member(session, org_id, user_id)
    return org, member


@contextlib.asynccontextmanager
async def _agent_profiles_transaction(
    org_id: UUID, user_id: str
) -> AsyncIterator[tuple[AsyncSession, Org, AgentProfiles]]:
    """Yield ``(session, org, agent_profiles)`` for a single locked mutation.

    Mirrors ``org_profiles._org_profiles_transaction``: ``SELECT ... FOR UPDATE``
    on the org row so concurrent profile mutations serialize at the DB level.
    The caller mutates ``agent_profiles`` in place (and may also write the acting
    member's ``active_agent_profile_id`` via the same session); on normal exit
    the helper serializes the collection back onto the org row and commits. This
    DB row lock *is* the store's ``lock()`` — hence the in-memory model's
    re-entrant no-op ``lock()``.

    The collection is written back ONLY when the caller actually changed it.
    Loading is best-effort (``_skip_invalid_profiles`` drops entries that fail
    to validate, e.g. after schema drift), so an unconditional write-back would
    let a mutation-free call such as ``/activate`` silently erase a stored
    profile it merely failed to parse.
    """
    await _get_org(org_id, user_id)
    async with a_session_maker() as session:
        result = await session.execute(
            select(Org).filter(Org.id == org_id).with_for_update()
        )
        org = result.scalars().first()
        if org is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Organization {org_id} not found',
            )
        profiles = load_agent_profiles(org)
        # The EncryptedJSON column is the at-rest boundary. expose_secrets=True
        # is now a no-op for AgentProfile (secret-free since #4017), kept for
        # parity with the write-back pattern the LLM-profile router shares.
        before = profiles.model_dump(mode='json', context={'expose_secrets': True})
        yield session, org, profiles
        after = profiles.model_dump(mode='json', context={'expose_secrets': True})
        if after != before:
            org.agent_profiles = after
        await session.commit()


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get('', response_model=AgentProfileListResponse)
async def list_agent_profiles(
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> AgentProfileListResponse:
    """List agent profiles and the effective active pointer for this member.

    No implicit writes: an org with no profiles yet just returns an empty
    list. Profile creation is always an explicit ``save_agent_profile`` call
    (``EDIT_ORG_SETTINGS``).
    """
    org, member = await _get_org_and_member(effective_org_id, user_id)
    profiles = load_agent_profiles(org)
    member_active = member.active_agent_profile_id if member is not None else None
    active_id = member_active or profiles.active

    return AgentProfileListResponse(
        profiles=[AgentProfileInfo(**s) for s in profiles.list_summaries()],
        active_agent_profile_id=active_id,
    )


@router.get('/{name}', response_model=AgentProfileDetailResponse)
async def get_agent_profile(
    name: ProfileName,
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> AgentProfileDetailResponse:
    """Get a stored profile — secret-free at rest, nothing to mask.

    Unlike the LLM-profile GET in ``org_profiles``, there is no
    ``X-Expose-Secrets`` handling here at all: the profile carries only
    references (``llm_profile_ref``, ``mcp_server_refs``) and a
    ``disabled_skills`` deny-list of names.
    """
    org = await _get_org(effective_org_id, user_id)
    profiles = load_agent_profiles(org)
    try:
        profile = profiles.load(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent profile '{name}' not found",
        )
    return AgentProfileDetailResponse(name=name, profile=profile)


@router.post(
    '/{name}',
    response_model=AgentProfileMutationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_agent_profile(
    name: ProfileName,
    body: dict[str, Any],
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> AgentProfileMutationResponse:
    """Create or update an agent profile under ``name`` (path name is authoritative).

    Server-managed id/revision: overwrite keeps the namesake's id and bumps
    revision; create mints a fresh id (``save_profile_preserving_identity``).
    Returns 422 on invalid payloads (secret-safe detail) and 409 when creating a
    new profile would exceed ``MAX_AGENT_PROFILES``.
    """
    async with _agent_profiles_transaction(effective_org_id, user_id) as (
        _session,
        _org,
        profiles,
    ):
        try:
            profile = validate_agent_profile({**body, 'name': name})
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=safe_validation_error_detail(e),
            )
        except Exception:
            # SkillValidationError / schema errors are client errors, never a
            # 500; stay generic — these messages can embed the input.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail='Invalid agent profile',
            )

        try:
            save_profile_preserving_identity(
                profiles, profile, max_profiles=MAX_AGENT_PROFILES
            )
        except ProfileLimitExceeded:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f'Agent profile limit reached ({MAX_AGENT_PROFILES}). '
                    'Delete a profile before saving a new one.'
                ),
            )

    logger.info("Saved agent profile '%s' for org %s", name, effective_org_id)
    return AgentProfileMutationResponse(
        name=name, message=f"Agent profile '{name}' saved"
    )


@router.delete('/{name}', response_model=AgentProfileMutationResponse)
async def delete_agent_profile(
    name: ProfileName,
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> AgentProfileMutationResponse:
    """Delete a profile (idempotent). Clears every org member's pointer to it.

    A missing name resolves 200, matching the ts-client ``AgentProfilesClient``
    contract and the local agent-server ``delete_profile`` (canvas's delete
    mutation has no 404 branch).
    """
    async with _agent_profiles_transaction(effective_org_id, user_id) as (
        session,
        _org,
        profiles,
    ):
        # Capture the id (if present) before deleting so the per-member pointer
        # clear still runs; ``profiles.delete`` is itself a no-op when absent.
        deleted_id: str | None = None
        with contextlib.suppress(FileNotFoundError):
            deleted_id = str(profiles.load(name).id)
        profiles.delete(name)
        if deleted_id is not None:
            # Clear the pointer for every member who had this profile active, not
            # just the acting member — activation is per-member, so any other
            # member could be pointing at the now-deleted id.
            await session.execute(
                update(OrgMember)
                .where(
                    OrgMember.org_id == effective_org_id,
                    OrgMember.active_agent_profile_id == deleted_id,
                )
                .values(active_agent_profile_id=None)
            )

    logger.info("Deleted agent profile '%s' for org %s", name, effective_org_id)
    return AgentProfileMutationResponse(
        name=name, message=f"Agent profile '{name}' deleted"
    )


@router.post('/{name}/rename', response_model=AgentProfileMutationResponse)
async def rename_agent_profile(
    name: ProfileName,
    request: RenameAgentProfileRequest = Body(...),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> AgentProfileMutationResponse:
    """Rename a profile atomically.

    The stable id is preserved, so active pointers (keyed on id) survive
    untouched.
    """
    async with _agent_profiles_transaction(effective_org_id, user_id) as (
        _session,
        _org,
        profiles,
    ):
        try:
            profiles.rename(name, request.new_name)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent profile '{name}' not found",
            )
        except FileExistsError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent profile '{request.new_name}' already exists",
            )

    return AgentProfileMutationResponse(
        name=request.new_name,
        message=f"Agent profile renamed from '{name}' to '{request.new_name}'",
    )


@router.post('/{profile_id}/activate', response_model=ActivateAgentProfileResponse)
async def activate_agent_profile(
    profile_id: ProfileId,
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> ActivateAgentProfileResponse:
    """Activate a profile for the calling member — pointer only.

    Writes the per-member ``OrgMember.active_agent_profile_id`` and nothing else
    (no ``agent_settings`` materialization, and the profiles collection itself
    is untouched — the transaction helper skips the write-back when nothing
    mutated; #15044 §3). Member-facing, so it requires only
    ``VIEW_ORG_SETTINGS`` (each member picks their own active profile); profile
    CRUD requires ``EDIT_ORG_SETTINGS``. 404 if no stored profile has that id.
    """
    async with _agent_profiles_transaction(effective_org_id, user_id) as (
        session,
        _org,
        profiles,
    ):
        known_ids = {s['id'] for s in profiles.list_summaries() if s.get('id')}
        if profile_id not in known_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent profile with id '{profile_id}' not found",
            )
        member = await _get_member(session, effective_org_id, user_id)
        if member is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Organization membership not found',
            )
        member.active_agent_profile_id = profile_id

    logger.info(
        "Activated agent profile id '%s' for member in org %s",
        profile_id,
        effective_org_id,
    )
    return ActivateAgentProfileResponse(
        id=profile_id, message=f"Agent profile '{profile_id}' activated"
    )


@router.post('/{name}/materialize', response_model=AgentProfileDiagnostics)
async def materialize_agent_profile(
    name: ProfileName,
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> AgentProfileDiagnostics:
    """Dry-run resolve a profile's LLM/MCP references into a diagnostics report.

    Dangling refs are reported in the body (``valid=False``), not raised; the
    only error status is 404 (unknown profile name). ``resolved_settings`` is
    redacted. Delegates entirely to ``resolve_agent_profile_dry_run``.
    """
    org, member = await _get_org_and_member(effective_org_id, user_id)
    profiles = load_agent_profiles(org)
    try:
        profile = profiles.load(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent profile '{name}' not found",
        )

    mcp_config = member_mcp_config(member) if member is not None else {}
    llm_store = OrgLLMProfileLoader(load_llm_profiles(org))

    try:
        return resolve_agent_profile_dry_run(
            profile,
            llm_store=llm_store,
            mcp_config=mcp_config,
            available_skills=None,
            cipher=None,
        )
    except Exception as exc:
        # The dry-run is contractually total, but SDK contract drift (e.g. a
        # new required kwarg raising TypeError) must surface as an invalid
        # diagnostics report, not a 500.
        logger.warning(
            "Agent profile dry-run resolve failed for '%s' in org %s: %s",
            name,
            effective_org_id,
            exc,
        )
        return AgentProfileDiagnostics(
            agent_kind=profile.agent_kind,
            valid=False,
            errors=[f'Failed to resolve profile: {exc}'],
        )
