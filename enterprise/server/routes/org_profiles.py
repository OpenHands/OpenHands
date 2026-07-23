"""Organization LLM profiles router.

Provides CRUD operations for org-level LLM profiles. Profiles are stored on
the organization and can be activated by members.

Permission model:
- CRUD (create, update, delete, rename): Requires EDIT_ORG_SETTINGS (owner/admin)
- Activate: Requires EDIT_ORG_SETTINGS — the handler also writes the org-wide
  ``profiles.active`` marker, so the permission must match the bigger of the
  two side effects rather than the per-member one.
"""

import contextlib
from typing import Any, AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field, ValidationError
from server.constants import LITE_LLM_API_URL
from server.routes.org_models import OrgNotFoundError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from storage.agent_profile_resolution import (
    OrgLLMProfileMutator,
    load_agent_profiles,
)
from storage.database import a_session_maker
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_service import OrgService

from openhands.app_server.settings.llm_profiles import (
    LLMProfiles,
    ProfileAlreadyExistsError,
    ProfileLimitExceededError,
    ProfileNotFoundError,
    StrictLLM,
)
from openhands.app_server.settings.settings_models import (
    _load_persisted_agent_settings,
)
from openhands.app_server.utils.llm import MASKED_API_KEY, is_openhands_model
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.llm import LLM
from openhands.sdk.profiles import (
    ProfileReferenced,
    delete_llm_profile,
    rename_llm_profile,
)
from openhands.sdk.profiles.agent_profile_store import PROFILE_NAME_PATTERN

from ..auth.authorization import Permission, require_permission

router = APIRouter(tags=['Organization Profiles'])


# ── Request/Response Models ────────────────────────────────────────────────


class ProfileInfo(BaseModel):
    """Summary info for a profile (no secrets)."""

    name: str
    model: str | None
    base_url: str | None
    api_key_set: bool


class ProfileListResponse(BaseModel):
    """Response for listing profiles."""

    profiles: list[ProfileInfo]
    active_profile: str | None


class ProfileDetailResponse(BaseModel):
    """Response for getting a single profile's details."""

    name: str
    llm: dict[str, Any]


class ProfileMutationResponse(BaseModel):
    """Response for profile mutations (save, delete, rename)."""

    name: str
    message: str


class ActivateProfileResponse(BaseModel):
    """Response for activating a profile."""

    name: str
    message: str
    llm: dict[str, Any]


class SaveProfileRequest(BaseModel):
    """Request body for saving a profile."""

    include_secrets: bool = True
    llm: StrictLLM | None = None
    # Set when the caller has no new key (UI key field left blank), so an
    # existing profile's stored key survives instead of the snapshotted one.
    preserve_existing_api_key: bool = False


class RenameProfileRequest(BaseModel):
    """Request body for renaming a profile.

    ``new_name`` is constrained to ``PROFILE_NAME_PATTERN`` because
    ``rename_llm_profile`` (the SDK FK-cascade helper backing this endpoint)
    validates the new name against that same pattern before renaming and
    repointing any referencing Agent Profiles — a name outside it always 422s
    there. Declaring the constraint here makes the schema honest and gives a
    field-level 422 instead of one raised deep inside the handler. ``save``
    (create/update) is intentionally left permissive: it never calls the FK
    helper, so it has no such requirement.
    """

    new_name: str = Field(
        ..., min_length=1, max_length=64, pattern=PROFILE_NAME_PATTERN
    )


# ── Helper Functions ────────────────────────────────────────────────────────


def _load_profiles(org: Org) -> LLMProfiles:
    """Load LLMProfiles from org row, defaulting to empty if not set."""
    if org.llm_profiles is None:
        return LLMProfiles()
    try:
        return LLMProfiles.model_validate(org.llm_profiles)
    except ValidationError as exc:
        # Schema drift / partially-invalid stored profiles: degrade to empty
        # rather than 500-ing. Other exceptions (DB decrypt failures, etc.)
        # bubble up so they're surfaced instead of silently masked.
        logger.warning('Failed to load org profiles for %s: %s', org.id, exc)
        return LLMProfiles()


async def _get_org(org_id: UUID, user_id: str) -> Org:
    """Get org, raising 404 if not found."""
    try:
        return await OrgService.get_org_by_id(org_id=org_id, user_id=user_id)
    except OrgNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@contextlib.asynccontextmanager
async def _org_profiles_transaction(
    org_id: UUID, user_id: str
) -> AsyncIterator[tuple[AsyncSession, Org, LLMProfiles]]:
    """Yield ``(session, org, profiles)`` for a single locked mutation.

    Wraps read → mutate → write in one session with ``SELECT ... FOR UPDATE``
    so concurrent profile mutations serialize at the database level instead
    of racing on the ``llm_profiles`` column (last-writer-wins would silently
    drop the loser's changes). The caller mutates ``profiles`` in place; on
    normal exit the helper serializes it back onto the org row and commits.
    Exceptions skip the commit, so partial state never lands — useful for
    multi-write endpoints like activate that also update ``OrgMember``.
    """
    # Membership/access check (perms are enforced by the route's Depends; this
    # is the same org-membership check the read endpoints do via _get_org).
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
        profiles = _load_profiles(org)
        yield session, org, profiles
        org.llm_profiles = profiles.model_dump(
            mode='json', context={'expose_secrets': True}
        )
        await session.commit()


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get('/{org_id}/profiles', response_model=ProfileListResponse)
async def list_profiles(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> ProfileListResponse:
    """List all LLM profiles for this organization."""
    org = await _get_org(org_id, user_id)
    profiles = _load_profiles(org)
    return ProfileListResponse(
        profiles=[
            ProfileInfo(**p)
            for p in profiles.summaries(managed_proxy_url=LITE_LLM_API_URL)
        ],
        active_profile=profiles.active,
    )


@router.get('/{org_id}/profiles/{name}', response_model=ProfileDetailResponse)
async def get_profile(
    org_id: UUID,
    name: str = Path(..., min_length=1),
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> ProfileDetailResponse:
    """Get details of a specific profile."""
    org = await _get_org(org_id, user_id)
    profiles = _load_profiles(org)
    llm = profiles.get(name)
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{name}' not found",
        )
    return ProfileDetailResponse(
        name=name,
        llm=llm.model_dump(mode='json', context={'expose_secrets': False}),
    )


@router.post('/{org_id}/profiles/{name}', response_model=ProfileMutationResponse)
async def save_profile(
    org_id: UUID,
    name: str = Path(..., min_length=1, max_length=100),
    request: SaveProfileRequest = SaveProfileRequest(),
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> ProfileMutationResponse:
    """Create or update an LLM profile.

    If ``llm`` is omitted, saves a copy of the current org LLM defaults.
    """
    async with _org_profiles_transaction(org_id, user_id) as (_session, org, profiles):
        existing = profiles.get(name)
        llm: LLM
        if request.llm is not None:
            llm = request.llm
            # Preserve the stored api_key when an update omits it (e.g. a
            # round-tripped GET response) — mirrors the personal profiles route.
            if llm.api_key is None and existing is not None:
                if existing.api_key is not None:
                    llm = llm.model_copy(update={'api_key': existing.api_key})
        else:
            # Snapshot current org LLM settings. Route through the persisted
            # loader so legacy/canonical ``agent_kind`` discriminator values
            # ('llm' vs 'openhands') both validate.
            llm = _load_persisted_agent_settings(org.agent_settings).llm
        if request.preserve_existing_api_key and existing is not None:
            # Caller has no new key: keep the profile's stored key (even "no
            # key") instead of the snapshotted one.
            llm = llm.model_copy(update={'api_key': existing.api_key})
        try:
            profiles.save(name, llm, include_secrets=request.include_secrets)
        except ProfileLimitExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc

    return ProfileMutationResponse(name=name, message=f"Profile '{name}' saved")


@router.delete('/{org_id}/profiles/{name}', response_model=ProfileMutationResponse)
async def delete_profile(
    org_id: UUID,
    name: str = Path(..., min_length=1),
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> ProfileMutationResponse:
    """Delete an LLM profile.

    Blocked with 409 if any of the org's Agent Profiles still reference this LLM
    profile by ``llm_profile_ref`` (the SDK ``find_referrers`` FK guard) — both
    collections live on the same org row, so the ``SELECT ... FOR UPDATE`` lock
    makes the referrer check and the delete atomic.
    """
    async with _org_profiles_transaction(org_id, user_id) as (
        session,
        org,
        profiles,
    ):
        if not profiles.has(name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile '{name}' not found",
            )
        agent_profiles = load_agent_profiles(org)
        try:
            delete_llm_profile(agent_profiles, OrgLLMProfileMutator(profiles), name)
        except ProfileReferenced as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        await session.execute(
            update(OrgMember)
            .where(
                OrgMember.org_id == org_id,
                OrgMember.title_llm_profile == name,
            )
            .values(title_llm_profile=None)
        )

    return ProfileMutationResponse(name=name, message=f"Profile '{name}' deleted")


@router.post(
    '/{org_id}/profiles/{name}/activate', response_model=ActivateProfileResponse
)
async def activate_profile(
    org_id: UUID,
    name: str = Path(..., min_length=1),
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> ActivateProfileResponse:
    """Activate a profile for the current user.

    Two side effects: updates the org-wide ``profiles.active`` marker and
    writes the profile's LLM into the calling member's
    ``agent_settings_diff``. Both writes share a single transaction so a
    failure in the second can't leave the org marker advanced without the
    member's settings catching up. Because the first effect is org-level
    state, this requires ``EDIT_ORG_SETTINGS`` — matching the CRUD endpoints
    rather than the read-only listing. For personal orgs the owner has the
    permission natively; for team orgs this scopes "set org default profile"
    to admins.
    """
    async with _org_profiles_transaction(org_id, user_id) as (
        session,
        _org,
        profiles,
    ):
        llm = profiles.get(name)
        if llm is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile '{name}' not found",
            )
        profiles.active = name

        # Same session as the org write so both side-effects commit atomically.
        # Cast ``user_id`` explicitly: Postgres' UUID type tolerates string
        # coercion, but SQLAlchemy's generic Uuid binding (used under SQLite
        # in tests) doesn't.
        member_result = await session.execute(
            select(OrgMember).filter(
                OrgMember.org_id == org_id, OrgMember.user_id == UUID(user_id)
            )
        )
        member = member_result.scalars().first()
        if member is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Organization membership not found',
            )
        # Apply the profile to the calling member. The profile's raw api_key
        # must never land in ``agent_settings_diff`` (a plain, unencrypted JSON
        # column): mask it there and lift the real value into the encrypted
        # ``_llm_api_key`` column, mirroring the main settings path
        # (OrgUpdate._lift_and_mask_llm_api_key / SaasSettingsStore.store). The
        # effective api_key is resolved from ``_llm_api_key`` /
        # ``has_custom_llm_api_key`` at load time, not from the diff — so a
        # profile's key only takes effect if written there.
        llm_dump = llm.model_dump(mode='json', context={'expose_secrets': True})
        profile_api_key = llm_dump.get('api_key')
        if profile_api_key and profile_api_key != MASKED_API_KEY:
            llm_dump['api_key'] = MASKED_API_KEY
            # Classify managed vs. BYOR exactly as SaasSettingsStore.store() so
            # billing attribution stays correct.
            base_url = llm_dump.get('base_url')
            normalized_base_url = base_url.rstrip('/') if base_url else None
            normalized_managed_base_url = LITE_LLM_API_URL.rstrip('/')
            uses_managed_llm_key = (
                normalized_base_url == normalized_managed_base_url
                or (
                    normalized_base_url is None
                    and is_openhands_model(llm_dump.get('model'))
                )
            )
            member.llm_api_key = profile_api_key
            member.has_custom_llm_api_key = not uses_managed_llm_key
        else:
            # No per-profile key: fall back to the org/managed default rather
            # than leaving a stale custom key from a previous activation in play.
            member.has_custom_llm_api_key = False

        member_diff = dict(member.agent_settings_diff or {})
        member_diff['llm'] = llm_dump
        member.agent_settings_diff = member_diff

    return ActivateProfileResponse(
        name=name,
        message=f"Profile '{name}' activated",
        llm=llm.model_dump(mode='json', context={'expose_secrets': False}),
    )


@router.post('/{org_id}/profiles/{name}/rename', response_model=ProfileMutationResponse)
async def rename_profile(
    org_id: UUID,
    name: str = Path(..., min_length=1),
    request: RenameProfileRequest = Body(...),
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> ProfileMutationResponse:
    """Rename an LLM profile, cascading the rename to any Agent Profiles that
    reference it.

    The SDK ``rename_llm_profile`` renames the LLM profile and repoints every
    ``agent_profiles.*.llm_profile_ref == name`` to ``new_name`` under the org
    row lock, so a referencing Agent Profile never dangles. Both collections are
    written back in the same transaction.
    """
    async with _org_profiles_transaction(org_id, user_id) as (
        session,
        org,
        profiles,
    ):
        agent_profiles = load_agent_profiles(org)
        # Loading is best-effort (invalid entries are dropped), so only write
        # the collection back when the cascade actually repointed a ref — an
        # unconditional write-back would erase a stored profile that merely
        # failed to parse.
        before = agent_profiles.model_dump(
            mode='json', context={'expose_secrets': True}
        )
        try:
            rename_llm_profile(
                agent_profiles,
                OrgLLMProfileMutator(profiles),
                name,
                request.new_name,
            )
        except ProfileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except ProfileAlreadyExistsError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except ValueError as exc:
            # rename_llm_profile validates new_name against PROFILE_NAME_PATTERN.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        # The cascade may have repointed agent-profile refs — persist them too.
        after = agent_profiles.model_dump(mode='json', context={'expose_secrets': True})
        if after != before:
            org.agent_profiles = after
        await session.execute(
            update(OrgMember)
            .where(
                OrgMember.org_id == org_id,
                OrgMember.title_llm_profile == name,
            )
            .values(title_llm_profile=request.new_name)
        )

    return ProfileMutationResponse(
        name=request.new_name,
        message=f"Profile renamed from '{name}' to '{request.new_name}'",
    )
