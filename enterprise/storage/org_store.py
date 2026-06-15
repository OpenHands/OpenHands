"""Store class for managing organizations."""

from typing import Any, Optional
from uuid import UUID

from pydantic import SecretStr
from server.constants import (
    DEFAULT_V1_ENABLED,
    LITE_LLM_API_URL,
    ORG_SETTINGS_VERSION,
    get_default_llm_base_url,
    get_default_llm_model,
)
from server.routes.org_models import (
    OrgMemberSettingsUpdate,
    OrgUpdate,
    OrphanedUserError,
)
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload
from storage.database import a_session_maker
from storage.lite_llm_manager import (
    LiteLlmManager,
    get_openhands_cloud_key_alias,
    get_org_team_alias,
)
from storage.org import Org
from storage.org_git_claim import OrgGitClaim
from storage.org_invitation import OrgInvitation
from storage.org_member import OrgMember
from storage.user import User
from storage.user_settings import UserSettings

from openhands.app_server.settings.settings_models import (
    Settings,
    _load_persisted_agent_settings,
    _load_persisted_conversation_settings,
)
from openhands.app_server.utils.jsonpatch_compat import deep_merge
from openhands.app_server.utils.llm import is_openhands_model
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.settings import (
    AgentSettingsConfig,
    ConversationSettings,
    OpenHandsAgentSettings,
    apply_agent_settings_diff,
)

_ORG_SETTINGS_EXCLUDED_FIELDS = {
    'id',
    'name',
    'contact_name',
    'contact_email',
    'org_version',
    'llm_api_key',
}
_ORG_SETTINGS_FIELDS = {
    normalized
    for column in Org.__table__.columns
    if (normalized := column.name.lstrip('_')) not in _ORG_SETTINGS_EXCLUDED_FIELDS
}


class OrgStore:
    """Store for managing organizations."""

    @staticmethod
    async def _delete_litellm_user_best_effort(user_id: str, org_id: UUID) -> None:
        """Delete the LiteLLM user record without blocking org deletion."""
        try:
            await LiteLlmManager.delete_user(user_id)
        except Exception as exc:
            logger.warning(
                'Failed to delete LiteLLM user during org cascade cleanup',
                extra={'org_id': str(org_id), 'user_id': user_id, 'error': str(exc)},
            )

    @staticmethod
    def get_agent_settings_from_org(org: Org) -> AgentSettingsConfig:
        # Route through the shared SDK loader: it applies persisted-settings
        # migrations (incl. the legacy ``agent_kind: 'llm'`` -> ``'openhands'``
        # rename) and returns the actual variant. ACP settings are returned as
        # ``ACPAgentSettings``, not coerced into the OpenHands shape — that
        # coercion 500s on ACP's nullable ``agent_context``.
        return _load_persisted_agent_settings(dict(org.agent_settings))

    @staticmethod
    def get_conversation_settings_from_org(org: Org) -> ConversationSettings:
        return _load_persisted_conversation_settings(dict(org.conversation_settings))

    @staticmethod
    def sync_agent_settings(org: Org) -> None:
        org.agent_settings = dict(org.agent_settings)

    @staticmethod
    def sync_conversation_settings(org: Org) -> None:
        org.conversation_settings = dict(org.conversation_settings)

    @staticmethod
    async def create_org(
        kwargs: dict,
    ) -> Org:
        """Create a new organization."""
        async with a_session_maker() as session:
            org = Org(**kwargs)
            org.org_version = ORG_SETTINGS_VERSION
            agent_settings = org.agent_settings or {}
            llm_settings = agent_settings.get('llm', {})
            org.agent_settings = deep_merge(
                agent_settings,
                {
                    'llm': {
                        'model': llm_settings.get('model') or get_default_llm_model(),
                        'base_url': llm_settings.get('base_url')
                        or get_default_llm_base_url(),
                    }
                },
            )
            if org.v1_enabled is None:
                org.v1_enabled = DEFAULT_V1_ENABLED
            session.add(org)
            await session.commit()
            await session.refresh(org)
            return org

    @staticmethod
    async def get_org_by_id(org_id: UUID) -> Org | None:
        """Get organization by ID."""
        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
        return await OrgStore._validate_org_version(org)

    @staticmethod
    async def enable_byor_export(org_id: UUID) -> Org | None:
        """Persist BYOR export enablement for an organization."""
        return await OrgStore._update_org_kwargs(
            org_id,
            {'byor_export_enabled': True},
        )

    @staticmethod
    async def get_orgs_by_ids(org_ids: list[UUID]) -> list[Org]:
        """Get multiple organizations by IDs in a single query.

        Args:
            org_ids: List of organization UUIDs to fetch.

        Returns:
            List of Org objects (may be fewer than input if some IDs don't exist).
        """
        if not org_ids:
            return []
        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.id.in_(org_ids)))
            orgs = list(result.scalars().all())
        # Validate org versions for all returned orgs
        validated_orgs = []
        for org in orgs:
            validated = await OrgStore._validate_org_version(org)
            if validated:
                validated_orgs.append(validated)
        return validated_orgs

    @staticmethod
    async def get_current_org_from_keycloak_user_id(
        keycloak_user_id: str,
    ) -> Org | None:
        async with a_session_maker() as session:
            result = await session.execute(
                select(User)
                .options(joinedload(User.org_members))
                .filter(User.id == UUID(keycloak_user_id))
            )
            user = result.scalars().first()
            if not user:
                logger.warning(f'User not found for ID {keycloak_user_id}')
                return None
            org_id = user.current_org_id
            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
            if not org:
                logger.warning(
                    f'Org not found for ID {org_id} as the current org for user {keycloak_user_id}'
                )
                return None
            return await OrgStore._validate_org_version(org)

    @staticmethod
    async def get_org_by_name(name: str) -> Org | None:
        """Get organization by name."""
        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.name == name))
            org = result.scalars().first()
        return await OrgStore._validate_org_version(org)

    @staticmethod
    async def get_default_org() -> Org | None:
        """Get the org flagged as the install's bootstrapped default org."""
        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.is_default))
            org = result.scalars().first()
        return await OrgStore._validate_org_version(org)

    @staticmethod
    async def mark_org_as_default(org_id: UUID) -> Org | None:
        """Flag an org as the install's default org.

        Returns the org on success (or if already flagged). Returns None if
        the org does not exist or another org is already flagged (the partial
        unique index allows at most one default org).
        """
        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
            if not org:
                return None
            if not org.is_default:
                org.is_default = True
                try:
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    return None
        return org

    @staticmethod
    async def list_team_orgs(limit: int | None = None) -> list[Org]:
        """List orgs that are not personal workspaces (see count_team_orgs)."""
        async with a_session_maker() as session:
            stmt = select(Org).where(~select(User.id).where(User.id == Org.id).exists())
            if limit is not None:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def count_team_orgs() -> int:
        """Count orgs that are not personal workspaces.

        A personal workspace shares its id with its user, so team orgs are
        the orgs whose id has no matching user.
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Org)
                .where(~select(User.id).where(User.id == Org.id).exists())
            )
            return int(result.scalar() or 0)

    @staticmethod
    async def _validate_org_version(org: Org | None) -> Org | None:
        """Check if we need to update org version."""
        if org and org.org_version < ORG_SETTINGS_VERSION:
            org = await OrgStore._update_org_kwargs(
                org.id,
                {
                    'org_version': ORG_SETTINGS_VERSION,
                    'agent_settings_diff': {
                        'llm': {
                            'model': get_default_llm_model(),
                            'base_url': get_default_llm_base_url(),
                        },
                    },
                },
            )
        return org

    @staticmethod
    async def list_orgs() -> list[Org]:
        """List all organizations."""
        async with a_session_maker() as session:
            result = await session.execute(select(Org))
            orgs = result.scalars().all()
            return list(orgs)

    @staticmethod
    async def get_user_orgs_paginated(
        user_id: UUID, page_id: str | None = None, limit: int = 100
    ) -> tuple[list[Org], str | None]:
        """Get paginated list of organizations for a user.

        Args:
            user_id: User UUID
            page_id: Optional page ID (offset as string) for pagination
            limit: Maximum number of organizations to return

        Returns:
            Tuple of (list of Org objects, next_page_id or None)
        """
        async with a_session_maker() as session:
            # Build query joining OrgMember with Org
            query = (
                select(Org)
                .join(OrgMember, Org.id == OrgMember.org_id)
                .filter(OrgMember.user_id == user_id)
                .order_by(Org.name)
            )

            # Apply pagination offset
            if page_id is not None:
                try:
                    offset = int(page_id)
                    query = query.offset(offset)
                except ValueError:
                    # If page_id is not a valid integer, start from beginning
                    offset = 0
            else:
                offset = 0

            # Fetch limit + 1 to check if there are more results
            query = query.limit(limit + 1)
            result = await session.execute(query)
            orgs = list(result.scalars().all())

            # Check if there are more results
            has_more = len(orgs) > limit
            if has_more:
                orgs = orgs[:limit]

            # Calculate next page ID
            next_page_id = None
            if has_more:
                next_page_id = str(offset + limit)

            # Validate org versions
            validated_orgs = []
            for org in orgs:
                if org:
                    validated = await OrgStore._validate_org_version(org)
                    if validated is not None:
                        validated_orgs.append(validated)

            return validated_orgs, next_page_id

    @staticmethod
    def _merge_and_validate_settings(
        current_settings: dict[str, Any],
        settings_diff: dict[str, Any],
        settings_type: type[OpenHandsAgentSettings] | type[ConversationSettings],
    ) -> AgentSettingsConfig | ConversationSettings:
        """Apply a sparse settings diff to the persisted base and validate it.

        Agent settings delegate to the SDK's :func:`apply_agent_settings_diff`,
        which owns the discriminated-union merge (replace on ``agent_kind``
        change, deep-merge within a variant) and returns the correct variant
        (OpenHands or ACP) rather than a coerced OpenHands shape.
        """
        if settings_type is OpenHandsAgentSettings:
            return apply_agent_settings_diff(current_settings or {}, settings_diff)

        base_settings = _load_persisted_conversation_settings(current_settings)  # type: ignore[assignment]
        merged_settings = deep_merge(
            base_settings.model_dump(mode='json'), settings_diff
        )
        return ConversationSettings.model_validate(merged_settings)

    @staticmethod
    async def update_org(
        org_id: UUID,
        update_data: OrgUpdate,
        user_id: str | None = None,
    ) -> Optional[Org]:
        """Update organization details from a validated OrgUpdate payload."""
        return await OrgStore._update_org_kwargs(
            org_id,
            update_data.model_update_dict(),
            user_id=user_id,
            update_data=update_data,
        )

    @staticmethod
    async def _update_org_kwargs(
        org_id: UUID,
        org_kwargs: dict[str, Any],
        user_id: str | None = None,
        update_data: OrgUpdate | None = None,
    ) -> Optional[Org]:
        """Internal helper for updating organization fields from raw kwargs."""
        from storage.org_member_store import OrgMemberStore

        org_kwargs = dict(org_kwargs)

        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
            if not org:
                return None

            old_name = org.name

            if 'id' in org_kwargs:
                org_kwargs.pop('id')

            # Pop the diff-style kwargs before the setattr loop — otherwise
            # ``hasattr(org, 'agent_settings')`` is True and the loop would
            # *overwrite* the JSON column instead of deep-merging into it.
            agent_settings_diff = (
                update_data.agent_settings_diff
                if update_data is not None
                else org_kwargs.pop('agent_settings_diff', None)
            )
            conversation_settings_diff = (
                update_data.conversation_settings_diff
                if update_data is not None
                else org_kwargs.pop('conversation_settings_diff', None)
            )
            for key, value in org_kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

            if agent_settings_diff is not None:
                org.agent_settings = OrgStore._merge_and_validate_settings(
                    org.agent_settings,
                    agent_settings_diff,
                    OpenHandsAgentSettings,
                ).model_dump(mode='json', exclude_unset=True)

            if conversation_settings_diff is not None:
                org.conversation_settings = OrgStore._merge_and_validate_settings(
                    org.conversation_settings,
                    conversation_settings_diff,
                    ConversationSettings,
                ).model_dump(mode='json', exclude_unset=True)

            if update_data is not None and update_data.touches_org_defaults():
                if user_id is None:
                    raise ValueError(
                        'user_id is required when updating organization defaults'
                    )

                member_updates = update_data.get_member_updates()
                effective_managed_key = (
                    await OrgStore._maybe_get_managed_llm_key_for_user(
                        session,
                        org,
                        user_id,
                    )
                )
                should_reset_custom_key_flag = (
                    update_data.llm_api_key is not None
                    or effective_managed_key is not None
                )
                if effective_managed_key is not None:
                    if member_updates is None:
                        member_updates = OrgMemberSettingsUpdate()
                    member_updates.llm_api_key = SecretStr(effective_managed_key)

                if member_updates is not None:
                    if should_reset_custom_key_flag:
                        member_updates.has_custom_llm_api_key = False
                    await OrgMemberStore.update_all_members_settings_async(
                        session, org_id, member_updates
                    )

            await session.commit()
            await session.refresh(org)

            # Keep the LiteLLM team_alias in sync with the org's display name so
            # the proxy dashboard stays readable after a rename. Best-effort —
            # never fail an org update because the proxy is briefly unreachable.
            if org.name != old_name:
                try:
                    await LiteLlmManager.update_team(
                        str(org.id),
                        get_org_team_alias(str(org.id), org.name, user_id),
                        None,
                    )
                except Exception:
                    logger.warning(
                        'Failed to propagate org rename to LiteLLM team_alias',
                        extra={'org_id': str(org.id)},
                    )
            return org

    @staticmethod
    def get_kwargs_from_settings(settings: Settings):
        dumped = settings.model_dump(mode='json', context={'expose_secrets': True})
        return {
            field: dumped[field] for field in _ORG_SETTINGS_FIELDS if field in dumped
        }

    @staticmethod
    def get_kwargs_from_user_settings(user_settings: UserSettings):
        kwargs = {
            field: getattr(user_settings, field)
            for field in _ORG_SETTINGS_FIELDS
            if hasattr(user_settings, field)
        }
        kwargs['org_version'] = user_settings.user_version
        return kwargs

    @staticmethod
    async def persist_org_with_owner(
        org: Org,
        org_member: OrgMember,
    ) -> Org:
        """Persist organization and owner membership in a single transaction.

        Args:
            org: Organization entity to persist
            org_member: Organization member entity to persist

        Returns:
            Org: The persisted organization object

        Raises:
            Exception: If database operations fail
        """
        async with a_session_maker() as session:
            session.add(org)
            session.add(org_member)
            await session.commit()
            await session.refresh(org)
            return org

    @staticmethod
    async def delete_org_cascade(
        org_id: UUID, requester_user_id: str | None = None
    ) -> Org | None:
        """Delete organization and all associated data in cascade, including external LiteLLM cleanup.

        Users that belong to the org being deleted are handled in three ways:

        * Users with a membership in at least one other org have their
          ``current_org_id`` reassigned to one of those alternative orgs.
        * If the *requester themselves* is the only orphaned user (sole member
          of the org being deleted — the personal-org self-service case), the
          requester's user row is cascade-deleted in the same transaction. The
          Keycloak account is left untouched, so on the user's next login
          ``UserStore.create_user`` re-onboards them as a brand-new user. The
          new ``User.id`` and ``Org.id`` are derived from the Keycloak ``sub``
          claim, which is stable across logins, so the re-created personal-org
          identity matches the deleted one (``User.id == Org.id ==
          UUID(keycloak.sub)``) and downstream automations that key on
          ``keycloak_user_id`` continue to resolve correctly.
        * If any orphan is **not** the requester (i.e., a multi-user org where
          another member has no other org), ``OrphanedUserError`` is raised
          and the whole transaction is rolled back. The org owner must
          transfer or remove those members before deletion can proceed —
          we refuse to silently destroy accounts that did not consent.

        Args:
            org_id: UUID of the organization to delete
            requester_user_id: Keycloak ``sub`` of the user initiating the
                deletion. Required for the sole-orphan personal-org case;
                when ``None`` (e.g., internal callers), any orphan triggers
                ``OrphanedUserError``.

        Returns:
            Org: The deleted organization object, or None if not found

        Raises:
            OrphanedUserError: If any non-requester member of the org would be
                left without any organization by the deletion.
            Exception: If database operations or LiteLLM cleanup fail
        """
        async with a_session_maker() as session:
            # First get the organization to return it
            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
            if not org:
                return None

            try:
                # Preflight orphan check — fail fast before any writes.
                #
                # The orphan SELECT only reads ``user`` and ``org_member``,
                # neither of which is modified by the org-data cleanup
                # below, so we hoist it to the top of the transaction. If
                # the check fails we raise immediately and the rollback
                # has essentially no write work to undo.
                #
                # Running it inside the transaction (rather than as a
                # separate pre-flight call before opening the session) is
                # deliberate: it ensures the orphan computation shares the
                # same snapshot/locks as the destructive writes below, so
                # another session cannot create a new orphan between
                # check-and-act.
                #
                # No row-level lock (``FOR UPDATE``) is acquired here. A
                # concurrent session could in principle insert or remove
                # an ``org_member`` row for the requester between this
                # read and the ``DELETE User`` write below; we accept
                # that as an unlikely edge case for the personal-org
                # self-service path, where the requester is the only
                # actor with permission to mutate their own memberships.
                # Promote to ``FOR UPDATE`` if this code starts running
                # on behalf of multi-actor flows.
                orphaned_result = await session.execute(
                    text("""
                        SELECT u.id
                        FROM "user" u
                        WHERE u.current_org_id = :org_id
                        AND NOT EXISTS (
                            SELECT 1 FROM org_member om
                            WHERE om.user_id = u.id AND om.org_id != :org_id
                        )
                    """),
                    {'org_id': str(org_id)},
                )
                orphaned_user_ids = [str(row[0]) for row in orphaned_result.fetchall()]

                # Split the orphaned users into the requester and everyone
                # else. Only the requester is cascade-deleted: by calling
                # DELETE on their own org they've consented to losing their
                # account. Other members have not consented, so if any
                # non-requester would be orphaned we raise OrphanedUserError
                # and let the transaction roll back.
                other_orphans = [
                    uid for uid in orphaned_user_ids if uid != requester_user_id
                ]
                if other_orphans:
                    raise OrphanedUserError(other_orphans)

                requester_orphan_ids = [
                    uid for uid in orphaned_user_ids if uid == requester_user_id
                ]

                # 1. Delete conversation data for organization conversations
                await session.execute(
                    text("""
                    DELETE FROM conversation_metadata
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_metadata_saas WHERE org_id = :org_id
                    )
                    """),
                    {'org_id': str(org_id)},
                )

                await session.execute(
                    text("""
                    DELETE FROM app_conversation_start_task
                    WHERE app_conversation_id IN (
                        SELECT conversation_id::uuid FROM conversation_metadata_saas WHERE org_id = :org_id
                    )
                    """),
                    {'org_id': str(org_id)},
                )

                # 2. Delete organization-owned data tables (direct org_id foreign keys)
                await session.execute(
                    text('DELETE FROM billing_sessions WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                await session.execute(
                    text(
                        'DELETE FROM conversation_metadata_saas WHERE org_id = :org_id'
                    ),
                    {'org_id': str(org_id)},
                )
                await session.execute(
                    text('DELETE FROM custom_secrets WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                await session.execute(
                    text('DELETE FROM api_keys WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                await session.execute(
                    text('DELETE FROM slack_conversation WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                await session.execute(
                    text('DELETE FROM slack_users WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                await session.execute(
                    text('DELETE FROM stripe_customers WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )

                # 3. Handle users with this as current_org_id BEFORE deleting
                # memberships. The orphan partition was computed in the
                # preflight check at the top of this try block;
                # ``requester_orphan_ids`` is set iff the requester is a
                # sole-org member of this org.

                # 3a. Cascade-delete the requester if they are a sole-org user
                # (personal-org self-service path). Their personal-org identity
                # is preserved on re-login because UserStore.create_user derives
                # both User.id and Org.id from the Keycloak ``sub`` claim (which
                # is stable across logins), so a re-onboarded user receives the
                # same UUIDs they had before. Downstream systems keyed on
                # ``keycloak_user_id``/``org_id`` continue to resolve correctly.
                #
                # We must remove the user row BEFORE deleting the org row,
                # because ``user.current_org_id`` is a NOT NULL FK to
                # ``org.id``. That in turn requires releasing other inbound
                # FKs to ``user.id`` first: ``org_invitation`` and
                # ``org_git_claim`` cascade on ``org.id`` in the DB schema,
                # but the org row still exists at this point, so we clean
                # those references explicitly here.
                #
                # FK edges on ``user.id`` cleared before ``DELETE User``:
                #   * ``org_invitation.inviter_id``
                #   * ``org_invitation.accepted_by_user_id``
                #   * ``org_git_claim.claimed_by``
                #   * ``org_member.user_id``
                # *** If a future migration adds another table with a
                # FK to ``user.id`` (e.g. ``user_api_key``, ``audit_log``)
                # this block MUST be updated to release it, or the
                # ``DELETE User`` below will raise a runtime FK violation
                # with no obvious pointer back to this site. ***
                if requester_orphan_ids:
                    await session.execute(
                        delete(OrgInvitation).where(
                            OrgInvitation.inviter_id.in_(requester_orphan_ids)
                            | OrgInvitation.accepted_by_user_id.in_(
                                requester_orphan_ids
                            )
                        )
                    )
                    await session.execute(
                        delete(OrgGitClaim).where(
                            OrgGitClaim.claimed_by.in_(requester_orphan_ids)
                        )
                    )
                    # All of the orphan's memberships are for the org being
                    # deleted (definition of "orphan") and so would be removed
                    # by step 4 anyway, but we have to release them now to drop
                    # the user_id FK before the user row goes away.
                    await session.execute(
                        delete(OrgMember).where(
                            OrgMember.user_id.in_(requester_orphan_ids)
                        )
                    )
                    await session.execute(
                        delete(User).where(User.id.in_(requester_orphan_ids))
                    )

                # 3b. Batch update: reassign current_org_id to an alternative org
                # for any remaining users. Orphan rows are gone at this point, so
                # the subquery is guaranteed to return a non-null org for every
                # row touched by the UPDATE.
                await session.execute(
                    text("""
                        UPDATE "user"
                        SET current_org_id = (
                            SELECT om.org_id FROM org_member om
                            WHERE om.user_id = "user".id AND om.org_id != :org_id
                            LIMIT 1
                        )
                        WHERE "user".current_org_id = :org_id
                    """),
                    {'org_id': str(org_id)},
                )

                # 4. Delete organization memberships (now safe)
                await session.execute(
                    text('DELETE FROM org_member WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )

                # 5. Finally delete the organization.
                # ``AsyncSession.delete`` is a coroutine; without ``await``
                # it is a silent no-op — the ORM never flushes the DELETE
                # and the ``org`` row survives the transaction even though
                # every preceding step committed. Forgetting the ``await``
                # here would leave the next sign-in colliding on
                # ``org_pkey`` in ``UserStore.create_user``, because both
                # the surviving row and the new row are keyed on the same
                # stable Keycloak ``sub``. Awaited explicitly to make that
                # invariant load-bearing rather than incidental.
                await session.delete(org)

                # 6. Clean up LiteLLM team before committing transaction
                logger.info(
                    'Deleting LiteLLM team within database transaction',
                    extra={'org_id': str(org_id)},
                )
                await LiteLlmManager.delete_team(str(org_id))

                if requester_orphan_ids:
                    for user_id in requester_orphan_ids:
                        await OrgStore._delete_litellm_user_best_effort(user_id, org_id)

                # 7. Commit all changes only if everything succeeded
                await session.commit()

                logger.info(
                    'Successfully deleted organization and all associated data including LiteLLM team',
                    extra={'org_id': str(org_id), 'org_name': org.name},
                )

                return org

            except Exception as e:
                await session.rollback()
                logger.error(
                    'Failed to delete organization - transaction rolled back',
                    extra={'org_id': str(org_id), 'error': str(e)},
                )
                raise

    @staticmethod
    async def get_org_by_id_async(org_id: UUID) -> Org | None:
        """Get organization by ID (async version).

        Note: This method is kept for backwards compatibility but simply
        delegates to get_org_by_id which is now async.
        """
        return await OrgStore.get_org_by_id(org_id)

    @staticmethod
    async def _maybe_get_managed_llm_key_for_user(
        session,
        updated_org: Org,
        user_id: str,
    ) -> str | None:
        """Return the managed LLM key every member row should carry, if any."""
        llm_settings = OrgStore.get_agent_settings_from_org(updated_org).llm
        llm_model = llm_settings.model
        llm_base_url = llm_settings.base_url
        normalized_llm_base_url = llm_base_url.rstrip('/') if llm_base_url else None
        normalized_managed_base_url = LITE_LLM_API_URL.rstrip('/')
        openhands_type = is_openhands_model(llm_model)
        uses_managed_llm_key = (
            normalized_llm_base_url == normalized_managed_base_url
            or (normalized_llm_base_url is None and openhands_type)
        )
        if not uses_managed_llm_key:
            return None

        result = await session.execute(
            select(OrgMember).where(
                OrgMember.org_id == updated_org.id,
                OrgMember.user_id == UUID(user_id),
            )
        )
        acting_member = result.scalars().first()
        if acting_member is None:
            logger.error(
                'Acting member row not found during managed LLM key '
                'rotation; skipping managed-key propagation. Members may '
                'retain stale keys until they save personal settings.',
                extra={'user_id': user_id, 'org_id': str(updated_org.id)},
            )
            return None

        existing_key = acting_member.llm_api_key
        existing_key_raw = existing_key.get_secret_value() if existing_key else None
        if existing_key_raw and await LiteLlmManager.verify_existing_key(
            existing_key_raw,
            user_id,
            str(updated_org.id),
            openhands_type=openhands_type,
        ):
            return existing_key_raw

        # One managed key per (user, org) under the same deterministic alias,
        # deleting any prior key first — symmetric across openhands/* and BYOR
        # defaults so switching between them never orphans a key.
        key_alias = get_openhands_cloud_key_alias(user_id, str(updated_org.id))
        await LiteLlmManager.delete_key_by_alias(key_alias=key_alias)
        logger.info(
            'Generated managed LLM key for acting user on org-defaults save',
            extra={'user_id': user_id, 'org_id': str(updated_org.id)},
        )
        return await LiteLlmManager.generate_key(
            user_id,
            str(updated_org.id),
            key_alias,
            {'type': 'openhands'} if openhands_type else None,
        )

    @staticmethod
    async def update_org_defaults_async(
        org_id: UUID,
        update_data: OrgUpdate,
        user_id: str,
    ) -> Org | None:
        """Backward-compatible wrapper for org-defaults updates."""
        return await OrgStore.update_org(org_id, update_data, user_id)
