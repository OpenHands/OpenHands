"""Service class for managing organization LLM settings.

Separates business logic from route handlers.
Uses dependency injection for db_session and user_context.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import Request
from pydantic import SecretStr
from server.constants import LITE_LLM_API_URL
from server.routes.org_models import (
    OrgLLMSettingsResponse,
    OrgLLMSettingsUpdate,
    OrgMemberLLMSettings,
    OrgNotFoundError,
)
from sqlalchemy import select
from storage.lite_llm_manager import LiteLlmManager, get_openhands_cloud_key_alias
from storage.org import Org
from storage.org_llm_settings_store import OrgLLMSettingsStore
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore

from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.core.logger import openhands_logger as logger
from openhands.utils.llm import is_openhands_model


@dataclass
class OrgLLMSettingsService:
    """Service for org LLM settings with injected dependencies."""

    store: OrgLLMSettingsStore
    user_context: UserContext

    async def get_org_llm_settings(self) -> OrgLLMSettingsResponse:
        """Get LLM settings for user's current organization.

        User ID is obtained from the injected user_context.

        Returns:
            OrgLLMSettingsResponse: The organization's LLM settings

        Raises:
            ValueError: If user is not authenticated
            OrgNotFoundError: If current organization not found
        """
        user_id = await self.user_context.get_user_id()
        if not user_id:
            raise ValueError('User is not authenticated')

        logger.info(
            'Getting organization LLM settings',
            extra={'user_id': user_id},
        )

        org = await self.store.get_current_org_by_user_id(user_id)

        if not org:
            raise OrgNotFoundError('No current organization')

        return OrgLLMSettingsResponse.from_org(org)

    async def update_org_llm_settings(
        self,
        update_data: OrgLLMSettingsUpdate,
    ) -> OrgLLMSettingsResponse:
        """Update LLM settings for user's current organization.

        Only updates fields that are explicitly provided in update_data.
        User ID is obtained from the injected user_context.
        Session auto-commits at request end via DbSessionInjector.

        Args:
            update_data: The update data from the request

        Returns:
            OrgLLMSettingsResponse: The updated organization's LLM settings

        Raises:
            ValueError: If user is not authenticated
            OrgNotFoundError: If current organization not found
        """
        user_id = await self.user_context.get_user_id()
        if not user_id:
            raise ValueError('User is not authenticated')

        logger.info(
            'Updating organization LLM settings',
            extra={'user_id': user_id},
        )

        # Check if any fields are provided
        if not update_data.has_updates():
            # No fields to update, just return current settings
            return await self.get_org_llm_settings()

        # Get user's current org first
        org = await self.store.get_current_org_by_user_id(user_id)
        if not org:
            raise OrgNotFoundError('No current organization')

        # Update the org LLM settings
        updated_org = await self.store.update_org_llm_settings(
            org_id=org.id,
            update_data=update_data,
        )

        if not updated_org:
            raise OrgNotFoundError(str(org.id))

        # Build the member-propagation payload from the update diff, then
        # let the managed-key rotation merge in a freshly generated /
        # reused managed key (OpenHands / LiteLLM proxy case). A single
        # ``update_all_members_llm_settings_async`` call at the end writes
        # the ``agent_settings_diff``, ``llm_api_key``, and a
        # ``has_custom_llm_api_key=False`` reset to every member row in one
        # pass. The flag reset is load-bearing: org-defaults saves are an
        # org-wide "use the org default" signal, so any lingering
        # "I have a personal BYOR key" marker on a member row would make
        # ``_get_effective_llm_api_key`` return the wrong key at read time.
        member_updates = update_data.get_member_updates()

        effective_managed_key = await self._maybe_rotate_managed_llm_key_for_user(
            updated_org=updated_org,
            user_id=user_id,
        )
        if effective_managed_key is not None:
            if member_updates is None:
                member_updates = OrgMemberLLMSettings()
            member_updates.llm_api_key = SecretStr(effective_managed_key)

        if member_updates is not None:
            member_updates.has_custom_llm_api_key = False
            await OrgMemberStore.update_all_members_llm_settings_async(
                self.store.db_session,
                org.id,
                member_updates,
            )
            logger.info(
                'Propagated org LLM settings to members',
                extra={'user_id': user_id, 'org_id': str(org.id)},
            )

        logger.info(
            'Organization LLM settings updated successfully',
            extra={'user_id': user_id, 'org_id': str(org.id)},
        )

        return OrgLLMSettingsResponse.from_org(updated_org)

    async def _maybe_rotate_managed_llm_key_for_user(
        self,
        updated_org: Org,
        user_id: str,
    ) -> str | None:
        """Return the managed LLM key every member row should carry, or
        ``None`` if the org isn't in managed mode.

        When the updated org defaults target a managed LLM (OpenHands
        provider or the LiteLLM proxy base URL), this reuses the acting
        user's (any admin or owner with ``EDIT_LLM_SETTINGS``) current
        managed key if ``verify_existing_key`` confirms it's still valid,
        otherwise generates a fresh one. The returned key is what every
        member's ``_llm_api_key`` column should hold — the caller bundles
        it into the single ``update_all_members_llm_settings_async`` call
        alongside the ``agent_settings_diff`` and a
        ``has_custom_llm_api_key=False`` reset so one DB pass covers all
        three. Detection matches ``SaasSettingsStore.store`` so the two
        save paths agree on when managed mode is in play; propagation
        semantics differ because org-defaults saves are intentionally an
        org-wide operation.
        """
        llm = (updated_org.agent_settings or {}).get('llm') or {}
        llm_model = llm.get('model')
        llm_base_url = llm.get('base_url')
        normalized_llm_base_url = llm_base_url.rstrip('/') if llm_base_url else None
        normalized_managed_base_url = LITE_LLM_API_URL.rstrip('/')
        openhands_type = is_openhands_model(llm_model)
        uses_managed_llm_key = (
            normalized_llm_base_url == normalized_managed_base_url
            or (normalized_llm_base_url is None and openhands_type)
        )
        if not uses_managed_llm_key:
            return None

        result = await self.store.db_session.execute(
            select(OrgMember).where(
                OrgMember.org_id == updated_org.id,
                OrgMember.user_id == uuid.UUID(user_id),
            )
        )
        acting_member = result.scalars().first()
        if acting_member is None:
            # Shouldn't happen — the caller already resolved the user's
            # current org via ``get_current_org_by_user_id`` before calling
            # us, so the ``OrgMember`` row must exist. If it's missing
            # anyway, the org-wide managed-key propagation skips the
            # ``llm_api_key`` write (``effective_managed_key`` returns
            # ``None``) and members keep whatever was in their columns.
            # Log loudly so this data-consistency issue surfaces instead of
            # silently leaving stale keys on member rows.
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
            # Reuse the acting user's still-valid managed key — no need to
            # burn a LiteLLM key rotation on a no-op save.
            effective_key = existing_key_raw
            rotated = False
        else:
            if openhands_type:
                effective_key = await LiteLlmManager.generate_key(
                    user_id,
                    str(updated_org.id),
                    None,
                    {'type': 'openhands'},
                )
            else:
                key_alias = get_openhands_cloud_key_alias(user_id, str(updated_org.id))
                await LiteLlmManager.delete_key_by_alias(key_alias=key_alias)
                effective_key = await LiteLlmManager.generate_key(
                    user_id,
                    str(updated_org.id),
                    key_alias,
                    None,
                )
            rotated = True

        # The caller merges ``effective_key`` into ``member_updates`` and
        # issues a single ``update_all_members_llm_settings_async`` call
        # that writes the key column AND resets ``has_custom_llm_api_key``
        # on every member — including this acting row — so we don't touch
        # the row directly here.

        if rotated:
            logger.info(
                'Generated managed LLM key for acting user on org-defaults save',
                extra={'user_id': user_id, 'org_id': str(updated_org.id)},
            )

        return effective_key


class OrgLLMSettingsServiceInjector(Injector[OrgLLMSettingsService]):
    """Injector that composes store and user_context for OrgLLMSettingsService."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[OrgLLMSettingsService, None]:
        # Local imports to avoid circular dependencies
        from openhands.app_server.config import get_db_session, get_user_context

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            store = OrgLLMSettingsStore(db_session=db_session)
            yield OrgLLMSettingsService(store=store, user_context=user_context)
