"""Store class for managing organization app settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timezone
from uuid import UUID

from server.constants import (
    ORG_SETTINGS_VERSION,
    get_default_llm_base_url,
    get_default_llm_model,
)
from server.routes.org_models import (
    OrgAppSettingsUpdate,
    OrgConcurrentModificationError,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.org import Org
from storage.user import User

from openhands.app_server.utils.jsonpatch_compat import deep_merge


@dataclass
class OrgAppSettingsStore:
    """Store for organization app settings with injected db_session."""

    db_session: AsyncSession

    async def get_current_org_by_user_id(self, user_id: str) -> Org | None:
        """Get the current organization for a user.

        Args:
            user_id: The user's ID (Keycloak user ID)

        Returns:
            Org: The organization object, or None if not found
        """
        # Get user with their current_org_id
        user_result = await self.db_session.execute(
            select(User).filter(User.id == UUID(user_id))
        )
        user = user_result.scalars().first()

        if not user:
            return None

        org_id = user.current_org_id
        if not org_id:
            return None

        return await self.get_org_by_id(org_id)

    async def get_org_by_id(self, org_id: UUID) -> Org | None:
        """Get an organization by its id, validating the org version.

        Args:
            org_id: The organization's UUID.

        Returns:
            Org: The organization object, or None if not found.
        """
        org_result = await self.db_session.execute(select(Org).filter(Org.id == org_id))
        org = org_result.scalars().first()
        if not org:
            return None
        return await self._validate_org_version(org)

    async def _validate_org_version(self, org: Org) -> Org:
        """Check if we need to update org version.

        Args:
            org: The organization to validate

        Returns:
            Org: The validated (and potentially updated) organization
        """
        if org.org_version < ORG_SETTINGS_VERSION:
            org.org_version = ORG_SETTINGS_VERSION
            org.agent_settings = deep_merge(
                org.agent_settings,
                {
                    'llm': {
                        'model': get_default_llm_model(),
                        'base_url': get_default_llm_base_url(),
                    },
                },
            )
            await self.db_session.flush()
            await self.db_session.refresh(org)

        return org

    async def update_org_app_settings(
        self, org_id: UUID, update_data: OrgAppSettingsUpdate
    ) -> Org | None:
        """Update organization app settings.

        Only updates fields that are explicitly provided in update_data.
        Uses flush() - commit happens at request end via DbSessionInjector.

        Implements optimistic locking: if last_known_updated_at is provided and
        doesn't match the current DB version, raises OrgConcurrentModificationError.

        Args:
            org_id: The organization's ID
            update_data: Pydantic model with fields to update

        Returns:
            Org: The updated organization object, or None if not found

        Raises:
            OrgConcurrentModificationError: If optimistic locking detects a conflict
        """
        logger = logging.getLogger(__name__)

        result = await self.db_session.execute(
            select(Org).filter(Org.id == org_id).with_for_update()
        )
        org = result.scalars().first()

        if not org:
            return None

        # Optimistic locking: check if record was modified since client read it
        if update_data.last_known_updated_at is not None:
            current_updated_at = org.updated_at
            expected_updated_at = update_data.last_known_updated_at

            # Ensure timezone-aware comparison
            if current_updated_at.tzinfo is None:
                current_updated_at = current_updated_at.replace(tzinfo=timezone.utc)
            if expected_updated_at.tzinfo is None:
                expected_updated_at = expected_updated_at.replace(tzinfo=timezone.utc)

            # Raise conflict error if versions don't match
            if current_updated_at != expected_updated_at:
                logger.warning(
                    f"Org '{org.name}' concurrent modification detected. "
                    f'Expected: {expected_updated_at.isoformat()}, '
                    f'Current: {current_updated_at.isoformat()}. '
                    f'Raising conflict error.'
                )
                raise OrgConcurrentModificationError(
                    org_id=str(org.id),
                    expected_version=expected_updated_at,
                    actual_version=current_updated_at,
                )

        # Handle registered_marketplaces separately (dedicated column)
        update_dict = update_data.model_dump(
            exclude_unset=True,
            exclude={'last_known_updated_at'},  # Don't save this field
        )
        if 'registered_marketplaces' in update_dict:
            from openhands.app_server.settings.settings_models import (
                MarketplaceRegistration,
                MarketplaceScope,
            )

            # Inject scope='org' for all marketplaces saved at org level
            marketplaces = update_dict.pop('registered_marketplaces')
            validated_marketplaces: list[MarketplaceRegistration] = []
            for mp in marketplaces:
                if isinstance(mp, dict):
                    # Strip scope from incoming request - backend will set it
                    mp_dict = {k: v for k, v in mp.items() if k != 'scope'}
                    # Ensure auto_load defaults to False if not provided
                    if 'auto_load' not in mp_dict:
                        mp_dict['auto_load'] = False
                    mp_obj = MarketplaceRegistration.model_validate(mp_dict)
                    mp_obj.scope = MarketplaceScope.ORG
                    validated_marketplaces.append(mp_obj)
                elif isinstance(mp, MarketplaceRegistration):
                    mp.scope = MarketplaceScope.ORG
                    validated_marketplaces.append(mp)
                else:
                    # Already validated dict from DB - reconstruct as MarketplaceRegistration
                    db_mp = dict(mp)  # Copy to avoid mutation
                    db_mp['scope'] = MarketplaceScope.ORG
                    if 'auto_load' not in db_mp:
                        db_mp['auto_load'] = False
                    validated_marketplaces.append(
                        MarketplaceRegistration.model_validate(db_mp)
                    )

            # Names are the marketplace identity; reject duplicates before persist
            # so a bad write can't break loading/resolution later.
            from openhands.app_server.settings.marketplace_composition import (
                duplicate_marketplace_names,
            )

            conflicts = duplicate_marketplace_names(validated_marketplaces)
            if conflicts:
                raise ValueError(
                    'Duplicate marketplace name(s) not allowed: '
                    + ', '.join(sorted(conflicts))
                )

            # Convert to JSON-ready dicts for the JSON column.
            org.registered_marketplaces = [
                mp.model_dump(mode='json') for mp in validated_marketplaces
            ]

        # Update regular org fields
        for field, value in update_dict.items():
            setattr(org, field, value)

        # flush instead of commit - DbSessionInjector auto-commits at request end
        await self.db_session.flush()
        await self.db_session.refresh(org)
        return org
