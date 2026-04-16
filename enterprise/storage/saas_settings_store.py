from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.constants import LITE_LLM_API_URL
from server.logger import logger
from server.routes.org_models import OrgMemberLLMSettings
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from storage.database import a_session_maker
from storage.lite_llm_manager import LiteLlmManager, get_openhands_cloud_key_alias
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore
from storage.org_store import OrgStore
from storage.user import User
from storage.user_settings import UserSettings
from storage.user_store import UserStore

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.settings import Settings
from openhands.storage.settings.settings_store import SettingsStore
from openhands.utils.jsonpatch_compat import deep_merge
from openhands.utils.llm import is_openhands_model


@dataclass
class SaasSettingsStore(SettingsStore):
    user_id: str
    config: OpenHandsConfig

    async def _get_user_settings_by_keycloak_id_async(
        self, keycloak_user_id: str, session=None
    ) -> UserSettings | None:
        """
        Get UserSettings by keycloak_user_id (async version).

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
        return org.llm_api_key or org_member.llm_api_key

    @staticmethod
    def _get_persisted_agent_settings(item: Settings) -> dict[str, Any]:
        return item.agent_settings.model_dump(mode='json')

    async def load(self) -> Settings | None:
        user = await UserStore.get_user_by_id(self.user_id)
        if not user:
            logger.error(f'User not found for ID {self.user_id}')
            return None

        org_id = user.current_org_id
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
        merged_agent_settings = deep_merge(
            org_agent_settings.model_dump(mode='json'),
            member_agent_settings_diff,
        )
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

        return Settings(**kwargs)

    async def store(self, item: Settings):
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

            org_id = user.current_org_id

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

            if uses_managed_llm_key:
                await self._ensure_api_key(
                    item, str(org_id), openhands_type=is_openhands_model(llm_model)
                )

            effective_agent_settings_diff = self._get_persisted_agent_settings(item)
            org.agent_settings = deep_merge(
                OrgStore.get_agent_settings_from_org(org).model_dump(mode='json'),
                effective_agent_settings_diff,
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

            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
                if hasattr(org, key) and key not in {
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
                current_member_llm_api_key.get_secret_value()
                if current_member_llm_api_key
                else None
            )

            await OrgMemberStore.update_all_members_llm_settings_async(
                session,
                org_id,
                OrgMemberLLMSettings(
                    agent_settings_diff=effective_agent_settings_diff,
                    conversation_settings_diff=effective_conversation_diff,
                    llm_api_key=(
                        current_member_llm_api_key_raw
                        if not uses_managed_llm_key
                        else None
                    ),
                ),
            )

            if uses_managed_llm_key and current_member_llm_api_key is not None:
                # Managed/proxy key — store on this member but mark as org-managed
                org_member.llm_api_key = current_member_llm_api_key
                org_member.has_custom_llm_api_key = False
            elif current_member_llm_api_key_raw is not None:
                # BYOR: member supplied their own (non-managed) API key
                org_member.llm_api_key = current_member_llm_api_key
                org_member.has_custom_llm_api_key = True
            elif org_default_llm_api_key_raw is not None:
                # No member key, falling back to org default
                org_member.has_custom_llm_api_key = False

            await session.commit()

    @classmethod
    async def get_instance(
        cls,
        config: OpenHandsConfig,
        user_id: str,  # type: ignore[override]
    ) -> SaasSettingsStore:
        logger.debug(f'saas_settings_store.get_instance::{user_id}')
        return SaasSettingsStore(user_id, config)

    async def _ensure_api_key(
        self, item: Settings, org_id: str, openhands_type: bool = False
    ) -> None:
        """Generate and set the OpenHands API key for the given settings.

        First checks if an existing key exists for the user and verifies it
        is valid in LiteLLM. If valid, reuses it. Otherwise, generates a new key.
        """

        llm_api_key = item.agent_settings.llm.api_key

        # First, check if our current key is valid
        if llm_api_key and not await LiteLlmManager.verify_existing_key(
            llm_api_key.get_secret_value(),
            self.user_id,
            org_id,
            openhands_type=openhands_type,
        ):
            if openhands_type:
                generated_key = await LiteLlmManager.generate_key(
                    self.user_id,
                    org_id,
                    None,
                    {'type': 'openhands'},
                )
            else:
                # Must delete any existing key with the same alias first
                key_alias = get_openhands_cloud_key_alias(self.user_id, org_id)
                await LiteLlmManager.delete_key_by_alias(key_alias=key_alias)
                generated_key = await LiteLlmManager.generate_key(
                    self.user_id,
                    org_id,
                    key_alias,
                    None,
                )

            item.agent_settings.llm.api_key = SecretStr(generated_key)
            logger.info(
                'saas_settings_store:store:generated_openhands_key',
                extra={'user_id': self.user_id},
            )
