"""Bootstrap a configured default OpenHands organization for OHE installs."""

import os
import re
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from pydantic import SecretStr
from server.constants import ROLE_MEMBER, ROLE_OWNER
from server.routes.org_models import OrgNameExistsError
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore
from storage.org_service import OrgService
from storage.org_store import OrgStore
from storage.role_store import RoleStore
from storage.user import User
from storage.user_store import UserStore

from openhands.app_server.utils.logger import openhands_logger as logger

_TRUTHY_VALUES = {'1', 'true', 'yes', 'on'}
_EMAIL_SPLIT_RE = re.compile(r'[\s,;]+')


class MembershipOutcome(Enum):
    """Result of ensuring a user's membership in the default org."""

    ADDED = 'added'
    PROMOTED = 'promoted'
    UNCHANGED = 'unchanged'


@dataclass(frozen=True)
class DefaultOrgConfig:
    enabled: bool
    org_name: str
    owner_emails: frozenset[str]
    auto_add_users: bool


def _env_value(name: str, *aliases: str, default: str = '') -> str:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value is not None:
            return value
    return default


def _env_truthy(name: str, *aliases: str, default: str = 'false') -> bool:
    return _env_value(name, *aliases, default=default).strip().lower() in _TRUTHY_VALUES


def _parse_owner_emails(raw_value: str) -> frozenset[str]:
    return frozenset(
        token.strip().lower()
        for token in _EMAIL_SPLIT_RE.split(raw_value)
        if token.strip()
    )


def get_default_org_config() -> DefaultOrgConfig:
    return DefaultOrgConfig(
        enabled=_env_truthy(
            'OPENHANDS_DEFAULT_ORG_ENABLED',
            'OH_DEFAULT_ORG_ENABLED',
        ),
        org_name=_env_value(
            'OPENHANDS_DEFAULT_ORG_NAME',
            'OH_DEFAULT_ORG_NAME',
        ).strip(),
        owner_emails=_parse_owner_emails(
            _env_value(
                'OPENHANDS_DEFAULT_ORG_OWNER_EMAILS',
                'OH_DEFAULT_ORG_OWNER_EMAILS',
            )
        ),
        auto_add_users=_env_truthy(
            'OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS',
            'OH_DEFAULT_ORG_AUTO_ADD_USERS',
        ),
    )


class DefaultOrgBootstrapService:
    """Apply additive default organization membership rules on user login."""

    @staticmethod
    async def apply_for_user(user: User, is_new_user: bool) -> User:
        config = get_default_org_config()
        if not config.enabled:
            return user

        if not config.org_name:
            logger.warning('default_org_bootstrap:missing_org_name')
            return user

        if not config.owner_emails:
            logger.warning(
                'default_org_bootstrap:missing_owner_emails',
                extra={'org_name': config.org_name},
            )
            return user

        user_email = (user.email or '').strip().lower()
        if not user_email:
            logger.warning(
                'default_org_bootstrap:user_missing_email',
                extra={'user_id': str(user.id), 'org_name': config.org_name},
            )
            return user

        is_configured_owner = user_email in config.owner_emails
        org, org_created_by_user = await DefaultOrgBootstrapService._get_or_create_org(
            config=config,
            current_user=user,
            current_user_is_owner=is_configured_owner,
        )

        if not org:
            logger.info(
                'default_org_bootstrap:pending_owner_login',
                extra={'user_id': str(user.id), 'org_name': config.org_name},
            )
            return user

        outcome = await DefaultOrgBootstrapService._ensure_membership(
            org=org,
            user=user,
            is_configured_owner=is_configured_owner,
            auto_add_users=config.auto_add_users,
        )

        # Move the user into the default org on their first join (signup,
        # first auto-add, or owner-created org); never on later logins, so a
        # deliberate switch back to the personal workspace sticks.
        should_set_current_org = outcome is MembershipOutcome.ADDED or (
            (is_new_user or org_created_by_user)
            and await OrgMemberStore.get_org_member(org.id, user.id) is not None
        )
        if should_set_current_org:
            updated_user = await UserStore.update_current_org(str(user.id), org.id)
            if updated_user:
                logger.info(
                    'default_org_bootstrap:set_current_org',
                    extra={
                        'user_id': str(user.id),
                        'org_id': str(org.id),
                        'is_new_user': is_new_user,
                    },
                )
                return await UserStore.get_user_by_id(str(user.id)) or updated_user

        return await UserStore.get_user_by_id(str(user.id)) or user

    @staticmethod
    async def _get_or_create_org(
        config: DefaultOrgConfig,
        current_user: User,
        current_user_is_owner: bool,
    ) -> tuple[Org | None, bool]:
        """Find or lazily create the default org.

        Returns (org, created_by_current_user) where the flag is True only
        when the org was created in this call with the current user as its
        owner — not on adoption, race recovery, or creation under a
        different configured owner.
        """
        org = await OrgStore.get_org_by_name(config.org_name)
        if org:
            if DefaultOrgBootstrapService._is_personal_workspace_org(org):
                logger.warning(
                    'default_org_bootstrap:refusing_personal_workspace_org',
                    extra={'org_id': str(org.id), 'org_name': org.name},
                )
                return None, False

            logger.info(
                'default_org_bootstrap:adopting_existing_org',
                extra={'org_id': str(org.id), 'org_name': org.name},
            )
            return org, False

        owner_user = current_user if current_user_is_owner else None
        if owner_user is None:
            owner_user = await DefaultOrgBootstrapService._find_existing_owner_user(
                config.owner_emails
            )

        if owner_user is None:
            return None, False

        owner_email = (owner_user.email or '').strip().lower()
        try:
            created_org = await OrgService.create_org_with_owner(
                name=config.org_name,
                contact_name=owner_email or 'Default organization owner',
                contact_email=owner_email,
                user_id=str(owner_user.id),
            )
            return created_org, owner_user.id == current_user.id
        except OrgNameExistsError:
            # A concurrent login may have created the org after our first lookup.
            org = await OrgStore.get_org_by_name(config.org_name)
            if org and DefaultOrgBootstrapService._is_personal_workspace_org(org):
                logger.warning(
                    'default_org_bootstrap:refusing_personal_workspace_org',
                    extra={'org_id': str(org.id), 'org_name': org.name},
                )
                return None, False
            return org, False

    @staticmethod
    def _is_personal_workspace_org(org: Org) -> bool:
        return org.name == f'user_{org.id}_org'

    @staticmethod
    async def _find_existing_owner_user(owner_emails: frozenset[str]) -> User | None:
        for owner_email in sorted(owner_emails):
            owner_user = await UserStore.get_user_by_email(owner_email)
            if owner_user:
                return owner_user
        return None

    @staticmethod
    async def _ensure_membership(
        org: Org,
        user: User,
        is_configured_owner: bool,
        auto_add_users: bool,
    ) -> MembershipOutcome:
        membership = await OrgMemberStore.get_org_member(org.id, user.id)
        desired_role_name = ROLE_OWNER if is_configured_owner else ROLE_MEMBER

        if membership:
            if is_configured_owner:
                return await DefaultOrgBootstrapService._promote_to_owner_if_needed(
                    org_id=org.id,
                    user_id=user.id,
                    membership=membership,
                )
            return MembershipOutcome.UNCHANGED

        if not is_configured_owner and not auto_add_users:
            return MembershipOutcome.UNCHANGED

        role = await RoleStore.get_role_by_name(desired_role_name)
        if not role:
            logger.error(
                'default_org_bootstrap:role_not_found',
                extra={'role': desired_role_name, 'org_id': str(org.id)},
            )
            return MembershipOutcome.UNCHANGED

        llm_api_key = await DefaultOrgBootstrapService._create_member_litellm_api_key(
            org_id=org.id,
            user_id=user.id,
        )

        await OrgMemberStore.add_user_to_org(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key=llm_api_key,
            status='active',
            agent_settings_diff={},
            conversation_settings_diff={},
        )
        logger.info(
            'default_org_bootstrap:member_added',
            extra={
                'user_id': str(user.id),
                'org_id': str(org.id),
                'role': desired_role_name,
            },
        )
        return MembershipOutcome.ADDED

    @staticmethod
    async def _create_member_litellm_api_key(org_id: UUID, user_id: UUID) -> str:
        """Provision org-scoped LiteLLM access and return the member API key."""
        settings = await OrgService.create_litellm_integration(org_id, str(user_id))
        llm_api_key = settings.agent_settings.llm.api_key
        if isinstance(llm_api_key, SecretStr):
            return llm_api_key.get_secret_value()
        return llm_api_key or ''

    @staticmethod
    async def _promote_to_owner_if_needed(
        org_id: UUID,
        user_id: UUID,
        membership: OrgMember,
    ) -> MembershipOutcome:
        current_role = await RoleStore.get_role_by_id(membership.role_id)
        if current_role and current_role.name == ROLE_OWNER:
            return MembershipOutcome.UNCHANGED

        owner_role = await RoleStore.get_role_by_name(ROLE_OWNER)
        if not owner_role:
            logger.error(
                'default_org_bootstrap:role_not_found',
                extra={'role': ROLE_OWNER, 'org_id': str(org_id)},
            )
            return MembershipOutcome.UNCHANGED

        await OrgMemberStore.update_user_role_in_org(
            org_id=org_id,
            user_id=user_id,
            role_id=owner_role.id,
            status='active',
        )
        logger.info(
            'default_org_bootstrap:member_promoted_to_owner',
            extra={'user_id': str(user_id), 'org_id': str(org_id)},
        )
        return MembershipOutcome.PROMOTED
