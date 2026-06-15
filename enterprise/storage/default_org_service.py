"""Bootstrap a default OpenHands organization for OHE installs."""

import os
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from pydantic import SecretStr
from server.constants import ROLE_MEMBER
from server.routes.org_models import OrgNameExistsError
from storage.org import Org
from storage.org_member_store import OrgMemberStore
from storage.org_service import OrgService
from storage.org_store import OrgStore
from storage.role_store import RoleStore
from storage.user import User
from storage.user_store import UserStore

from openhands.app_server.utils.logger import openhands_logger as logger

_TRUTHY_VALUES = {'1', 'true', 'yes', 'on'}

DEFAULT_ORG_NAME = 'Enterprise Org'


class MembershipOutcome(Enum):
    """Result of ensuring a user's membership in the default org."""

    ADDED = 'added'
    UNCHANGED = 'unchanged'


@dataclass(frozen=True)
class DefaultOrgConfig:
    enabled: bool
    org_name: str
    auto_add_users: bool
    hide_personal_workspaces: bool


def _env_value(name: str, *aliases: str, default: str = '') -> str:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value is not None:
            return value
    return default


def _env_truthy(name: str, *aliases: str, default: str = 'false') -> bool:
    return _env_value(name, *aliases, default=default).strip().lower() in _TRUTHY_VALUES


def get_default_org_config() -> DefaultOrgConfig:
    return DefaultOrgConfig(
        enabled=_env_truthy(
            'OPENHANDS_DEFAULT_ORG_ENABLED',
            'OH_DEFAULT_ORG_ENABLED',
        ),
        # Only used as the initial name when the org is first created; owners
        # can rename it in the app afterwards (the org is tracked by its
        # is_default flag, not its name).
        org_name=_env_value(
            'OPENHANDS_DEFAULT_ORG_NAME',
            'OH_DEFAULT_ORG_NAME',
        ).strip()
        or DEFAULT_ORG_NAME,
        auto_add_users=_env_truthy(
            'OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS',
            'OH_DEFAULT_ORG_AUTO_ADD_USERS',
        ),
        # Same env var the web client config injector reads to hide personal
        # workspaces in the UI; here it additionally moves users out of a
        # personal current org on login (there is no visible way to be there).
        hide_personal_workspaces=_env_truthy('HIDE_PERSONAL_WORKSPACES'),
    )


def is_personal_workspace_org(org: Org) -> bool:
    """A personal workspace org is auto-created and named after its user."""
    return org.name == f'user_{org.id}_org'


class DefaultOrgBootstrapService:
    """Apply additive default organization membership rules on user login.

    The first user to sign in after the feature is enabled creates the
    default org and becomes its owner. The org is tracked by the is_default
    flag, so it can be freely renamed in the app; ownership and membership
    are likewise managed in the app afterwards (this service never demotes
    or removes anyone).
    """

    @staticmethod
    async def apply_for_user(user: User, is_new_user: bool) -> User:
        config = get_default_org_config()
        if not config.enabled:
            return user

        user_email = (user.email or '').strip().lower()
        if not user_email:
            logger.warning(
                'default_org_bootstrap:user_missing_email',
                extra={'user_id': str(user.id)},
            )
            return user

        org, org_created_by_user = await DefaultOrgBootstrapService._get_or_create_org(
            config=config,
            current_user=user,
        )

        if not org:
            return user

        outcome = await DefaultOrgBootstrapService._ensure_membership(
            org=org,
            user=user,
            auto_add_users=config.auto_add_users,
        )

        # Move the user into the default org on their first join (signup,
        # first auto-add, or having just created the org); never on later
        # logins, so a deliberate switch back to the personal workspace
        # sticks.
        should_set_current_org = outcome is MembershipOutcome.ADDED or (
            (is_new_user or org_created_by_user)
            and await OrgMemberStore.get_org_member(org.id, user.id) is not None
        )

        # When personal workspaces are hidden there is no visible way to be
        # in one, so members parked on their personal org (e.g. from before
        # the flag was enabled) are moved on every login.
        if (
            not should_set_current_org
            and config.hide_personal_workspaces
            and user.current_org_id == user.id
        ):
            should_set_current_org = (
                await OrgMemberStore.get_org_member(org.id, user.id) is not None
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
    ) -> tuple[Org | None, bool]:
        """Find or lazily create the default org.

        Returns (org, created_by_current_user) where the flag is True only
        when the org was created in this call with the current user as its
        owner — not on adoption or race recovery.
        """
        org = await OrgStore.get_default_org()
        if org:
            return org, False

        # An install can predate the is_default flag (an org bootstrapped by
        # an earlier name-keyed version). Team orgs cannot be created by
        # users on OHE installs, so a sole team org is the default org:
        # adopt it instead of creating a duplicate.
        team_orgs = await OrgStore.list_team_orgs(limit=2)
        if len(team_orgs) == 1:
            adopted = await OrgStore.mark_org_as_default(team_orgs[0].id)
            if adopted is None:
                # A concurrent login flagged an org first; use that one.
                return await OrgStore.get_default_org(), False
            logger.info(
                'default_org_bootstrap:adopting_existing_org',
                extra={'org_id': str(adopted.id), 'org_name': adopted.name},
            )
            return adopted, False
        if len(team_orgs) > 1:
            logger.warning(
                'default_org_bootstrap:ambiguous_existing_orgs',
                extra={'team_org_count': len(team_orgs)},
            )
            return None, False

        # No team org exists yet: the current user is the first one through
        # the door — create the org with them as its owner.
        user_email = (current_user.email or '').strip().lower()
        try:
            created_org = await OrgService.create_org_with_owner(
                name=config.org_name,
                contact_name=user_email or 'Default organization owner',
                contact_email=user_email,
                user_id=str(current_user.id),
            )
        except OrgNameExistsError:
            # A concurrent login may have created the org after our lookup.
            org = await OrgStore.get_default_org()
            if org:
                return org, False
            org = await OrgStore.get_org_by_name(config.org_name)
            if org and not is_personal_workspace_org(org):
                return await OrgStore.mark_org_as_default(org.id), False
            return None, False

        flagged_org = await OrgStore.mark_org_as_default(created_org.id)
        if flagged_org is None:
            # Another org got flagged concurrently; defer to it.
            return await OrgStore.get_default_org(), False
        logger.info(
            'default_org_bootstrap:org_created',
            extra={
                'org_id': str(flagged_org.id),
                'org_name': flagged_org.name,
                'owner_user_id': str(current_user.id),
            },
        )
        return flagged_org, True

    @staticmethod
    async def _ensure_membership(
        org: Org,
        user: User,
        auto_add_users: bool,
    ) -> MembershipOutcome:
        membership = await OrgMemberStore.get_org_member(org.id, user.id)
        if membership:
            return MembershipOutcome.UNCHANGED

        if not auto_add_users:
            return MembershipOutcome.UNCHANGED

        role = await RoleStore.get_role_by_name(ROLE_MEMBER)
        if not role:
            logger.error(
                'default_org_bootstrap:role_not_found',
                extra={'role': ROLE_MEMBER, 'org_id': str(org.id)},
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
                'role': ROLE_MEMBER,
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
