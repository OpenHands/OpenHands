"""Resolve which OpenHands organization workspace a resolver conversation should be created in.

This module provides a reusable utility for routing resolver conversations
(GitHub, GitLab, Bitbucket, Slack, etc.) to the correct OpenHands organization
workspace based on claimed Git organizations.
"""

from uuid import UUID

from storage.org_git_claim_store import OrgGitClaimStore
from storage.org_member_store import OrgMemberStore

from openhands.core.logger import openhands_logger as logger


async def resolve_org_for_repo(
    provider: str,
    full_repo_name: str,
    keycloak_user_id: str | None = None,
) -> UUID | None:
    """Determine the OpenHands org_id for a resolver conversation.

    If the repo's git organization is claimed by an OpenHands org, returns the
    claiming org's ID. When keycloak_user_id is provided, also verifies the user
    is a member of that org.

    Args:
        provider: Git provider name ("github", "gitlab", "bitbucket")
        full_repo_name: Full repository name (e.g., "OpenHands/foo")
        keycloak_user_id: The user's Keycloak UUID string (optional). If provided,
            membership is verified before returning the org_id.

    Returns:
        The org_id if the repo's org is claimed (and user is a member when
        keycloak_user_id is provided), else None
    """
    git_org = full_repo_name.split('/')[0].lower()

    try:
        claim = await OrgGitClaimStore.get_claim_by_provider_and_git_org(
            provider, git_org
        )
        if not claim:
            logger.debug(
                f'[OrgResolver] No claim found for {provider}/{git_org}',
            )
            return None

        # Skip membership check if no user_id provided
        if keycloak_user_id is None:
            logger.info(
                f'[OrgResolver] Resolved org {claim.org_id} '
                f'for {provider}/{git_org} (no user membership check)',
            )
            return claim.org_id

        member = await OrgMemberStore.get_org_member(
            claim.org_id, UUID(keycloak_user_id)
        )
        if not member:
            logger.debug(
                f'[OrgResolver] User {keycloak_user_id} is not a member of org '
                f'{claim.org_id} (claimed {provider}/{git_org}). '
                f'Falling back to personal workspace.',
            )
            return None

        logger.info(
            f'[OrgResolver] Routing conversation to org {claim.org_id} '
            f'for {provider}/{git_org} (user {keycloak_user_id})',
        )
        return claim.org_id
    except Exception as e:
        logger.error(
            f'[OrgResolver] Error resolving org for {provider}/{git_org}: {e}',
            exc_info=True,
        )
        return None
