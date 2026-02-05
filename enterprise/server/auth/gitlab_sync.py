import asyncio

from pydantic import SecretStr
from sqlalchemy import select

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import ProviderType
from openhands.server.types import AppMode


async def _user_has_gitlab_provider(user_id: str) -> bool:
    """Check if the user has authenticated with GitLab.

    Args:
        user_id: The Keycloak user ID

    Returns:
        True if the user has a GitLab provider token, False otherwise
    """
    # Lazy import to avoid circular dependency issues at module load time
    from storage.auth_tokens import AuthTokens
    from storage.database import a_session_maker

    async with a_session_maker() as session:
        result = await session.execute(
            select(AuthTokens).where(
                AuthTokens.keycloak_user_id == user_id,
                AuthTokens.identity_provider == ProviderType.GITLAB.value,
            )
        )
        return result.scalars().first() is not None


def schedule_gitlab_repo_sync(
    user_id: str, keycloak_access_token: SecretStr | None = None
) -> None:
    """Schedule a background sync of GitLab repositories and webhook tracking.

    Because the outer call is already a background task, we instruct the service
    to store repository data synchronously (store_in_background=False) to avoid
    nested background tasks while still keeping the overall operation async.

    The sync is only performed if the user has authenticated with GitLab.
    """

    async def _run():
        try:
            # Check if the user has a GitLab provider token before syncing
            if not await _user_has_gitlab_provider(user_id):
                logger.debug(
                    'gitlab_repo_sync_skipped: user has no GitLab provider',
                    extra={'user_id': user_id},
                )
                return

            # Lazy import to avoid circular dependency:
            # middleware -> gitlab_sync -> integrations.gitlab.gitlab_service
            # -> openhands.integrations.gitlab.gitlab_service -> get_impl
            # -> integrations.gitlab.gitlab_service (circular)
            from integrations.gitlab.gitlab_service import SaaSGitLabService

            service = SaaSGitLabService(
                external_auth_id=user_id, external_auth_token=keycloak_access_token
            )
            await service.get_all_repositories(
                'pushed', AppMode.SAAS, store_in_background=False
            )
        except Exception:
            logger.warning('gitlab_repo_sync_failed', exc_info=True)

    asyncio.create_task(_run())
