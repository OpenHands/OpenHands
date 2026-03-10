from integrations.github.github_service import SaaSGitHubService
from pydantic import SecretStr
from server.auth.auth_utils import user_verifier

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.github_types import GitHubUser


def is_user_allowed(user_login: str):
    if user_verifier.is_active() and not user_verifier.is_user_allowed(user_login):
        logger.warning(f'GitHub user {user_login} not in allow list')
        return False

    return True


async def authenticate_github_user_id(auth_user_id: str) -> GitHubUser | None:
    logger.debug('Checking auth status for GitHub user')

    if not auth_user_id:
        logger.warning('No GitHub User ID provided')
        return None

    gh_service = SaaSGitHubService(user_id=auth_user_id)
    try:
        user: GitHubUser = await gh_service.get_user()
        if is_user_allowed(user.login):
            return user

        return None
    except:  # noqa: E722
        logger.warning("GitHub user doens't have valid token")
        return None


async def authenticate_github_user_token(access_token: str):
    if not access_token:
        logger.warning('No GitHub User ID provided')
        return None

    gh_service = SaaSGitHubService(token=SecretStr(access_token))
    try:
        user: GitHubUser = await gh_service.get_user()
        if is_user_allowed(user.login):
            return user

        return None
    except:  # noqa: E722
        logger.warning("GitHub user doens't have valid token")
        return None
