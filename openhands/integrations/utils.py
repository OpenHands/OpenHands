from pydantic import SecretStr

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl as AzureDevOpsService,
)
from openhands.integrations.bitbucket.bitbucket_service import BitBucketService
from openhands.integrations.github.github_service import GitHubService
from openhands.integrations.gitlab.gitlab_service import GitLabService
from openhands.integrations.provider import ProviderType


async def validate_provider_token(
    token: SecretStr, base_domain: str | None = None
) -> ProviderType | None:
    """Determine whether a token is for GitHub, GitLab, Bitbucket, or Azure DevOps by attempting to get user info from the services.

    Args:
        token: The token to check
        base_domain: Optional base domain for the service

    Returns:
        'github' if it's a GitHub token
        'gitlab' if it's a GitLab token
        'bitbucket' if it's a Bitbucket token
        'azure_devops' if it's an Azure DevOps token
        None if the token is invalid for all services
    """
    # Skip validation for empty tokens
    if token is None:
        return None  # type: ignore[unreachable]

    # Try GitHub first
    github_error = None
    try:
        github_service = GitHubService(token=token, base_domain=base_domain)
        await github_service.verify_access()
        return ProviderType.GITHUB
    except Exception as e:
        github_error = e

    # Try GitLab next
    gitlab_error = None
    try:
        gitlab_service = GitLabService(token=token, base_domain=base_domain)
        await gitlab_service.get_user()
        return ProviderType.GITLAB
    except Exception as e:
        gitlab_error = e

    # Try Bitbucket next
    bitbucket_error = None
    try:
        bitbucket_service = BitBucketService(token=token, base_domain=base_domain)
        await bitbucket_service.get_user()
        return ProviderType.BITBUCKET
    except Exception as e:
        bitbucket_error = e

    # Try Azure DevOps last
    azure_devops_error = None
    try:
        azure_devops_service = AzureDevOpsService(token=token, base_domain=base_domain)
        await azure_devops_service.get_user()
        return ProviderType.AZURE_DEVOPS
    except Exception as e:
        azure_devops_error = e

    logger.debug(
        f'Failed to validate token: {github_error} \n {gitlab_error} \n {bitbucket_error} \n {azure_devops_error}'
    )

    return None
