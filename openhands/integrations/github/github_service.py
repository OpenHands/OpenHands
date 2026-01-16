import os

from pydantic import SecretStr

from openhands.integrations.github.service import (
    GitHubBranchesMixin,
    GitHubFeaturesMixin,
    GitHubPRsMixin,
    GitHubReposMixin,
    GitHubResolverMixin,
)
from openhands.integrations.service_types import (
    BaseGitService,
    GitService,
    InstallationsService,
    ProviderType,
)
from openhands.utils.import_utils import get_impl


class GitHubService(
    GitHubBranchesMixin,
    GitHubFeaturesMixin,
    GitHubPRsMixin,
    GitHubReposMixin,
    GitHubResolverMixin,
    BaseGitService,
    GitService,
    InstallationsService,
):
    """
    Assembled GitHub service class combining mixins by feature area.

    TODO: This doesn't seem a good candidate for the get_impl() pattern. What are the abstract methods we should actually separate and implement here?
    This is an extension point in OpenHands that allows applications to customize GitHub
    integration behavior. Applications can substitute their own implementation by:
    1. Creating a class that inherits from GitService
    2. Implementing all required methods
    3. Setting server_config.github_service_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.
    """

    BASE_URL = 'https://api.github.com'
    GRAPHQL_URL = 'https://api.github.com/graphql'

    def __init__(
        self,
        user_id: str | None = None,
        external_auth_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
    ) -> None:
        self.user_id = user_id
        self.external_token_manager = external_token_manager

        if token:
            self.token = token

        if base_domain and base_domain != 'github.com':
            self.BASE_URL = f'https://{base_domain}/api/v3'
            self.GRAPHQL_URL = f'https://{base_domain}/api/graphql'

        self.external_auth_id = external_auth_id
        self.external_auth_token = external_auth_token

    @property
    def provider(self) -> str:
        return ProviderType.GITHUB.value


github_service_cls = os.environ.get(
    'OPENHANDS_GITHUB_SERVICE_CLS',
    'openhands.integrations.github.github_service.GitHubService',
)

# Lazy loading to avoid circular imports
_github_service_impl = None


def get_github_service_impl():
    """Get the GitHub service implementation with lazy loading."""
    global _github_service_impl
    if _github_service_impl is None:
        _github_service_impl = get_impl(GitHubService, github_service_cls)
    return _github_service_impl


# For backward compatibility, provide the implementation as a property
class _GitHubServiceImplProxy:
    """Proxy class to provide lazy loading for GithubServiceImpl."""

    def __getattr__(self, name):
        impl = get_github_service_impl()
        return getattr(impl, name)

    def __call__(self, *args, **kwargs):
        impl = get_github_service_impl()
        return impl(*args, **kwargs)


GithubServiceImpl: type[GitHubService] = _GitHubServiceImplProxy()  # type: ignore[assignment]
