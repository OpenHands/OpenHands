import os

from pydantic import SecretStr

from openhands.integrations.gitlab.service import (
    GitLabBranchesMixin,
    GitLabFeaturesMixin,
    GitLabPRsMixin,
    GitLabReposMixin,
    GitLabResolverMixin,
)
from openhands.integrations.service_types import (
    BaseGitService,
    GitService,
    ProviderType,
)
from openhands.utils.import_utils import get_impl


class GitLabService(
    GitLabBranchesMixin,
    GitLabFeaturesMixin,
    GitLabPRsMixin,
    GitLabReposMixin,
    GitLabResolverMixin,
    BaseGitService,
    GitService,
):
    """
    Assembled GitLab service class combining mixins by feature area.

    TODO: This doesn't seem a good candidate for the get_impl() pattern. What are the abstract methods we should actually separate and implement here?
    This is an extension point in OpenHands that allows applications to customize GitLab
    integration behavior. Applications can substitute their own implementation by:
    1. Creating a class that inherits from GitService
    2. Implementing all required methods
    3. Setting server_config.gitlab_service_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.
    """

    BASE_URL = 'https://gitlab.com/api/v4'
    GRAPHQL_URL = 'https://gitlab.com/api/graphql'

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
        self.external_auth_id = external_auth_id
        self.external_auth_token = external_auth_token

        if token:
            self.token = token

        if base_domain:
            # Check if protocol is already included
            if base_domain.startswith(('http://', 'https://')):
                # Use the provided protocol
                self.BASE_URL = f'{base_domain}/api/v4'
                self.GRAPHQL_URL = f'{base_domain}/api/graphql'
            else:
                # Default to https if no protocol specified
                self.BASE_URL = f'https://{base_domain}/api/v4'
                self.GRAPHQL_URL = f'https://{base_domain}/api/graphql'

    @property
    def provider(self) -> str:
        return ProviderType.GITLAB.value


gitlab_service_cls = os.environ.get(
    'OPENHANDS_GITLAB_SERVICE_CLS',
    'openhands.integrations.gitlab.gitlab_service.GitLabService',
)
GitLabServiceImpl = get_impl(GitLabService, gitlab_service_cls)
