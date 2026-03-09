import os

from pydantic import SecretStr

from openhands.integrations.bitbucket_data_center.service import (
    BitbucketDCBranchesMixin,
    BitbucketDCFeaturesMixin,
    BitbucketDCPRsMixin,
    BitbucketDCReposMixin,
    BitbucketDCResolverMixin,
)
from openhands.integrations.service_types import (
    GitService,
    InstallationsService,
    ProviderType,
)
from openhands.utils.import_utils import get_impl


class BitbucketDCService(
    BitbucketDCResolverMixin,
    BitbucketDCBranchesMixin,
    BitbucketDCFeaturesMixin,
    BitbucketDCPRsMixin,
    BitbucketDCReposMixin,
    GitService,
    InstallationsService,
):
    """Default implementation of GitService for Bitbucket data center integration.

    This is an extension point in OpenHands that allows applications to customize Bitbucket data center
    integration behavior. Applications can substitute their own implementation by:
    1. Creating a class that inherits from GitService
    2. Implementing all required methods
    3. Setting server_config.bitbucket_service_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.
    """

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
        self.base_domain = base_domain
        self.BASE_URL = f'https://{base_domain}/rest/api/1.0' if base_domain else ''

        if token:
            token_val = token.get_secret_value()
            if ':' not in token_val:
                token = SecretStr(f'x-token-auth:{token_val}')
            self.token = token

        # Derive user_id from token when not explicitly provided.
        if not user_id and token:
            token_val = token.get_secret_value()
            if not token_val.startswith('x-token-auth:'):
                user_id = token_val.split(':', 1)[0]

        self.user_id = user_id

    @property
    def provider(self) -> str:
        return ProviderType.BITBUCKET_DATA_CENTER.value


bitbucket_dc_service_cls = os.environ.get(
    'OPENHANDS_BITBUCKET_DATA_CENTER_SERVICE_CLS',
    'openhands.integrations.bitbucket_data_center.bitbucket_dc_service.BitbucketDCService',
)

# Lazy loading to avoid circular imports
_bitbucket_dc_service_impl = None


def get_bitbucket_dc_service_impl():
    """Get the BitBucket data center service implementation with lazy loading."""
    global _bitbucket_dc_service_impl
    if _bitbucket_dc_service_impl is None:
        _bitbucket_dc_service_impl = get_impl(
            BitbucketDCService, bitbucket_dc_service_cls
        )
    return _bitbucket_dc_service_impl


# For backward compatibility, provide the implementation as a property
class _BitbucketDCServiceImplProxy:
    """Proxy class to provide lazy loading for BitbucketDCServiceImpl."""

    def __getattr__(self, name):
        impl = get_bitbucket_dc_service_impl()
        return getattr(impl, name)

    def __call__(self, *args, **kwargs):
        impl = get_bitbucket_dc_service_impl()
        return impl(*args, **kwargs)


BitbucketDCServiceImpl: type[BitbucketDCService] = _BitbucketDCServiceImplProxy()  # type: ignore[assignment]
