import os

from pydantic import SecretStr

from openhands.integrations.bitbucket.service import (
    BitBucketBranchesMixin,
    BitBucketFeaturesMixin,
    BitBucketPRsMixin,
    BitBucketReposMixin,
)
from openhands.integrations.service_types import (
    GitService,
    InstallationsService,
    ProviderType,
)
from openhands.utils.import_utils import get_impl


class BitBucketService(
    BitBucketReposMixin,
    BitBucketBranchesMixin,
    BitBucketPRsMixin,
    BitBucketFeaturesMixin,
    GitService,
    InstallationsService,
):
    """Default implementation of GitService for Bitbucket integration.

    This is an extension point in OpenHands that allows applications to customize Bitbucket
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
        self.base_domain = base_domain or 'bitbucket.org'

        if token:
            self.token = token
        if base_domain:
            self.BASE_URL = f'https://api.{base_domain}/2.0'

    @property
    def provider(self) -> str:
        return ProviderType.BITBUCKET.value


bitbucket_service_cls = os.environ.get(
    'OPENHANDS_BITBUCKET_SERVICE_CLS',
    'openhands.integrations.bitbucket.bitbucket_service.BitBucketService',
)

# Lazy loading to avoid circular imports
_bitbucket_service_impl = None


def get_bitbucket_service_impl():
    """Get the BitBucket service implementation with lazy loading."""
    global _bitbucket_service_impl
    if _bitbucket_service_impl is None:
        _bitbucket_service_impl = get_impl(BitBucketService, bitbucket_service_cls)
    return _bitbucket_service_impl


# For backward compatibility, provide the implementation as a property
class _BitBucketServiceImplProxy:
    """Proxy class to provide lazy loading for BitBucketServiceImpl."""

    def __getattr__(self, name):
        impl = get_bitbucket_service_impl()
        return getattr(impl, name)

    def __call__(self, *args, **kwargs):
        impl = get_bitbucket_service_impl()
        return impl(*args, **kwargs)


BitBucketServiceImpl: type[BitBucketService] = _BitBucketServiceImplProxy()  # type: ignore[assignment]
