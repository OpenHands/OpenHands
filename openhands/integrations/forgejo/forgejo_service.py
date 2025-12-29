from __future__ import annotations

import os

from pydantic import SecretStr

from openhands.integrations.forgejo.service import (
    ForgejoBranchesMixin,
    ForgejoFeaturesMixin,
    ForgejoMixinBase,
    ForgejoPRsMixin,
    ForgejoReposMixin,
    ForgejoResolverMixin,
)
from openhands.integrations.service_types import GitService
from openhands.utils.import_utils import get_impl


class ForgejoService(
    ForgejoBranchesMixin,
    ForgejoFeaturesMixin,
    ForgejoPRsMixin,
    ForgejoReposMixin,
    ForgejoResolverMixin,
    ForgejoMixinBase,
    GitService,
):
    """Assembled Forgejo service combining mixins by feature area."""

    def __init__(
        self,
        user_id: str | None = None,
        external_auth_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
        base_url: str | None = None,
    ) -> None:
        ForgejoMixinBase.__init__(
            self,
            user_id=user_id,
            external_auth_id=external_auth_id,
            external_auth_token=external_auth_token,
            token=token,
            external_token_manager=external_token_manager,
            base_domain=base_domain,
            base_url=base_url,
        )


forgejo_service_cls = os.environ.get(
    'OPENHANDS_FORGEJO_SERVICE_CLS',
    'openhands.integrations.forgejo.forgejo_service.ForgejoService',
)
ForgejoServiceImpl = get_impl(ForgejoService, forgejo_service_cls)
