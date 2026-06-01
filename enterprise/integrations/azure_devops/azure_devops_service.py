from pydantic import SecretStr
from server.auth.constants import AZURE_DEVOPS_ORGANIZATION
from server.auth.token_manager import TokenManager

from openhands.app_server.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsService,
)
from openhands.app_server.integrations.service_types import ProviderType
from openhands.app_server.utils.logger import openhands_logger as logger


class SaaSAzureDevOpsService(AzureDevOpsService):
    def __init__(
        self,
        user_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        external_auth_id: str | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
    ):
        configured_org = AZURE_DEVOPS_ORGANIZATION or None
        super().__init__(
            user_id=user_id,
            external_auth_token=external_auth_token,
            external_auth_id=external_auth_id,
            token=token,
            external_token_manager=external_token_manager,
            base_domain=base_domain or configured_org,
        )

        self.external_auth_token = external_auth_token
        self.external_auth_id = external_auth_id
        self.token_manager = TokenManager(external=external_token_manager)
        self.refresh = True

    async def get_latest_token(self) -> SecretStr | None:
        azure_devops_token = None
        if self.external_auth_token:
            azure_devops_token = SecretStr(
                await self.token_manager.get_idp_token(
                    self.external_auth_token.get_secret_value(),
                    idp=ProviderType.AZURE_DEVOPS,
                )
            )
            logger.debug('Got Azure DevOps token via external_auth_token')
        elif self.external_auth_id:
            offline_token = await self.token_manager.load_offline_token(
                self.external_auth_id
            )
            azure_devops_token_str: str | None = (
                await self.token_manager.get_idp_token_from_offline_token(
                    offline_token, ProviderType.AZURE_DEVOPS
                )
                if offline_token
                else None
            )
            azure_devops_token = (
                SecretStr(azure_devops_token_str) if azure_devops_token_str else None
            )
            logger.debug('Got Azure DevOps token via external_auth_id')
        elif self.user_id:
            azure_devops_token_str = (
                await self.token_manager.get_idp_token_from_idp_user_id(
                    self.user_id, ProviderType.AZURE_DEVOPS
                )
            )
            azure_devops_token = (
                SecretStr(azure_devops_token_str) if azure_devops_token_str else None
            )
            logger.debug('Got Azure DevOps token via user_id')
        else:
            logger.warning('external_auth_token and user_id not set!')
        if azure_devops_token:
            self.token = azure_devops_token
        return azure_devops_token

    async def get_installations(self) -> list[str]:
        if self.organization:
            return [self.organization]

        profile_url = (
            'https://app.vssps.visualstudio.com/_apis/profile/profiles/me'
            '?api-version=7.1-preview.3'
        )
        profile, _ = await self._make_request(profile_url)
        member_id = profile.get('id')
        if not member_id:
            return []

        accounts_url = (
            'https://app.vssps.visualstudio.com/_apis/accounts'
            f'?memberId={member_id}&api-version=7.1-preview.1'
        )
        accounts, _ = await self._make_request(accounts_url)
        account_values = accounts.get('value') or accounts.get('accounts') or []
        return [
            account['accountName']
            for account in account_values
            if account.get('accountName')
        ]

    async def get_paginated_repos(
        self,
        page: int,
        per_page: int,
        sort: str,
        installation_id: str | None,
        query: str | None = None,
    ):
        if installation_id:
            self.organization = installation_id
        elif not self.organization:
            installations = await self.get_installations()
            if installations:
                self.organization = installations[0]

        return await super().get_paginated_repos(
            page=page,
            per_page=per_page,
            sort=sort,
            installation_id=installation_id,
            query=query,
        )
