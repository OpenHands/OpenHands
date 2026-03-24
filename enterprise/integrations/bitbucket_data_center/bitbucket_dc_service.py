from pydantic import SecretStr
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)
from openhands.integrations.service_types import ProviderType


class SaaSBitbucketDCService(BitbucketDCService):
    def __init__(
        self,
        user_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        external_auth_id: str | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
    ):
        logger.debug(
            f'SaaSBitbucketDCService created with user_id {user_id}, external_auth_id {external_auth_id}, external_auth_token {'set' if external_auth_token else 'None'}, token {'set' if token else 'None'}, external_token_manager {external_token_manager}'
        )
        super().__init__(
            user_id=user_id,
            external_auth_token=external_auth_token,
            external_auth_id=external_auth_id,
            token=token,
            external_token_manager=external_token_manager,
            base_domain=base_domain,
        )

        self.token_manager = TokenManager(external=external_token_manager)
        self.refresh = True

    async def get_latest_token(self) -> SecretStr | None:
        bitbucket_dc_token = None
        if self.external_auth_token:
            bitbucket_dc_token = SecretStr(
                await self.token_manager.get_idp_token(
                    self.external_auth_token.get_secret_value(),
                    idp=ProviderType.BITBUCKET_DATA_CENTER,
                )
            )
            logger.debug('Got Bitbucket DC token via external_auth_token')
        elif self.external_auth_id:
            offline_token = await self.token_manager.load_offline_token(
                self.external_auth_id
            )
            bitbucket_dc_token = SecretStr(
                await self.token_manager.get_idp_token_from_offline_token(
                    offline_token, ProviderType.BITBUCKET_DATA_CENTER
                )
            )
            logger.debug('Got Bitbucket DC token via external_auth_id')
        elif self.user_id:
            bitbucket_dc_token = SecretStr(
                await self.token_manager.get_idp_token_from_idp_user_id(
                    self.user_id, ProviderType.BITBUCKET_DATA_CENTER
                )
            )
            logger.debug('Got Bitbucket DC token via user_id')
        else:
            logger.warning('external_auth_token and user_id not set!')
        return bitbucket_dc_token
