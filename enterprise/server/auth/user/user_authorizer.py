import logging
from abc import ABC, abstractmethod

from fastapi import Depends
from pydantic import BaseModel
from server.auth.token_manager import KeycloakUserInfo

from openhands.agent_server.env_parser import from_env
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin

logger = logging.getLogger(__name__)


class UserAuthorizationResponse(BaseModel):
    success: bool
    error_detail: str | None = None


class UserAuthorizer(ABC):
    """Class determining whether a user may be authorized."""

    @abstractmethod
    async def authorize_user(
        self, user_info: KeycloakUserInfo
    ) -> UserAuthorizationResponse:
        """Determine whether the info given is permitted."""


class UserAuthorizerInjector(DiscriminatedUnionMixin, Injector[UserAuthorizer], ABC):
    pass


def depends_user_authorizer():
    from server.auth.user.default_user_authorizer import (
        DefaultUserAuthorizerInjector,
    )

    try:
        injector: UserAuthorizerInjector = from_env(
            UserAuthorizerInjector, 'OH_USER_AUTHORIZER'
        )
    except Exception as ex:
        print(ex)
        logger.info('Using default UserAuthorizer')
        injector = DefaultUserAuthorizerInjector()

    return Depends(injector.depends)
