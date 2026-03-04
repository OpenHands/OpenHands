import logging
from abc import ABC, abstractmethod

from fastapi import Depends
from pydantic import BaseModel
from server.auth.token_manager import KeycloakUserInfo

from openhands.agent_server.env_parser import from_env
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin

logger = logging.getLogger(__name__)


class UserCreateAuthorization(BaseModel):
    success: bool
    error_detail: str | None = None


class UserCreateAuthorizer(ABC):
    """Class determining whether a user may be created."""

    @abstractmethod
    async def authorize_user_create(
        self, user_info: KeycloakUserInfo
    ) -> UserCreateAuthorization:
        """Determine whether the info given is permitted when creating an account."""


class UserCreateAuthorizerInjector(
    DiscriminatedUnionMixin, Injector[UserCreateAuthorizer], ABC
):
    pass


def depends_user_create_authorizer():
    try:
        injector: UserCreateAuthorizerInjector = from_env(
            UserCreateAuthorizerInjector, 'OH_USER_CREATE_AUTHORIZER'
        )
    except Exception:
        logger.info('Using default UserCreateAuthorizer')

        from server.auth.user_create.default_user_create_authorizer import (
            DefaultUserCreateAuthorizerInjector,
        )

        injector = DefaultUserCreateAuthorizerInjector()

    return Depends(injector.depends)
