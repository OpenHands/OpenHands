import logging
from typing import AsyncGenerator

from fastapi import Request
from server.auth.auth_utils import user_verifier
from server.auth.domain_blocker import domain_blocker
from server.auth.token_manager import KeycloakUserInfo, TokenManager

from enterprise.server.auth.user_create.user_create_authorizer import (
    UserCreateAuthorization,
    UserCreateAuthorizer,
    UserCreateAuthorizerInjector,
)
from openhands.app_server.services.injector import InjectorState

logger = logging.getLogger(__name__)
token_manager = TokenManager()


class DefaultUserCreateAuthorizer(UserCreateAuthorizer):
    """Class determining whether a user may be created."""

    async def authorize_user_create(
        self, user_info: KeycloakUserInfo
    ) -> UserCreateAuthorization:
        user_id = user_info.sub
        email = user_info.email
        try:
            if not email:
                logger.warning(f'No email provided for user_id: {user_id}')
                return UserCreateAuthorization(success=False, detail='missing_email')

            if await domain_blocker.is_domain_blocked(email):
                logger.warning(
                    f'Blocked authentication attempt for email: {email}, user_id: {user_id}'
                )
                return UserCreateAuthorization(success=False, detail='email_blocked')

            has_duplicate = await token_manager.check_duplicate_base_email(
                email, user_id
            )
            if has_duplicate:
                logger.warning(
                    f'Blocked signup attempt for email {email} - duplicate base email found',
                    extra={'user_id': user_id, 'email': email},
                )
                return UserCreateAuthorization(success=False, detail='duplicate_email')

            username = user_info.preferred_username
            if user_verifier.is_active() and not user_verifier.is_user_allowed(
                username
            ):
                return UserCreateAuthorization(success=False, detail='blocked')

            return True
        except Exception:
            logger.exception('error authorizing user', extra={'user_id': user_id})
            return UserCreateAuthorization(success=False)


class DefaultUserCreateAuthorizerInjector(UserCreateAuthorizerInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[UserCreateAuthorizer, None]:
        yield DefaultUserCreateAuthorizer()
