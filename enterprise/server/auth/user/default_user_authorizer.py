import logging
import re
from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import Request
from pydantic import BaseModel, Field
from server.auth.domain_blocker import domain_blocker
from server.auth.token_manager import KeycloakUserInfo, TokenManager
from server.auth.user.user_authorizer import (
    UserAuthorization,
    UserAuthorizer,
    UserAuthorizerInjector,
)

from openhands.app_server.services.injector import InjectorState
from openhands.integrations.service_types import ProviderType

logger = logging.getLogger(__name__)
token_manager = TokenManager()


class UserMatch(BaseModel):
    email_pattern: str | None = None
    provider: ProviderType | None = None

    def match(self, user_info: KeycloakUserInfo) -> bool:
        return self.match_email(user_info.email) and self.match_provider(
            user_info.identity_provider
        )

    def match_email(self, email: str | None) -> bool:
        if not self.email_pattern:
            return True
        if not email:
            return False
        return bool(re.match(self.email_pattern, email))

    def match_provider(self, identity_provider: str | None) -> bool:
        if not self.provider:
            return True
        if not identity_provider:
            return False
        return self.provider.value == identity_provider


@dataclass
class DefaultUserAuthorizer(UserAuthorizer):
    """Class determining whether a user may be authorized."""

    prevent_duplicates: bool
    whitelist: list[UserMatch]
    blacklist: list[UserMatch]

    async def authorize_user(self, user_info: KeycloakUserInfo) -> UserAuthorization:
        user_id = user_info.sub
        email = user_info.email
        try:
            if not email:
                logger.warning(f'No email provided for user_id: {user_id}')
                return UserAuthorization(success=False, error_detail='missing_email')

            if self.prevent_duplicates:
                has_duplicate = await token_manager.check_duplicate_base_email(
                    email, user_id
                )
                if has_duplicate:
                    logger.warning(
                        f'Blocked signup attempt for email {email} - duplicate base email found',
                        extra={'user_id': user_id, 'email': email},
                    )
                    return UserAuthorization(
                        success=False, error_detail='duplicate_email'
                    )

            if DefaultUserAuthorizer._has_match(self.whitelist, user_info):
                return UserAuthorization(success=True)

            if DefaultUserAuthorizer._has_match(self.blacklist, user_info):
                return UserAuthorization(success=False, error_detail='blocked')

            if await domain_blocker.is_domain_blocked(email):
                logger.warning(
                    f'Blocked authentication attempt for email: {email}, user_id: {user_id}'
                )
                return UserAuthorization(success=False, error_detail='blocked')

            return UserAuthorization(success=True)
        except Exception:
            logger.exception('error authorizing user', extra={'user_id': user_id})
            return UserAuthorization(success=False)

    @staticmethod
    def _has_match(
        matches: list[UserMatch] | None, user_info: KeycloakUserInfo
    ) -> bool:
        if not matches:
            return False
        for user_match in matches:
            if user_match.match(user_info):
                return True
        return False


class DefaultUserAuthorizerInjector(UserAuthorizerInjector):
    prevent_duplicates: bool = Field(
        default=True, description='Whether duplicate emails (containing +) are filtered'
    )
    whitelist: list[UserMatch] = Field(
        default_factory=list, description='Whitelist for emails'
    )
    blacklist: list[UserMatch] = Field(
        default_factory=list, description=' Blacklist for emails'
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[UserAuthorizer, None]:
        yield DefaultUserAuthorizer(
            prevent_duplicates=self.prevent_duplicates,
            whitelist=self.whitelist,
            blacklist=self.blacklist,
        )
