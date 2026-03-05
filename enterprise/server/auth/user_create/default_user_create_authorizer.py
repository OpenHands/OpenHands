from dataclasses import dataclass
import logging
import re
from typing import AsyncGenerator

from fastapi import Request
from openhands.integrations.service_types import ProviderType
from server.auth.domain_blocker import domain_blocker
from server.auth.token_manager import KeycloakUserInfo, TokenManager
from pydantic import BaseModel, Field
from server.auth.user_create.user_create_authorizer import (
    UserCreateAuthorization,
    UserCreateAuthorizer,
    UserCreateAuthorizerInjector,
)
from openhands.app_server.services.injector import InjectorState

logger = logging.getLogger(__name__)
token_manager = TokenManager()


class UserMatch(BaseModel):
    email_pattern: str | None = None
    provider: ProviderType | None = None

    def match(self, user_info: KeycloakUserInfo) -> bool:
        return self.match_email(user_info.email) and self.match_provider(user_info.identity_provider)

    def match_email(self, email: str) -> bool:
        if not self.email_pattern:
            return True
        return re.match(self.email_pattern, email)

    def match_provider(self, identity_provider: str):
        if not self.provider:
            return True
        return self.provider.value == identity_provider


@dataclass
class DefaultUserCreateAuthorizer(UserCreateAuthorizer):
    """Class determining whether a user may be created."""
    prevent_duplicates: bool
    whitelist: list[UserMatch]
    blacklist: list[UserMatch]

    async def authorize_user_create(
        self, user_info: KeycloakUserInfo
    ) -> UserCreateAuthorization:
        user_id = user_info.sub
        email = user_info.email
        try:
            if not email:
                logger.warning(f'No email provided for user_id: {user_id}')
                return UserCreateAuthorization(success=False, detail='missing_email')

            if self.prevent_duplicates:
                has_duplicate = await token_manager.check_duplicate_base_email(
                    email, user_id
                )
                if has_duplicate:
                    logger.warning(
                        f'Blocked signup attempt for email {email} - duplicate base email found',
                        extra={'user_id': user_id, 'email': email},
                    )
                    return UserCreateAuthorization(success=False, detail='duplicate_email')

            if DefaultUserCreateAuthorizer._has_match(self.whitelist, user_info):
                return UserCreateAuthorization(success=True)

            if DefaultUserCreateAuthorizer._has_match(self.blacklist, user_info):
                return UserCreateAuthorization(success=False, detail='blocked')

            if await domain_blocker.is_domain_blocked(email):
                logger.warning(
                    f'Blocked authentication attempt for email: {email}, user_id: {user_id}'
                )
                return UserCreateAuthorization(success=False, detail='blocked')

            return UserCreateAuthorization(success=True)
        except Exception:
            logger.exception('error authorizing user', extra={'user_id': user_id})
            return UserCreateAuthorization(success=False)

    @staticmethod
    def _has_match(matches: list[UserMatch] | None, user_info: KeycloakUserInfo) -> bool:
        if not matches:
            return False
        for user_match in matches:
            if user_match.match(user_info):
                return True
        return False


class DefaultUserCreateAuthorizerInjector(UserCreateAuthorizerInjector):
    prevent_duplicates: bool = Field(default=True, description="Whether duplicate emails (containing +) are filtered")
    whitelist: list[UserMatch] = Field(default_factory=list, description="Whitelist for emails")
    blacklist: list[UserMatch] = Field(default_factory=list, description=" Blacklist for emails")

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[UserCreateAuthorizer, None]:
        yield DefaultUserCreateAuthorizer(
            prevent_duplicates=self.prevent_duplicates,
            whitelist=self.whitelist,
            blacklist=self.blacklist,
        )
