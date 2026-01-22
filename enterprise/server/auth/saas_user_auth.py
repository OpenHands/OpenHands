import time
from dataclasses import dataclass
from types import MappingProxyType

import jwt
from fastapi import Request
from keycloak.exceptions import KeycloakError
from pydantic import SecretStr
from server.auth.auth_error import (
    AuthError,
    BearerTokenError,
    CookieError,
    ExpiredError,
    NoCredentialsError,
)
from server.auth.domain_blocker import domain_blocker
from server.auth.token_manager import TokenManager
from server.config import get_config
from server.logger import logger
from server.rate_limit import RateLimiter, create_redis_rate_limiter
from storage.api_key_store import ApiKeyStore
from storage.auth_tokens import AuthTokens
from storage.database import session_maker
from storage.saas_secrets_store import SaasSecretsStore
from storage.saas_settings_store import SaasSettingsStore
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderToken,
    ProviderType,
)
from openhands.server.settings import Settings
from openhands.server.user_auth.user_auth import AuthType, UserAuth
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.settings.settings_store import SettingsStore

token_manager = TokenManager()


rate_limiter: RateLimiter = create_redis_rate_limiter('10/second; 100/minute')


@dataclass
class SaasUserAuth(UserAuth):
    refresh_token: SecretStr
    user_id: str
    email: str | None = None
    email_verified: bool | None = None
    access_token: SecretStr | None = None
    provider_tokens: PROVIDER_TOKEN_TYPE | None = None
    refreshed: bool = False
    settings_store: SaasSettingsStore | None = None
    secrets_store: SaasSecretsStore | None = None
    _settings: Settings | None = None
    _secrets: Secrets | None = None
    accepted_tos: bool | None = None
    auth_type: AuthType = AuthType.COOKIE

    async def get_user_id(self) -> str | None:
        return self.user_id

    async def get_user_email(self) -> str | None:
        return self.email

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(KeycloakError),
    )
    async def refresh(self):
        if self._is_token_expired(self.refresh_token):
            logger.debug('saas_user_auth_refresh:expired')
            raise ExpiredError()

        tokens = await token_manager.refresh(self.refresh_token.get_secret_value())
        self.access_token = SecretStr(tokens['access_token'])
        self.refresh_token = SecretStr(tokens['refresh_token'])
        self.refreshed = True
        if not self.email or not self.email_verified or not self.user_id:
            # We don't need to verify the signature here because we just refreshed
            # this token from the IDP via token_manager.refresh()
            access_token_payload = jwt.decode(
                tokens['access_token'], options={'verify_signature': False}
            )
            self.user_id = access_token_payload['sub']
            self.email = access_token_payload['email']
            self.email_verified = access_token_payload['email_verified']

    def _is_token_expired(self, token: SecretStr):
        logger.debug('saas_user_auth_is_token_expired')
        # Decode token payload - works with both access and refresh tokens
        payload = jwt.decode(
            token.get_secret_value(), options={'verify_signature': False}
        )

        # Sanity check - make sure we refer to current user
        assert payload['sub'] == self.user_id

        # Check token expiration
        expiration = payload.get('exp')
        if expiration:
            logger.debug('saas_user_auth_is_token_expired expiration is %d', expiration)
        return expiration and expiration < time.time()

    def get_auth_type(self) -> AuthType | None:
        return self.auth_type

    async def get_user_settings(self) -> Settings | None:
        settings = self._settings
        if settings:
            return settings
        settings_store = await self.get_user_settings_store()
        settings = await settings_store.load()
        if settings:
            settings.email = self.email
            settings.email_verified = self.email_verified
            self._settings = settings
        return settings

    async def get_secrets_store(self):
        logger.debug('saas_user_auth_get_secrets_store')
        secrets_store = self.secrets_store
        if secrets_store:
            return secrets_store
        user_id = await self.get_user_id()
        secrets_store = SaasSecretsStore(user_id, session_maker, get_config())
        self.secrets_store = secrets_store
        return secrets_store

    async def get_secrets(self):
        user_secrets = self._secrets
        if user_secrets:
            return user_secrets
        secrets_store = await self.get_secrets_store()
        user_secrets = await secrets_store.load()
        self._secrets = user_secrets
        return user_secrets

    async def get_access_token(self) -> SecretStr | None:
        logger.debug('saas_user_auth_get_access_token')
        try:
            if self.access_token is None or self._is_token_expired(self.access_token):
                await self.refresh()
            return self.access_token
        except AuthError:
            raise
        except Exception as e:
            raise AuthError() from e

    async def get_provider_tokens(self) -> PROVIDER_TOKEN_TYPE | None:
        logger.debug('saas_user_auth_get_provider_tokens')
        if self.provider_tokens is not None:
            return self.provider_tokens
        provider_tokens = {}
        access_token = await self.get_access_token()
        if not access_token:
            raise AuthError()

        user_secrets = await self.get_secrets()

        try:
            # TODO: I think we can do this in a single request if we refactor
            with session_maker() as session:
                tokens = (
                    session.query(AuthTokens)
                    .where(AuthTokens.keycloak_user_id == self.user_id)
                    .all()
                )

            for token in tokens:
                idp_type = ProviderType(token.identity_provider)
                try:
                    host = None
                    if user_secrets and idp_type in user_secrets.provider_tokens:
                        host = user_secrets.provider_tokens[idp_type].host

                    provider_token = await token_manager.get_idp_token(
                        access_token.get_secret_value(),
                        idp=idp_type,
                    )
                    # TODO: Currently we don't store the IDP user id in our refresh table. We should.
                    provider_tokens[idp_type] = ProviderToken(
                        token=SecretStr(provider_token), user_id=None, host=host
                    )
                except Exception as e:
                    # If there was a problem with a refresh token we log and delete it
                    logger.error(
                        f'Error refreshing provider_token token: {e}',
                        extra={
                            'user_id': self.user_id,
                            'idp_type': token.identity_provider,
                        },
                    )
                    with session_maker() as session:
                        session.query(AuthTokens).filter(
                            AuthTokens.id == token.id
                        ).delete()
                        session.commit()
                    raise

            self.provider_tokens = MappingProxyType(provider_tokens)
            return self.provider_tokens
        except Exception as e:
            # Any error refreshing tokens means we need to log in again
            raise AuthError() from e

    async def get_user_settings_store(self) -> SettingsStore:
        settings_store = self.settings_store
        if settings_store:
            return settings_store
        user_id = await self.get_user_id()
        settings_store = SaasSettingsStore(user_id, session_maker, get_config())
        self.settings_store = settings_store
        return settings_store

    async def get_mcp_api_key(self) -> str:
        api_key_store = ApiKeyStore.get_instance()
        mcp_api_key = api_key_store.retrieve_mcp_api_key(self.user_id)
        if not mcp_api_key:
            mcp_api_key = api_key_store.create_api_key(
                self.user_id, 'MCP_API_KEY', None
            )
        return mcp_api_key

    @classmethod
    async def get_instance(cls, request: Request) -> UserAuth:
        logger.debug('saas_user_auth_get_instance')
        # First we check for for an API Key...
        logger.debug('saas_user_auth_get_instance:check_bearer')
        instance = await saas_user_auth_from_bearer(request)
        if instance is None:
            logger.debug('saas_user_auth_get_instance:check_cookie')
            instance = await saas_user_auth_from_cookie(request)
        if instance is None:
            logger.debug('saas_user_auth_get_instance:no_credentials')
            raise NoCredentialsError('failed to authenticate')
        if not getattr(request.state, 'user_rate_limit_processed', False):
            user_id = await instance.get_user_id()
            if user_id:
                # Ensure requests are only counted once
                request.state.user_rate_limit_processed = True
                # Will raise if rate limit is reached.
                await rate_limiter.hit('auth_uid', user_id)
        return instance

    @classmethod
    async def get_for_user(cls, user_id: str) -> UserAuth:
        offline_token = await token_manager.load_offline_token(user_id)
        assert offline_token is not None
        return SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr(offline_token),
            auth_type=AuthType.BEARER,
        )


def get_api_key_from_header(request: Request):
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.replace('Bearer ', '')

    # This is a temp hack
    # Streamable HTTP MCP Client works via redirect requests, but drops the Authorization header for reason
    # We include `X-Session-API-Key` header by default due to nested runtimes, so it used as a drop in replacement here
    session_api_key = request.headers.get('X-Session-API-Key')
    if session_api_key:
        return session_api_key

    # Fallback to X-Access-Token header as an additional option
    return request.headers.get('X-Access-Token')


async def saas_user_auth_from_bearer(request: Request) -> SaasUserAuth | None:
    try:
        api_key = get_api_key_from_header(request)
        if not api_key:
            return None

        api_key_store = ApiKeyStore.get_instance()
        user_id = api_key_store.validate_api_key(api_key)
        if not user_id:
            return None
        offline_token = await token_manager.load_offline_token(user_id)
        saas_user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr(offline_token),
            auth_type=AuthType.BEARER,
        )
        await saas_user_auth.refresh()
        return saas_user_auth
    except Exception as exc:
        raise BearerTokenError from exc


async def saas_user_auth_from_cookie(request: Request) -> SaasUserAuth | None:
    try:
        signed_token = request.cookies.get('keycloak_auth')
        if not signed_token:
            return None
        return await saas_user_auth_from_signed_token(signed_token)
    except Exception as exc:
        raise CookieError from exc


async def saas_user_auth_from_signed_token(signed_token: str) -> SaasUserAuth:
    logger.debug('saas_user_auth_from_signed_token')
    jwt_secret = get_config().jwt_secret.get_secret_value()
    decoded = jwt.decode(signed_token, jwt_secret, algorithms=['HS256'])
    logger.debug('saas_user_auth_from_signed_token:decoded')
    access_token = decoded['access_token']
    refresh_token = decoded['refresh_token']
    logger.debug(
        'saas_user_auth_from_signed_token',
        extra={
            'access_token': access_token,
            'refresh_token': refresh_token,
        },
    )
    accepted_tos = decoded.get('accepted_tos')

    # The access token was encoded using HS256 on keycloak. Since we signed it, we can trust is was
    # created by us. So we can grab the user_id and expiration from it without going back to keycloak.
    access_token_payload = jwt.decode(access_token, options={'verify_signature': False})
    user_id = access_token_payload['sub']
    email = access_token_payload['email']
    email_verified = access_token_payload['email_verified']

    # Check if email domain is blocked
    if email and domain_blocker.is_domain_blocked(email):
        logger.warning(
            f'Blocked authentication attempt for existing user with email: {email}'
        )
        raise AuthError(
            'Access denied: Your email domain is not allowed to access this service'
        )

    logger.debug('saas_user_auth_from_signed_token:return')

    return SaasUserAuth(
        access_token=SecretStr(access_token),
        refresh_token=SecretStr(refresh_token),
        user_id=user_id,
        email=email,
        email_verified=email_verified,
        accepted_tos=accepted_tos,
        auth_type=AuthType.COOKIE,
    )


async def get_user_auth_from_keycloak_id(keycloak_user_id: str) -> UserAuth:
    offline_token = await token_manager.load_offline_token(keycloak_user_id)
    if offline_token is None:
        logger.info('no_offline_token_found')

    user_auth = SaasUserAuth(
        user_id=keycloak_user_id,
        refresh_token=SecretStr(offline_token),
    )
    return user_auth
