import time
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

import jwt
from fastapi import HTTPException, Request
from keycloak.exceptions import KeycloakError
from pydantic import SecretStr
from server.auth.auth_error import (
    AuthError,
    BearerTokenError,
    CookieError,
    ExpiredError,
    NoCredentialsError,
)
from server.auth.authorization import (
    get_role_permissions,
    get_user_org_role,
)
from server.auth.constants import BITBUCKET_DATA_CENTER_HOST
from server.auth.token_manager import TokenManager
from server.logger import logger
from server.rate_limit import RateLimiter, create_redis_rate_limiter
from sqlalchemy import delete, select
from storage.api_key_store import ApiKeyStore
from storage.auth_tokens import AuthTokens
from storage.database import a_session_maker
from storage.org_store import OrgStore
from storage.saas_secrets_store import SaasSecretsStore
from storage.saas_settings_store import SaasSettingsStore
from storage.user_authorization import UserAuthorizationType
from storage.user_authorization_store import UserAuthorizationStore
from storage.user_store import UserStore
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from openhands.app_server.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    CustomSecret,
    ProviderToken,
    ProviderType,
)
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.user_auth.user_auth import AuthType, UserAuth

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
    # API key context fields - populated when authenticated via API key
    api_key_org_id: UUID | None = None  # Org bound to the API key used for auth
    api_key_id: int | None = None
    api_key_name: str | None = None
    # Organization context fields - populated lazily via get_org_info()
    _org_id: str | None = None
    _org_name: str | None = None
    _role: str | None = None
    _permissions: list[str] | None = None
    _org_info_loaded: bool = False
    # Per-request `X-Org-Id` header (raw, unvalidated); see
    # `enterprise/server/auth/org_context.py` for resolution rules.
    _x_org_id_header: str | None = None
    # Cached result of `get_effective_org_id()`. The `_resolved` flag is
    # needed to distinguish "not yet computed" from "computed and None".
    _effective_org_id: UUID | None = None
    _effective_org_id_resolved: bool = False

    def get_api_key_org_id(self) -> UUID | None:
        """Get the organization ID bound to the API key used for authentication.

        Returns:
            The org_id if authenticated via API key with org binding, None otherwise
            (cookie auth or legacy API keys without org binding).
        """
        return self.api_key_org_id

    async def get_effective_org_id(self) -> UUID | None:
        """Resolve the effective organization ID for this request.

        Precedence (highest first):

        1. ``api_key_org_id`` — if the request is authenticated with an
           org-bound API key, that org wins. If the caller also sent an
           ``X-Org-Id`` header that disagrees, raise 403.
        2. ``X-Org-Id`` header — explicit, per-request override. The
           authenticated user must be a member of that org or we raise
           403. Malformed UUIDs raise 400.
        3. ``user.current_org_id`` — server-side default.

        The resolved value is cached on the auth instance for the rest
        of the request, so callers can invoke this freely.

        Raises:
            HTTPException: 400 for a malformed header, 403 for
                membership / API-key conflicts.
        """
        if self._effective_org_id_resolved:
            return self._effective_org_id

        # Import locally to avoid a circular import via authorization.py.
        from fastapi import status
        from storage.org_member_store import OrgMemberStore

        header_value = self._x_org_id_header
        requested: UUID | None = None
        if header_value:
            try:
                requested = UUID(header_value)
            except ValueError as exc:
                logger.warning(
                    'x_org_id_invalid',
                    extra={'user_id': self.user_id, 'header': header_value},
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Invalid X-Org-Id header (must be a UUID)',
                ) from exc

        # Case 1: API key binds the org.
        if self.api_key_org_id is not None:
            if requested is not None and requested != self.api_key_org_id:
                logger.warning(
                    'x_org_id_api_key_mismatch',
                    extra={
                        'user_id': self.user_id,
                        'api_key_org_id': str(self.api_key_org_id),
                        'x_org_id': str(requested),
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='API key is not authorized for this organization',
                )
            self._effective_org_id = self.api_key_org_id
            self._effective_org_id_resolved = True
            return self._effective_org_id

        # Case 2: X-Org-Id override; verify membership.
        if requested is not None:
            try:
                user_uuid = UUID(self.user_id)
            except ValueError as exc:
                # Shouldn't happen, but treat as not-a-member.
                logger.error(
                    'x_org_id_invalid_user_id',
                    extra={'user_id': self.user_id},
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='User is not a member of the requested organization',
                ) from exc
            member = await OrgMemberStore.get_org_member(requested, user_uuid)
            if member is None:
                logger.warning(
                    'x_org_id_not_a_member',
                    extra={
                        'user_id': self.user_id,
                        'x_org_id': str(requested),
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='User is not a member of the requested organization',
                )
            self._effective_org_id = requested
            self._effective_org_id_resolved = True
            return self._effective_org_id

        # Case 3: Fall back to the user's currently-selected org.
        user = await UserStore.get_user_by_id(self.user_id)
        if user is not None:
            self._effective_org_id = user.current_org_id
        self._effective_org_id_resolved = True
        return self._effective_org_id

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

    async def get_secrets_store(self) -> SaasSecretsStore:
        logger.debug('saas_user_auth_get_secrets_store')
        secrets_store = self.secrets_store
        if secrets_store:
            return secrets_store
        # Scope secrets to the request's effective org so that callers
        # using an API key bound to org A — or supplying an X-Org-Id
        # header — read/write secrets under that org, not under whatever
        # `user.current_org_id` happens to point at.
        effective_org_id = await self.get_effective_org_id()
        secrets_store = await SaasSecretsStore.get_instance(
            self.user_id,
            effective_org_id=effective_org_id,
        )
        self.secrets_store = secrets_store
        return secrets_store

    async def get_secrets(self):
        user_secrets = self._secrets
        if user_secrets:
            return user_secrets
        secrets_store = await self.get_secrets_store()
        user_secrets = await secrets_store.load()

        # Inject OPENHANDS_API_KEY (system-level, lazily generated)
        openhands_api_key = await self._get_openhands_api_key()
        if openhands_api_key:
            custom_secrets = dict(user_secrets.custom_secrets) if user_secrets else {}
            custom_secrets['OPENHANDS_API_KEY'] = CustomSecret(
                secret=SecretStr(openhands_api_key),
                description='OpenHands Cloud API Key for automations and integrations (system-managed)',
            )
            user_secrets = Secrets(
                custom_secrets=custom_secrets,
                provider_tokens=user_secrets.provider_tokens if user_secrets else {},
            )

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
            async with a_session_maker() as session:
                result = await session.execute(
                    select(AuthTokens).where(
                        AuthTokens.keycloak_user_id == self.user_id
                    )
                )
                tokens = result.scalars().all()

            for token in tokens:
                idp_type = ProviderType(token.identity_provider)
                try:
                    host = None
                    if user_secrets and idp_type in user_secrets.provider_tokens:
                        host = user_secrets.provider_tokens[idp_type].host

                    if idp_type == ProviderType.BITBUCKET_DATA_CENTER and not host:
                        host = BITBUCKET_DATA_CENTER_HOST or None

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
                    async with a_session_maker() as session:
                        await session.execute(
                            delete(AuthTokens).where(AuthTokens.id == token.id)
                        )
                        await session.commit()
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
        # Scope settings to the request's effective org. See
        # `get_secrets_store` for the same rationale: the store mutates
        # the resolved Org row (and per-member overrides), so the
        # effective org must flow through here rather than letting the
        # store fall back to `user.current_org_id`.
        effective_org_id = await self.get_effective_org_id()
        settings_store = SaasSettingsStore(
            self.user_id, effective_org_id=effective_org_id
        )
        self.settings_store = settings_store
        return settings_store

    async def get_mcp_api_key(self) -> str:
        api_key_store = ApiKeyStore.get_instance()
        # Scope MCP_API_KEY to the request's effective org so that an
        # X-Org-Id override or API-key binding produces an MCP key in
        # the correct org context. Falls back to user.current_org_id
        # when no SAAS auth or effective org can be resolved.
        effective_org_id = await self.get_effective_org_id()
        mcp_api_key = await api_key_store.retrieve_mcp_api_key(
            self.user_id, org_id=effective_org_id
        )
        if not mcp_api_key:
            mcp_api_key = await api_key_store.create_api_key(
                self.user_id,
                'MCP_API_KEY',
                None,
                org_id=effective_org_id,
            )
        return mcp_api_key

    async def _get_openhands_api_key(self) -> str:
        """Get or create the user's OPENHANDS_API_KEY (system-level, non-deletable).

        This key is automatically generated on first access and stored as a system
        key that users cannot delete or modify. It is used for automations and
        integrations.

        The key is scoped to the request's *effective* organization (honoring
        an ``X-Org-Id`` override or API-key binding) rather than the user's
        persisted ``current_org_id``.
        """
        effective_org_id = await self.get_effective_org_id()
        if effective_org_id is None:
            raise ValueError(f'User {self.user_id} has no current organization')

        api_key_store = ApiKeyStore.get_instance()
        openhands_api_key = await api_key_store.get_or_create_system_api_key(
            user_id=self.user_id,
            org_id=effective_org_id,
            name='OPENHANDS_API_KEY',
        )
        return openhands_api_key

    async def get_org_info(self) -> dict | None:
        """Get organization info for the current user.

        Lazily loads and caches organization data including:
        - org_id: Current organization ID
        - org_name: Current organization name
        - role: User's role in the organization
        - permissions: List of permission names for the role

        Returns:
            dict with org_id, org_name, role, permissions or None if not available
        """
        if self._org_info_loaded:
            if self._org_id is None:
                return None
            return {
                'org_id': self._org_id,
                'org_name': self._org_name,
                'role': self._role,
                'permissions': self._permissions,
            }

        # Mark as loaded to avoid repeated attempts on failure
        self._org_info_loaded = True

        try:
            # Use the effective org id so that requests carrying an
            # X-Org-Id override (or an org-bound API key) see info for
            # the org they're actually operating in, not the user's
            # persisted current_org_id.
            effective_org_id = await self.get_effective_org_id()
            if effective_org_id is None:
                logger.warning(
                    f'No effective org for user {self.user_id} in get_org_info'
                )
                return None

            org = await OrgStore.get_org_by_id(effective_org_id)
            if not org:
                logger.warning(
                    f'Organization {effective_org_id} not found for user {self.user_id}'
                )
                return None

            # Get user's role in that org
            role = await get_user_org_role(self.user_id, effective_org_id)
            role_name = role.name if role else None

            # Get permissions for the role
            permissions: list[str] = []
            if role_name:
                role_permissions = get_role_permissions(role_name)
                permissions = [p.value for p in role_permissions]

            # Cache the results
            self._org_id = str(effective_org_id)
            self._org_name = org.name
            self._role = role_name
            self._permissions = permissions

            return {
                'org_id': self._org_id,
                'org_name': self._org_name,
                'role': self._role,
                'permissions': self._permissions,
            }
        except HTTPException:
            # Propagate validation errors raised by get_effective_org_id().
            raise
        except Exception as e:
            logger.error(f'Error fetching org info for user {self.user_id}: {e}')
            return None

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
        # Capture the raw X-Org-Id header (if any) so it can be validated
        # lazily by `get_effective_org_id()` the first time the request
        # needs an org context. See `server.auth.org_context`.
        instance._x_org_id_header = request.headers.get('X-Org-Id')
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
        validation_result = await api_key_store.validate_api_key(api_key)
        if not validation_result:
            return None
        offline_token = await token_manager.load_offline_token(
            validation_result.user_id
        )
        saas_user_auth = SaasUserAuth(
            user_id=validation_result.user_id,
            refresh_token=SecretStr(offline_token),
            auth_type=AuthType.BEARER,
            api_key_org_id=validation_result.org_id,
            api_key_id=validation_result.key_id,
            api_key_name=validation_result.key_name,
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
    from storage.encrypt_utils import get_jwt_service

    decoded = get_jwt_service().verify_jws_token(signed_token)
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

    # Check if email is blacklisted (whitelist takes precedence)
    if email:
        auth_type = await UserAuthorizationStore.get_authorization_type(email, None)
        if auth_type == UserAuthorizationType.BLACKLIST:
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
