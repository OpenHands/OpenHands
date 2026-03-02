from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict

from server.auth.auth_error import TokenRefreshError
from sqlalchemy import select, text, update
from sqlalchemy.exc import OperationalError
from storage.auth_tokens import AuthTokens
from storage.database import a_session_maker

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import ProviderType

# Time buffer (in seconds) before actual expiration to consider token expired
# This ensures tokens are refreshed before they actually expire. The
# github default is 8 hours, so 15 minutes leeway is ~3% of this.
ACCESS_TOKEN_EXPIRY_BUFFER = 900  # 15 minutes

# Database lock timeout to prevent indefinite blocking
LOCK_TIMEOUT_SECONDS = 5


@dataclass
class AuthTokenStore:
    keycloak_user_id: str
    idp: ProviderType

    @property
    def identity_provider_value(self) -> str:
        return self.idp.value

    def _is_token_expired(
        self, access_token_expires_at: int, refresh_token_expires_at: int
    ) -> tuple[bool, bool]:
        """Check if access and refresh tokens are expired.

        Args:
            access_token_expires_at: Expiration time for access token (seconds since epoch)
            refresh_token_expires_at: Expiration time for refresh token (seconds since epoch)

        Returns:
            Tuple of (access_expired, refresh_expired)
        """
        current_time = int(time.time())
        access_expired = (
            False
            if access_token_expires_at == 0
            else access_token_expires_at < current_time + ACCESS_TOKEN_EXPIRY_BUFFER
        )
        refresh_expired = (
            False
            if refresh_token_expires_at == 0
            else refresh_token_expires_at < current_time
        )
        return access_expired, refresh_expired

    async def store_tokens(
        self,
        access_token: str,
        refresh_token: str,
        access_token_expires_at: int,
        refresh_token_expires_at: int,
    ) -> None:
        """Store auth tokens in the database.

        Args:
            access_token: The access token to store
            refresh_token: The refresh token to store
            access_token_expires_at: Expiration time for access token (seconds since epoch)
            refresh_token_expires_at: Expiration time for refresh token (seconds since epoch)
        """
        async with a_session_maker() as session:
            async with session.begin():  # Explicitly start a transaction
                result = await session.execute(
                    select(AuthTokens).where(
                        AuthTokens.keycloak_user_id == self.keycloak_user_id,
                        AuthTokens.identity_provider == self.identity_provider_value,
                    )
                )
                token_record = result.scalars().first()

                if token_record:
                    token_record.access_token = access_token
                    token_record.refresh_token = refresh_token
                    token_record.access_token_expires_at = access_token_expires_at
                    token_record.refresh_token_expires_at = refresh_token_expires_at
                else:
                    token_record = AuthTokens(
                        keycloak_user_id=self.keycloak_user_id,
                        identity_provider=self.identity_provider_value,
                        access_token=access_token,
                        refresh_token=refresh_token,
                        access_token_expires_at=access_token_expires_at,
                        refresh_token_expires_at=refresh_token_expires_at,
                    )
                    session.add(token_record)

            await session.commit()  # Commit after transaction block

    async def load_tokens(
        self,
        check_expiration_and_refresh: Callable[
            [ProviderType, str, int, int], Awaitable[Dict[str, str | int]]
        ]
        | None = None,
    ) -> Dict[str, str | int] | None:
        """Load authentication tokens from the database and refresh them if necessary.

        This method uses a double-checked locking pattern to minimize lock contention:
        1. First, check if the token is valid WITHOUT acquiring a lock (fast path)
        2. If refresh is needed, acquire a lock with a timeout
        3. Double-check if refresh is still needed (another request may have refreshed)
        4. Perform the refresh if still needed

        The row-level lock ensures that only one refresh operation is performed per
        refresh token, which is important because most IDPs invalidate the old refresh
        token after it's used once.

        Args:
            check_expiration_and_refresh: A function that checks if the tokens have
                expired and attempts to refresh them. It should return a dictionary
                containing the new access_token, refresh_token, and their respective
                expiration timestamps. If no refresh is needed, it should return None.

        Returns:
            A dictionary containing the access_token, refresh_token,
            access_token_expires_at, and refresh_token_expires_at.
            If no token record is found, returns None.

        Raises:
            TokenRefreshError: If the lock cannot be acquired within the timeout
                period. This typically means another request is holding the lock
                for an extended period. Callers should handle this by returning
                a 401 response to prompt the user to re-authenticate.
        """
        # FAST PATH: Check without lock first to avoid unnecessary lock contention
        async with a_session_maker() as session:
            result = await session.execute(
                select(AuthTokens).filter(
                    AuthTokens.keycloak_user_id == self.keycloak_user_id,
                    AuthTokens.identity_provider == self.identity_provider_value,
                )
            )
            token_record = result.scalars().one_or_none()

            if not token_record:
                return None

            # Check if token needs refresh
            access_expired, _ = self._is_token_expired(
                token_record.access_token_expires_at,
                token_record.refresh_token_expires_at,
            )

            # If token is still valid, return it without acquiring a lock
            if not access_expired or check_expiration_and_refresh is None:
                return {
                    'access_token': token_record.access_token,
                    'refresh_token': token_record.refresh_token,
                    'access_token_expires_at': token_record.access_token_expires_at,
                    'refresh_token_expires_at': token_record.refresh_token_expires_at,
                }

        # SLOW PATH: Token needs refresh, acquire lock
        try:
            async with a_session_maker() as session:
                async with session.begin():
                    # Set a lock timeout to prevent indefinite blocking
                    # This ensures we don't hold connections forever if something goes wrong
                    await session.execute(
                        text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT_SECONDS}s'")
                    )

                    # Acquire row-level lock to prevent concurrent refresh attempts
                    result = await session.execute(
                        select(AuthTokens)
                        .filter(
                            AuthTokens.keycloak_user_id == self.keycloak_user_id,
                            AuthTokens.identity_provider
                            == self.identity_provider_value,
                        )
                        .with_for_update()
                    )
                    token_record = result.scalars().one_or_none()

                    if not token_record:
                        return None

                    # Double-check: another request may have refreshed while we waited for the lock
                    access_expired, _ = self._is_token_expired(
                        token_record.access_token_expires_at,
                        token_record.refresh_token_expires_at,
                    )

                    if not access_expired:
                        # Token was refreshed by another request while we waited
                        logger.debug(
                            'Token was refreshed by another request while waiting for lock'
                        )
                        return {
                            'access_token': token_record.access_token,
                            'refresh_token': token_record.refresh_token,
                            'access_token_expires_at': token_record.access_token_expires_at,
                            'refresh_token_expires_at': token_record.refresh_token_expires_at,
                        }

                    # We're the one doing the refresh
                    token_refresh = await check_expiration_and_refresh(
                        self.idp,
                        token_record.refresh_token,
                        token_record.access_token_expires_at,
                        token_record.refresh_token_expires_at,
                    )

                    if token_refresh:
                        await session.execute(
                            update(AuthTokens)
                            .where(AuthTokens.id == token_record.id)
                            .values(
                                access_token=token_refresh['access_token'],
                                refresh_token=token_refresh['refresh_token'],
                                access_token_expires_at=token_refresh[
                                    'access_token_expires_at'
                                ],
                                refresh_token_expires_at=token_refresh[
                                    'refresh_token_expires_at'
                                ],
                            )
                        )
                        await session.commit()

                    return (
                        token_refresh
                        if token_refresh
                        else {
                            'access_token': token_record.access_token,
                            'refresh_token': token_record.refresh_token,
                            'access_token_expires_at': token_record.access_token_expires_at,
                            'refresh_token_expires_at': token_record.refresh_token_expires_at,
                        }
                    )
        except OperationalError as e:
            # Lock timeout - another request is holding the lock for too long
            logger.warning(
                f'Token refresh lock timeout for user {self.keycloak_user_id}: {e}'
            )
            raise TokenRefreshError(
                'Unable to refresh token due to lock timeout. Please try again.'
            ) from e

    async def is_access_token_valid(self) -> bool:
        """Check if the access token is still valid.

        Returns:
            True if the access token exists and is not expired, False otherwise
        """
        tokens = await self.load_tokens()
        if not tokens:
            return False

        access_token_expires_at = tokens['access_token_expires_at']
        current_time = int(time.time())

        # Return True if the token is not expired (with a small buffer)
        return int(access_token_expires_at) > (current_time + 30)

    async def is_refresh_token_valid(self) -> bool:
        """Check if the refresh token is still valid.

        Returns:
            True if the refresh token exists and is not expired, False otherwise
        """
        tokens = await self.load_tokens()
        if not tokens:
            return False

        refresh_token_expires_at = tokens['refresh_token_expires_at']
        current_time = int(time.time())

        # Return True if the token is not expired (with a small buffer)
        return int(refresh_token_expires_at) > (current_time + 30)

    @classmethod
    async def get_instance(
        cls, keycloak_user_id: str, idp: ProviderType
    ) -> AuthTokenStore:
        """Get an instance of the AuthTokenStore.

        Args:
            keycloak_user_id: The Keycloak user ID
            idp: The identity provider type

        Returns:
            An instance of AuthTokenStore
        """
        logger.debug(f'auth_token_store.get_instance::{keycloak_user_id}')
        if keycloak_user_id:
            keycloak_user_id = str(keycloak_user_id)
        return AuthTokenStore(keycloak_user_id=keycloak_user_id, idp=idp)
