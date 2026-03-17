"""Store class for managing user authorizations."""

from typing import Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.database import a_session_maker
from storage.user_authorization import UserAuthorization, UserAuthorizationType


class UserAuthorizationStore:
    """Store for managing user authorization rules."""

    @staticmethod
    async def _get_matching_authorizations(
        email: str,
        provider_type: str | None,
        session: AsyncSession,
    ) -> list[UserAuthorization]:
        """Get all authorization rules that match the given email and provider.

        Uses SQL LIKE for pattern matching:
        - email_pattern is NULL matches all emails
        - provider_type is NULL matches all providers
        - email LIKE email_pattern for pattern matching

        Args:
            email: The user's email address
            provider_type: The identity provider type (e.g., 'github', 'gitlab')
            session: Database session

        Returns:
            List of matching UserAuthorization objects
        """
        # Build query using SQLAlchemy ORM
        # We need: (email_pattern IS NULL OR LOWER(email) LIKE LOWER(email_pattern))
        #      AND (provider_type IS NULL OR provider_type = :provider_type)
        email_condition = or_(
            UserAuthorization.email_pattern.is_(None),
            func.lower(email).like(func.lower(UserAuthorization.email_pattern)),
        )
        provider_condition = or_(
            UserAuthorization.provider_type.is_(None),
            UserAuthorization.provider_type == provider_type,
        )

        query = select(UserAuthorization).where(email_condition, provider_condition)
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_matching_authorizations(
        email: str,
        provider_type: str | None,
        session: Optional[AsyncSession] = None,
    ) -> list[UserAuthorization]:
        """Get all authorization rules that match the given email and provider.

        Args:
            email: The user's email address
            provider_type: The identity provider type (e.g., 'github', 'gitlab')
            session: Optional database session

        Returns:
            List of matching UserAuthorization objects
        """
        if session is not None:
            return await UserAuthorizationStore._get_matching_authorizations(
                email, provider_type, session
            )
        async with a_session_maker() as new_session:
            return await UserAuthorizationStore._get_matching_authorizations(
                email, provider_type, new_session
            )

    @staticmethod
    async def get_authorization_type(
        email: str,
        provider_type: str | None,
        session: Optional[AsyncSession] = None,
    ) -> UserAuthorizationType | None:
        """Get the authorization type for the given email and provider.

        Checks matching authorization rules and returns the effective authorization
        type. Whitelist rules take precedence over blacklist rules.

        Args:
            email: The user's email address
            provider_type: The identity provider type (e.g., 'github', 'gitlab')
            session: Optional database session

        Returns:
            UserAuthorizationType.WHITELIST if a whitelist rule matches,
            UserAuthorizationType.BLACKLIST if a blacklist rule matches (and no whitelist),
            None if no rules match
        """
        authorizations = await UserAuthorizationStore.get_matching_authorizations(
            email, provider_type, session
        )

        has_whitelist = any(
            auth.type == UserAuthorizationType.WHITELIST.value
            for auth in authorizations
        )
        if has_whitelist:
            return UserAuthorizationType.WHITELIST

        has_blacklist = any(
            auth.type == UserAuthorizationType.BLACKLIST.value
            for auth in authorizations
        )
        if has_blacklist:
            return UserAuthorizationType.BLACKLIST

        return None

    @staticmethod
    async def _create_authorization(
        email_pattern: str | None,
        provider_type: str | None,
        auth_type: UserAuthorizationType,
        session: AsyncSession,
    ) -> UserAuthorization:
        """Create a new user authorization rule."""
        authorization = UserAuthorization(
            email_pattern=email_pattern,
            provider_type=provider_type,
            type=auth_type.value,
        )
        session.add(authorization)
        await session.flush()
        await session.refresh(authorization)
        return authorization

    @staticmethod
    async def create_authorization(
        email_pattern: str | None,
        provider_type: str | None,
        auth_type: UserAuthorizationType,
        session: Optional[AsyncSession] = None,
    ) -> UserAuthorization:
        """Create a new user authorization rule.

        Args:
            email_pattern: SQL LIKE pattern for email matching (e.g., '%@openhands.dev')
            provider_type: Provider type to match (e.g., 'github'), or None for all
            auth_type: WHITELIST or BLACKLIST
            session: Optional database session

        Returns:
            The created UserAuthorization object
        """
        if session is not None:
            return await UserAuthorizationStore._create_authorization(
                email_pattern, provider_type, auth_type, session
            )
        async with a_session_maker() as new_session:
            auth = await UserAuthorizationStore._create_authorization(
                email_pattern, provider_type, auth_type, new_session
            )
            await new_session.commit()
            return auth

    @staticmethod
    async def _delete_authorization(
        authorization_id: int,
        session: AsyncSession,
    ) -> bool:
        """Delete an authorization rule by ID."""
        result = await session.execute(
            select(UserAuthorization).where(UserAuthorization.id == authorization_id)
        )
        authorization = result.scalars().first()
        if authorization:
            await session.delete(authorization)
            return True
        return False

    @staticmethod
    async def delete_authorization(
        authorization_id: int,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """Delete an authorization rule by ID.

        Args:
            authorization_id: The ID of the authorization to delete
            session: Optional database session

        Returns:
            True if deleted, False if not found
        """
        if session is not None:
            return await UserAuthorizationStore._delete_authorization(
                authorization_id, session
            )
        async with a_session_maker() as new_session:
            deleted = await UserAuthorizationStore._delete_authorization(
                authorization_id, new_session
            )
            if deleted:
                await new_session.commit()
            return deleted
