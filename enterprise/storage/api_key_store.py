from __future__ import annotations

import secrets
import string
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.orm import sessionmaker
from storage.api_key import ApiKey
from storage.database import session_maker
from storage.user_store import UserStore

from openhands.core.logger import openhands_logger as logger
from openhands.utils.async_utils import call_sync_from_async


@dataclass
class ApiKeyStore:
    session_maker: sessionmaker

    API_KEY_PREFIX = 'sk-oh-'

    def generate_api_key(self, length: int = 32) -> str:
        """Generate a random API key with the sk-oh- prefix."""
        alphabet = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
        return f'{self.API_KEY_PREFIX}{random_part}'

    async def create_api_key(
        self, user_id: str, name: str | None = None, expires_at: datetime | None = None
    ) -> str:
        """Create a new API key for a user.

        Args:
            user_id: The ID of the user to create the key for
            name: Optional name for the key
            expires_at: Optional expiration date for the key

        Returns:
            The generated API key
        """
        api_key = self.generate_api_key()
        user = await UserStore.get_user_by_id_async(user_id)
        org_id = user.current_org_id
        await call_sync_from_async(
            self._store_api_key, user_id, org_id, api_key, name, expires_at
        )

        return api_key

    def _store_api_key(
        self,
        user_id: str,
        org_id: str,
        api_key: str,
        name: str | None,
        expires_at: datetime | None = None,
    ) -> None:
        """Store an existing API key in the database."""
        with self.session_maker() as session:
            key_record = ApiKey(
                key=api_key,
                user_id=user_id,
                org_id=org_id,
                name=name,
                expires_at=expires_at,
            )
            session.add(key_record)
            session.commit()

    def validate_api_key(self, api_key: str) -> str | None:
        """Validate an API key and return the associated user_id if valid."""
        now = datetime.now(UTC)

        with self.session_maker() as session:
            key_record = session.query(ApiKey).filter(ApiKey.key == api_key).first()

            if not key_record:
                return None

            # Check if the key has expired
            if key_record.expires_at:
                # Handle timezone-naive datetime from database by assuming it's UTC
                expires_at = key_record.expires_at
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)

                if expires_at < now:
                    logger.info(f'API key has expired: {key_record.id}')
                    return None

            # Update last_used_at timestamp
            session.execute(
                update(ApiKey)
                .where(ApiKey.id == key_record.id)
                .values(last_used_at=now)
            )
            session.commit()

            return key_record.user_id

    def delete_api_key(self, api_key: str) -> bool:
        """Delete an API key by the key value."""
        with self.session_maker() as session:
            key_record = session.query(ApiKey).filter(ApiKey.key == api_key).first()

            if not key_record:
                return False

            session.delete(key_record)
            session.commit()

            return True

    def delete_api_key_by_id(self, key_id: int) -> bool:
        """Delete an API key by its ID."""
        with self.session_maker() as session:
            key_record = session.query(ApiKey).filter(ApiKey.id == key_id).first()

            if not key_record:
                return False

            session.delete(key_record)
            session.commit()

            return True

    async def list_api_keys(self, user_id: str) -> list[dict]:
        """List all API keys for a user."""
        user = await UserStore.get_user_by_id_async(user_id)
        org_id = user.current_org_id
        return await call_sync_from_async(self._list_api_keys_from_db, user_id, org_id)

    def _list_api_keys_from_db(self, user_id: str, org_id: str) -> list[ApiKey]:
        with self.session_maker() as session:
            keys = (
                session.query(ApiKey)
                .filter(ApiKey.user_id == user_id)
                .filter(ApiKey.org_id == org_id)
                .all()
            )

            return [
                {
                    'id': key.id,
                    'name': key.name,
                    'created_at': key.created_at,
                    'last_used_at': key.last_used_at,
                    'expires_at': key.expires_at,
                }
                for key in keys
                if 'MCP_API_KEY' != key.name
            ]

    async def retrieve_mcp_api_key(self, user_id: str) -> str | None:
        user = await UserStore.get_user_by_id_async(user_id)
        org_id = user.current_org_id
        return await call_sync_from_async(
            self._retrieve_mcp_api_key_from_db, user_id, org_id
        )

    def _retrieve_mcp_api_key_from_db(self, user_id: str, org_id: str) -> str | None:
        with self.session_maker() as session:
            keys: list[ApiKey] = (
                session.query(ApiKey)
                .filter(ApiKey.user_id == user_id)
                .filter(ApiKey.org_id == org_id)
                .all()
            )
            for key in keys:
                if key.name == 'MCP_API_KEY':
                    return key.key

        return None

    def retrieve_api_key_by_name(self, user_id: str, name: str) -> str | None:
        """Retrieve an API key by name for a specific user."""
        with self.session_maker() as session:
            key_record = (
                session.query(ApiKey)
                .filter(ApiKey.user_id == user_id, ApiKey.name == name)
                .first()
            )
            return key_record.key if key_record else None

    def delete_api_key_by_name(self, user_id: str, name: str) -> bool:
        """Delete an API key by name for a specific user."""
        with self.session_maker() as session:
            key_record = (
                session.query(ApiKey)
                .filter(ApiKey.user_id == user_id, ApiKey.name == name)
                .first()
            )

            if not key_record:
                return False

            session.delete(key_record)
            session.commit()

            return True

    @classmethod
    def get_instance(cls) -> ApiKeyStore:
        """Get an instance of the ApiKeyStore."""
        logger.debug('api_key_store.get_instance')
        return ApiKeyStore(session_maker)
