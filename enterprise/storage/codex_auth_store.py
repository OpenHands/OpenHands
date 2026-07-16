import hashlib
import hmac
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from storage.database import a_session_maker
from storage.stored_custom_secrets import StoredCustomSecrets

from openhands.app_server.services.jwt_service import JwtService

_SECRET_NAME = 'CODEX_AUTH_JSON'


@dataclass
class CodexAuthStore:
    user_id: str
    org_id: UUID
    _jwt_svc: JwtService = field(repr=False)

    async def get_value(self) -> str | None:
        async with a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets)
                .filter(
                    StoredCustomSecrets.keycloak_user_id == self.user_id,
                    StoredCustomSecrets.org_id == self.org_id,
                    StoredCustomSecrets.secret_name == _SECRET_NAME,
                )
                .order_by(StoredCustomSecrets.id.desc())
                .limit(1)
            )
            stored = result.scalars().first()
            if stored is None:
                return None
            return self._jwt_svc.decrypt_value(stored.secret_value)

    async def compare_and_swap(self, expected_digest: str, value: str) -> bool:
        async with a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets)
                .filter(
                    StoredCustomSecrets.keycloak_user_id == self.user_id,
                    StoredCustomSecrets.org_id == self.org_id,
                    StoredCustomSecrets.secret_name == _SECRET_NAME,
                )
                .order_by(StoredCustomSecrets.id.desc())
                .with_for_update()
            )
            stored_rows = result.scalars().all()
            if not stored_rows:
                raise KeyError(_SECRET_NAME)
            current = self._jwt_svc.decrypt_value(stored_rows[0].secret_value)
            current_digest = hashlib.sha256(current.encode()).hexdigest()
            if not hmac.compare_digest(current_digest, expected_digest):
                return False
            encrypted = self._jwt_svc.encrypt_value(value)
            for stored in stored_rows:
                stored.secret_value = encrypted
            await session.commit()
            return True

    @classmethod
    async def get_instance(cls, user_id: str, org_id: UUID) -> 'CodexAuthStore':
        from storage.encrypt_utils import get_jwt_service

        return cls(user_id, org_id, get_jwt_service())
