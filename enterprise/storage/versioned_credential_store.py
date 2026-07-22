import hashlib
import hmac
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from storage.database import a_session_maker
from storage.stored_custom_secrets import StoredCustomSecrets

from openhands.app_server.secrets.secrets_store import CredentialVersionConflict
from openhands.app_server.services.jwt_service import JwtService


def credential_version(row: StoredCustomSecrets) -> str:
    source = f'{row.id}\0{row.secret_value}'
    return hashlib.sha256(source.encode()).hexdigest()


@dataclass
class SaasVersionedCredentialStore:
    user_id: str
    organization_id: UUID
    jwt_service: JwtService = field(repr=False)

    def _query(self, name: str):
        return (
            select(StoredCustomSecrets)
            .filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id,
                StoredCustomSecrets.org_id == self.organization_id,
                StoredCustomSecrets.secret_name == name,
            )
            .order_by(StoredCustomSecrets.id.desc())
        )

    async def load(self, name: str) -> tuple[str, str]:
        async with a_session_maker() as session:
            result = await session.execute(self._query(name).limit(1))
            row = result.scalars().first()
            if row is None:
                raise KeyError(name)
            return self.jwt_service.decrypt_value(row.secret_value), credential_version(
                row
            )

    async def replace(self, name: str, expected_version: str, value: str) -> str:
        async with a_session_maker() as session:
            result = await session.execute(self._query(name).with_for_update())
            rows = result.scalars().all()
            if not rows:
                raise KeyError(name)
            if not hmac.compare_digest(
                credential_version(rows[0]).encode(),
                expected_version.encode(errors='surrogatepass'),
            ):
                raise CredentialVersionConflict
            encrypted = self.jwt_service.encrypt_value(value)
            for row in rows:
                row.secret_value = encrypted
            await session.commit()
            return credential_version(rows[0])
