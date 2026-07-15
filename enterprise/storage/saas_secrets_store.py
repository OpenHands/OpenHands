from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from storage.database import a_session_maker
from storage.stored_custom_secrets import StoredCustomSecrets
from storage.user_store import UserStore

from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.logger import openhands_logger as logger


@dataclass
class SaasSecretsStore(SecretsStore):
    user_id: str
    _jwt_svc: JwtService = field(repr=False)
    # When set, overrides the user's `current_org_id` for both load and
    # store. Used to honor a request's effective org (api_key_org_id >
    # X-Org-Id header > user.current_org_id). Secrets are stored per
    # (user_id, org_id), so the effective org must flow through here for
    # the right rows to be read/written.
    effective_org_id: UUID | None = None
    _loaded_custom_secrets: dict[UUID | None, dict[str, tuple[str, str | None]]] = (
        field(default_factory=dict, init=False, repr=False)
    )

    async def get_custom_secret_value(self, secret_name: str) -> str | None:
        user = await UserStore.get_user_by_id(self.user_id)
        if user is None:
            raise ValueError(f'User not found: {self.user_id}')
        org_id = self.effective_org_id or user.current_org_id
        async with a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets).filter(
                    StoredCustomSecrets.keycloak_user_id == self.user_id,
                    StoredCustomSecrets.org_id == org_id,
                    StoredCustomSecrets.secret_name == secret_name,
                )
            )
            stored = result.scalars().one_or_none()
            if stored is None:
                return None
            return self._jwt_svc.decrypt_value(stored.secret_value)

    async def compare_and_swap_custom_secret(
        self, secret_name: str, expected_digest: str, value: str
    ) -> bool:
        user = await UserStore.get_user_by_id(self.user_id)
        if user is None:
            raise ValueError(f'User not found: {self.user_id}')
        org_id = self.effective_org_id or user.current_org_id
        async with a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets)
                .filter(
                    StoredCustomSecrets.keycloak_user_id == self.user_id,
                    StoredCustomSecrets.org_id == org_id,
                    StoredCustomSecrets.secret_name == secret_name,
                )
                .with_for_update()
            )
            stored = result.scalars().one_or_none()
            if stored is None:
                raise KeyError(secret_name)
            current_value = self._jwt_svc.decrypt_value(stored.secret_value)
            current_digest = hashlib.sha256(current_value.encode()).hexdigest()
            if not hmac.compare_digest(current_digest, expected_digest):
                return False
            stored.secret_value = self._jwt_svc.encrypt_value(value)
            await session.commit()
            return True

    async def load(self) -> Secrets | None:
        if not self.user_id:
            return None
        user = await UserStore.get_user_by_id(self.user_id)
        org_id = self.effective_org_id or (user.current_org_id if user else None)

        async with a_session_maker() as session:
            # Fetch all secrets for the given user ID
            query = select(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id
            )
            if org_id is not None:
                query = query.filter(StoredCustomSecrets.org_id == org_id)
            result = await session.execute(query)
            settings = result.scalars().all()

            if not settings:
                self._loaded_custom_secrets[org_id] = {}
                return Secrets()

            kwargs = {}
            for secret in settings:
                kwargs[secret.secret_name] = {
                    'secret': secret.secret_value,
                    'description': secret.description,
                }

            self._decrypt_kwargs(kwargs)
            self._loaded_custom_secrets[org_id] = {
                name: (value['secret'], value.get('description'))
                for name, value in kwargs.items()
            }

            return Secrets(custom_secrets=kwargs)  # type: ignore[arg-type]

    async def store(self, item: Secrets):
        user = await UserStore.get_user_by_id(self.user_id)
        if user is None:
            raise ValueError(f'User not found: {self.user_id}')
        org_id = self.effective_org_id or user.current_org_id
        desired = {
            name: (secret.secret.get_secret_value(), secret.description)
            for name, secret in item.custom_secrets.items()
        }

        async with a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets)
                .filter(
                    StoredCustomSecrets.keycloak_user_id == self.user_id,
                    StoredCustomSecrets.org_id == org_id,
                )
                .with_for_update()
            )
            stored_by_name = {row.secret_name: row for row in result.scalars().all()}
            current = {
                name: (
                    self._jwt_svc.decrypt_value(row.secret_value),
                    self._jwt_svc.decrypt_value(row.description)
                    if row.description is not None
                    else None,
                )
                for name, row in stored_by_name.items()
            }
            baseline = self._loaded_custom_secrets.get(org_id, current)
            changed_names = {
                name
                for name in baseline.keys() | desired.keys()
                if baseline.get(name) != desired.get(name)
            }

            for name in changed_names:
                stored = stored_by_name.get(name)
                updated = desired.get(name)
                if updated is None:
                    if stored is not None:
                        await session.delete(stored)
                    current.pop(name, None)
                    continue

                value, description = updated
                encrypted_value = self._jwt_svc.encrypt_value(value)
                encrypted_description = (
                    self._jwt_svc.encrypt_value(description)
                    if description is not None
                    else None
                )
                if stored is None:
                    session.add(
                        StoredCustomSecrets(
                            keycloak_user_id=self.user_id,
                            org_id=org_id,
                            secret_name=name,
                            secret_value=encrypted_value,
                            description=encrypted_description,
                        )
                    )
                else:
                    stored.secret_value = encrypted_value
                    stored.description = encrypted_description
                current[name] = updated

            await session.commit()
            self._loaded_custom_secrets[org_id] = current

    def _decrypt_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self._decrypt_kwargs(value)
                continue

            if value is None:
                kwargs[key] = value
            else:
                kwargs[key] = self._jwt_svc.decrypt_value(value)

    def _encrypt_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self._encrypt_kwargs(value)
                continue

            if value is None:
                kwargs[key] = value
            else:
                kwargs[key] = self._jwt_svc.encrypt_value(value)

    @classmethod
    async def get_instance(  # type: ignore[override]
        cls,
        user_id: str,
        effective_org_id: UUID | None = None,
    ) -> SaasSecretsStore:
        """Get a SaasSecretsStore instance for the given user.

        Args:
            user_id: Keycloak user id.
            effective_org_id: Optional org id resolved from the request
                (see SaasUserAuth.get_effective_org_id). When None the
                store falls back to ``user.current_org_id`` to preserve
                legacy behavior for background / non-request callers
                (e.g. webhook resolvers).

        TODO: This method should be replaced with dependency injection.
        """
        logger.debug(f'saas_secrets_store.get_instance::{user_id}')
        from storage.encrypt_utils import get_jwt_service

        return SaasSecretsStore(
            user_id,
            get_jwt_service(),
            effective_org_id=effective_org_id,
        )
