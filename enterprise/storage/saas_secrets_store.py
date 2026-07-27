from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import delete, select
from storage.database import a_session_maker
from storage.org_member_store import OrgMemberStore
from storage.stored_custom_secrets import StoredCustomSecrets
from storage.user_store import UserStore

from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import (
    CredentialVersionConflict,
    SecretsStore,
)
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.logger import openhands_logger as logger

_CODEX_AUTH_SECRET_NAME = 'CODEX_AUTH_JSON'


def _credential_version(row: StoredCustomSecrets) -> str:
    state = f'{row.id}\0{row.secret_value}'.encode()
    return hashlib.sha256(state).hexdigest()


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
    _loaded_codex_auth: dict[UUID | None, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    async def load(self) -> Secrets | None:
        if not self.user_id:
            return None
        user = await UserStore.get_user_by_id(self.user_id)
        org_id = self.effective_org_id or (user.current_org_id if user else None)

        async with a_session_maker() as session:
            query = select(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id
            )
            if org_id is not None:
                query = query.filter(StoredCustomSecrets.org_id == org_id)
            result = await session.execute(query.order_by(StoredCustomSecrets.id))
            settings = result.scalars().all()

            if not settings:
                self._loaded_codex_auth[org_id] = None
                return Secrets()

            kwargs = {}
            for secret in settings:
                kwargs[secret.secret_name] = {
                    'secret': secret.secret_value,
                    'description': secret.description,
                }

            self._decrypt_kwargs(kwargs)
            codex_info = kwargs.get(_CODEX_AUTH_SECRET_NAME)
            codex_value = (
                codex_info.get('secret') if isinstance(codex_info, dict) else None
            )
            self._loaded_codex_auth[org_id] = (
                codex_value if isinstance(codex_value, str) else None
            )

            return Secrets(custom_secrets=kwargs)  # type: ignore[arg-type]

    async def store(self, item: Secrets):
        user = await UserStore.get_user_by_id(self.user_id)
        if user is None:
            raise ValueError(f'User not found: {self.user_id}')
        org_id = self.effective_org_id or user.current_org_id

        kwargs = item.model_dump(context={'expose_secrets': True})
        del kwargs['provider_tokens']
        secrets_json = kwargs.get('custom_secrets', {})
        codex_info = secrets_json.get(_CODEX_AUTH_SECRET_NAME)
        submitted_codex = (
            codex_info.get('secret') if isinstance(codex_info, dict) else None
        )
        if not isinstance(submitted_codex, str):
            submitted_codex = None

        has_baseline = org_id in self._loaded_codex_auth
        preserve_codex = not has_baseline and codex_info is None
        if has_baseline and submitted_codex == self._loaded_codex_auth[org_id]:
            preserve_codex = True

        async with a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets)
                .filter(
                    StoredCustomSecrets.keycloak_user_id == self.user_id,
                    StoredCustomSecrets.org_id == org_id,
                )
                # Lock in the same order as _versioned_query so a concurrent
                # replace_versioned over the duplicate Codex rows cannot deadlock.
                .order_by(StoredCustomSecrets.id.desc())
                .with_for_update()
            )
            codex_rows = [
                row
                for row in result.scalars().all()
                if row.secret_name == _CODEX_AUTH_SECRET_NAME
            ]
            if preserve_codex:
                secrets_json.pop(_CODEX_AUTH_SECRET_NAME, None)
                if codex_rows and isinstance(codex_info, dict):
                    description = codex_info.get('description')
                    encrypted_description = (
                        self._jwt_svc.encrypt_value(description)
                        if description is not None
                        else None
                    )
                    for row in codex_rows:
                        row.description = encrypted_description

            delete_query = delete(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id,
                StoredCustomSecrets.org_id == org_id,
            )
            if preserve_codex:
                delete_query = delete_query.filter(
                    StoredCustomSecrets.secret_name != _CODEX_AUTH_SECRET_NAME
                )
            await session.execute(delete_query)

            self._encrypt_kwargs(kwargs)

            secret_tuples = []
            for secret_name, secret_info in secrets_json.items():
                secret_value = secret_info.get('secret')
                description = secret_info.get('description')

                secret_tuples.append((secret_name, secret_value, description))

            for secret_name, secret_value, description in secret_tuples:
                new_secret = StoredCustomSecrets(
                    keycloak_user_id=self.user_id,
                    org_id=org_id,
                    secret_name=secret_name,
                    secret_value=secret_value,
                    description=description,
                )
                session.add(new_secret)

            await session.commit()
            if not preserve_codex:
                self._loaded_codex_auth[org_id] = submitted_codex

    async def load_versioned(
        self,
        name: str,
        organization_id: UUID | None = None,
    ) -> tuple[str, str]:
        org_id = await self._require_organization_id(name, organization_id)
        async with a_session_maker() as session:
            result = await session.execute(self._versioned_query(name, org_id).limit(1))
            row = result.scalars().first()
            if row is None:
                raise KeyError(name)
            return self._jwt_svc.decrypt_value(row.secret_value), _credential_version(
                row
            )

    async def replace_versioned(
        self,
        name: str,
        expected_version: str,
        value: str,
        organization_id: UUID | None = None,
    ) -> str:
        org_id = await self._require_organization_id(name, organization_id)
        async with a_session_maker() as session:
            result = await session.execute(
                self._versioned_query(name, org_id).with_for_update()
            )
            rows = result.scalars().all()
            if not rows:
                result = await session.execute(
                    self._versioned_query(name, org_id).limit(1)
                )
                if result.scalars().first() is not None:
                    raise CredentialVersionConflict
                raise KeyError(name)

            current_version = _credential_version(rows[0])
            if not hmac.compare_digest(
                current_version.encode(),
                expected_version.encode(errors='surrogatepass'),
            ):
                raise CredentialVersionConflict

            encrypted = self._jwt_svc.encrypt_value(value)
            for row in rows:
                row.secret_value = encrypted
            successor = _credential_version(rows[0])
            await session.commit()
            return successor

    async def _require_organization_id(
        self,
        name: str,
        organization_id: UUID | None,
    ) -> UUID:
        org_id = organization_id or self.effective_org_id
        if org_id is None:
            user = await UserStore.get_user_by_id(self.user_id)
            org_id = user.current_org_id if user else None
        if org_id is None:
            raise KeyError(name)
        try:
            user_id = UUID(self.user_id)
        except (TypeError, ValueError) as exc:
            raise KeyError(name) from exc
        if await OrgMemberStore.get_org_member(org_id, user_id) is None:
            raise KeyError(name)
        return org_id

    def _versioned_query(self, name: str, organization_id: UUID):
        return (
            select(StoredCustomSecrets)
            .filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id,
                StoredCustomSecrets.org_id == organization_id,
                StoredCustomSecrets.secret_name == name,
            )
            .order_by(StoredCustomSecrets.id.desc())
        )

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
