from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import delete, select
from storage.database import a_session_maker
from storage.stored_custom_secrets import StoredCustomSecrets
from storage.user_store import UserStore

from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.logger import openhands_logger as logger

_CODEX_AUTH_SECRET_NAME = 'CODEX_AUTH_JSON'


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
    _loaded_codex_auth_digests: dict[UUID | None, str] = field(
        default_factory=dict, init=False, repr=False
    )

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
                self._loaded_codex_auth_digests.pop(org_id, None)
                return Secrets()

            kwargs = {}
            for secret in settings:
                kwargs[secret.secret_name] = {
                    'secret': secret.secret_value,
                    'description': secret.description,
                }

            self._decrypt_kwargs(kwargs)
            codex_auth = kwargs.get(_CODEX_AUTH_SECRET_NAME, {}).get('secret')
            if isinstance(codex_auth, str):
                self._loaded_codex_auth_digests[org_id] = hashlib.sha256(
                    codex_auth.encode()
                ).hexdigest()
            else:
                self._loaded_codex_auth_digests.pop(org_id, None)

            return Secrets(custom_secrets=kwargs)  # type: ignore[arg-type]

    async def store(self, item: Secrets):
        user = await UserStore.get_user_by_id(self.user_id)
        if user is None:
            raise ValueError(f'User not found: {self.user_id}')
        org_id = self.effective_org_id or user.current_org_id
        kwargs = item.model_dump(context={'expose_secrets': True})
        del kwargs['provider_tokens']
        secrets_json = kwargs.get('custom_secrets', {})
        codex_auth_info = secrets_json.get(_CODEX_AUTH_SECRET_NAME)
        codex_auth = (
            codex_auth_info.get('secret') if isinstance(codex_auth_info, dict) else None
        )
        loaded_codex_known = org_id in self._loaded_codex_auth_digests
        loaded_codex_digest = self._loaded_codex_auth_digests.get(org_id)
        codex_auth_digest = (
            hashlib.sha256(codex_auth.encode()).hexdigest()
            if isinstance(codex_auth, str)
            else None
        )
        preserve_codex_auth = not loaded_codex_known and codex_auth_info is None
        stored_codex_auth = None
        async with a_session_maker() as session:
            if loaded_codex_known and loaded_codex_digest == codex_auth_digest:
                result = await session.execute(
                    select(StoredCustomSecrets)
                    .filter(
                        StoredCustomSecrets.keycloak_user_id == self.user_id,
                        StoredCustomSecrets.org_id == org_id,
                        StoredCustomSecrets.secret_name == _CODEX_AUTH_SECRET_NAME,
                    )
                    .order_by(StoredCustomSecrets.id.desc())
                    .limit(1)
                    .with_for_update()
                )
                stored_codex_auth = result.scalar_one_or_none()
                preserve_codex_auth = stored_codex_auth is not None

            if preserve_codex_auth:
                secrets_json.pop(_CODEX_AUTH_SECRET_NAME, None)
                if (
                    loaded_codex_known
                    and stored_codex_auth is not None
                    and isinstance(codex_auth_info, dict)
                ):
                    description = codex_auth_info.get('description')
                    stored_codex_auth.description = (
                        self._jwt_svc.encrypt_value(description)
                        if description is not None
                        else None
                    )

            # Incoming secrets are always the most updated ones
            # Delete existing records for this user AND organization only
            # org_id is always set: it's either the effective org from
            # the request or the user's non-nullable current_org_id.
            delete_query = delete(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id,
                StoredCustomSecrets.org_id == org_id,
            )
            if preserve_codex_auth:
                delete_query = delete_query.filter(
                    StoredCustomSecrets.secret_name != _CODEX_AUTH_SECRET_NAME
                )
            await session.execute(delete_query)

            self._encrypt_kwargs(kwargs)

            # Extract the secrets into tuples for insertion or updating
            secret_tuples = []
            for secret_name, secret_info in secrets_json.items():
                secret_value = secret_info.get('secret')
                description = secret_info.get('description')

                secret_tuples.append((secret_name, secret_value, description))

            # Add the new secrets
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
            if not preserve_codex_auth:
                if isinstance(codex_auth, str):
                    assert codex_auth_digest is not None
                    self._loaded_codex_auth_digests[org_id] = codex_auth_digest
                else:
                    self._loaded_codex_auth_digests.pop(org_id, None)

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
