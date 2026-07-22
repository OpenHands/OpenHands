from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import delete, select
from storage.database import a_session_maker
from storage.stored_custom_secrets import StoredCustomSecrets
from storage.user_store import UserStore
from storage.versioned_credential_store import (
    SaasVersionedCredentialStore,
    credential_version,
)

from openhands.app_server.secrets.credential_binding_models import (
    is_runtime_managed_credential,
)
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import (
    ManagedCredentialStore,
    SecretsStore,
)
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.logger import openhands_logger as logger


@dataclass
class SaasSecretsStore(SecretsStore, ManagedCredentialStore):
    user_id: str
    _jwt_svc: JwtService = field(repr=False)
    # When set, overrides the user's `current_org_id` for both load and
    # store. Used to honor a request's effective org (api_key_org_id >
    # X-Org-Id header > user.current_org_id). Secrets are stored per
    # (user_id, org_id), so the effective org must flow through here for
    # the right rows to be read/written.
    effective_org_id: UUID | None = None
    is_managed_credential: Callable[[str], bool] = field(
        default=is_runtime_managed_credential,
        repr=False,
    )
    _loaded_managed_credentials: dict[UUID | None, dict[str, tuple[str, str]]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def managed_credentials(self) -> ManagedCredentialStore:
        return self

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
            result = await session.execute(query.order_by(StoredCustomSecrets.id))
            settings = result.scalars().all()

            if not settings:
                self._loaded_managed_credentials.pop(org_id, None)
                return Secrets()

            kwargs = {}
            managed_rows = {}
            for secret in settings:
                kwargs[secret.secret_name] = {
                    'secret': secret.secret_value,
                    'description': secret.description,
                }
                if self.is_managed_credential(secret.secret_name):
                    managed_rows[secret.secret_name] = secret

            self._decrypt_kwargs(kwargs)
            loaded_managed = {
                name: (value, credential_version(row))
                for name, row in managed_rows.items()
                if isinstance(value := kwargs.get(name, {}).get('secret'), str)
            }
            if loaded_managed:
                self._loaded_managed_credentials[org_id] = loaded_managed
            else:
                self._loaded_managed_credentials.pop(org_id, None)

            return Secrets(custom_secrets=kwargs)  # type: ignore[arg-type]

    async def store(self, item: Secrets):
        user = await UserStore.get_user_by_id(self.user_id)
        if user is None:
            raise ValueError(f'User not found: {self.user_id}')
        org_id = self.effective_org_id or user.current_org_id
        kwargs = item.model_dump(context={'expose_secrets': True})
        del kwargs['provider_tokens']
        secrets_json = kwargs.get('custom_secrets', {})
        submitted_managed = {
            name: value
            for name, info in secrets_json.items()
            if self.is_managed_credential(name)
            and isinstance(info, dict)
            and isinstance(value := info.get('secret'), str)
        }
        loaded_managed = dict(self._loaded_managed_credentials.get(org_id, {}))
        async with a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets)
                .filter(
                    StoredCustomSecrets.keycloak_user_id == self.user_id,
                    StoredCustomSecrets.org_id == org_id,
                )
                .order_by(StoredCustomSecrets.id)
                .with_for_update()
            )
            managed_rows: dict[str, list[StoredCustomSecrets]] = {}
            for row in result.scalars().all():
                if self.is_managed_credential(row.secret_name):
                    managed_rows.setdefault(row.secret_name, []).append(row)

            preserved_names = set()
            for name in set(managed_rows) | set(loaded_managed):
                info = secrets_json.get(name)
                submitted = submitted_managed.get(name)
                loaded = loaded_managed.get(name)
                preserve = loaded is None and info is None
                if loaded is not None and submitted == loaded[0]:
                    preserve = True
                if not preserve:
                    continue
                preserved_names.add(name)
                secrets_json.pop(name, None)
                if managed_rows.get(name) and isinstance(info, dict):
                    description = info.get('description')
                    encrypted_description = (
                        self._jwt_svc.encrypt_value(description)
                        if description is not None
                        else None
                    )
                    for row in managed_rows[name]:
                        row.description = encrypted_description

            # Incoming secrets are always the most updated ones
            # Delete existing records for this user AND organization only
            # org_id is always set: it's either the effective org from
            # the request or the user's non-nullable current_org_id.
            delete_query = delete(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id,
                StoredCustomSecrets.org_id == org_id,
            )
            if preserved_names:
                delete_query = delete_query.filter(
                    StoredCustomSecrets.secret_name.not_in(preserved_names)
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
            cached = loaded_managed
            for name in (
                set(managed_rows) | set(loaded_managed) | set(submitted_managed)
            ):
                if name in preserved_names:
                    continue
                submitted = submitted_managed.get(name)
                if submitted is not None:
                    cached[name] = (submitted, '')
                else:
                    cached.pop(name, None)
            if cached:
                self._loaded_managed_credentials[org_id] = cached
            else:
                self._loaded_managed_credentials.pop(org_id, None)

    async def load_managed(
        self,
        name: str,
        organization_id: UUID | None = None,
    ) -> tuple[str, str]:
        org_id = organization_id or self.effective_org_id
        if org_id is None:
            user = await UserStore.get_user_by_id(self.user_id)
            org_id = user.current_org_id if user else None
        if org_id is None:
            raise KeyError(name)
        return await SaasVersionedCredentialStore(
            self.user_id,
            org_id,
            self._jwt_svc,
        ).load(name)

    async def replace_managed(
        self,
        name: str,
        expected_version: str,
        value: str,
        organization_id: UUID | None = None,
    ) -> str:
        org_id = organization_id or self.effective_org_id
        if org_id is None:
            user = await UserStore.get_user_by_id(self.user_id)
            org_id = user.current_org_id if user else None
        if org_id is None:
            raise KeyError(name)
        return await SaasVersionedCredentialStore(
            self.user_id,
            org_id,
            self._jwt_svc,
        ).replace(name, expected_version, value)

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
