from __future__ import annotations

import hashlib
from base64 import b64decode, b64encode
from dataclasses import dataclass

from cryptography.fernet import Fernet
from sqlalchemy.orm import sessionmaker
from storage.database import session_maker
from storage.stored_custom_secrets import StoredCustomSecrets

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.secrets.secrets_store import SecretsStore


@dataclass
class SaasSecretsStore(SecretsStore):
    user_id: str
    session_maker: sessionmaker
    config: OpenHandsConfig

    async def load(self) -> Secrets | None:
        if not self.user_id:
            return None

        with self.session_maker() as session:
            # Fetch all secrets for the given user ID
            settings = (
                session.query(StoredCustomSecrets)
                .filter(StoredCustomSecrets.keycloak_user_id == self.user_id)
                .all()
            )

            if not settings:
                return Secrets()

            kwargs = {}
            for secret in settings:
                kwargs[secret.secret_name] = {
                    'secret': secret.secret_value,
                    'description': secret.description,
                }

            self._decrypt_kwargs(kwargs)

            return Secrets(custom_secrets=kwargs)  # type: ignore[arg-type]

    async def store(self, item: Secrets):
        with self.session_maker() as session:
            # Incoming secrets are always the most updated ones
            # Delete all existing records and override with incoming ones
            session.query(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == self.user_id
            ).delete()

            # Prepare the new secrets data
            kwargs = item.model_dump(context={'expose_secrets': True})
            del kwargs[
                'provider_tokens'
            ]  # Assuming provider_tokens is not part of custom_secrets
            self._encrypt_kwargs(kwargs)

            secrets_json = kwargs.get('custom_secrets', {})

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
                    secret_name=secret_name,
                    secret_value=secret_value,
                    description=description,
                )
                session.add(new_secret)

            session.commit()

    def _decrypt_kwargs(self, kwargs: dict):
        fernet = self._fernet()
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self._decrypt_kwargs(value)
                continue

            if value is None:
                kwargs[key] = value
            else:
                value = fernet.decrypt(b64decode(value.encode())).decode()
                kwargs[key] = value

    def _encrypt_kwargs(self, kwargs: dict):
        fernet = self._fernet()
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self._encrypt_kwargs(value)
                continue

            if value is None:
                kwargs[key] = value
            else:
                encrypted_value = b64encode(fernet.encrypt(value.encode())).decode()
                kwargs[key] = encrypted_value

    def _fernet(self):
        if not self.config.jwt_secret:
            raise Exception('config.jwt_secret must be set')
        jwt_secret = self.config.jwt_secret.get_secret_value()
        fernet_key = b64encode(hashlib.sha256(jwt_secret.encode()).digest())
        return Fernet(fernet_key)

    @classmethod
    async def get_instance(
        cls,
        config: OpenHandsConfig,
        user_id: str | None,
    ) -> SaasSecretsStore:
        if not user_id:
            raise Exception('SaasSecretsStore cannot be constructed with no user_id')
        logger.debug(f'saas_secrets_store.get_instance::{user_id}')
        return SaasSecretsStore(user_id, session_maker, config)
