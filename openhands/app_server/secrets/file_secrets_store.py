from __future__ import annotations

import json
import secrets as secrets_module
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import (
    CredentialVersionConflict,
    SecretsStore,
)
from openhands.app_server.utils.async_utils import call_sync_from_async

_CODEX_AUTH_SECRET_NAME = 'CODEX_AUTH_JSON'
_CREDENTIAL_VERSIONS_KEY = '_credential_versions'


@dataclass(frozen=True)
class _LoadedCredential:
    value: str | None


@dataclass
class FileSecretsStore(SecretsStore):
    file_store: FileStore
    path: str = 'secrets.json'
    _loaded_codex_auth: _LoadedCredential | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def _read_data(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.file_store.read(self.path))
        except FileNotFoundError:
            return None
        if not isinstance(data, dict):
            raise ValueError('Invalid secrets file')
        return data

    @staticmethod
    def _entries(data: dict[str, Any], key: str) -> dict[str, Any]:
        entries = data.get(key)
        if entries is None:
            return {}
        if not isinstance(entries, dict):
            raise ValueError('Invalid secrets file')
        return entries

    @classmethod
    def _secrets(cls, data: dict[str, Any]) -> Secrets:
        provider_tokens = {
            name: value
            for name, value in cls._entries(data, 'provider_tokens').items()
            if isinstance(value, dict) and value.get('token')
        }
        kwargs = dict(data)
        kwargs['provider_tokens'] = provider_tokens
        kwargs['custom_secrets'] = cls._entries(data, 'custom_secrets')
        return Secrets(**kwargs)

    @staticmethod
    def _versions(data: dict[str, Any]) -> dict[str, Any]:
        versions = data.get(_CREDENTIAL_VERSIONS_KEY)
        if versions is None:
            return {}
        if not isinstance(versions, dict):
            raise ValueError('Invalid credential versions')
        return dict(versions)

    @classmethod
    def _raw_secret(
        cls,
        data: dict[str, Any],
        name: str,
    ) -> tuple[dict[str, Any], str]:
        current = cls._entries(data, 'custom_secrets').get(name)
        if not isinstance(current, dict):
            raise KeyError(name)
        value = current.get('secret')
        if not isinstance(value, str):
            raise KeyError(name)
        return current, value

    @staticmethod
    def _merge_entries(
        original: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        merged = {}
        for name, value in incoming.items():
            current = original.get(name)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[name] = {**current, **value}
            else:
                merged[name] = value
        return merged

    @staticmethod
    def _new_version(previous: str | None = None) -> str:
        while True:
            version = secrets_module.token_urlsafe(24)
            if version != previous:
                return version

    def _write_data(self, data: dict[str, Any]) -> None:
        self.file_store.write(self.path, json.dumps(data))

    async def load(self) -> Secrets | None:
        if not self.file_store.supports_locked_update:
            data = await call_sync_from_async(self._read_data)
            return self._secrets(data) if data is not None else None

        def load_locked() -> Secrets | None:
            data = self._read_data()
            if data is None:
                self._loaded_codex_auth = _LoadedCredential(None)
                return None
            loaded = self._secrets(data)
            current = loaded.custom_secrets.get(_CODEX_AUTH_SECRET_NAME)
            value = current.secret.get_secret_value() if current is not None else None
            self._loaded_codex_auth = _LoadedCredential(value)
            return loaded

        return await call_sync_from_async(
            self.file_store.locked_update,
            self.path,
            load_locked,
        )

    async def store(self, secrets: Secrets) -> None:
        if not self.file_store.supports_locked_update:
            json_str = secrets.model_dump_json(context={'expose_secrets': True})
            await call_sync_from_async(self.file_store.write, self.path, json_str)
            return

        serialized = secrets.model_dump(
            mode='json',
            context={'expose_secrets': True},
        )

        def store_locked() -> None:
            data = self._read_data() or {}
            original_provider_tokens = self._entries(data, 'provider_tokens')
            original_custom_secrets = self._entries(data, 'custom_secrets')
            incoming_provider_tokens = serialized['provider_tokens']
            incoming_custom_secrets = serialized['custom_secrets']
            versions = self._versions(data)

            submitted_info = incoming_custom_secrets.get(_CODEX_AUTH_SECRET_NAME)
            submitted_value = (
                submitted_info.get('secret')
                if isinstance(submitted_info, dict)
                else None
            )
            if not isinstance(submitted_value, str):
                submitted_value = None

            baseline = self._loaded_codex_auth
            preserve = baseline is None and submitted_info is None
            if baseline is not None and submitted_value == baseline.value:
                preserve = True

            updated_custom_secrets = self._merge_entries(
                original_custom_secrets,
                incoming_custom_secrets,
            )
            if preserve:
                try:
                    current_info, current_value = self._raw_secret(
                        data,
                        _CODEX_AUTH_SECRET_NAME,
                    )
                except KeyError:
                    updated_custom_secrets.pop(_CODEX_AUTH_SECRET_NAME, None)
                else:
                    if isinstance(submitted_info, dict):
                        current_info = {**current_info, **submitted_info}
                    updated_custom_secrets[_CODEX_AUTH_SECRET_NAME] = {
                        **current_info,
                        'secret': current_value,
                    }
            elif submitted_value is None:
                versions.pop(_CODEX_AUTH_SECRET_NAME, None)
            else:
                versions[_CODEX_AUTH_SECRET_NAME] = self._new_version(
                    versions.get(_CODEX_AUTH_SECRET_NAME)
                    if isinstance(versions.get(_CODEX_AUTH_SECRET_NAME), str)
                    else None
                )

            updated = dict(data)
            updated['provider_tokens'] = self._merge_entries(
                original_provider_tokens,
                incoming_provider_tokens,
            )
            updated['custom_secrets'] = updated_custom_secrets
            if versions or _CREDENTIAL_VERSIONS_KEY in data:
                updated[_CREDENTIAL_VERSIONS_KEY] = versions
            else:
                updated.pop(_CREDENTIAL_VERSIONS_KEY, None)
            self._write_data(updated)

            if not preserve:
                self._loaded_codex_auth = _LoadedCredential(submitted_value)

        await call_sync_from_async(
            self.file_store.locked_update,
            self.path,
            store_locked,
        )

    async def load_versioned(
        self,
        name: str,
        organization_id: UUID | None = None,
    ) -> tuple[str, str]:
        del organization_id
        if not self.file_store.supports_locked_update:
            raise NotImplementedError

        def load_locked() -> tuple[str, str]:
            data = self._read_data()
            if data is None:
                raise KeyError(name)
            _, value = self._raw_secret(data, name)
            versions = self._versions(data)
            version = versions.get(name)
            if not isinstance(version, str) or not version:
                version = self._new_version()
                versions[name] = version
                updated = dict(data)
                updated[_CREDENTIAL_VERSIONS_KEY] = versions
                self._write_data(updated)
            return value, version

        return await call_sync_from_async(
            self.file_store.locked_update,
            self.path,
            load_locked,
        )

    async def replace_versioned(
        self,
        name: str,
        expected_version: str,
        value: str,
        organization_id: UUID | None = None,
    ) -> str:
        del organization_id
        if not self.file_store.supports_locked_update:
            raise NotImplementedError

        def replace_locked() -> str:
            data = self._read_data()
            if data is None:
                raise KeyError(name)
            current, _ = self._raw_secret(data, name)
            versions = self._versions(data)
            version = versions.get(name)
            if (
                not isinstance(version, str)
                or not version
                or not secrets_module.compare_digest(
                    version.encode(),
                    expected_version.encode(errors='surrogatepass'),
                )
            ):
                raise CredentialVersionConflict

            successor = self._new_version(version)
            custom_secrets = dict(self._entries(data, 'custom_secrets'))
            custom_secrets[name] = {**current, 'secret': value}
            versions[name] = successor
            updated = dict(data)
            updated['custom_secrets'] = custom_secrets
            updated[_CREDENTIAL_VERSIONS_KEY] = versions
            self._write_data(updated)
            return successor

        return await call_sync_from_async(
            self.file_store.locked_update,
            self.path,
            replace_locked,
        )

    @classmethod
    async def get_instance(cls, user_id: str | None) -> FileSecretsStore:
        """Get a FileSecretsStore instance using the global config's file_store.

        TODO: This method should be replaced with dependency injection.
        """
        from openhands.app_server.config import get_global_config

        file_store = get_global_config().file_store
        return FileSecretsStore(file_store)
