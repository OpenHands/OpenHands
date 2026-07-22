from __future__ import annotations

import importlib
import json
import os
import secrets as secrets_module
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.file_store.memory import InMemoryFileStore
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import (
    CredentialVersionConflict,
    SecretsStore,
)
from openhands.app_server.utils.async_utils import call_sync_from_async

fcntl: Any = None
msvcrt: Any = None
if sys.platform == 'win32':
    msvcrt = importlib.import_module('msvcrt')
else:
    fcntl = importlib.import_module('fcntl')


_CREDENTIAL_VERSIONS_KEY = '_credential_versions'
_CODEX_AUTH_SECRET_NAME = 'CODEX_AUTH_JSON'
_process_lock = threading.RLock()


def _supports_atomic_versioned_writes(file_store: FileStore) -> bool:
    return isinstance(file_store, InMemoryFileStore) or callable(
        getattr(file_store, 'get_full_path', None)
    )


@contextmanager
def _file_lock(file_store: FileStore, path: str) -> Iterator[None]:
    get_full_path = getattr(file_store, 'get_full_path', None)
    if not callable(get_full_path):
        with _process_lock:
            yield
        return

    lock_path = get_full_path(f'{path}.lock')
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        elif msvcrt is not None:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        elif msvcrt is not None:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)


@dataclass
class FileSecretsStore(SecretsStore):
    file_store: FileStore
    path: str = 'secrets.json'
    _loaded_credentials: dict[str, tuple[str | None, str | None]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def supports_versioned_credentials(self) -> bool:
        return _supports_atomic_versioned_writes(self.file_store)

    def _read_data(self) -> dict[str, Any]:
        try:
            data = json.loads(self.file_store.read(self.path))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise ValueError('Invalid secrets file')
        return data

    @staticmethod
    def _versions(data: dict[str, Any]) -> dict[str, str]:
        versions = data.get(_CREDENTIAL_VERSIONS_KEY)
        if not isinstance(versions, dict):
            return {}
        return {
            name: version
            for name, version in versions.items()
            if isinstance(name, str) and isinstance(version, str) and version
        }

    @staticmethod
    def _secrets(data: dict[str, Any]) -> Secrets:
        provider_tokens = {
            name: value
            for name, value in (data.get('provider_tokens') or {}).items()
            if value.get('token')
        }
        return Secrets(
            provider_tokens=provider_tokens,
            custom_secrets=data.get('custom_secrets') or {},
        )

    def _write(self, secrets: Secrets, versions: dict[str, str]) -> None:
        data = secrets.model_dump(
            mode='json',
            context={'expose_secrets': True},
        )
        if versions:
            data[_CREDENTIAL_VERSIONS_KEY] = versions
        self.file_store.write(self.path, json.dumps(data))

    async def load(self) -> Secrets | None:
        def load_locked() -> Secrets | None:
            with _file_lock(self.file_store, self.path):
                data = self._read_data()
                if not data:
                    return None
                secrets = self._secrets(data)
                versions = self._versions(data)
                managed_names = set(versions)
                if _CODEX_AUTH_SECRET_NAME in secrets.custom_secrets:
                    managed_names.add(_CODEX_AUTH_SECRET_NAME)
                for name in managed_names:
                    current = secrets.custom_secrets.get(name)
                    value = (
                        current.secret.get_secret_value()
                        if current is not None
                        else None
                    )
                    self._loaded_credentials[name] = (value, versions.get(name))
                return secrets

        return await call_sync_from_async(load_locked)

    async def store(self, secrets: Secrets) -> None:
        def store_locked() -> None:
            with _file_lock(self.file_store, self.path):
                data = self._read_data()
                current = self._secrets(data) if data else Secrets()
                versions = self._versions(data)
                incoming = dict(secrets.custom_secrets)
                managed_names = set(versions) | set(self._loaded_credentials)
                if _CODEX_AUTH_SECRET_NAME in current.custom_secrets:
                    managed_names.add(_CODEX_AUTH_SECRET_NAME)

                for name in managed_names:
                    submitted = incoming.get(name)
                    submitted_value = (
                        submitted.secret.get_secret_value()
                        if submitted is not None
                        else None
                    )
                    loaded = self._loaded_credentials.get(name)
                    preserve = loaded is None and submitted is None
                    if loaded is not None and submitted_value == loaded[0]:
                        preserve = True
                    current_secret = current.custom_secrets.get(name)
                    current_value = (
                        current_secret.secret.get_secret_value()
                        if current_secret is not None
                        else None
                    )
                    if preserve:
                        if current_secret is None:
                            incoming.pop(name, None)
                        elif submitted is None:
                            incoming[name] = current_secret
                        else:
                            incoming[name] = submitted.model_copy(
                                update={'secret': current_secret.secret}
                            )
                    elif submitted_value != current_value:
                        if submitted is None:
                            versions.pop(name, None)
                        else:
                            versions[name] = secrets_module.token_urlsafe(24)

                updated = secrets.model_copy(update={'custom_secrets': incoming})
                self._write(updated, versions)
                for name in managed_names | set(versions):
                    stored = updated.custom_secrets.get(name)
                    value = (
                        stored.secret.get_secret_value() if stored is not None else None
                    )
                    self._loaded_credentials[name] = (value, versions.get(name))

        await call_sync_from_async(store_locked)

    async def load_versioned(
        self,
        name: str,
        organization_id: UUID | None = None,
    ) -> tuple[str, str]:
        del organization_id
        if not _supports_atomic_versioned_writes(self.file_store):
            raise NotImplementedError

        def load_locked() -> tuple[str, str]:
            with _file_lock(self.file_store, self.path):
                data = self._read_data()
                secrets = self._secrets(data)
                current = secrets.custom_secrets.get(name)
                if current is None:
                    raise KeyError(name)
                versions = self._versions(data)
                version = versions.get(name)
                if version is None:
                    version = secrets_module.token_urlsafe(24)
                    versions[name] = version
                    self._write(secrets, versions)
                value = current.secret.get_secret_value()
                self._loaded_credentials[name] = (value, version)
                return value, version

        return await call_sync_from_async(load_locked)

    async def replace_versioned(
        self,
        name: str,
        expected_version: str,
        value: str,
        organization_id: UUID | None = None,
    ) -> str:
        del organization_id
        if not _supports_atomic_versioned_writes(self.file_store):
            raise NotImplementedError

        def replace_locked() -> str:
            with _file_lock(self.file_store, self.path):
                data = self._read_data()
                secrets = self._secrets(data)
                current = secrets.custom_secrets.get(name)
                if current is None:
                    raise KeyError(name)
                versions = self._versions(data)
                if versions.get(name) != expected_version:
                    raise CredentialVersionConflict
                custom_secrets = dict(secrets.custom_secrets)
                custom_secrets[name] = current.model_copy(
                    update={'secret': type(current.secret)(value)}
                )
                successor = secrets_module.token_urlsafe(24)
                versions[name] = successor
                updated = secrets.model_copy(update={'custom_secrets': custom_secrets})
                self._write(updated, versions)
                self._loaded_credentials[name] = (value, successor)
                return successor

        return await call_sync_from_async(replace_locked)

    @classmethod
    async def get_instance(cls, user_id: str | None) -> FileSecretsStore:
        from openhands.app_server.config import get_global_config

        file_store = get_global_config().file_store
        return FileSecretsStore(file_store)
