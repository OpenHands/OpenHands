from __future__ import annotations

import errno
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
from openhands.app_server.secrets.credential_binding_models import (
    is_runtime_managed_credential,
)
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
_process_lock = threading.RLock()


def _supports_atomic_versioned_writes(file_store: FileStore) -> bool:
    return isinstance(file_store, InMemoryFileStore) or callable(
        getattr(file_store, 'get_full_path', None)
    )


@contextmanager
def _file_lock(file_store: FileStore, path: str) -> Iterator[None]:
    if isinstance(file_store, InMemoryFileStore):
        with _process_lock:
            yield
        return

    get_full_path = getattr(file_store, 'get_full_path', None)
    assert callable(get_full_path)
    lock_path = get_full_path(f'{path}.lock')
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
        elif msvcrt is not None:
            os.lseek(descriptor, 0, os.SEEK_SET)
            while True:
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
            locked = True
        yield
    finally:
        try:
            if locked and fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif locked and msvcrt is not None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        finally:
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
    """Track managed values last observed by this instance to preserve stale saves."""

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

    @staticmethod
    def _raw_secret(data: dict[str, Any], name: str) -> tuple[dict[str, Any], str]:
        custom_secrets = data.get('custom_secrets')
        if not isinstance(custom_secrets, dict):
            raise KeyError(name)
        current = custom_secrets.get(name)
        if not isinstance(current, dict):
            raise KeyError(name)
        value = current.get('secret')
        if not isinstance(value, str):
            raise KeyError(name)
        return current, value

    @staticmethod
    def _raw_versions(data: dict[str, Any]) -> dict[str, Any]:
        versions = data.get(_CREDENTIAL_VERSIONS_KEY)
        if versions is None:
            return {}
        if not isinstance(versions, dict):
            raise ValueError('Invalid credential versions')
        return dict(versions)

    def _write(
        self,
        secrets: Secrets,
        versions: dict[str, str],
        original: dict[str, Any],
    ) -> None:
        data = dict(original)
        data.update(
            secrets.model_dump(
                mode='json',
                context={'expose_secrets': True},
            )
        )
        if versions:
            data[_CREDENTIAL_VERSIONS_KEY] = versions
        else:
            data.pop(_CREDENTIAL_VERSIONS_KEY, None)
        self.file_store.write(self.path, json.dumps(data))

    async def load(self) -> Secrets | None:
        if not _supports_atomic_versioned_writes(self.file_store):
            data = await call_sync_from_async(self._read_data)
            return self._secrets(data) if data else None

        def load_locked() -> Secrets | None:
            with _file_lock(self.file_store, self.path):
                data = self._read_data()
                if not data:
                    return None
                secrets = self._secrets(data)
                versions = self._versions(data)
                managed_names = set(versions) | {
                    name
                    for name in secrets.custom_secrets
                    if is_runtime_managed_credential(name)
                }
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
        """Persist secrets; without a baseline, submitted managed values are edits."""
        if not _supports_atomic_versioned_writes(self.file_store):
            json_str = secrets.model_dump_json(context={'expose_secrets': True})
            await call_sync_from_async(self.file_store.write, self.path, json_str)
            return

        def store_locked() -> None:
            with _file_lock(self.file_store, self.path):
                try:
                    data = self._read_data()
                    current = self._secrets(data) if data else Secrets()
                except (AttributeError, TypeError, ValueError):
                    data = {}
                    current = Secrets()
                versions = self._versions(data)
                incoming = dict(secrets.custom_secrets)
                preserved_names = set()
                managed_names = (
                    set(versions)
                    | set(self._loaded_credentials)
                    | {
                        name
                        for name in current.custom_secrets
                        if is_runtime_managed_credential(name)
                    }
                )

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
                        preserved_names.add(name)
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
                for name in updated.custom_secrets:
                    if is_runtime_managed_credential(name) and name not in versions:
                        versions[name] = secrets_module.token_urlsafe(24)
                self._write(updated, versions, data)
                for name in managed_names | set(versions):
                    if name in preserved_names:
                        continue
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

        def load_current() -> tuple[str, str]:
            data = self._read_data()
            _, value = self._raw_secret(data, name)
            version = self._raw_versions(data).get(name)
            if not isinstance(version, str) or not version:
                with _file_lock(self.file_store, self.path):
                    data = self._read_data()
                    _, value = self._raw_secret(data, name)
                    versions = self._raw_versions(data)
                    version = versions.get(name)
                    if not isinstance(version, str) or not version:
                        version = secrets_module.token_urlsafe(24)
                        versions[name] = version
                        updated = dict(data)
                        updated[_CREDENTIAL_VERSIONS_KEY] = versions
                        self.file_store.write(self.path, json.dumps(updated))
            self._loaded_credentials[name] = (value, version)
            return value, version

        return await call_sync_from_async(load_current)

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
                current, _ = self._raw_secret(data, name)
                versions = self._raw_versions(data)
                if versions.get(name) != expected_version:
                    raise CredentialVersionConflict
                custom_secrets = dict(data['custom_secrets'])
                custom_secrets[name] = {**current, 'secret': value}
                successor = secrets_module.token_urlsafe(24)
                versions[name] = successor
                updated = {
                    **data,
                    'custom_secrets': custom_secrets,
                    _CREDENTIAL_VERSIONS_KEY: versions,
                }
                self.file_store.write(self.path, json.dumps(updated))
                self._loaded_credentials[name] = (value, successor)
                return successor

        return await call_sync_from_async(replace_locked)

    @classmethod
    async def get_instance(cls, user_id: str | None) -> FileSecretsStore:
        from openhands.app_server.config import get_global_config

        file_store = get_global_config().file_store
        return FileSecretsStore(file_store)
