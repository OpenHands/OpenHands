from __future__ import annotations

import json
import secrets as secrets_module
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.secrets.credential_binding_models import (
    is_runtime_managed_credential,
)
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import (
    CredentialVersionConflict,
    SecretsStore,
)
from openhands.app_server.utils.async_utils import call_sync_from_async

_CREDENTIAL_VERSIONS_KEY = '_credential_versions'


@dataclass(frozen=True)
class Unloaded:
    pass


@dataclass(frozen=True)
class Loaded:
    value: str | None
    version: str | None


ManagedBaseline = Unloaded | Loaded
_UNLOADED = Unloaded()


class _ManagedSaveDecision(Enum):
    PRESERVE = 'preserve'
    EDIT = 'edit'
    DELETE = 'delete'


def _managed_save_decision(
    baseline: ManagedBaseline,
    submitted_value: str | None,
) -> _ManagedSaveDecision:
    if isinstance(baseline, Unloaded):
        return (
            _ManagedSaveDecision.PRESERVE
            if submitted_value is None
            else _ManagedSaveDecision.EDIT
        )
    if submitted_value == baseline.value:
        return _ManagedSaveDecision.PRESERVE
    return (
        _ManagedSaveDecision.DELETE
        if submitted_value is None
        else _ManagedSaveDecision.EDIT
    )


@dataclass
class FileSecretsStore(SecretsStore):
    file_store: FileStore
    path: str = 'secrets.json'
    _managed_baselines: dict[str, ManagedBaseline] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

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
        raw_provider_tokens = data.get('provider_tokens', {})
        raw_custom_secrets = data.get('custom_secrets', {})
        if raw_provider_tokens is None:
            raw_provider_tokens = {}
        if raw_custom_secrets is None:
            raw_custom_secrets = {}
        if not isinstance(raw_provider_tokens, dict) or not isinstance(
            raw_custom_secrets, dict
        ):
            raise ValueError('Invalid secrets file')
        provider_tokens = {
            name: value
            for name, value in raw_provider_tokens.items()
            if value.get('token')
        }
        return Secrets(
            provider_tokens=provider_tokens,
            custom_secrets=raw_custom_secrets,
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
        raw_versions = original.get(_CREDENTIAL_VERSIONS_KEY)
        merged_versions = dict(raw_versions) if isinstance(raw_versions, dict) else {}
        for name, version in tuple(merged_versions.items()):
            if isinstance(version, str) and version and name not in versions:
                merged_versions.pop(name)
        merged_versions.update(versions)
        if merged_versions:
            data[_CREDENTIAL_VERSIONS_KEY] = merged_versions
        else:
            data.pop(_CREDENTIAL_VERSIONS_KEY, None)
        self.file_store.write(self.path, json.dumps(data))

    async def load(self) -> Secrets | None:
        if not self.file_store.supports_locked_update:
            data = await call_sync_from_async(self._read_data)
            return self._secrets(data) if data else None

        def load_locked() -> Secrets | None:
            data = self._read_data()
            if not data:
                self._managed_baselines = {}
                return None
            secrets = self._secrets(data)
            versions = self._versions(data)
            managed_names = set(versions) | {
                name
                for name in secrets.custom_secrets
                if is_runtime_managed_credential(name)
            }
            baselines: dict[str, ManagedBaseline] = {}
            for name in managed_names:
                current = secrets.custom_secrets.get(name)
                value = (
                    current.secret.get_secret_value() if current is not None else None
                )
                baselines[name] = Loaded(value, versions.get(name))
            self._managed_baselines = baselines
            return secrets

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

        def store_locked() -> None:
            data = self._read_data()
            current = self._secrets(data) if data else Secrets()
            versions = self._versions(data)
            incoming = dict(secrets.custom_secrets)
            preserved_names = set()
            managed_names = (
                set(versions)
                | set(self._managed_baselines)
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
                decision = _managed_save_decision(
                    self._managed_baselines.get(name, _UNLOADED),
                    submitted_value,
                )
                current_secret = current.custom_secrets.get(name)
                current_value = (
                    current_secret.secret.get_secret_value()
                    if current_secret is not None
                    else None
                )
                if decision == _ManagedSaveDecision.PRESERVE:
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
                    if decision == _ManagedSaveDecision.DELETE:
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
                value = stored.secret.get_secret_value() if stored is not None else None
                self._managed_baselines[name] = Loaded(value, versions.get(name))

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

        def load_current() -> tuple[str, str]:
            data = self._read_data()
            _, value = self._raw_secret(data, name)
            version = self._raw_versions(data).get(name)
            if not isinstance(version, str) or not version:

                def bootstrap_version() -> tuple[str, str]:
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
                    return value, version

                return self.file_store.locked_update(self.path, bootstrap_version)
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
        if not self.file_store.supports_locked_update:
            raise NotImplementedError

        def replace_locked() -> str:
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
            return successor

        return await call_sync_from_async(
            self.file_store.locked_update,
            self.path,
            replace_locked,
        )

    @classmethod
    async def get_instance(cls, user_id: str | None) -> FileSecretsStore:
        from openhands.app_server.config import get_global_config

        file_store = get_global_config().file_store
        return FileSecretsStore(file_store)
