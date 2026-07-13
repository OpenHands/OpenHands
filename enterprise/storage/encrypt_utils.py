import binascii
import hashlib
import json
from base64 import b64decode, b64encode
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, SecretStr
from sqlalchemy import String, TypeDecorator
from sqlalchemy.engine.interfaces import Dialect

_jwt_service = None
_fernet = None
_settings_cipher = None
_FERNET_TOKEN_PREFIX = 'gAAAA'


class _SettingsReadCipher:
    def __init__(self, cipher):
        self._cipher = cipher

    def encrypt(self, secret: SecretStr | None) -> str | None:
        return self._cipher.encrypt(secret)

    def decrypt(self, secret: str | None) -> SecretStr | None:
        if secret is None:
            return None
        if not secret.startswith(_FERNET_TOKEN_PREFIX):
            return SecretStr(secret)
        decrypted = self._cipher.try_decrypt_str(secret)
        return SecretStr(decrypted if decrypted is not None else secret)

    def try_decrypt_str(self, secret: str) -> str | None:
        return self._cipher.try_decrypt_str(secret)


def encrypt_value(value: str | SecretStr) -> str:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    return get_jwt_service().encrypt_value(raw)


def decrypt_value(value: str | SecretStr) -> str:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    return get_jwt_service().decrypt_value(raw)


def get_jwt_service():
    from openhands.app_server.config import get_global_config

    global _jwt_service
    if _jwt_service is None:
        jwt_service_injector = get_global_config().jwt
        assert jwt_service_injector is not None
        _jwt_service = jwt_service_injector.get_jwt_service()
    return _jwt_service


def get_settings_cipher():
    """Return the SDK cipher used for field-level settings encryption."""
    from openhands.sdk.utils.cipher import Cipher

    global _settings_cipher
    if _settings_cipher is None:
        jwt_svc = get_jwt_service()
        default_key = jwt_svc.get_key(jwt_svc._default_key_id)
        _settings_cipher = Cipher(default_key.key.get_secret_value())
    return _settings_cipher


def get_settings_cipher_context() -> dict[str, Any]:
    return {'cipher': _SettingsReadCipher(get_settings_cipher())}


def get_settings_storage_context() -> dict[str, Any]:
    return {'expose_secrets': 'encrypted', 'cipher': get_settings_cipher()}


def decrypt_legacy_model(decrypt_keys: list, model_instance) -> dict:
    return decrypt_legacy_kwargs(decrypt_keys, model_to_kwargs(model_instance))


def decrypt_legacy_kwargs(encrypt_keys: list, kwargs: dict) -> dict:
    for key, value in kwargs.items():
        try:
            if value is None:
                continue
            if key in encrypt_keys:
                value = decrypt_legacy_value(value)
                kwargs[key] = value
        except binascii.Error:
            pass  # Key is in legacy format...
        except InvalidToken:
            pass  # Key not encrypted...
    return kwargs


def decrypt_legacy_value(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        return (
            get_fernet().decrypt(b64decode(value.get_secret_value().encode())).decode()
        )
    else:
        return get_fernet().decrypt(b64decode(value.encode())).decode()


def encrypt_legacy_value(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        return b64encode(
            get_fernet().encrypt(value.get_secret_value().encode())
        ).decode()
    else:
        return b64encode(get_fernet().encrypt(value.encode())).decode()


def get_fernet():
    global _fernet
    if _fernet is None:
        jwt_svc = get_jwt_service()
        default_key = jwt_svc.get_key(jwt_svc._default_key_id)
        secret = default_key.key.get_secret_value()
        fernet_key = b64encode(hashlib.sha256(secret.encode()).digest())
        _fernet = Fernet(fernet_key)
    return _fernet


def model_to_kwargs(model_instance):
    return {
        column.name: getattr(model_instance, column.name)
        for column in model_instance.__table__.columns
    }


class EncryptedJSON(TypeDecorator[dict[str, Any]]):
    """JSON column whose serialized payload is encrypted at rest."""

    impl = String
    cache_ok = True

    def process_bind_param(
        self, value: BaseModel | dict[str, Any] | None, dialect: Dialect
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            value = value.model_dump(mode='json', context={'expose_secrets': True})
        return encrypt_value(json.dumps(value))

    def process_result_value(
        self, value: str | None, dialect: Dialect
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return json.loads(decrypt_value(value))


class SecretAwareJSON(TypeDecorator[dict[str, Any]]):
    """JSON string column that encrypts nested Pydantic secret fields only.

    The database value remains parseable JSON for non-secret operational data,
    while SDK serializers encrypt fields such as LLM ``api_key`` values.
    """

    impl = String
    cache_ok = True

    def process_bind_param(
        self, value: BaseModel | dict[str, Any] | None, dialect: Dialect
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            value = value.model_dump(
                mode='json', context=get_settings_storage_context()
            )
        else:
            value = _encrypt_known_settings_payload(value)
        return json.dumps(value)

    def process_result_value(
        self, value: str | None, dialect: Dialect
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return json.loads(value)


def _encrypt_known_settings_payload(value: dict[str, Any]) -> dict[str, Any]:
    if 'profiles' in value:
        from openhands.app_server.settings.llm_profiles import LLMProfiles

        return LLMProfiles.model_validate(
            value, context=get_settings_cipher_context()
        ).model_dump(mode='json', context=get_settings_storage_context())
    return value
